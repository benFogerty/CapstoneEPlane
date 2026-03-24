from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_workspace.soh_forecast.common import ModelArtifacts, SplitFrames, TargetSpec, make_feature_frame, metric_table


@dataclass
class PhysicsHybridConfig:
    hidden_dim: int = 64
    physics_hidden_dim: int = 48
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 10
    device: str = "cpu"
    stress_col: str = "rolling_stress_index_mean_5"
    residual_scale: float = 0.05
    degradation_weight: float = 0.35
    residual_weight: float = 0.02
    smoothness_weight: float = 0.03
    interaction_scale: float = 0.02
    smoothness_gap_days: float = 14.0
    context_feature_candidates: tuple[str, ...] = (
        "observed_soh_pct",
        "latent_soh_filter_pct",
        "latent_soh_filter_std_pct",
        "measurement_sigma_pct",
        "condition_multiplier",
        "current_abs_mean_a",
        "p95_abs_current_a",
        "avg_cell_temp_mean_c",
        "avg_cell_temp_span_c",
        "soc_mean_pct",
        "soc_span_pct",
        "event_duration_s",
        "delta_days",
        "rolling_condition_mean_5",
        "rolling_temp_mean_5",
        "rolling_soc_mean_5",
        "rolling_duration_mean_5",
        "rolling_sigma_mean_5",
        "rolling_gap_days_mean_5",
        "rolling_flight_frac_5",
        "prev_observed_soh_pct",
        "prev_latent_filter_pct",
        "observed_soh_delta_1",
        "latent_filter_delta_1",
        "rolling_observed_delta_mean_5",
        "rolling_latent_filter_delta_mean_5",
        "current_temp_stress_index",
        "soc_stress_index",
        "duration_stress_index",
        "rolling_stress_index_mean_5",
        "rolling_stress_index_max_5",
        "cumulative_efc",
        "cumulative_ah",
    )


@dataclass
class PhysicsTensorBundle:
    features: torch.Tensor
    physics_context: torch.Tensor
    drives: torch.Tensor
    current: torch.Tensor
    target: torch.Tensor
    delta_days: torch.Tensor
    battery_index: torch.Tensor
    event_ids: np.ndarray
    splits: np.ndarray


class PhysicsHybridForecastNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        n_drives: int,
        hidden_dim: int = 64,
        physics_hidden_dim: int = 48,
        residual_scale: float = 0.05,
        interaction_scale: float = 0.15,
    ):
        super().__init__()
        self.n_drives = n_drives
        self.residual_scale = residual_scale
        self.interaction_scale = interaction_scale

        self.feature_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.physics_encoder = nn.Sequential(
            nn.Linear(context_dim, physics_hidden_dim),
            nn.Tanh(),
            nn.Linear(physics_hidden_dim, physics_hidden_dim),
            nn.Tanh(),
        )
        self.condition_gate = nn.Sequential(
            nn.Linear(hidden_dim + physics_hidden_dim, physics_hidden_dim),
            nn.ReLU(),
            nn.Linear(physics_hidden_dim, n_drives),
        )
        self.parameter_head = nn.Sequential(
            nn.Linear(physics_hidden_dim, physics_hidden_dim),
            nn.ReLU(),
            nn.Linear(physics_hidden_dim, n_drives + 2),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim + physics_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        physics_context: torch.Tensor,
        drives: torch.Tensor,
        current_soh: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feature_hidden = self.feature_backbone(x)
        physics_hidden = self.physics_encoder(physics_context)
        merged_hidden = torch.cat([feature_hidden, physics_hidden], dim=1)

        raw_params = self.parameter_head(physics_hidden)
        params = F.softplus(raw_params - 4.0) + 1e-4
        sensitivities = params[:, : self.n_drives]
        interaction_gain = params[:, self.n_drives]
        aging_gain = params[:, self.n_drives + 1].unsqueeze(-1)

        health_gap = torch.clamp((100.0 - current_soh) / 100.0, min=0.0).unsqueeze(-1)
        gates = 0.75 + 0.5 * torch.sigmoid(self.condition_gate(merged_hidden))
        effective_sensitivities = sensitivities * (1.0 + aging_gain * health_gap)
        gated_drives = torch.clamp(drives, min=0.0) * gates

        base_degradation = torch.sum(effective_sensitivities * gated_drives, dim=1)
        interaction_drive = (
            gated_drives[:, 0] * gated_drives[:, 1]
            + gated_drives[:, 1] * gated_drives[:, 2]
            + gated_drives[:, 2] * gated_drives[:, 3]
            + gated_drives[:, 0] * gated_drives[:, 4]
        )
        interaction_degradation = self.interaction_scale * interaction_gain * interaction_drive * (1.0 + health_gap.squeeze(-1))

        residual_degradation = self.residual_scale * F.softplus(self.residual_head(merged_hidden).squeeze(-1))
        total_degradation = base_degradation + interaction_degradation + residual_degradation
        next_soh = torch.clamp(current_soh - total_degradation, min=0.0, max=100.0)

        return {
            "next_soh": next_soh,
            "total_degradation": total_degradation,
            "base_degradation": base_degradation,
            "interaction_degradation": interaction_degradation,
            "residual_degradation": residual_degradation,
            "gates": gates,
            "sensitivities": effective_sensitivities,
            "health_gap": health_gap.squeeze(-1),
        }


def _sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sort_cols = [col for col in ["plane_id", "battery_id", "event_datetime", "flight_id"] if col in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True).copy()


def _zero_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index, dtype=float)


def _coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return _zero_series(df)
    return pd.to_numeric(df[col], errors="coerce")


def _first_available(df: pd.DataFrame, candidates: tuple[str, ...], default: float = 0.0, absolute: bool = False) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").fillna(default)
            return series.abs() if absolute else series
    return pd.Series(default, index=df.index, dtype=float)


def _build_drive_frame(df: pd.DataFrame, config: PhysicsHybridConfig) -> pd.DataFrame:
    delta_days = _coerce_numeric(df, "delta_days").fillna(0.0).clip(lower=0.0)
    condition = _first_available(df, ("condition_multiplier", "rolling_condition_mean_5"), default=0.0).clip(lower=0.0)
    storage = _first_available(df, ("storage_stress_proxy",), default=0.0).clip(lower=0.0)
    throughput = _first_available(df, ("throughput_stress_proxy",), default=0.0).clip(lower=0.0)
    current_rms = _first_available(df, ("current_rms_proxy_a", "p95_abs_current_a", "current_abs_mean_a"), default=0.0).clip(lower=0.0)
    duration_min = _coerce_numeric(df, "event_duration_s").fillna(0.0).clip(lower=0.0) / 60.0
    soc_span = _first_available(df, ("soc_span_pct", "rolling_soc_span_mean_5"), default=0.0).clip(lower=0.0)
    arrhenius = _first_available(df, ("arrhenius_temp_proxy",), default=0.0).clip(lower=0.0)
    thermal = _first_available(df, ("thermal_severity_proxy",), default=0.0).clip(lower=0.0)
    time_above_40 = _first_available(df, ("time_above_40c_proxy_min",), default=0.0).clip(lower=0.0)
    current_temp = _first_available(df, ("current_temp_stress_index",), default=0.0).clip(lower=0.0)
    resistance = _first_available(df, ("internal_resistance_proxy_ohm",), default=0.0).clip(lower=0.0)
    voltage_sag = _first_available(df, ("voltage_sag_proxy_v",), default=0.0).clip(lower=0.0)
    coulomb_gap = _first_available(df, ("coulomb_gap_abs_pct", "kalman_coulomb_gap_mean_pct"), default=0.0, absolute=True).clip(lower=0.0)
    reset_risk = _first_available(df, ("estimation_reset_risk", "measurement_sigma_pct"), default=0.0).clip(lower=0.0)
    degradation_stress = _first_available(df, ("degradation_stress_proxy", config.stress_col), default=0.0).clip(lower=0.0)
    rolling_stress = _first_available(df, ("rolling_stress_index_mean_5", config.stress_col), default=0.0).clip(lower=0.0)
    cumulative_efc = _first_available(df, ("cumulative_efc", "cumulative_cycles"), default=0.0).clip(lower=0.0)
    cumulative_ah = _first_available(df, ("cumulative_ah",), default=0.0).clip(lower=0.0)

    fallback_cycle = current_rms * duration_min * (1.0 + soc_span / 100.0)
    drive_frame = pd.DataFrame(
        {
            "calendar_drive": np.log1p(delta_days * (1.0 + 0.15 * condition) + 0.10 * storage),
            "cycle_drive": np.log1p(throughput + 0.25 * fallback_cycle),
            "thermal_drive": np.log1p(arrhenius + thermal + time_above_40 + 0.02 * current_temp),
            "resistance_drive": np.log1p(1000.0 * resistance + 10.0 * voltage_sag + coulomb_gap + reset_risk),
            "history_drive": np.log1p(degradation_stress + rolling_stress + cumulative_efc + 0.02 * cumulative_ah),
        },
        index=df.index,
    )
    return drive_frame.fillna(0.0).clip(lower=0.0)


def _build_context_frame(df: pd.DataFrame, config: PhysicsHybridConfig) -> tuple[pd.DataFrame, list[str]]:
    cols = [col for col in config.context_feature_candidates if col in df.columns]
    if not cols:
        return pd.DataFrame({"physics_context_bias": np.zeros(len(df), dtype=float)}, index=df.index), ["physics_context_bias"]
    context = df[cols].apply(pd.to_numeric, errors="coerce")
    return context, cols


def _scale_context_frames(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    config: PhysicsHybridConfig,
):
    train_ctx, context_cols = _build_context_frame(train_df, config)
    train_medians = train_ctx.median().fillna(0.0)
    train_ctx = train_ctx.fillna(train_medians).fillna(0.0)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        ctx, _ = _build_context_frame(frame, config)
        for col in train_ctx.columns:
            if col not in ctx.columns:
                ctx[col] = np.nan
        return ctx[train_ctx.columns].fillna(train_medians).fillna(0.0)

    valid_ctx = transform(valid_df)
    test_ctx = transform(test_df)
    holdout_ctx = transform(holdout_df) if not holdout_df.empty else pd.DataFrame(columns=train_ctx.columns)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_ctx)
    valid_scaled = scaler.transform(valid_ctx)
    test_scaled = scaler.transform(test_ctx) if not test_df.empty else np.empty((0, train_ctx.shape[1]), dtype=float)
    holdout_scaled = scaler.transform(holdout_ctx) if not holdout_df.empty else np.empty((0, train_ctx.shape[1]), dtype=float)

    return train_scaled, valid_scaled, test_scaled, holdout_scaled, context_cols, scaler, train_medians


def _make_bundle(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    physics_context_scaled: np.ndarray,
    target_spec: TargetSpec,
    config: PhysicsHybridConfig,
) -> PhysicsTensorBundle:
    if df.empty:
        return PhysicsTensorBundle(
            features=torch.empty((0, X_scaled.shape[1]), dtype=torch.float32),
            physics_context=torch.empty((0, physics_context_scaled.shape[1]), dtype=torch.float32),
            drives=torch.empty((0, 5), dtype=torch.float32),
            current=torch.empty(0, dtype=torch.float32),
            target=torch.empty(0, dtype=torch.float32),
            delta_days=torch.empty(0, dtype=torch.float32),
            battery_index=torch.empty(0, dtype=torch.long),
            event_ids=np.array([], dtype=object),
            splits=np.array([], dtype=object),
        )

    drives_df = _build_drive_frame(df, config)
    battery_keys = (
        df["plane_id"].astype(str) + "__" + df["battery_id"].astype("Int64").astype(str)
        if "plane_id" in df.columns and "battery_id" in df.columns
        else pd.Series(np.arange(len(df), dtype=int), index=df.index).astype(str)
    )
    battery_codes, _ = pd.factorize(battery_keys, sort=False)
    delta_days = _coerce_numeric(df, "delta_days").fillna(0.0).clip(lower=0.0)
    return PhysicsTensorBundle(
        features=torch.tensor(X_scaled, dtype=torch.float32),
        physics_context=torch.tensor(physics_context_scaled, dtype=torch.float32),
        drives=torch.tensor(drives_df.to_numpy(dtype=float), dtype=torch.float32),
        current=torch.tensor(df[target_spec.current_col].to_numpy(dtype=float), dtype=torch.float32),
        target=torch.tensor(df[target_spec.next_col].to_numpy(dtype=float), dtype=torch.float32),
        delta_days=torch.tensor(delta_days.to_numpy(dtype=float), dtype=torch.float32),
        battery_index=torch.tensor(battery_codes.astype(np.int64), dtype=torch.long),
        event_ids=df["event_id"].to_numpy(),
        splits=df["split"].astype(str).to_numpy(),
    )


def _bundle_to_loader(bundle: PhysicsTensorBundle, batch_size: int, shuffle: bool) -> DataLoader | None:
    if bundle.features.numel() == 0:
        return None
    dataset = TensorDataset(bundle.features, bundle.physics_context, bundle.drives, bundle.current, bundle.target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _smoothness_loss(outputs: dict[str, torch.Tensor], bundle: PhysicsTensorBundle, config: PhysicsHybridConfig) -> torch.Tensor:
    if bundle.features.shape[0] < 2:
        return torch.tensor(0.0, device=outputs["next_soh"].device)
    same_battery = bundle.battery_index[1:] == bundle.battery_index[:-1]
    if not torch.any(same_battery):
        return torch.tensor(0.0, device=outputs["next_soh"].device)
    smoothness_scale = torch.exp(-bundle.delta_days[1:].to(outputs["next_soh"].device) / max(config.smoothness_gap_days, 1e-3))
    param_diff = outputs["sensitivities"][1:] - outputs["sensitivities"][:-1]
    param_penalty = torch.mean(param_diff.pow(2), dim=1)
    weighted = param_penalty * smoothness_scale
    return weighted[same_battery.to(outputs["next_soh"].device)].mean()


def _predict(bundle: PhysicsTensorBundle, model: nn.Module, device: torch.device, model_name: str) -> pd.DataFrame:
    if bundle.features.numel() == 0:
        return pd.DataFrame(columns=["event_id", "split", model_name])
    model.eval()
    with torch.no_grad():
        outputs = model(
            bundle.features.to(device),
            bundle.physics_context.to(device),
            bundle.drives.to(device),
            bundle.current.to(device),
        )
    return pd.DataFrame({"event_id": bundle.event_ids, "split": bundle.splits, model_name: outputs["next_soh"].cpu().numpy()})


def train_physics_hybrid_nn(
    split_frames: SplitFrames,
    target_spec: TargetSpec,
    feature_cols: list[str],
    model_name: str = "physics_hybrid_nn",
    config: PhysicsHybridConfig | None = None,
) -> ModelArtifacts:
    config = config or PhysicsHybridConfig()
    device = torch.device(config.device)

    train_df = _sort_frame(split_frames.train)
    valid_df = _sort_frame(split_frames.valid)
    test_df = _sort_frame(split_frames.test)
    holdout_df = _sort_frame(split_frames.holdout)

    train_x, medians, dummy_cols = make_feature_frame(train_df, feature_cols)
    valid_x, _, _ = make_feature_frame(valid_df, feature_cols, medians, dummy_cols)
    test_x, _, _ = make_feature_frame(test_df, feature_cols, medians, dummy_cols)
    holdout_x, _, _ = (
        make_feature_frame(holdout_df, feature_cols, medians, dummy_cols)
        if not holdout_df.empty
        else (pd.DataFrame(columns=train_x.columns), medians, dummy_cols)
    )

    feature_scaler = StandardScaler()
    train_x_s = feature_scaler.fit_transform(train_x)
    valid_x_s = feature_scaler.transform(valid_x)
    test_x_s = feature_scaler.transform(test_x) if not test_df.empty else np.empty((0, train_x.shape[1]), dtype=float)
    holdout_x_s = feature_scaler.transform(holdout_x) if not holdout_df.empty else np.empty((0, train_x.shape[1]), dtype=float)

    (
        train_ctx_s,
        valid_ctx_s,
        test_ctx_s,
        holdout_ctx_s,
        context_cols,
        context_scaler,
        context_medians,
    ) = _scale_context_frames(train_df, valid_df, test_df, holdout_df, config)

    train_bundle = _make_bundle(train_df, train_x_s, train_ctx_s, target_spec, config)
    valid_bundle = _make_bundle(valid_df, valid_x_s, valid_ctx_s, target_spec, config)
    test_bundle = _make_bundle(test_df, test_x_s, test_ctx_s, target_spec, config)
    holdout_bundle = _make_bundle(holdout_df, holdout_x_s, holdout_ctx_s, target_spec, config)

    train_loader = _bundle_to_loader(train_bundle, config.batch_size, True)
    if train_loader is None or valid_bundle.features.numel() == 0:
        return ModelArtifacts(model_name=model_name, predictions=pd.DataFrame(columns=["event_id", "split", model_name]), metrics=pd.DataFrame())

    model = PhysicsHybridForecastNet(
        input_dim=train_x.shape[1],
        context_dim=train_ctx_s.shape[1],
        n_drives=train_bundle.drives.shape[1],
        hidden_dim=config.hidden_dim,
        physics_hidden_dim=config.physics_hidden_dim,
        residual_scale=config.residual_scale,
        interaction_scale=config.interaction_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: list[dict[str, float]] = []
    best_state = None
    best_valid_mae = np.inf
    epochs_no_improve = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_parts: list[dict[str, float]] = []
        for xb, physics_context, drives, current, target in train_loader:
            optimizer.zero_grad()
            outputs = model(xb.to(device), physics_context.to(device), drives.to(device), current.to(device))
            target_deg = torch.relu(current.to(device) - target.to(device))
            data_loss = F.mse_loss(outputs["next_soh"], target.to(device))
            degradation_loss = F.mse_loss(outputs["total_degradation"], target_deg)
            residual_loss = outputs["residual_degradation"].pow(2).mean()
            loss = data_loss + config.degradation_weight * degradation_loss + config.residual_weight * residual_loss
            loss.backward()
            optimizer.step()
            train_loss_parts.append(
                {
                    "loss": float(loss.item()),
                    "data_loss": float(data_loss.item()),
                    "degradation_loss": float(degradation_loss.item()),
                    "residual_loss": float(residual_loss.item()),
                }
            )

        model.eval()
        with torch.no_grad():
            train_outputs = model(
                train_bundle.features.to(device),
                train_bundle.physics_context.to(device),
                train_bundle.drives.to(device),
                train_bundle.current.to(device),
            )
            smoothness_loss = _smoothness_loss(train_outputs, train_bundle, config)
            valid_outputs = model(
                valid_bundle.features.to(device),
                valid_bundle.physics_context.to(device),
                valid_bundle.drives.to(device),
                valid_bundle.current.to(device),
            )

        if config.smoothness_weight > 0.0:
            optimizer.zero_grad()
            outputs = model(
                train_bundle.features.to(device),
                train_bundle.physics_context.to(device),
                train_bundle.drives.to(device),
                train_bundle.current.to(device),
            )
            smooth_penalty = _smoothness_loss(outputs, train_bundle, config)
            (config.smoothness_weight * smooth_penalty).backward()
            optimizer.step()
            smoothness_value = float(smooth_penalty.item())
        else:
            smoothness_value = float(smoothness_loss.item())

        valid_pred_np = valid_outputs["next_soh"].cpu().numpy()
        valid_true_np = valid_df[target_spec.next_col].to_numpy(dtype=float)
        valid_mae = mean_absolute_error(valid_true_np, valid_pred_np)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean([row["loss"] for row in train_loss_parts])),
                "train_data_loss": float(np.mean([row["data_loss"] for row in train_loss_parts])),
                "train_degradation_loss": float(np.mean([row["degradation_loss"] for row in train_loss_parts])),
                "train_residual_loss": float(np.mean([row["residual_loss"] for row in train_loss_parts])),
                "train_smoothness_loss": smoothness_value,
                "valid_mae": float(valid_mae),
                "mean_calendar_sensitivity": float(train_outputs["sensitivities"][:, 0].mean().item()),
                "mean_cycle_sensitivity": float(train_outputs["sensitivities"][:, 1].mean().item()),
                "mean_thermal_sensitivity": float(train_outputs["sensitivities"][:, 2].mean().item()),
                "mean_resistance_sensitivity": float(train_outputs["sensitivities"][:, 3].mean().item()),
                "mean_history_sensitivity": float(train_outputs["sensitivities"][:, 4].mean().item()),
            }
        )
        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    prediction_parts = [
        _predict(train_bundle, model, device, model_name),
        _predict(valid_bundle, model, device, model_name),
        _predict(test_bundle, model, device, model_name),
        _predict(holdout_bundle, model, device, model_name),
    ]
    predictions = pd.concat([part for part in prediction_parts if not part.empty], ignore_index=True) if prediction_parts else pd.DataFrame()

    metrics_rows = []
    for split_name, frame in [
        ("train", train_df),
        ("valid", valid_df),
        ("test", test_df),
        ("holdout", holdout_df),
    ]:
        if frame.empty:
            continue
        pred_frame = predictions.loc[predictions["split"].eq(split_name), ["event_id", model_name]]
        eval_frame = frame.merge(pred_frame, on="event_id", how="inner")
        metrics_rows.append(
            {
                "model": model_name,
                "eval_split": split_name,
                **metric_table(
                    eval_frame[target_spec.next_col].to_numpy(dtype=float),
                    eval_frame[model_name].to_numpy(dtype=float),
                    eval_frame[target_spec.current_col].to_numpy(dtype=float),
                ),
            }
        )

    diagnostics = {
        "history": pd.DataFrame(history),
        "feature_scaler": feature_scaler,
        "context_scaler": context_scaler,
        "feature_medians": medians,
        "feature_dummy_cols": dummy_cols,
        "context_medians": context_medians,
        "context_feature_names": context_cols,
        "drive_feature_names": ["calendar_drive", "cycle_drive", "thermal_drive", "resistance_drive", "history_drive"],
    }

    return ModelArtifacts(
        model_name=model_name,
        predictions=predictions,
        metrics=pd.DataFrame(metrics_rows),
        model=model,
        feature_names=list(train_x.columns),
        diagnostics=diagnostics,
    )
