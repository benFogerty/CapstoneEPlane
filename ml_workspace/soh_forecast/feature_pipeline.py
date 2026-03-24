from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml_workspace.latent_soh.build_latent_soh import build_latent_soh_labels
from ml_workspace.soh_forecast.common import SplitFrames, TargetSpec


RAW_FEATURE_COLS = [
    "observed_soh_pct",
    "current_abs_mean_a",
    "p95_abs_current_a",
    "current_span_a",
    "avg_cell_temp_mean_c",
    "avg_cell_temp_min_c",
    "avg_cell_temp_max_c",
    "avg_cell_temp_span_c",
    "soc_mean_pct",
    "soc_min_pct",
    "soc_max_pct",
    "soc_span_pct",
    "event_duration_s",
    "n_rows",
    "measurement_sigma_pct",
    "condition_multiplier",
    "delta_days",
    "flight_event_flag",
    "charge_event_flag",
    "cumulative_event_count",
    "cumulative_flight_count",
]

LATENT_FEATURE_COLS = [
    "latent_soh_filter_pct",
    "latent_soh_filterpy_filter_pct",
    "_filterpy_pred_state_pct",
    "_filterpy_pred_var_pct2",
    "latent_soh_filter_std_pct",
    "measurement_sigma_pct",
    "condition_multiplier",
]

OPERATING_FEATURE_COLS = [
    "current_abs_mean_a",
    "p95_abs_current_a",
    "current_span_a",
    "p95_abs_dcurrent_a_per_s",
    "avg_cell_temp_mean_c",
    "avg_cell_temp_min_c",
    "avg_cell_temp_max_c",
    "avg_cell_temp_span_c",
    "p95_abs_dtemp_c_per_min",
    "soc_mean_pct",
    "soc_min_pct",
    "soc_max_pct",
    "soc_span_pct",
    "voltage_mean_v",
    "voltage_max_v",
    "event_duration_s",
    "n_rows",
    "delta_days",
    "event_efc",
    "event_ah",
    "cumulative_efc",
    "cumulative_ah",
    "measurement_sigma_pct",
    "condition_multiplier",
    "observed_soh_iqr_pct",
    "observed_soh_span_pct",
    "kalman_coulomb_gap_mean_pct",
    "kalman_coulomb_gap_span_pct",
    "cap_est_delta_raw",
    "cap_est_span_raw",
    "score_current",
    "score_didt",
    "score_dtemp",
    "score_soc_edge",
    "score_observation_instability",
    "score_gap",
    "score_switch",
    "score_event_type",
]

PHYSICS_STRESS_FEATURE_COLS = [
    "arrhenius_temp_proxy",
    "time_above_40c_proxy_min",
    "thermal_severity_proxy",
    "current_rms_proxy_a",
    "throughput_stress_proxy",
    "voltage_sag_proxy_v",
    "internal_resistance_proxy_ohm",
    "soc_edge_stress_proxy",
    "storage_stress_proxy",
    "coulomb_gap_abs_pct",
    "estimation_reset_risk",
    "degradation_stress_proxy",
    "instant_stress_index",
    "current_temp_stress_index",
    "soc_stress_index",
    "duration_stress_index",
]

STATIC_NUMERIC_FEATURE_COLS = [
    "battery_id",
    "flight_event_flag",
    "charge_event_flag",
    "flag_new_est_batt_cap_any",
    "flag_rst_coulomb_any",
]

STATIC_CATEGORICAL_FEATURE_COLS = [
    "plane_id",
    "battery_id_str",
    "event_type",
]

HISTORY_FEATURE_COLS = [
    "prev_observed_soh_pct",
    "prev2_observed_soh_pct",
    "prev_latent_filter_pct",
    "prev2_latent_filter_pct",
    "prev_delta_days",
    "time_since_prev_event_days",
    "observed_soh_delta_1",
    "observed_soh_delta_2",
    "latent_filter_delta_1",
    "latent_filter_delta_2",
    "observed_soh_slope_pct_per_day_1",
    "rolling_observed_soh_mean_3",
    "rolling_observed_soh_std_3",
    "rolling_observed_delta_mean_5",
    "rolling_observed_delta_std_5",
    "rolling_latent_filter_mean_3",
    "rolling_latent_filter_delta_mean_5",
    "rolling_latent_filter_delta_std_5",
    "rolling_current_abs_mean_5",
    "rolling_current_abs_max_5",
    "rolling_temp_mean_5",
    "rolling_temp_span_mean_5",
    "rolling_soc_mean_5",
    "rolling_soc_span_mean_5",
    "rolling_duration_mean_5",
    "rolling_sigma_mean_5",
    "rolling_condition_mean_5",
    "rolling_gap_days_mean_5",
    "rolling_flight_frac_5",
    "cumulative_efc",
    "cumulative_ah",
    "current_temp_stress_index",
    "soc_stress_index",
    "duration_stress_index",
    "instant_stress_index",
    "rolling_stress_index_mean_5",
    "rolling_stress_index_max_5",
]

MULTISCALE_BASE_SIGNAL_COLS = [
    "observed_soh_pct",
    "latent_soh_filter_pct",
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
    "cumulative_event_count",
    "cumulative_flight_count",
    "instant_stress_index",
    "current_temp_stress_index",
    "soc_stress_index",
    "duration_stress_index",
    "arrhenius_temp_proxy",
    "thermal_severity_proxy",
    "throughput_stress_proxy",
    "internal_resistance_proxy_ohm",
    "storage_stress_proxy",
    "degradation_stress_proxy",
]

DEFAULT_MULTI_HORIZON_CONFIGS: tuple[dict[str, object], ...] = (
    {"kind": "flight", "value": 1, "label": "flight_1", "title": "Next flight"},
    {"kind": "flight", "value": 5, "label": "flight_5", "title": "Next 5 flights"},
    {"kind": "flight", "value": 10, "label": "flight_10", "title": "Next 10 flights"},
    {"kind": "flight", "value": 15, "label": "flight_15", "title": "Next 15 flights"},
    {"kind": "flight", "value": 20, "label": "flight_20", "title": "Next 20 flights"},
)

EVENT_GROUP_KEY_COLS = ["plane_id", "flight_id", "event_datetime"]


def ensure_latent_outputs(
    repo_root: Path,
    plane_id: str,
    run_latent_pipeline: bool,
    rt_profile: str,
    q_day_sigma_pct: float,
    latent_root: Path | None = None,
    timeseries_path: Path | None = None,
    spec_path: Path | None = None,
) -> Path:
    latent_root = latent_root or (repo_root / "ml_workspace" / "latent_soh" / "output")
    timeseries_path = timeseries_path or (repo_root / "data" / "event_timeseries_corrected.parquet")
    spec_path = spec_path or (repo_root / "ml_workspace" / "battery_specs.yaml")

    plane_dir = latent_root / f"plane_{plane_id}"
    latent_csv = plane_dir / "latent_soh_event_table.csv"
    if latent_csv.exists():
        return plane_dir
    if not run_latent_pipeline:
        raise FileNotFoundError(f"Missing latent output for plane {plane_id}: {latent_csv}")

    plane_dir.mkdir(parents=True, exist_ok=True)
    result = build_latent_soh_labels(
        plane_id=plane_id,
        timeseries_path=timeseries_path,
        spec_path=spec_path,
        output_dir=plane_dir,
        q_day_sigma_pct=q_day_sigma_pct,
        rt_profile=rt_profile,
    )
    print(json.dumps(result, indent=2))
    return plane_dir


def load_plane_latent(
    repo_root: Path,
    plane_id: str,
    run_latent_pipeline: bool = False,
    rt_profile: str = "balanced",
    q_day_sigma_pct: float = 0.05,
    latent_root: Path | None = None,
    timeseries_path: Path | None = None,
    spec_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    plane_dir = ensure_latent_outputs(
        repo_root=repo_root,
        plane_id=plane_id,
        run_latent_pipeline=run_latent_pipeline,
        rt_profile=rt_profile,
        q_day_sigma_pct=q_day_sigma_pct,
        latent_root=latent_root,
        timeseries_path=timeseries_path,
        spec_path=spec_path,
    )
    latent_df = pd.read_csv(plane_dir / "latent_soh_event_table.csv", parse_dates=["event_datetime"])
    summary_path = plane_dir / "diagnostics" / "smoother_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    latent_df["plane_id"] = latent_df["plane_id"].astype(str)
    latent_df["battery_id"] = pd.to_numeric(latent_df["battery_id"], errors="coerce")
    return latent_df, summary


def load_latent_dataset(
    repo_root: Path,
    primary_plane: str,
    holdout_plane: str | None = None,
    run_latent_pipeline: bool = False,
    rt_profile: str = "balanced",
    q_day_sigma_pct: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    latent_root = repo_root / "ml_workspace" / "latent_soh" / "output"
    timeseries_candidates = [
        repo_root / "data" / "event_timeseries_corrected.parquet",
        repo_root / "data" / "event_timeseries.parquet",
    ]
    timeseries_path = next((path for path in timeseries_candidates if path.exists()), None)
    spec_path = repo_root / "ml_workspace" / "battery_specs.yaml"

    frames: list[pd.DataFrame] = []
    summaries: dict[str, dict] = {}
    primary_df, primary_summary = load_plane_latent(
        repo_root=repo_root,
        plane_id=primary_plane,
        run_latent_pipeline=run_latent_pipeline,
        rt_profile=rt_profile,
        q_day_sigma_pct=q_day_sigma_pct,
        latent_root=latent_root,
        timeseries_path=timeseries_path,
        spec_path=spec_path,
    )
    frames.append(primary_df)
    summaries[str(primary_plane)] = primary_summary

    if holdout_plane:
        holdout_csv = latent_root / f"plane_{holdout_plane}" / "latent_soh_event_table.csv"
        if holdout_csv.exists() or run_latent_pipeline:
            holdout_df, holdout_summary = load_plane_latent(
                repo_root=repo_root,
                plane_id=holdout_plane,
                run_latent_pipeline=run_latent_pipeline,
                rt_profile=rt_profile,
                q_day_sigma_pct=q_day_sigma_pct,
                latent_root=latent_root,
                timeseries_path=timeseries_path,
                spec_path=spec_path,
            )
            frames.append(holdout_df)
            summaries[str(holdout_plane)] = holdout_summary

    latent_df = pd.concat(frames, ignore_index=True)
    latent_df["battery_id_str"] = latent_df["battery_id"].astype("Int64").astype(str)
    latent_df["event_id"] = (
        latent_df["plane_id"].astype(str)
        + "_"
        + latent_df["battery_id_str"]
        + "_"
        + latent_df["flight_id"].astype(str)
        + "_"
        + latent_df["event_datetime"].dt.strftime("%Y%m%d%H%M%S")
    )
    return latent_df, summaries


def add_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    out_frames = []
    for (_, _), group in df.groupby(["plane_id", "battery_id"], sort=False):
        g = group.sort_values(["event_datetime", "flight_id"]).copy()

        numeric_cols = [
            "observed_soh_pct",
            "observed_soh_iqr_pct",
            "observed_soh_span_pct",
            "latent_soh_filter_pct",
            "latent_soh_filterpy_filter_pct",
            "latent_soh_filter_std_pct",
            "_filterpy_pred_state_pct",
            "_filterpy_pred_var_pct2",
            "measurement_sigma_pct",
            "condition_multiplier",
            "current_abs_mean_a",
            "p95_abs_current_a",
            "current_span_a",
            "p95_abs_dcurrent_a_per_s",
            "avg_cell_temp_mean_c",
            "avg_cell_temp_min_c",
            "avg_cell_temp_max_c",
            "avg_cell_temp_span_c",
            "p95_abs_dtemp_c_per_min",
            "soc_mean_pct",
            "soc_min_pct",
            "soc_max_pct",
            "soc_span_pct",
            "voltage_mean_v",
            "voltage_max_v",
            "event_duration_s",
            "n_rows",
            "delta_days",
            "kalman_coulomb_gap_mean_pct",
            "kalman_coulomb_gap_span_pct",
            "cap_est_delta_raw",
            "cap_est_span_raw",
            "score_current",
            "score_didt",
            "score_dtemp",
            "score_soc_edge",
            "score_observation_instability",
            "score_gap",
            "score_switch",
            "score_event_type",
            "flag_new_est_batt_cap_any",
            "flag_rst_coulomb_any",
        ]
        for col in numeric_cols:
            if col in g.columns:
                g[col] = pd.to_numeric(g[col], errors="coerce")

        g["flight_event_flag"] = g["event_type"].astype(str).str.lower().eq("flight").astype(int)
        g["charge_event_flag"] = g["event_type"].astype(str).str.lower().eq("charge").astype(int)
        g["cumulative_event_count"] = np.arange(1, len(g) + 1, dtype=int)
        g["cumulative_flight_count"] = g["flight_event_flag"].cumsum()
        g["event_efc"] = g["soc_span_pct"].abs().fillna(0.0) / 100.0
        g["event_ah"] = g["current_abs_mean_a"].abs().fillna(0.0) * g["event_duration_s"].fillna(0.0) / 3600.0
        g["cumulative_efc"] = g["event_efc"].cumsum()
        g["cumulative_ah"] = g["event_ah"].cumsum()

        g["time_since_prev_event_days"] = g["event_datetime"].diff().dt.total_seconds().div(86400.0)
        g["prev_delta_days"] = g["delta_days"].shift(1)
        g["prev_observed_soh_pct"] = g["observed_soh_pct"].shift(1)
        g["prev2_observed_soh_pct"] = g["observed_soh_pct"].shift(2)
        g["prev_latent_filter_pct"] = g["latent_soh_filter_pct"].shift(1)
        g["prev2_latent_filter_pct"] = g["latent_soh_filter_pct"].shift(2)

        g["observed_soh_delta_1"] = g["observed_soh_pct"] - g["prev_observed_soh_pct"]
        g["observed_soh_delta_2"] = g["prev_observed_soh_pct"] - g["prev2_observed_soh_pct"]
        g["latent_filter_delta_1"] = g["latent_soh_filter_pct"] - g["prev_latent_filter_pct"]
        g["latent_filter_delta_2"] = g["prev_latent_filter_pct"] - g["prev2_latent_filter_pct"]
        g["observed_soh_slope_pct_per_day_1"] = g["observed_soh_delta_1"] / g["time_since_prev_event_days"].clip(lower=1e-3)

        g["rolling_observed_soh_mean_3"] = g["observed_soh_pct"].shift(1).rolling(3, min_periods=1).mean()
        g["rolling_observed_soh_std_3"] = g["observed_soh_pct"].shift(1).rolling(3, min_periods=1).std()
        g["rolling_observed_delta_mean_5"] = g["observed_soh_delta_1"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_observed_delta_std_5"] = g["observed_soh_delta_1"].shift(1).rolling(5, min_periods=1).std()

        g["rolling_latent_filter_mean_3"] = g["latent_soh_filter_pct"].shift(1).rolling(3, min_periods=1).mean()
        g["rolling_latent_filter_delta_mean_5"] = g["latent_filter_delta_1"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_latent_filter_delta_std_5"] = g["latent_filter_delta_1"].shift(1).rolling(5, min_periods=1).std()

        g["rolling_current_abs_mean_5"] = g["current_abs_mean_a"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_current_abs_max_5"] = g["p95_abs_current_a"].shift(1).rolling(5, min_periods=1).max()
        g["rolling_temp_mean_5"] = g["avg_cell_temp_mean_c"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_temp_span_mean_5"] = g["avg_cell_temp_span_c"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_soc_mean_5"] = g["soc_mean_pct"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_soc_span_mean_5"] = g["soc_span_pct"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_duration_mean_5"] = g["event_duration_s"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_sigma_mean_5"] = g["measurement_sigma_pct"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_condition_mean_5"] = g["condition_multiplier"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_gap_days_mean_5"] = g["time_since_prev_event_days"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_flight_frac_5"] = g["flight_event_flag"].shift(1).rolling(5, min_periods=1).mean()

        g["current_temp_stress_index"] = g["current_abs_mean_a"].clip(lower=0.0) * g["avg_cell_temp_mean_c"].clip(lower=0.0)
        g["soc_stress_index"] = g["soc_mean_pct"].clip(lower=0.0) * g["soc_span_pct"].clip(lower=0.0)
        g["duration_stress_index"] = g["event_duration_s"].clip(lower=0.0) * g["condition_multiplier"].clip(lower=0.0)
        g["instant_stress_index"] = g["current_temp_stress_index"] + g["soc_stress_index"] + 0.001 * g["duration_stress_index"]
        g["current_rms_proxy_a"] = np.sqrt(0.5 * (g["current_abs_mean_a"].clip(lower=0.0) ** 2 + g["p95_abs_current_a"].clip(lower=0.0) ** 2))
        k = np.log(2.0) / 10.0
        avg_temp = g["avg_cell_temp_mean_c"].clip(lower=-20.0, upper=100.0)
        g["arrhenius_temp_proxy"] = np.exp(k * avg_temp) * (g["event_duration_s"].clip(lower=0.0) / 60.0)
        tmin = g["avg_cell_temp_min_c"]
        tmax = g["avg_cell_temp_max_c"]
        duration_min = g["event_duration_s"].clip(lower=0.0) / 60.0
        frac = ((tmax - 40.0) / (tmax - tmin)).clip(lower=0.0, upper=1.0)
        frac = frac.where((tmax - tmin).abs() > 1e-6, np.where(tmax >= 40.0, 1.0, 0.0))
        g["time_above_40c_proxy_min"] = np.where(
            tmax <= 40.0,
            0.0,
            np.where(tmin >= 40.0, duration_min, duration_min * frac),
        )
        g["thermal_severity_proxy"] = g["arrhenius_temp_proxy"] * (1.0 + g["avg_cell_temp_span_c"].clip(lower=0.0))
        g["throughput_stress_proxy"] = g["current_rms_proxy_a"] * duration_min * (1.0 + g["soc_span_pct"].clip(lower=0.0) / 100.0)
        g["voltage_sag_proxy_v"] = g["voltage_max_v"] - g["voltage_mean_v"]
        g["internal_resistance_proxy_ohm"] = g["voltage_sag_proxy_v"] / g["p95_abs_current_a"].replace(0.0, np.nan)
        g["soc_edge_stress_proxy"] = np.minimum(g["soc_mean_pct"], 100.0 - g["soc_mean_pct"]).rsub(50.0).clip(lower=0.0) * (g["soc_span_pct"].clip(lower=0.0) / 100.0)
        g["storage_stress_proxy"] = g["delta_days"].clip(lower=0.0) * (g["soc_mean_pct"].clip(lower=0.0) / 100.0)
        g["coulomb_gap_abs_pct"] = g["kalman_coulomb_gap_mean_pct"].abs()
        g["estimation_reset_risk"] = g["flag_new_est_batt_cap_any"].fillna(0.0) + g["flag_rst_coulomb_any"].fillna(0.0) + g["score_switch"].fillna(0.0)
        g["degradation_stress_proxy"] = (
            0.35 * g["thermal_severity_proxy"].fillna(0.0)
            + 0.25 * g["throughput_stress_proxy"].fillna(0.0)
            + 0.15 * g["storage_stress_proxy"].fillna(0.0)
            + 0.15 * g["internal_resistance_proxy_ohm"].fillna(0.0).clip(lower=0.0)
            + 0.10 * g["coulomb_gap_abs_pct"].fillna(0.0)
        )
        g["rolling_stress_index_mean_5"] = g["instant_stress_index"].shift(1).rolling(5, min_periods=1).mean()
        g["rolling_stress_index_max_5"] = g["instant_stress_index"].shift(1).rolling(5, min_periods=1).max()

        g["next_observed_soh_pct"] = g["observed_soh_pct"].shift(-1)
        g["next_observed_delta_pct"] = g["next_observed_soh_pct"] - g["observed_soh_pct"]
        g["next_latent_soh_smooth_pct"] = g["latent_soh_smooth_pct"].shift(-1)
        g["next_latent_soh_causal_pct"] = g["latent_soh_filter_pct"].shift(-1)
        g["next_latent_soh_rts_pct"] = g["next_latent_soh_smooth_pct"]
        g["next_latent_soh_smooth_pct"] = g["next_latent_soh_causal_pct"]
        g["next_latent_delta_pct"] = g["next_latent_soh_causal_pct"] - g["latent_soh_filter_pct"]
        g["next_cumulative_efc"] = g["cumulative_efc"].shift(-1)
        g["next_cumulative_ah"] = g["cumulative_ah"].shift(-1)

        out_frames.append(g)

    out = pd.concat(out_frames, ignore_index=True)
    for col in [
        "rolling_observed_soh_std_3",
        "rolling_observed_delta_std_5",
        "rolling_latent_filter_delta_std_5",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
    return add_multi_horizon_targets(out)


def _multi_horizon_target_positions(g: pd.DataFrame, kind: str, value: float) -> np.ndarray:
    n = len(g)
    base_idx = np.arange(n, dtype=int)
    if kind == "step":
        offset = int(value)
        target_idx = base_idx + offset
        target_idx[target_idx >= n] = -1
        return target_idx

    if kind == "days":
        age_days = (g["event_datetime"] - g["event_datetime"].iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        target_age = age_days + float(value)
        target_idx = np.searchsorted(age_days, target_age, side="left")
        target_idx[target_idx >= n] = -1
        return np.where(target_idx > base_idx, target_idx, -1)

    if kind == "efc":
        cumulative_efc = pd.to_numeric(g["cumulative_efc"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        target_efc = cumulative_efc + float(value)
        target_idx = np.searchsorted(cumulative_efc, target_efc, side="left")
        target_idx[target_idx >= n] = -1
        return np.where(target_idx > base_idx, target_idx, -1)

    if kind == "flight":
        cumulative_flights = pd.to_numeric(g["cumulative_flight_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        target_flights = cumulative_flights + float(value)
        target_idx = np.searchsorted(cumulative_flights, target_flights, side="left")
        target_idx[target_idx >= n] = -1
        return np.where(target_idx > base_idx, target_idx, -1)

    raise ValueError(f"Unsupported horizon kind: {kind}")


def add_multi_horizon_targets(
    df: pd.DataFrame,
    horizon_configs: tuple[dict[str, object], ...] | None = None,
) -> pd.DataFrame:
    horizon_configs = horizon_configs or DEFAULT_MULTI_HORIZON_CONFIGS
    out_frames: list[pd.DataFrame] = []

    for (_, _), group in df.groupby(["plane_id", "battery_id"], sort=False):
        g = group.sort_values(["event_datetime", "flight_id"]).copy()
        observed_soh = pd.to_numeric(g["observed_soh_pct"], errors="coerce").to_numpy(dtype=float)
        latent_soh_smooth = pd.to_numeric(g["latent_soh_smooth_pct"], errors="coerce").to_numpy(dtype=float)
        latent_soh_causal = pd.to_numeric(g["latent_soh_filter_pct"], errors="coerce").to_numpy(dtype=float)
        current_latent = pd.to_numeric(g["latent_soh_filter_pct"], errors="coerce").to_numpy(dtype=float)
        age_days = (g["event_datetime"] - g["event_datetime"].iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        cumulative_efc = pd.to_numeric(g["cumulative_efc"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        cumulative_flights = pd.to_numeric(g["cumulative_flight_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        row_idx = np.arange(len(g), dtype=int)

        for cfg in horizon_configs:
            kind = str(cfg["kind"])
            value = float(cfg["value"])
            label = str(cfg["label"])
            if kind == "step" and int(round(value)) == 1:
                continue

            target_idx = _multi_horizon_target_positions(g, kind, value)
            valid = target_idx >= 0
            safe_idx = np.where(valid, target_idx, 0)

            observed_target = np.where(valid, observed_soh[safe_idx], np.nan)
            latent_target_smooth = np.where(valid, latent_soh_smooth[safe_idx], np.nan)
            latent_target_causal = np.where(valid, latent_soh_causal[safe_idx], np.nan)
            realized_days = np.where(valid, age_days[safe_idx] - age_days[row_idx], np.nan)
            realized_efc = np.where(valid, cumulative_efc[safe_idx] - cumulative_efc[row_idx], np.nan)
            realized_events = np.where(valid, safe_idx - row_idx, np.nan)
            realized_flights = np.where(valid, cumulative_flights[safe_idx] - cumulative_flights[row_idx], np.nan)

            g[f"next_observed_soh_{label}_pct"] = observed_target
            g[f"next_observed_delta_{label}_pct"] = observed_target - observed_soh
            g[f"next_latent_soh_smooth_{label}_pct"] = latent_target_smooth
            g[f"next_latent_soh_causal_{label}_pct"] = latent_target_causal
            g[f"next_latent_soh_rts_{label}_pct"] = latent_target_smooth
            g[f"next_latent_soh_smooth_{label}_pct"] = latent_target_causal
            g[f"next_latent_delta_{label}_pct"] = latent_target_causal - current_latent
            g[f"horizon_{label}_days"] = realized_days
            g[f"horizon_{label}_efc"] = realized_efc
            g[f"horizon_{label}_events"] = realized_events
            g[f"horizon_{label}_flights"] = realized_flights

        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True) if out_frames else df.copy()


def make_multi_horizon_target_specs(
    horizon_configs: tuple[dict[str, object], ...] | None = None,
    include_observed: bool = True,
    include_latent: bool = True,
) -> dict[str, TargetSpec]:
    horizon_configs = horizon_configs or DEFAULT_MULTI_HORIZON_CONFIGS
    specs: dict[str, TargetSpec] = {}
    for cfg in horizon_configs:
        kind = str(cfg["kind"])
        value = float(cfg["value"])
        label = str(cfg["label"])
        title = str(cfg["title"])

        if include_latent:
            next_col = (
                "next_latent_soh_causal_pct"
                if kind == "step" and int(round(value)) == 1
                else f"next_latent_soh_causal_{label}_pct"
            )
            delta_col = "next_latent_delta_pct" if kind == "step" and int(round(value)) == 1 else f"next_latent_delta_{label}_pct"
            specs[f"latent_{label}"] = TargetSpec(
                name=f"latent_{label}",
                current_col="latent_soh_filter_pct",
                next_col=next_col,
                delta_col=delta_col,
                title_label=f"Latent SOH ({title})",
            )

        if include_observed:
            next_col = "next_observed_soh_pct" if kind == "step" and int(round(value)) == 1 else f"next_observed_soh_{label}_pct"
            delta_col = "next_observed_delta_pct" if kind == "step" and int(round(value)) == 1 else f"next_observed_delta_{label}_pct"
            specs[f"observed_{label}"] = TargetSpec(
                name=f"observed_{label}",
                current_col="observed_soh_pct",
                next_col=next_col,
                delta_col=delta_col,
                title_label=f"Observed SOH ({title})",
            )
    return specs


def available_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "raw": [col for col in RAW_FEATURE_COLS if col in df.columns],
        "operating": [col for col in OPERATING_FEATURE_COLS if col in df.columns],
        "latent": [col for col in LATENT_FEATURE_COLS if col in df.columns],
        "physics": [col for col in PHYSICS_STRESS_FEATURE_COLS if col in df.columns],
        "static_numeric": [col for col in STATIC_NUMERIC_FEATURE_COLS if col in df.columns],
        "static_categorical": [col for col in STATIC_CATEGORICAL_FEATURE_COLS if col in df.columns],
        "history": [col for col in HISTORY_FEATURE_COLS if col in df.columns],
    }


def time_split(df: pd.DataFrame, train_frac: float, valid_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("event_datetime").copy()
    n = len(ordered)
    n_train = max(1, int(round(n * train_frac)))
    n_valid = max(1, int(round(n * valid_frac))) if n >= 3 else max(0, n - n_train)
    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1) if n >= 3 else max(0, n - n_train)
    return ordered.iloc[:n_train].copy(), ordered.iloc[n_train : n_train + n_valid].copy(), ordered.iloc[n_train + n_valid :].copy()


def _event_group_frame(df: pd.DataFrame) -> pd.DataFrame:
    group_df = df[EVENT_GROUP_KEY_COLS].drop_duplicates().copy()
    return group_df.sort_values(["event_datetime", "flight_id"]).reset_index(drop=True)


def _walk_forward_boundaries(
    total_groups: int,
    final_test_frac: float,
    backtest_folds: int,
    fold_valid_frac: float,
) -> tuple[int, int, list[dict[str, int]]]:
    if total_groups < 4:
        raise ValueError("Need at least 4 eligible event groups for walk-forward benchmarking")
    if not (0.0 < final_test_frac < 1.0):
        raise ValueError("final_test_frac must be between 0 and 1")
    if not (0.0 < fold_valid_frac < 1.0):
        raise ValueError("fold_valid_frac must be between 0 and 1")
    if backtest_folds < 1:
        raise ValueError("backtest_folds must be at least 1")

    final_test_count = max(1, int(round(total_groups * final_test_frac)))
    dev_count = total_groups - final_test_count
    valid_count = max(1, int(round(total_groups * fold_valid_frac)))
    initial_train_count = dev_count - backtest_folds * valid_count
    if initial_train_count < 1:
        raise ValueError(
            "Not enough eligible event groups for the requested walk-forward layout. "
            f"groups={total_groups}, final_test_count={final_test_count}, valid_count={valid_count}, folds={backtest_folds}"
        )

    folds: list[dict[str, int]] = []
    train_end = initial_train_count
    for fold_id in range(1, backtest_folds + 1):
        valid_start = train_end
        valid_end = valid_start + valid_count
        if valid_end > dev_count:
            raise ValueError("Walk-forward validation windows exceed development range")
        folds.append(
            {
                "fold_id": fold_id,
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
            }
        )
        train_end = valid_end

    return final_test_count, valid_count, folds


def assign_walk_forward_splits(
    predictive_df: pd.DataFrame,
    primary_plane: str,
    holdout_plane: str | None,
    final_test_frac: float = 0.15,
    backtest_folds: int = 3,
    fold_valid_frac: float = 0.10,
    required_target_cols: list[str] | None = None,
) -> pd.DataFrame:
    required_target_cols = required_target_cols or []
    usable = predictive_df.copy()
    for col in required_target_cols:
        usable = usable.loc[usable[col].notna()].copy()

    usable["final_split"] = "unused"
    usable["refit_role"] = "unused"
    for fold_id in range(1, backtest_folds + 1):
        usable[f"backtest_fold_{fold_id}_role"] = "unused"

    primary_rows = usable.loc[usable["plane_id"].eq(primary_plane)].copy()
    group_df = _event_group_frame(primary_rows)
    final_test_count, _valid_count, folds = _walk_forward_boundaries(
        total_groups=len(group_df),
        final_test_frac=final_test_frac,
        backtest_folds=backtest_folds,
        fold_valid_frac=fold_valid_frac,
    )
    dev_count = len(group_df) - final_test_count

    group_df["final_split"] = "train_dev"
    group_df.loc[group_df.index >= dev_count, "final_split"] = "final_test"

    group_df["refit_role"] = "train"
    last_fold = folds[-1]
    group_df.loc[group_df.index >= dev_count, "refit_role"] = "test"
    group_df.loc[(group_df.index >= last_fold["valid_start"]) & (group_df.index < last_fold["valid_end"]), "refit_role"] = "valid"

    for fold in folds:
        col = f"backtest_fold_{fold['fold_id']}_role"
        group_df[col] = "unused"
        group_df.loc[group_df.index < fold["train_end"], col] = "train"
        group_df.loc[(group_df.index >= fold["valid_start"]) & (group_df.index < fold["valid_end"]), col] = "valid"

    assignment_cols = ["final_split", "refit_role", *[f"backtest_fold_{fold['fold_id']}_role" for fold in folds]]
    usable = usable.merge(group_df[EVENT_GROUP_KEY_COLS + assignment_cols], on=EVENT_GROUP_KEY_COLS, how="left", suffixes=("", "_group"))
    for col in assignment_cols:
        group_col = f"{col}_group"
        if group_col in usable.columns:
            usable[col] = usable[group_col].fillna(usable[col])
            usable = usable.drop(columns=[group_col])

    if holdout_plane:
        holdout_mask = usable["plane_id"].eq(holdout_plane)
        usable.loc[holdout_mask, "final_split"] = "holdout"
        usable.loc[holdout_mask, "refit_role"] = "holdout"

    return usable


def split_frames_from_column(df: pd.DataFrame, split_col: str) -> SplitFrames:
    working = df.copy()
    working["split"] = working[split_col].fillna("unused")
    return split_frames_from_assigned(working)


def build_target_frame(
    predictive_df: pd.DataFrame,
    target_spec: TargetSpec,
    primary_plane: str,
    holdout_plane: str | None,
    train_frac: float,
    valid_frac: float,
) -> tuple[pd.DataFrame, SplitFrames]:
    usable = predictive_df.loc[predictive_df[target_spec.next_col].notna()].copy()
    usable["split"] = "unused"

    primary_rows = usable.loc[usable["plane_id"].eq(primary_plane)].copy()
    train_primary, valid_primary, test_primary = time_split(primary_rows, train_frac, valid_frac)
    usable.loc[train_primary.index, "split"] = "train"
    usable.loc[valid_primary.index, "split"] = "valid"
    usable.loc[test_primary.index, "split"] = "test"

    if holdout_plane:
        usable.loc[usable["plane_id"].eq(holdout_plane), "split"] = "holdout"

    split_frames = SplitFrames(
        train=usable.loc[usable["split"].eq("train")].copy(),
        valid=usable.loc[usable["split"].eq("valid")].copy(),
        test=usable.loc[usable["split"].eq("test")].copy(),
        holdout=usable.loc[usable["split"].eq("holdout")].copy(),
    )
    return usable, split_frames


def assign_shared_splits(
    predictive_df: pd.DataFrame,
    primary_plane: str,
    holdout_plane: str | None,
    train_frac: float,
    valid_frac: float,
    required_target_cols: list[str] | None = None,
) -> pd.DataFrame:
    required_target_cols = required_target_cols or []
    usable = predictive_df.copy()
    for col in required_target_cols:
        usable = usable.loc[usable[col].notna()].copy()
    usable["split"] = "unused"

    primary_rows = usable.loc[usable["plane_id"].eq(primary_plane)].copy()
    train_primary, valid_primary, test_primary = time_split(primary_rows, train_frac, valid_frac)
    usable.loc[train_primary.index, "split"] = "train"
    usable.loc[valid_primary.index, "split"] = "valid"
    usable.loc[test_primary.index, "split"] = "test"

    if holdout_plane:
        usable.loc[usable["plane_id"].eq(holdout_plane), "split"] = "holdout"
    return usable


def split_frames_from_assigned(df: pd.DataFrame) -> SplitFrames:
    return SplitFrames(
        train=df.loc[df["split"].eq("train")].copy(),
        valid=df.loc[df["split"].eq("valid")].copy(),
        test=df.loc[df["split"].eq("test")].copy(),
        holdout=df.loc[df["split"].eq("holdout")].copy(),
    )


def add_multiscale_history_features(
    df: pd.DataFrame,
    base_cols: list[str] | None = None,
    lag_steps: tuple[int, ...] = (1, 2, 3, 5, 8, 13),
    rolling_windows: tuple[int, ...] = (2, 3, 5, 8, 13, 21),
    ewm_spans: tuple[int, ...] = (3, 5, 8, 13),
    rate_steps: tuple[int, ...] = (1, 3, 5),
) -> tuple[pd.DataFrame, list[str]]:
    base_cols = base_cols or [col for col in MULTISCALE_BASE_SIGNAL_COLS if col in df.columns]
    out_frames: list[pd.DataFrame] = []
    candidate_cols: list[str] = []

    for (_, _), group in df.groupby(["plane_id", "battery_id"], sort=False):
        g = group.sort_values(["event_datetime", "flight_id"]).copy()
        for col in base_cols:
            if col not in g.columns:
                continue
            g[col] = pd.to_numeric(g[col], errors="coerce")
            shifted = g[col].shift(1)

            for lag in lag_steps:
                lag_col = f"{col}_lag{lag}"
                g[lag_col] = g[col].shift(lag)
                candidate_cols.append(lag_col)

            for step in rate_steps:
                diff_col = f"{col}_diff{step}"
                pct_col = f"{col}_pct_change{step}"
                rate_col = f"{col}_rate_per_day{step}"
                prior = g[col].shift(step)
                g[diff_col] = g[col] - prior
                g[pct_col] = (g[col] - prior) / prior.replace(0.0, np.nan)
                g[rate_col] = g[diff_col] / g["time_since_prev_event_days"].rolling(step, min_periods=1).sum().clip(lower=1e-3)
                candidate_cols.extend([diff_col, pct_col, rate_col])

            for window in rolling_windows:
                mean_col = f"{col}_rollmean_{window}"
                std_col = f"{col}_rollstd_{window}"
                min_col = f"{col}_rollmin_{window}"
                max_col = f"{col}_rollmax_{window}"
                slope_col = f"{col}_rollslope_{window}"
                g[mean_col] = shifted.rolling(window, min_periods=1).mean()
                g[std_col] = shifted.rolling(window, min_periods=1).std()
                g[min_col] = shifted.rolling(window, min_periods=1).min()
                g[max_col] = shifted.rolling(window, min_periods=1).max()
                g[slope_col] = g[mean_col] - g[f"{col}_rollmean_{max(1, window // 2)}"] if f"{col}_rollmean_{max(1, window // 2)}" in g.columns else np.nan
                candidate_cols.extend([mean_col, std_col, min_col, max_col, slope_col])

            for span in ewm_spans:
                ewm_mean_col = f"{col}_ewmmean_{span}"
                ewm_std_col = f"{col}_ewmstd_{span}"
                g[ewm_mean_col] = shifted.ewm(span=span, adjust=False, min_periods=1).mean()
                g[ewm_std_col] = shifted.ewm(span=span, adjust=False, min_periods=1).std()
                candidate_cols.extend([ewm_mean_col, ewm_std_col])

        out_frames.append(g)

    out = pd.concat(out_frames, ignore_index=True)
    candidate_cols = list(dict.fromkeys([col for col in candidate_cols if col in out.columns]))
    for col in candidate_cols:
        if out[col].dtype.kind in {"f", "i"}:
            if col.endswith("_rollstd_2") or "_rollstd_" in col or "_ewmstd_" in col:
                out[col] = out[col].fillna(0.0)
    return out, candidate_cols
