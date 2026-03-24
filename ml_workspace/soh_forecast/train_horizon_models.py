from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import pandas as pd
import torch


def _bootstrap_repo_root() -> None:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "ml_workspace").exists() and (candidate / "data").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return
    raise RuntimeError("Could not locate repo root for ml_workspace imports")


_bootstrap_repo_root()

from ml_workspace.soh_forecast.benchmarking import (
    build_final_and_holdout_metrics,
    build_full_and_common_metrics,
    build_truth_frame,
    build_winner_summary_row,
    combine_prediction_tables,
    gbdt_importance_frame,
    rename_prediction_splits,
    ridge_coefficient_frame,
    select_backtest_winner,
    summarize_backtest_fold_metrics,
    summarize_feature_correlation,
)
from ml_workspace.soh_forecast.common import ModelArtifacts, SplitFrames, TargetSpec, concat_frames, find_repo_root, set_seed
from ml_workspace.soh_forecast.feature_pipeline import (
    EVENT_GROUP_KEY_COLS,
    add_forecast_features,
    assign_shared_splits,
    assign_walk_forward_splits,
    available_feature_sets,
    load_latent_dataset,
    make_multi_horizon_target_specs,
    split_frames_from_assigned,
    split_frames_from_column,
)
from ml_workspace.soh_forecast.models.elastic_net_delta import train_elastic_net_delta
from ml_workspace.soh_forecast.models.gam_spline_delta import train_gam_spline_delta
from ml_workspace.soh_forecast.models.gru_sequence import train_gru_sequence
from ml_workspace.soh_forecast.models.hist_gbdt_delta import train_hist_gbdt_delta
from ml_workspace.soh_forecast.models.lstm_sequence import LSTMConfig, train_lstm_sequence
from ml_workspace.soh_forecast.models.naive_zero_delta import train_naive_zero_delta
from ml_workspace.soh_forecast.models.physics_hybrid_nn import PhysicsHybridConfig, train_physics_hybrid_nn
from ml_workspace.soh_forecast.models.physics_informed_nn import PhysicsInformedConfig, train_physics_informed_nn
from ml_workspace.soh_forecast.models.random_forest_delta import train_random_forest_delta
from ml_workspace.soh_forecast.models.ridge_delta import train_ridge_delta


TrainFn = Callable[[pd.DataFrame, SplitFrames], ModelArtifacts]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-horizon SOH models and select best per horizon.")
    parser.add_argument("--primary-plane", default="166", help="Primary plane id for train/valid/test")
    parser.add_argument("--holdout-plane", default="192", help="Holdout plane id")
    parser.add_argument("--run-latent-pipeline", action="store_true", help="Regenerate latent SOH tables if missing")
    parser.add_argument("--rt-profile", default="current", help="Latent SOH rt_profile")
    parser.add_argument("--q-day-sigma-pct", type=float, default=0.10, help="Latent SOH process sigma")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Train fraction for single_block compatibility mode")
    parser.add_argument("--valid-frac", type=float, default=0.10, help="Validation fraction for single_block compatibility mode")
    parser.add_argument("--split-scheme", choices=["walk_forward", "single_block"], default="walk_forward", help="Benchmark split scheme")
    parser.add_argument("--final-test-frac", type=float, default=0.15, help="Final untouched in-plane test fraction")
    parser.add_argument("--backtest-folds", type=int, default=3, help="Number of expanding walk-forward folds")
    parser.add_argument("--fold-valid-frac", type=float, default=0.10, help="Validation fraction per walk-forward fold")
    parser.add_argument("--lookback", type=int, default=20, help="Sequence model lookback window")
    parser.add_argument("--device", default="cpu", help="Torch device for sequence/physics models")
    parser.add_argument("--tune", action="store_true", help="Enable broader hyperparameter tuning")
    parser.add_argument(
        "--horizons",
        default="1,5,10,15,20",
        help="Comma-separated flight horizons (e.g. 1,5,10,15,20)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for metrics and predictions",
    )
    return parser.parse_args()


def _save_model(artifact: ModelArtifacts, output_dir: Path) -> str | None:
    model = artifact.model
    if model is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        model_path = output_dir / f"{artifact.model_name}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_names": artifact.feature_names,
                "diagnostics": artifact.diagnostics,
            },
            model_path,
        )
        return str(model_path)
    try:
        import joblib

        model_path = output_dir / f"{artifact.model_name}.joblib"
        joblib.dump(
            {
                "model": model,
                "feature_names": artifact.feature_names,
                "diagnostics": artifact.diagnostics,
            },
            model_path,
        )
        return str(model_path)
    except Exception:
        return None


def _sequence_features(
    target_df: pd.DataFrame,
    raw_features: list[str],
    latent_features: list[str],
) -> tuple[list[str], list[str]]:
    seq_feature_candidates_no_latent = [
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
        "delta_days",
        "event_efc",
        "event_ah",
        "cumulative_efc",
        "cumulative_ah",
        "cumulative_flight_count",
        "flight_event_flag",
        "charge_event_flag",
        "time_since_prev_event_days",
    ]
    seq_features_no_latent = [col for col in seq_feature_candidates_no_latent if col in target_df.columns]
    if len(seq_features_no_latent) < 3:
        seq_features_no_latent = list(dict.fromkeys(raw_features))

    seq_feature_candidates_with_latent = [
        "latent_soh_filter_pct",
        "latent_soh_filter_std_pct",
        "measurement_sigma_pct",
        "condition_multiplier",
        *seq_feature_candidates_no_latent,
    ]
    seq_features_with_latent = [col for col in seq_feature_candidates_with_latent if col in target_df.columns]
    if len(seq_features_with_latent) < 3:
        seq_features_with_latent = list(dict.fromkeys(raw_features + latent_features))
    return seq_features_no_latent, seq_features_with_latent


def _trainer_registry(
    target_df: pd.DataFrame,
    target_spec: TargetSpec,
    feature_sets: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, TrainFn]:
    raw_features = feature_sets.get("raw", [])
    operating_features = feature_sets.get("operating", [])
    latent_features = feature_sets.get("latent", [])
    physics_features = feature_sets.get("physics", [])
    static_numeric_features = feature_sets.get("static_numeric", [])
    history_features = feature_sets.get("history", [])
    all_features_with_latent = list(
        dict.fromkeys(
            raw_features
            + operating_features
            + latent_features
            + physics_features
            + static_numeric_features
            + history_features
        )
    )
    all_features_no_latent = list(
        dict.fromkeys(
            raw_features
            + operating_features
            + physics_features
            + static_numeric_features
            + history_features
        )
    )
    seq_features_no_latent, seq_features_with_latent = _sequence_features(target_df, raw_features, latent_features)

    if args.tune:
        ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 300.0]
        elastic_grid = [
            {"alpha": alpha, "l1_ratio": l1_ratio, "max_iter": 50000, "tol": 1e-3, "selection": "random", "random_state": 42}
            for alpha in [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
            for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]
        ]
        gbdt_grid = [
            {"learning_rate": 0.02, "max_depth": 3, "max_iter": 500, "min_samples_leaf": 10, "random_state": 42},
            {"learning_rate": 0.03, "max_depth": 3, "max_iter": 700, "min_samples_leaf": 8, "random_state": 42},
            {"learning_rate": 0.05, "max_depth": 4, "max_iter": 400, "min_samples_leaf": 8, "random_state": 42},
            {"learning_rate": 0.08, "max_depth": 4, "max_iter": 300, "min_samples_leaf": 5, "random_state": 42},
        ]
        rf_grid = [
            {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 3, "max_features": "sqrt", "random_state": 42, "n_jobs": -1},
            {"n_estimators": 600, "max_depth": 10, "min_samples_leaf": 3, "max_features": "sqrt", "random_state": 42, "n_jobs": -1},
            {"n_estimators": 800, "max_depth": 12, "min_samples_leaf": 5, "max_features": 0.5, "random_state": 42, "n_jobs": -1},
        ]
        gam_alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        gam_knots = [3, 4, 5]
        seq_config_grid = [
            LSTMConfig(lookback=args.lookback, hidden_dim=48, dropout=0.1, lr=1e-3, device=args.device),
            LSTMConfig(lookback=30, hidden_dim=48, dropout=0.1, lr=5e-4, device=args.device),
            LSTMConfig(lookback=args.lookback, hidden_dim=32, dropout=0.2, lr=1e-3, device=args.device),
            LSTMConfig(lookback=30, hidden_dim=32, dropout=0.2, lr=5e-4, device=args.device),
        ]
        hybrid_configs = [
            PhysicsHybridConfig(hidden_dim=64, physics_hidden_dim=48, lr=1e-3, weight_decay=1e-4, device=args.device),
            PhysicsHybridConfig(hidden_dim=96, physics_hidden_dim=64, lr=5e-4, weight_decay=3e-4, device=args.device),
        ]
        pinn_configs = [
            PhysicsInformedConfig(hidden_dim=96, lr=1e-3, weight_decay=1e-5, device=args.device),
            PhysicsInformedConfig(hidden_dim=128, lr=5e-4, weight_decay=3e-5, device=args.device),
        ]
    else:
        ridge_alphas = None
        elastic_grid = None
        gbdt_grid = None
        rf_grid = None
        gam_alphas = None
        gam_knots = None
        seq_config_grid = [LSTMConfig(lookback=args.lookback, device=args.device)]
        hybrid_configs = [PhysicsHybridConfig(device=args.device)]
        pinn_configs = [PhysicsInformedConfig(device=args.device)]

    trainers: dict[str, TrainFn] = {
        "naive_zero_delta": lambda run_df, split_frames: train_naive_zero_delta(split_frames, target_spec),
        "ridge_raw_only": lambda run_df, split_frames: train_ridge_delta(
            split_frames, target_spec, raw_features, model_name="ridge_raw_only", alphas=ridge_alphas
        ),
        "ridge_raw_plus_latent": lambda run_df, split_frames: train_ridge_delta(
            split_frames, target_spec, raw_features + latent_features, model_name="ridge_raw_plus_latent", alphas=ridge_alphas
        ),
        "ridge_raw_only_no_latent": lambda run_df, split_frames: train_ridge_delta(
            split_frames, target_spec, raw_features, model_name="ridge_raw_only_no_latent", alphas=ridge_alphas
        ),
        "elastic_with_latent": lambda run_df, split_frames: train_elastic_net_delta(
            split_frames, target_spec, all_features_with_latent, model_name="elastic_with_latent", grid=elastic_grid
        ),
        "elastic_no_latent": lambda run_df, split_frames: train_elastic_net_delta(
            split_frames, target_spec, all_features_no_latent, model_name="elastic_no_latent", grid=elastic_grid
        ),
        "ridge_with_latent": lambda run_df, split_frames: train_ridge_delta(
            split_frames, target_spec, all_features_with_latent, model_name="ridge_with_latent", alphas=ridge_alphas
        ),
        "ridge_no_latent": lambda run_df, split_frames: train_ridge_delta(
            split_frames, target_spec, all_features_no_latent, model_name="ridge_no_latent", alphas=ridge_alphas
        ),
        "gbdt_with_latent": lambda run_df, split_frames: train_hist_gbdt_delta(
            split_frames, target_spec, all_features_with_latent, model_name="gbdt_with_latent", param_grid=gbdt_grid
        ),
        "gbdt_no_latent": lambda run_df, split_frames: train_hist_gbdt_delta(
            split_frames, target_spec, all_features_no_latent, model_name="gbdt_no_latent", param_grid=gbdt_grid
        ),
        "random_forest_with_latent": lambda run_df, split_frames: train_random_forest_delta(
            split_frames, target_spec, all_features_with_latent, model_name="random_forest_with_latent", param_grid=rf_grid
        ),
        "random_forest_no_latent": lambda run_df, split_frames: train_random_forest_delta(
            split_frames, target_spec, all_features_no_latent, model_name="random_forest_no_latent", param_grid=rf_grid
        ),
        "gam_spline_with_latent": lambda run_df, split_frames: train_gam_spline_delta(
            split_frames, target_spec, all_features_with_latent, model_name="gam_spline_with_latent", alphas=gam_alphas, n_knots_list=gam_knots
        ),
        "gam_spline_no_latent": lambda run_df, split_frames: train_gam_spline_delta(
            split_frames, target_spec, all_features_no_latent, model_name="gam_spline_no_latent", alphas=gam_alphas, n_knots_list=gam_knots
        ),
    }

    for idx, cfg in enumerate(seq_config_grid):
        suffix = f"_tune{idx}" if args.tune else ""
        lstm_with_latent = f"lstm_sequence_with_latent{suffix}"
        gru_with_latent = f"gru_sequence_with_latent{suffix}"
        lstm_no_latent = f"lstm_sequence_no_latent{suffix}"
        gru_no_latent = f"gru_sequence_no_latent{suffix}"
        trainers[lstm_with_latent] = lambda run_df, split_frames, cfg=cfg, model_name=lstm_with_latent: train_lstm_sequence(
            run_df,
            split_frames,
            target_spec,
            seq_features_with_latent,
            model_name=model_name,
            config=cfg,
        )
        trainers[gru_with_latent] = lambda run_df, split_frames, cfg=cfg, model_name=gru_with_latent: train_gru_sequence(
            run_df,
            split_frames,
            target_spec,
            seq_features_with_latent,
            model_name=model_name,
            config=cfg,
        )
        trainers[lstm_no_latent] = lambda run_df, split_frames, cfg=cfg, model_name=lstm_no_latent: train_lstm_sequence(
            run_df,
            split_frames,
            target_spec,
            seq_features_no_latent,
            model_name=model_name,
            config=cfg,
        )
        trainers[gru_no_latent] = lambda run_df, split_frames, cfg=cfg, model_name=gru_no_latent: train_gru_sequence(
            run_df,
            split_frames,
            target_spec,
            seq_features_no_latent,
            model_name=model_name,
            config=cfg,
        )

    for idx, cfg in enumerate(hybrid_configs):
        suffix = f"_tune{idx}" if args.tune else ""
        hybrid_with_latent = f"physics_hybrid_with_latent{suffix}"
        hybrid_no_latent = f"physics_hybrid_no_latent{suffix}"
        trainers[hybrid_with_latent] = lambda run_df, split_frames, cfg=cfg, model_name=hybrid_with_latent: train_physics_hybrid_nn(
            split_frames,
            target_spec,
            all_features_with_latent,
            model_name=model_name,
            config=cfg,
        )
        trainers[hybrid_no_latent] = lambda run_df, split_frames, cfg=cfg, model_name=hybrid_no_latent: train_physics_hybrid_nn(
            split_frames,
            target_spec,
            all_features_no_latent,
            model_name=model_name,
            config=cfg,
        )

    for idx, cfg in enumerate(pinn_configs):
        suffix = f"_tune{idx}" if args.tune else ""
        pinn_with_latent = f"physics_informed_with_latent{suffix}"
        pinn_no_latent = f"physics_informed_no_latent{suffix}"
        trainers[pinn_with_latent] = lambda run_df, split_frames, cfg=cfg, model_name=pinn_with_latent: train_physics_informed_nn(
            split_frames,
            target_spec,
            all_features_with_latent,
            model_name=model_name,
            config=cfg,
        )
        trainers[pinn_no_latent] = lambda run_df, split_frames, cfg=cfg, model_name=pinn_no_latent: train_physics_informed_nn(
            split_frames,
            target_spec,
            all_features_no_latent,
            model_name=model_name,
            config=cfg,
        )

    return trainers


def _run_trainers(run_df: pd.DataFrame, split_frames: SplitFrames, trainers: dict[str, TrainFn]) -> list[ModelArtifacts]:
    return [trainer(run_df, split_frames) for trainer in trainers.values()]


def _split_assignment_columns(target_df: pd.DataFrame, backtest_folds: int) -> list[str]:
    cols = ["event_id", "plane_id", "battery_id", *EVENT_GROUP_KEY_COLS, "final_split", "refit_role"]
    cols.extend(f"backtest_fold_{fold_id}_role" for fold_id in range(1, backtest_folds + 1) if f"backtest_fold_{fold_id}_role" in target_df.columns)
    return list(dict.fromkeys([col for col in cols if col in target_df.columns]))


def _single_block_target_outputs(
    target_df: pd.DataFrame,
    split_frames: SplitFrames,
    target_spec: TargetSpec,
    target_output_dir: Path,
    trainers: dict[str, TrainFn],
) -> tuple[dict[str, object], dict[str, object] | None]:
    artifacts = _run_trainers(target_df, split_frames, trainers)
    artifacts_by_name = {artifact.model_name: artifact for artifact in artifacts}
    model_metrics = pd.concat([artifact.metrics for artifact in artifacts if not artifact.metrics.empty], ignore_index=True)
    predictions = combine_prediction_tables(artifacts)
    truth_frame = build_truth_frame(target_df, target_spec)
    benchmark_df = truth_frame.merge(predictions, on=["event_id", "split"], how="left")

    comparison_models = list(artifacts_by_name.keys())
    full_available_metrics, common_subset_metrics, _ = build_full_and_common_metrics(benchmark_df, target_spec, comparison_models)

    benchmark_df.to_csv(target_output_dir / f"{target_spec.name}_predictions.csv", index=False)
    model_metrics.to_csv(target_output_dir / f"{target_spec.name}_metrics_all_rows.csv", index=False)
    full_available_metrics.to_csv(target_output_dir / f"{target_spec.name}_metrics_by_available_predictions.csv", index=False)
    common_subset_metrics.to_csv(target_output_dir / f"{target_spec.name}_metrics_common_subset.csv", index=False)

    valid_rows = model_metrics.loc[model_metrics["eval_split"].eq("valid")].copy()
    best_row = valid_rows.sort_values(["level_mae", "delta_mae"]).iloc[0].to_dict() if not valid_rows.empty else None
    best_model_name = best_row["model"] if best_row else None
    best_model_path = None
    if best_model_name:
        best_model_path = _save_model(artifacts_by_name[best_model_name], target_output_dir / "best_model")

    summary_row = {
        "target": target_spec.name,
        "best_model": best_model_name,
        "best_model_path": best_model_path,
        **(best_row or {}),
    }
    best_config = None
    if best_model_name:
        best_config = {
            "model": best_model_name,
            "target": target_spec.name,
            "next_col": target_spec.next_col,
            "delta_col": target_spec.delta_col,
            "metrics": best_row,
            "model_path": best_model_path,
            "selection_split": "valid",
        }
    best_artifact = artifacts_by_name.get(best_model_name)
    if best_artifact is not None and best_artifact.model_name == "ridge_with_latent":
        ridge_coefficient_frame(best_artifact).to_csv(target_output_dir / f"{target_spec.name}_ridge_coefficients_with_latent.csv", index=False)
    if best_artifact is not None and best_artifact.model_name == "gbdt_with_latent":
        gbdt_importance_frame(best_artifact).to_csv(target_output_dir / f"{target_spec.name}_gbdt_importance_with_latent.csv", index=False)
    return summary_row, best_config


def _walk_forward_target_outputs(
    target_df: pd.DataFrame,
    target_spec: TargetSpec,
    target_output_dir: Path,
    trainers: dict[str, TrainFn],
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object] | None]:
    split_assignments = target_df[_split_assignment_columns(target_df, args.backtest_folds)].copy()
    split_assignments.to_csv(target_output_dir / f"{target_spec.name}_split_assignments.csv", index=False)

    fold_metric_parts = []
    for fold_id in range(1, args.backtest_folds + 1):
        split_col = f"backtest_fold_{fold_id}_role"
        if split_col not in target_df.columns:
            continue
        fold_df = target_df.copy()
        fold_df["split"] = fold_df[split_col]
        split_frames = split_frames_from_assigned(fold_df)
        if split_frames.train.empty or split_frames.valid.empty:
            continue
        artifacts = _run_trainers(fold_df, split_frames, trainers)
        fold_metrics = pd.concat([artifact.metrics for artifact in artifacts if not artifact.metrics.empty], ignore_index=True)
        if fold_metrics.empty:
            continue
        fold_metrics["fold_id"] = fold_id
        fold_metric_parts.append(fold_metrics)

    backtest_fold_metrics = concat_frames(fold_metric_parts)
    backtest_summary = summarize_backtest_fold_metrics(backtest_fold_metrics)
    winner_row = select_backtest_winner(backtest_summary)
    best_model_name = str(winner_row["model"]) if winner_row else None
    if not best_model_name:
        return {"target": target_spec.name, "status": "skipped", "reason": "no walk-forward winner"}, None

    refit_df = target_df.copy()
    refit_df["split"] = refit_df["refit_role"]
    refit_frames = split_frames_from_assigned(refit_df)
    final_artifacts = _run_trainers(refit_df, refit_frames, trainers)
    final_artifacts_by_name = {artifact.model_name: artifact for artifact in final_artifacts}
    winner_artifact = final_artifacts_by_name.get(best_model_name)
    if winner_artifact is None:
        return {"target": target_spec.name, "status": "skipped", "reason": "walk-forward winner missing from refit"}, None
    best_model_path = _save_model(winner_artifact, target_output_dir / "best_model")

    final_predictions = rename_prediction_splits(
        combine_prediction_tables(final_artifacts),
        {"train": "train_dev", "valid": "train_dev", "test": "final_test"},
    )
    final_truth_df = target_df.copy()
    final_truth_df["split"] = final_truth_df["final_split"]
    benchmark_df = build_truth_frame(final_truth_df, target_spec).merge(final_predictions, on=["event_id", "split"], how="left")
    final_metrics = build_final_and_holdout_metrics(
        benchmark_df,
        target_spec,
        [artifact.model_name for artifact in final_artifacts],
    )

    backtest_fold_metrics.to_csv(target_output_dir / f"{target_spec.name}_backtest_fold_metrics.csv", index=False)
    backtest_summary.to_csv(target_output_dir / f"{target_spec.name}_backtest_summary.csv", index=False)
    final_metrics.to_csv(target_output_dir / f"{target_spec.name}_final_metrics.csv", index=False)
    benchmark_df.to_csv(target_output_dir / f"{target_spec.name}_predictions.csv", index=False)
    backtest_fold_metrics.to_csv(target_output_dir / f"{target_spec.name}_metrics_all_rows.csv", index=False)
    final_metrics.to_csv(target_output_dir / f"{target_spec.name}_metrics_by_available_predictions.csv", index=False)
    final_metrics.to_csv(target_output_dir / f"{target_spec.name}_metrics_common_subset.csv", index=False)

    if winner_artifact is not None and winner_artifact.model_name == "ridge_with_latent":
        ridge_coefficient_frame(winner_artifact).to_csv(target_output_dir / f"{target_spec.name}_ridge_coefficients_with_latent.csv", index=False)
    if winner_artifact is not None and winner_artifact.model_name == "gbdt_with_latent":
        gbdt_importance_frame(winner_artifact).to_csv(target_output_dir / f"{target_spec.name}_gbdt_importance_with_latent.csv", index=False)

    summary_row = build_winner_summary_row(target_spec.name, winner_row, final_metrics, best_model_path)
    summary_metrics = {
        "level_mae": summary_row.get("final_test_level_mae", summary_row.get("backtest_mean_level_mae")),
        "delta_mae": summary_row.get("final_test_delta_mae", summary_row.get("backtest_mean_delta_mae")),
        "backtest_mean_level_mae": summary_row.get("backtest_mean_level_mae"),
        "backtest_std_level_mae": summary_row.get("backtest_std_level_mae"),
        "backtest_mean_delta_mae": summary_row.get("backtest_mean_delta_mae"),
        "final_test_level_mae": summary_row.get("final_test_level_mae"),
        "final_test_delta_mae": summary_row.get("final_test_delta_mae"),
        "holdout_level_mae": summary_row.get("holdout_level_mae"),
        "holdout_delta_mae": summary_row.get("holdout_delta_mae"),
    }
    best_config = {
        "model": best_model_name,
        "target": target_spec.name,
        "next_col": target_spec.next_col,
        "delta_col": target_spec.delta_col,
        "metrics": summary_metrics,
        "model_path": best_model_path,
        "selection_split": "backtest_valid_mean",
    }
    return summary_row, best_config


def main() -> None:
    args = _parse_args()
    set_seed(42)

    repo_root = find_repo_root(Path.cwd())
    output_root = Path(args.output_dir) if args.output_dir else (
        repo_root / "ml_workspace" / "soh_forecast" / "output" / f"multihorizon_runner_plane_{args.primary_plane}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    latent_df, summaries = load_latent_dataset(
        repo_root=repo_root,
        primary_plane=args.primary_plane,
        holdout_plane=args.holdout_plane,
        run_latent_pipeline=args.run_latent_pipeline,
        rt_profile=args.rt_profile,
        q_day_sigma_pct=args.q_day_sigma_pct,
    )
    forecast_df = add_forecast_features(latent_df)
    feature_sets = available_feature_sets(forecast_df)

    horizons = [int(v.strip()) for v in str(args.horizons).split(",") if v.strip()]
    horizon_configs = tuple(
        {"kind": "flight", "value": value, "label": f"flight_{value}", "title": f"Next {value} flights"}
        for value in horizons
    )
    target_specs = make_multi_horizon_target_specs(
        horizon_configs=horizon_configs,
        include_observed=False,
        include_latent=True,
    )

    shared_df = None
    if args.split_scheme == "single_block":
        shared_df = assign_shared_splits(
            forecast_df,
            primary_plane=args.primary_plane,
            holdout_plane=args.holdout_plane,
            train_frac=args.train_frac,
            valid_frac=args.valid_frac,
        )

    summary_rows = []
    best_config = {}

    for target_name, target_spec in target_specs.items():
        if args.split_scheme == "single_block":
            target_df = shared_df.loc[shared_df[target_spec.next_col].notna()].copy()
            split_frames = split_frames_from_assigned(target_df)
            if split_frames.train.empty or split_frames.valid.empty:
                summary_rows.append({"target": target_name, "status": "skipped", "reason": "insufficient train/valid rows"})
                continue
        else:
            target_df = assign_walk_forward_splits(
                forecast_df,
                primary_plane=args.primary_plane,
                holdout_plane=args.holdout_plane,
                final_test_frac=args.final_test_frac,
                backtest_folds=args.backtest_folds,
                fold_valid_frac=args.fold_valid_frac,
                required_target_cols=[target_spec.next_col],
            )
            final_frames = split_frames_from_column(target_df, "refit_role")
            if final_frames.train.empty or final_frames.valid.empty or final_frames.test.empty:
                summary_rows.append({"target": target_name, "status": "skipped", "reason": "insufficient walk-forward groups"})
                continue

        target_output_dir = output_root / target_spec.name
        target_output_dir.mkdir(parents=True, exist_ok=True)

        trainers = _trainer_registry(target_df, target_spec, feature_sets, args)
        corr_features = list(
            dict.fromkeys(
                feature_sets.get("raw", [])
                + feature_sets.get("operating", [])
                + feature_sets.get("latent", [])
                + feature_sets.get("physics", [])
                + feature_sets.get("static_numeric", [])
                + feature_sets.get("history", [])
            )
        )
        corr_df = summarize_feature_correlation(target_df, corr_features, target_spec)
        corr_df.to_csv(target_output_dir / f"{target_spec.name}_feature_correlations.csv", index=False)

        if args.split_scheme == "single_block":
            summary_row, config_row = _single_block_target_outputs(target_df, split_frames, target_spec, target_output_dir, trainers)
        else:
            summary_row, config_row = _walk_forward_target_outputs(target_df, target_spec, target_output_dir, trainers, args)

        summary_rows.append(summary_row)
        if config_row:
            best_config[target_name] = config_row

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "best_models_by_horizon.csv", index=False)
    (output_root / "best_models_by_horizon.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    (output_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "primary_plane": args.primary_plane,
                "holdout_plane": args.holdout_plane,
                "split_scheme": args.split_scheme,
                "rt_profile": args.rt_profile,
                "q_day_sigma_pct": args.q_day_sigma_pct,
                "train_frac": args.train_frac,
                "valid_frac": args.valid_frac,
                "final_test_frac": args.final_test_frac,
                "backtest_folds": args.backtest_folds,
                "fold_valid_frac": args.fold_valid_frac,
                "lookback": args.lookback,
                "device": args.device,
                "horizons": horizons,
                "output_dir": str(output_root),
                "latent_summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Finished. Best models summary:")
    print(summary_df)


if __name__ == "__main__":
    main()
