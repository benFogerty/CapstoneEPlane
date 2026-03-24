from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ml_workspace.soh_forecast.common import ModelArtifacts, SplitFrames, TargetSpec, concat_frames, metric_table


def combine_prediction_tables(artifacts: Iterable[ModelArtifacts]) -> pd.DataFrame:
    artifacts = list(artifacts)
    if not artifacts:
        return pd.DataFrame(columns=["event_id", "split"])
    combined = artifacts[0].predictions.copy()
    for artifact in artifacts[1:]:
        combined = combined.merge(artifact.predictions, on=["event_id", "split"], how="outer")
    return combined


def metrics_from_prediction_df(
    prediction_df: pd.DataFrame,
    split_frames: SplitFrames,
    target_spec: TargetSpec,
    model_name: str,
    pred_col: str,
) -> pd.DataFrame:
    rows = []
    for split_name, split_frame in [
        ("train", split_frames.train),
        ("valid", split_frames.valid),
        ("test", split_frames.test),
        ("holdout", split_frames.holdout),
    ]:
        if split_frame.empty:
            continue
        pred_frame = prediction_df.loc[(prediction_df["split"] == split_name) & prediction_df[pred_col].notna(), ["event_id", pred_col]]
        if pred_frame.empty:
            continue
        eval_frame = split_frame.merge(pred_frame, on="event_id", how="inner")
        rows.append(
            {
                "model": model_name,
                "eval_split": split_name,
                **metric_table(
                    eval_frame[target_spec.next_col].to_numpy(dtype=float),
                    eval_frame[pred_col].to_numpy(dtype=float),
                    eval_frame[target_spec.current_col].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_truth_frame(predictive_df: pd.DataFrame, target_spec: TargetSpec) -> pd.DataFrame:
    cols = list(
        dict.fromkeys(
            [
                "event_id",
                "split",
                "plane_id",
                "battery_id",
                "event_datetime",
                "cumulative_flight_count",
                "event_type",
                "observed_soh_pct",
                "latent_soh_filter_pct",
                target_spec.current_col,
                target_spec.next_col,
            ]
        )
    )
    return predictive_df[cols].copy()


def comparison_metrics(
    benchmark_df: pd.DataFrame,
    target_spec: TargetSpec,
    model_cols: list[str],
    split_name: str,
) -> pd.DataFrame:
    eval_df = benchmark_df.loc[benchmark_df["split"].eq(split_name)].copy()
    rows = []
    for model_col in model_cols:
        sub = eval_df.loc[eval_df[model_col].notna()].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "model": model_col,
                "eval_split": split_name,
                "coverage_rows": int(len(sub)),
                **metric_table(
                    sub[target_spec.next_col].to_numpy(dtype=float),
                    sub[model_col].to_numpy(dtype=float),
                    sub[target_spec.current_col].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_full_and_common_metrics(
    benchmark_df: pd.DataFrame,
    target_spec: TargetSpec,
    model_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_model_cols = [col for col in model_cols if col in benchmark_df.columns]
    common_subset = benchmark_df.dropna(subset=available_model_cols).copy() if available_model_cols else benchmark_df.iloc[0:0].copy()
    split_names = [value for value in benchmark_df["split"].dropna().unique().tolist() if value]
    full_available_metrics = concat_frames([comparison_metrics(benchmark_df, target_spec, available_model_cols, split_name) for split_name in split_names])
    common_subset_metrics = concat_frames([comparison_metrics(common_subset, target_spec, available_model_cols, split_name) for split_name in split_names])
    return full_available_metrics, common_subset_metrics, common_subset


def summarize_backtest_fold_metrics(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "backtest_folds",
                "backtest_mean_level_mae",
                "backtest_std_level_mae",
                "backtest_mean_delta_mae",
                "backtest_mean_level_rmse",
                "backtest_mean_delta_rmse",
                "backtest_mean_level_r2",
                "backtest_mean_delta_r2",
                "backtest_total_rows",
            ]
        )
    valid_rows = fold_metrics_df.loc[fold_metrics_df["eval_split"].eq("valid")].copy()
    if valid_rows.empty:
        return pd.DataFrame()
    summary = (
        valid_rows.groupby("model", as_index=False)
        .agg(
            backtest_folds=("fold_id", "nunique"),
            backtest_mean_level_mae=("level_mae", "mean"),
            backtest_std_level_mae=("level_mae", "std"),
            backtest_mean_delta_mae=("delta_mae", "mean"),
            backtest_mean_level_rmse=("level_rmse", "mean"),
            backtest_mean_delta_rmse=("delta_rmse", "mean"),
            backtest_mean_level_r2=("level_r2", "mean"),
            backtest_mean_delta_r2=("delta_r2", "mean"),
            backtest_total_rows=("n", "sum"),
        )
        .sort_values(
            ["backtest_mean_level_mae", "backtest_std_level_mae", "backtest_mean_delta_mae", "model"],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )
    summary["backtest_std_level_mae"] = summary["backtest_std_level_mae"].fillna(0.0)
    return summary


def select_backtest_winner(backtest_summary: pd.DataFrame) -> dict[str, object] | None:
    if backtest_summary.empty:
        return None
    best = backtest_summary.sort_values(
        ["backtest_mean_level_mae", "backtest_std_level_mae", "backtest_mean_delta_mae", "model"],
        ascending=[True, True, True, True],
    ).iloc[0]
    return best.to_dict()


def rename_prediction_splits(
    prediction_df: pd.DataFrame,
    split_map: dict[str, str],
) -> pd.DataFrame:
    renamed = prediction_df.copy()
    renamed["split"] = renamed["split"].replace(split_map)
    return renamed


def rename_metric_splits(
    metrics_df: pd.DataFrame,
    split_map: dict[str, str],
) -> pd.DataFrame:
    renamed = metrics_df.copy()
    renamed["eval_split"] = renamed["eval_split"].replace(split_map)
    return renamed


def build_winner_summary_row(
    target_name: str,
    winner_row: dict[str, object] | None,
    final_metrics: pd.DataFrame,
    model_path: str | None,
) -> dict[str, object]:
    row: dict[str, object] = {"target": target_name, "best_model_path": model_path}
    if winner_row:
        row["best_model"] = winner_row.get("model")
        row.update(winner_row)
    else:
        row["best_model"] = None
        return row

    final_test = final_metrics.loc[final_metrics["eval_split"].eq("final_test")]
    holdout = final_metrics.loc[final_metrics["eval_split"].eq("holdout")]
    if len(final_test):
        final_test_row = final_test.iloc[0].to_dict()
        for key, value in final_test_row.items():
            if key in {"model", "eval_split"}:
                continue
            row[f"final_test_{key}"] = value
    if len(holdout):
        holdout_row = holdout.iloc[0].to_dict()
        for key, value in holdout_row.items():
            if key in {"model", "eval_split"}:
                continue
            row[f"holdout_{key}"] = value
    row["selection_split"] = "backtest_valid_mean"
    row["eval_split"] = "backtest"
    row["level_mae"] = row.get("backtest_mean_level_mae")
    row["delta_mae"] = row.get("backtest_mean_delta_mae")
    return row


def build_final_and_holdout_metrics(
    benchmark_df: pd.DataFrame,
    target_spec: TargetSpec,
    model_names: str | list[str],
) -> pd.DataFrame:
    model_cols = [model_names] if isinstance(model_names, str) else list(model_names)
    return concat_frames(
        [
            comparison_metrics(benchmark_df, target_spec, model_cols, "final_test"),
            comparison_metrics(benchmark_df, target_spec, model_cols, "holdout"),
        ]
    )


def summarize_feature_correlation(
    predictive_df: pd.DataFrame,
    feature_cols: list[str],
    target_spec: TargetSpec,
) -> pd.DataFrame:
    corr_df = (
        predictive_df[feature_cols + [target_spec.delta_col, target_spec.next_col]]
        .apply(pd.to_numeric, errors="coerce")
        .corr(numeric_only=True)[target_spec.delta_col]
        .dropna()
        .drop(target_spec.delta_col)
        .rename(f"corr_with_{target_spec.delta_col}")
        .to_frame()
        .assign(abs_corr=lambda x: x.iloc[:, 0].abs())
        .sort_values("abs_corr", ascending=False)
    )
    return corr_df


def get_metric(metrics_df: pd.DataFrame, split_name: str, model_name: str, metric_name: str) -> float:
    sub = metrics_df.loc[metrics_df["eval_split"].eq(split_name) & metrics_df["model"].eq(model_name), metric_name]
    return float(sub.iloc[0]) if len(sub) else np.nan


def ridge_coefficient_frame(artifact: ModelArtifacts) -> pd.DataFrame:
    coef = getattr(artifact.model, "coef_", None)
    if coef is None:
        return pd.DataFrame(columns=["feature", "coef", "abs_coef"])
    return (
        pd.DataFrame({"feature": artifact.feature_names, "coef": coef, "abs_coef": np.abs(coef)})
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )


def gbdt_importance_frame(artifact: ModelArtifacts) -> pd.DataFrame:
    if artifact.model is None:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    test_frame = artifact.diagnostics.get("test_frame")
    test_target_level = artifact.diagnostics.get("test_target_level")
    test_current = artifact.diagnostics.get("test_current")
    if test_frame is None or test_target_level is None or test_current is None or len(test_frame) == 0:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    perm = permutation_importance(
        artifact.model,
        test_frame,
        test_target_level - test_current,
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    return (
        pd.DataFrame(
            {
                "feature": artifact.feature_names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
