from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class TargetSpec:
    name: str
    current_col: str
    next_col: str
    delta_col: str
    title_label: str


@dataclass
class SplitFrames:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    holdout: pd.DataFrame


@dataclass
class ModelArtifacts:
    model_name: str
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    model: Any | None = None
    feature_names: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "ml_workspace").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError("Could not locate repo root")


def resolve_timeseries_path(repo_root: Path) -> Path:
    preferred = repo_root / "data" / "event_timeseries_corrected.parquet"
    fallback = repo_root / "data" / "event_timeseries.parquet"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not find local event_timeseries parquet")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_feature_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    medians_ref: pd.Series | None = None,
    dummy_cols_ref: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cols = list(dict.fromkeys([col for col in feature_cols if col in df.columns]))
    numeric = df[cols].apply(pd.to_numeric, errors="coerce").copy()
    medians = numeric.median().fillna(0.0) if medians_ref is None else medians_ref
    numeric = numeric.fillna(medians).fillna(0.0)

    event_dummies = pd.get_dummies(df["event_type"].fillna("unknown"), prefix="event_type")
    battery_dummies = pd.get_dummies(df["battery_id_str"].fillna("unknown"), prefix="battery")
    plane_dummies = pd.get_dummies(df["plane_id"].fillna("unknown"), prefix="plane")
    dummies = pd.concat([event_dummies, battery_dummies, plane_dummies], axis=1)

    if dummy_cols_ref is None:
        dummy_cols = list(dummies.columns)
    else:
        dummy_cols = list(dummy_cols_ref)
        for col in dummy_cols:
            if col not in dummies.columns:
                dummies[col] = 0
    dummies = dummies[dummy_cols]
    return pd.concat([numeric.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1), medians, dummy_cols


def fit_best_linear(model_cls, X_train, y_train, X_valid, y_valid, grid):
    best_model = None
    best_score = np.inf
    for params in grid:
        model = model_cls(**params)
        model.fit(X_train, y_train)
        score = mean_absolute_error(y_valid, model.predict(X_valid))
        if score < best_score:
            best_model = model
            best_score = score
    return best_model


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if len(y_true) and denom > 1e-12 else np.nan


def metric_table(y_true_level: np.ndarray, y_pred_level: np.ndarray, current_level: np.ndarray) -> dict[str, float]:
    y_true_delta = y_true_level - current_level
    y_pred_delta = y_pred_level - current_level
    return {
        "n": int(len(y_true_level)),
        "level_mae": float(mean_absolute_error(y_true_level, y_pred_level)),
        "level_rmse": float(np.sqrt(mean_squared_error(y_true_level, y_pred_level))),
        "level_r2": r2_safe(y_true_level, y_pred_level),
        "delta_mae": float(mean_absolute_error(y_true_delta, y_pred_delta)),
        "delta_rmse": float(np.sqrt(mean_squared_error(y_true_delta, y_pred_delta))),
        "delta_r2": r2_safe(y_true_delta, y_pred_delta),
    }


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()


def split_frame_items(split_frames: SplitFrames) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("train", split_frames.train),
        ("valid", split_frames.valid),
        ("test", split_frames.test),
        ("holdout", split_frames.holdout),
    ]


def build_prediction_frame(
    split_frames: SplitFrames,
    model_name: str,
    split_predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    parts = []
    for split_name, frame in split_frame_items(split_frames):
        preds = split_predictions.get(split_name)
        if frame.empty or preds is None or len(preds) == 0:
            continue
        parts.append(pd.DataFrame({"event_id": frame["event_id"], "split": split_name, model_name: preds}))
    return concat_frames(parts)


def build_metric_frame(
    split_frames: SplitFrames,
    target_spec: TargetSpec,
    model_name: str,
    split_predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for split_name, frame in split_frame_items(split_frames):
        preds = split_predictions.get(split_name)
        if frame.empty or preds is None or len(preds) == 0:
            continue
        rows.append(
            {
                "model": model_name,
                "eval_split": split_name,
                **metric_table(
                    frame[target_spec.next_col].to_numpy(dtype=float),
                    np.asarray(preds, dtype=float),
                    frame[target_spec.current_col].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)
