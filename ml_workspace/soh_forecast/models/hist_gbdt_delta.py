from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ml_workspace.soh_forecast.common import (
    ModelArtifacts,
    SplitFrames,
    TargetSpec,
    build_metric_frame,
    build_prediction_frame,
    make_feature_frame,
)


def train_hist_gbdt_delta(
    split_frames: SplitFrames,
    target_spec: TargetSpec,
    feature_cols: list[str],
    model_name: str,
    param_grid: list[dict] | None = None,
) -> ModelArtifacts:
    train_x, medians, dummy_cols = make_feature_frame(split_frames.train, feature_cols)
    valid_x, _, _ = make_feature_frame(split_frames.valid, feature_cols, medians, dummy_cols)
    test_x, _, _ = make_feature_frame(split_frames.test, feature_cols, medians, dummy_cols)
    holdout_x, _, _ = (
        make_feature_frame(split_frames.holdout, feature_cols, medians, dummy_cols)
        if not split_frames.holdout.empty
        else (pd.DataFrame(), medians, dummy_cols)
    )

    y_train_level = split_frames.train[target_spec.next_col].to_numpy(dtype=float)
    y_valid_level = split_frames.valid[target_spec.next_col].to_numpy(dtype=float)
    y_test_level = split_frames.test[target_spec.next_col].to_numpy(dtype=float)
    y_holdout_level = split_frames.holdout[target_spec.next_col].to_numpy(dtype=float) if not split_frames.holdout.empty else np.array([], dtype=float)

    current_train = split_frames.train[target_spec.current_col].to_numpy(dtype=float)
    current_valid = split_frames.valid[target_spec.current_col].to_numpy(dtype=float)
    current_test = split_frames.test[target_spec.current_col].to_numpy(dtype=float)
    current_holdout = split_frames.holdout[target_spec.current_col].to_numpy(dtype=float) if not split_frames.holdout.empty else np.array([], dtype=float)

    y_train_delta = y_train_level - current_train
    y_valid_delta = y_valid_level - current_valid

    best_model = None
    best_score = np.inf
    grid = param_grid or [
        {"learning_rate": 0.03, "max_depth": 3, "max_iter": 300, "min_samples_leaf": 10, "random_state": 42},
        {"learning_rate": 0.05, "max_depth": 4, "max_iter": 300, "min_samples_leaf": 10, "random_state": 42},
        {"learning_rate": 0.05, "max_depth": 3, "max_iter": 500, "min_samples_leaf": 5, "random_state": 42},
    ]
    for params in grid:
        candidate = HistGradientBoostingRegressor(**params)
        candidate.fit(train_x, y_train_delta)
        score = float(np.mean(np.abs(y_valid_delta - candidate.predict(valid_x))))
        if score < best_score:
            best_model = candidate
            best_score = score

    pred_train_level = current_train + best_model.predict(train_x)
    pred_valid_level = current_valid + best_model.predict(valid_x)
    pred_test_level = current_test + best_model.predict(test_x) if not split_frames.test.empty else np.array([], dtype=float)
    pred_holdout_level = current_holdout + best_model.predict(holdout_x) if not split_frames.holdout.empty else np.array([], dtype=float)

    split_predictions = {
        "train": pred_train_level,
        "valid": pred_valid_level,
        "test": pred_test_level,
        "holdout": pred_holdout_level,
    }
    predictions = build_prediction_frame(split_frames, model_name, split_predictions)
    metrics = build_metric_frame(split_frames, target_spec, model_name, split_predictions)

    return ModelArtifacts(
        model_name=model_name,
        predictions=predictions,
        metrics=metrics,
        model=best_model,
        feature_names=list(train_x.columns),
        diagnostics={
            "test_frame": test_x,
            "test_target_level": y_test_level,
            "test_current": current_test,
        },
    )
