from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from ml_workspace.soh_forecast.common import (
    ModelArtifacts,
    SplitFrames,
    TargetSpec,
    build_metric_frame,
    build_prediction_frame,
    fit_best_linear,
    make_feature_frame,
)


def train_elastic_net_delta(
    split_frames: SplitFrames,
    target_spec: TargetSpec,
    feature_cols: list[str],
    model_name: str,
    grid: list[dict] | None = None,
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

    scaler = StandardScaler()
    train_x_s = scaler.fit_transform(train_x)
    valid_x_s = scaler.transform(valid_x)
    test_x_s = scaler.transform(test_x) if not split_frames.test.empty else np.empty((0, train_x.shape[1]))
    holdout_x_s = scaler.transform(holdout_x) if not split_frames.holdout.empty else np.empty((0, train_x.shape[1]))

    if grid is None:
        grid = [
            {
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "max_iter": 50000,
                "tol": 1e-3,
                "selection": "random",
                "random_state": 42,
            }
            for alpha in [0.01, 0.1, 1.0, 10.0]
            for l1_ratio in [0.1, 0.5, 0.9]
        ]
    model = fit_best_linear(ElasticNet, train_x_s, y_train_delta, valid_x_s, y_valid_delta, grid)

    pred_train_level = current_train + model.predict(train_x_s)
    pred_valid_level = current_valid + model.predict(valid_x_s)
    pred_test_level = current_test + model.predict(test_x_s) if not split_frames.test.empty else np.array([], dtype=float)
    pred_holdout_level = current_holdout + model.predict(holdout_x_s) if not split_frames.holdout.empty else np.array([], dtype=float)

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
        model=model,
        feature_names=list(train_x.columns),
        diagnostics={"scaler": scaler},
    )
