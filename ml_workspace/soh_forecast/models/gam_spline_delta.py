from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from ml_workspace.soh_forecast.common import (
    ModelArtifacts,
    SplitFrames,
    TargetSpec,
    build_metric_frame,
    build_prediction_frame,
    make_feature_frame,
)


def _is_binary(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float).unique()
    return len(values) <= 2 and set(np.round(values, 10)).issubset({0.0, 1.0})


def _build_gam_preprocessor(train_x: pd.DataFrame, n_knots: int = 4) -> tuple[ColumnTransformer, list[str]]:
    binary_cols = [col for col in train_x.columns if _is_binary(train_x[col])]
    continuous_cols = [col for col in train_x.columns if col not in binary_cols]
    transformers = []
    if continuous_cols:
        transformers.append(
            (
                "continuous_splines",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("spline", SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)),
                    ]
                ),
                continuous_cols,
            )
        )
    if binary_cols:
        transformers.append(("binary_passthrough", "passthrough", binary_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    return preprocessor, continuous_cols


def train_gam_spline_delta(
    split_frames: SplitFrames,
    target_spec: TargetSpec,
    feature_cols: list[str],
    model_name: str,
    alphas: list[float] | None = None,
    n_knots_list: list[int] | None = None,
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

    gam_alphas = alphas or [0.01, 0.1, 1.0, 10.0]
    knots_list = n_knots_list or [4]
    preprocessor, continuous_cols = _build_gam_preprocessor(train_x, n_knots=knots_list[0])
    best_model = None
    best_score = np.inf
    for n_knots in knots_list:
        preprocessor, _ = _build_gam_preprocessor(train_x, n_knots=n_knots)
        for alpha in gam_alphas:
            candidate = Pipeline([("preprocess", preprocessor), ("ridge", Ridge(alpha=alpha))])
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
            "continuous_cols": continuous_cols,
            "test_frame": test_x,
            "test_target_level": y_test_level,
            "test_current": current_test,
        },
    )
