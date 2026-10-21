"""Unit tests for the FlowCast Day 25 model pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.day25.model_pipeline import (
    FEATURES,
    TARGET,
    build_pipeline,
    prepare_xy,
    regression_metrics,
    validate_feature_contract,
)


def valid_frame(rows: int = 5) -> pd.DataFrame:
    data = {name: np.arange(rows, dtype=float) + 1 for name in FEATURES}
    data[TARGET] = np.arange(rows, dtype=float) + 10
    data["vehicle_count"] = np.arange(rows, dtype=float) + 100
    return pd.DataFrame(data)


def test_feature_contract_uses_exact_allowlist():
    frame = valid_frame()
    X, y = prepare_xy(frame, "unit-test")
    assert list(X.columns) == FEATURES
    assert TARGET not in X.columns
    assert "vehicle_count" not in X.columns
    assert len(X) == len(y)


def test_missing_feature_fails_fast():
    frame = valid_frame().drop(columns=[FEATURES[0]])
    with pytest.raises(ValueError, match="missing required model features"):
        validate_feature_contract(frame, "unit-test")


def test_invalid_target_rows_are_removed():
    frame = valid_frame()
    frame.loc[0, TARGET] = np.nan
    frame.loc[1, TARGET] = -1
    X, y = prepare_xy(frame, "unit-test")
    assert len(X) == 3
    assert len(y) == 3
    assert (y >= 0).all()


def test_non_finite_feature_values_become_missing_for_pipeline():
    frame = valid_frame()
    frame.loc[0, FEATURES[0]] = np.inf
    X, _ = prepare_xy(frame, "unit-test")
    assert np.isnan(X.loc[0, FEATURES[0]])


def test_pipeline_contains_train_fitted_imputer_and_regressor():
    pipeline = build_pipeline()
    assert list(pipeline.named_steps) == ["imputer", "regressor"]


def test_regression_metrics_are_finite_and_guard_zero_mape():
    metrics = regression_metrics(
        np.array([0.0, 10.0, 20.0]),
        np.array([1.0, 12.0, 18.0]),
    )
    assert metrics["rows"] == 3
    assert metrics["mae"] == pytest.approx(5 / 3)
    assert metrics["zero_target_rows_excluded_from_mape"] == 1
    assert np.isfinite(metrics["rmse"])
