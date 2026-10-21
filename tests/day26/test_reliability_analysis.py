import numpy as np
import pandas as pd
import pytest

from src.day26.reliability_analysis import (
    regression_metrics,
    add_diagnostics,
    grouped_metrics,
    build_feature_availability_audit,
)


def test_regression_metrics_exact_predictions():
    result = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert result["rows"] == 3
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0
    assert result["mean_error_predicted_minus_actual"] == 0.0


def test_regression_metrics_rejects_non_finite_predictions():
    with pytest.raises(ValueError):
        regression_metrics([1.0, 2.0], [1.0, np.nan])


def test_mape_excludes_zero_targets():
    result = regression_metrics([0.0, 10.0], [5.0, 12.0])
    assert result["mean_absolute_percentage_error_denominator_rows_excluded"] == 1
    assert result["mape_percent_nonzero_targets_only"] == pytest.approx(20.0)


def test_add_diagnostics_preserves_index_and_direction():
    source = pd.DataFrame({"road_id": ["A", "B"]}, index=[4, 8])
    actual = pd.Series([10.0, 20.0], index=[4, 8])
    result = add_diagnostics(source, actual, np.array([12.0, 17.0]))
    assert result.index.tolist() == [4, 8]
    assert result["prediction_direction"].tolist() == ["overprediction", "underprediction"]
    assert result["absolute_error"].tolist() == [2.0, 3.0]


def test_grouped_metrics_orders_worst_rmse_first():
    frame = pd.DataFrame({
        "road": ["A", "A", "B", "B"],
        "actual_traffic_volume": [10.0, 10.0, 10.0, 10.0],
        "predicted_traffic_volume": [10.0, 10.0, 20.0, 0.0],
    })
    result = grouped_metrics(frame, "road", minimum_rows=1)
    assert result.iloc[0]["road"] == "B"
    assert result.iloc[0]["rmse"] > result.iloc[1]["rmse"]


def test_feature_audit_flags_manual_review():
    result = build_feature_availability_audit([
        "public_holiday", "event_flag", "roadwork_flag", "traffic_lag_30m"
    ])
    assert result["status"] == "REQUIRES_MANUAL_VERIFICATION"
    assert len(result["manual_review"]) == 3
    assert "traffic_lag_30m" in result["lag_and_rolling_features"]["features"]
