"""
FlowCast Day 15
Tests for forecast error analysis and performance intelligence.
"""

import pandas as pd
import pytest

from src.day15.error_analysis import (
    build_common_population_comparison,
    build_road_analysis,
    build_temporal_analysis,
    calculate_metrics,
    validate_predictions,
)


def sample_predictions():
    """Create a small deterministic prediction dataset."""

    records = [
        {
            "road_id": "NL-001",
            "timestamp": "2025-04-07 00:00:00",
            "actual": 10.0,
            "predicted": 11.0,
            "reference_timestamp": "2025-04-06 23:30:00",
            "model": "naive_30min",
        },
        {
            "road_id": "NL-001",
            "timestamp": "2025-04-07 00:00:00",
            "actual": 10.0,
            "predicted": 12.0,
            "reference_timestamp": "2025-04-06 00:00:00",
            "model": "seasonal_naive_24h",
        },
        {
            "road_id": "NL-001",
            "timestamp": "2025-04-07 00:30:00",
            "actual": 20.0,
            "predicted": None,
            "reference_timestamp": "2025-04-07 00:00:00",
            "model": "naive_30min",
        },
        {
            "road_id": "NL-001",
            "timestamp": "2025-04-07 00:30:00",
            "actual": 20.0,
            "predicted": 19.0,
            "reference_timestamp": "2025-04-06 00:30:00",
            "model": "seasonal_naive_24h",
        },
        {
            "road_id": "NL-002",
            "timestamp": "2025-04-07 01:00:00",
            "actual": 30.0,
            "predicted": 29.0,
            "reference_timestamp": "2025-04-07 00:30:00",
            "model": "naive_30min",
        },
        {
            "road_id": "NL-002",
            "timestamp": "2025-04-07 01:00:00",
            "actual": 30.0,
            "predicted": 31.0,
            "reference_timestamp": "2025-04-06 01:00:00",
            "model": "seasonal_naive_24h",
        },
        {
            "road_id": "NL-002",
            "timestamp": "2025-04-07 01:30:00",
            "actual": 40.0,
            "predicted": 39.0,
            "reference_timestamp": "2025-04-07 01:00:00",
            "model": "naive_30min",
        },
        {
            "road_id": "NL-002",
            "timestamp": "2025-04-07 01:30:00",
            "actual": 40.0,
            "predicted": None,
            "reference_timestamp": "2025-04-06 01:30:00",
            "model": "seasonal_naive_24h",
        },
    ]

    return pd.DataFrame(records)


def test_validation_converts_timestamps():
    dataframe = validate_predictions(
        sample_predictions()
    )

    assert pd.api.types.is_datetime64_any_dtype(
        dataframe["timestamp"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        dataframe["reference_timestamp"]
    )


def test_validation_allows_missing_predictions():
    dataframe = validate_predictions(
        sample_predictions()
    )

    assert dataframe["predicted"].isna().sum() == 2


def test_validation_rejects_missing_required_columns():
    dataframe = sample_predictions().drop(
        columns=["road_id"]
    )

    with pytest.raises(ValueError):
        validate_predictions(dataframe)


def test_validation_rejects_empty_dataframe():
    dataframe = sample_predictions().iloc[0:0]

    with pytest.raises(ValueError):
        validate_predictions(dataframe)


def test_validation_rejects_duplicate_prediction_records():
    dataframe = sample_predictions()

    duplicate = dataframe.iloc[[0]]

    dataframe = pd.concat(
        [dataframe, duplicate],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        validate_predictions(dataframe)


def test_validation_rejects_missing_actual_values():
    dataframe = sample_predictions()

    dataframe.loc[0, "actual"] = float("nan")

    with pytest.raises(ValueError):
        validate_predictions(dataframe)


def test_metrics_exclude_missing_predictions():
    actual = pd.Series([10.0, 20.0, 30.0])

    predicted = pd.Series([12.0, float("nan"), 27.0])

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    assert metrics["valid_predictions"] == 2
    assert metrics["mae"] == pytest.approx(2.5)
    assert metrics["rmse"] == pytest.approx(
    (13.0 / 2) ** 0.5
    )
    assert metrics["bias"] == pytest.approx(-0.5)


def test_metrics_return_nan_when_no_predictions_are_valid():
    actual = pd.Series([10.0, 20.0])

    predicted = pd.Series(
        [float("nan"), float("nan")]
    )

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    assert metrics["valid_predictions"] == 0
    assert pd.isna(metrics["mae"])
    assert pd.isna(metrics["rmse"])


def test_common_population_uses_same_observations():
    dataframe = validate_predictions(
        sample_predictions()
    )

    comparison, common = (
        build_common_population_comparison(
            dataframe
        )
    )

    assert len(common) == 2

    assert set(
        comparison["common_valid_observations"]
    ) == {2}

    assert set(
        comparison["model"]
    ) == {
        "naive_30min",
        "seasonal_naive_24h",
    }


def test_common_population_metrics_are_correct():
    dataframe = validate_predictions(
        sample_predictions()
    )

    comparison, _ = (
        build_common_population_comparison(
            dataframe
        )
    )

    naive = comparison.loc[
        comparison["model"] == "naive_30min"
    ].iloc[0]

    seasonal = comparison.loc[
        comparison["model"] == "seasonal_naive_24h"
    ].iloc[0]

    assert naive["mae"] == pytest.approx(1.0)
    assert naive["rmse"] == pytest.approx(1.0)

    assert seasonal["mae"] == pytest.approx(1.5)
    assert seasonal["rmse"] == pytest.approx(
        (2.5 ** 0.5)
    )


def test_road_analysis_reports_prediction_coverage():
    dataframe = validate_predictions(
        sample_predictions()
    )

    roads = build_road_analysis(
        dataframe
    )

    assert len(roads) == 4

    naive_nl001 = roads.loc[
        (roads["road_id"] == "NL-001")
        & (roads["model"] == "naive_30min")
    ].iloc[0]

    assert naive_nl001["validation_rows"] == 2
    assert naive_nl001["valid_predictions"] == 1
    assert naive_nl001["missing_predictions"] == 1
    assert naive_nl001["coverage_pct"] == pytest.approx(
        50.0
    )


def test_temporal_analysis_creates_hourly_and_daily_results():
    dataframe = validate_predictions(
        sample_predictions()
    )

    hourly, daily = build_temporal_analysis(
        dataframe
    )

    assert not hourly.empty
    assert not daily.empty

    assert {
        "model",
        "hour",
        "observations",
        "mae",
        "rmse",
    }.issubset(hourly.columns)

    assert {
        "model",
        "day_of_week",
        "observations",
        "mae",
        "rmse",
    }.issubset(daily.columns)