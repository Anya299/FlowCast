import pandas as pd
import pytest

from src.day16.evaluation_audit import (
    audit_chronological_split,
    audit_unique_road_timestamps,
    audit_feature_target_separation,
    audit_prediction_alignment,
    build_common_evaluation_population,
)


def test_valid_chronological_split_passes():
    train = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-04-06 23:00",
            "2025-04-06 23:30",
        ])
    })

    validation = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-04-07 00:00",
            "2025-04-07 00:30",
        ])
    })

    result = audit_chronological_split(
        train,
        validation,
    )

    assert result.passed is True


def test_overlapping_chronological_split_fails():
    train = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-04-07 00:00",
        ])
    })

    validation = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-04-07 00:00",
        ])
    })

    result = audit_chronological_split(
        train,
        validation,
    )

    assert result.passed is False


def test_empty_split_raises_error():
    train = pd.DataFrame({
        "timestamp": pd.to_datetime([])
    })

    validation = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-04-07 00:00",
        ])
    })

    with pytest.raises(ValueError):
        audit_chronological_split(
            train,
            validation,
        )


def test_duplicate_road_timestamp_fails():
    df = pd.DataFrame({
        "road_id": ["R1", "R1"],
        "timestamp": pd.to_datetime([
            "2025-04-07 00:00",
            "2025-04-07 00:00",
        ]),
    })

    result = audit_unique_road_timestamps(df)

    assert result.passed is False


def test_unique_road_timestamp_passes():
    df = pd.DataFrame({
        "road_id": ["R1", "R1"],
        "timestamp": pd.to_datetime([
            "2025-04-07 00:00",
            "2025-04-07 00:30",
        ]),
    })

    result = audit_unique_road_timestamps(df)

    assert result.passed is True


def test_target_in_feature_list_fails():
    result = audit_feature_target_separation([
        "traffic_lag_30m",
        "traffic_volume",
    ])

    assert result.passed is False


def test_forbidden_feature_fails():
    result = audit_feature_target_separation(
        ["traffic_lag_30m", "future_traffic_volume"],
        forbidden_columns=["future_traffic_volume"],
    )

    assert result.passed is False


def test_clean_feature_list_passes():
    result = audit_feature_target_separation([
        "traffic_lag_30m",
        "hour_of_day",
    ])

    assert result.passed is True


def test_duplicate_feature_names_fail():
    result = audit_feature_target_separation([
        "traffic_lag_30m",
        "traffic_lag_30m",
    ])

    assert result.passed is False


def test_prediction_with_wrong_horizon_fails():
    predictions = pd.DataFrame({
        "road_id": ["R1"],
        "timestamp": pd.to_datetime([
            "2025-04-07 01:00",
        ]),
        "reference_timestamp": pd.to_datetime([
            "2025-04-07 00:00",
        ]),
        "actual": [100.0],
        "predicted": [95.0],
    })

    result = audit_prediction_alignment(
        predictions,
        forecast_horizon_minutes=30,
    )

    assert result.passed is False


def test_prediction_with_correct_horizon_passes():
    predictions = pd.DataFrame({
        "road_id": ["R1"],
        "timestamp": pd.to_datetime([
            "2025-04-07 01:00",
        ]),
        "reference_timestamp": pd.to_datetime([
            "2025-04-07 00:30",
        ]),
        "actual": [100.0],
        "predicted": [95.0],
    })

    result = audit_prediction_alignment(
        predictions,
        forecast_horizon_minutes=30,
    )

    assert result.passed is True


def test_common_population_keeps_only_shared_valid_rows():
    predictions = pd.DataFrame({
        "road_id": ["R1", "R1", "R1", "R1"],
        "timestamp": pd.to_datetime([
            "2025-04-07 00:00",
            "2025-04-07 00:00",
            "2025-04-07 00:30",
            "2025-04-07 00:30",
        ]),
        "model": [
            "naive",
            "seasonal",
            "naive",
            "seasonal",
        ],
        "actual": [100, 100, 110, 110],
        "predicted": [95, 90, 105, None],
    })

    result = build_common_evaluation_population(
        predictions
    )

    assert len(result) == 2
    assert result["timestamp"].nunique() == 1
    assert set(result["model"]) == {
        "naive",
        "seasonal",
    }


def test_common_population_rejects_disagreeing_actuals():
    predictions = pd.DataFrame({
        "road_id": ["R1", "R1"],
        "timestamp": pd.to_datetime([
            "2025-04-07 00:00",
            "2025-04-07 00:00",
        ]),
        "model": ["naive", "seasonal"],
        "actual": [100, 101],
        "predicted": [95, 90],
    })

    with pytest.raises(ValueError):
        build_common_evaluation_population(
            predictions
        )


def test_common_population_rejects_duplicate_model_rows():
    predictions = pd.DataFrame({
        "road_id": ["R1", "R1"],
        "timestamp": pd.to_datetime([
            "2025-04-07 00:00",
            "2025-04-07 00:00",
        ]),
        "model": ["naive", "naive"],
        "actual": [100, 100],
        "predicted": [95, 96],
    })

    with pytest.raises(ValueError):
        build_common_evaluation_population(
            predictions
        )