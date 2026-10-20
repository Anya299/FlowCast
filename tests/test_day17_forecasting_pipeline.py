import numpy as np
import pandas as pd
import pytest

from src.day17.forecasting_pipeline import (
    audit_predictions,
    calculate_metrics,
    compare_on_common_population,
    generate_reference_predictions,
    load_dataset,
    validate_chronological_split,
)


def make_dataset(
    timestamps,
    values,
    road_id="R1",
):
    return pd.DataFrame(
        {
            "road_id": [road_id] * len(timestamps),
            "timestamp": pd.to_datetime(timestamps),
            "traffic_volume": values,
        }
    )


def test_load_dataset_accepts_valid_data(tmp_path):
    path = tmp_path / "valid.csv"

    pd.DataFrame(
        {
            "road_id": ["R1", "R1"],
            "timestamp": [
                "2025-01-01 00:00:00",
                "2025-01-01 00:30:00",
            ],
            "traffic_volume": [100, 120],
        }
    ).to_csv(path, index=False)

    result = load_dataset(path, "test")

    assert len(result) == 2
    assert str(result["timestamp"].dtype).startswith(
        "datetime64"
    )


def test_load_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "invalid.csv"

    pd.DataFrame(
        {
            "road_id": ["R1"],
            "timestamp": ["2025-01-01 00:00:00"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path, "test")


def test_load_dataset_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicates.csv"

    pd.DataFrame(
        {
            "road_id": ["R1", "R1"],
            "timestamp": [
                "2025-01-01 00:00:00",
                "2025-01-01 00:00:00",
            ],
            "traffic_volume": [100, 101],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path, "test")


def test_chronological_split_passes():
    train = make_dataset(
        ["2025-01-01 00:00:00"],
        [100],
    )

    validation = make_dataset(
        ["2025-01-01 00:30:00"],
        [110],
    )

    result = validate_chronological_split(
        train,
        validation,
    )

    assert result["chronological_split_passed"] is True


def test_chronological_split_rejects_overlap():
    train = make_dataset(
        ["2025-01-01 00:00:00"],
        [100],
    )

    validation = make_dataset(
        ["2025-01-01 00:00:00"],
        [110],
    )

    with pytest.raises(ValueError, match="Chronological split failed"):
        validate_chronological_split(
            train,
            validation,
        )


def test_naive_prediction_uses_30_minute_reference():
    history = make_dataset(
        [
            "2025-01-01 00:00:00",
            "2025-01-01 00:30:00",
        ],
        [100, 120],
    )

    forecast_rows = make_dataset(
        ["2025-01-01 00:30:00"],
        [120],
    )

    result = generate_reference_predictions(
        history=history,
        forecast_rows=forecast_rows,
        horizon_minutes=30,
        model_name="naive_30min",
    )

    assert result.iloc[0]["predicted"] == 100
    assert result.iloc[0]["reference_timestamp"] == pd.Timestamp(
        "2025-01-01 00:00:00"
    )


def test_missing_reference_stays_missing():
    history = make_dataset(
        ["2025-01-01 00:00:00"],
        [100],
    )

    forecast_rows = make_dataset(
        ["2025-01-01 01:00:00"],
        [130],
    )

    result = generate_reference_predictions(
        history=history,
        forecast_rows=forecast_rows,
        horizon_minutes=30,
        model_name="naive_30min",
    )

    assert pd.isna(result.iloc[0]["predicted"])
    assert (
        result.iloc[0]["prediction_status"]
        == "missing_historical_reference"
    )


def test_metrics_calculate_correct_values():
    predictions = pd.DataFrame(
        {
            "model": ["test", "test", "test"],
            "actual": [100, 200, 300],
            "predicted": [110, 180, 330],
        }
    )

    result = calculate_metrics(predictions)

    assert result["valid_predictions"] == 3
    assert result["coverage_percent"] == 100.0
    assert result["mae"] == pytest.approx(20.0)
    assert result["rmse"] == pytest.approx(
        np.sqrt((100 + 400 + 900) / 3)
    )
    assert result["bias"] == pytest.approx(20 / 3)


def test_metrics_handle_missing_predictions():
    predictions = pd.DataFrame(
        {
            "model": ["test", "test"],
            "actual": [100, 200],
            "predicted": [110, np.nan],
        }
    )

    result = calculate_metrics(predictions)

    assert result["valid_predictions"] == 1
    assert result["missing_predictions"] == 1
    assert result["coverage_percent"] == 50.0
    assert result["mae"] == pytest.approx(10.0)


def test_prediction_audit_passes_valid_predictions():
    predictions = pd.DataFrame(
        {
            "road_id": ["R1"],
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:30:00"]
            ),
            "actual": [120.0],
            "predicted": [100.0],
            "reference_timestamp": pd.to_datetime(
                ["2025-01-01 00:00:00"]
            ),
            "horizon_minutes": [30],
            "model": ["naive_30min"],
        }
    )

    result = audit_predictions(predictions)

    assert result["passed"] is True
    assert result["prediction_rows"] == 1


def test_prediction_audit_rejects_wrong_reference():
    predictions = pd.DataFrame(
        {
            "road_id": ["R1"],
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:30:00"]
            ),
            "actual": [120.0],
            "predicted": [100.0],
            "reference_timestamp": pd.to_datetime(
                ["2025-01-01 00:15:00"]
            ),
            "horizon_minutes": [30],
            "model": ["naive_30min"],
        }
    )

    with pytest.raises(
        ValueError,
        match="reference timestamps",
    ):
        audit_predictions(predictions)


def test_common_population_uses_shared_valid_rows():
    predictions = pd.DataFrame(
        {
            "road_id": ["R1", "R1", "R1", "R1"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:30:00",
                    "2025-01-01 00:30:00",
                    "2025-01-01 01:00:00",
                    "2025-01-01 01:00:00",
                ]
            ),
            "actual": [100, 100, 200, 200],
            "predicted": [110, 105, 190, np.nan],
            "model": [
                "model_a",
                "model_b",
                "model_a",
                "model_b",
            ],
        }
    )

    common, comparison = compare_on_common_population(
        predictions
    )

    assert len(common) == 2
    assert set(common["model"]) == {
        "model_a",
        "model_b",
    }

    assert comparison["valid_predictions"].tolist() == [1, 1]


def test_common_population_rejects_disagreeing_actuals():
    predictions = pd.DataFrame(
        {
            "road_id": ["R1", "R1"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:30:00",
                    "2025-01-01 00:30:00",
                ]
            ),
            "actual": [100, 101],
            "predicted": [110, 105],
            "model": ["model_a", "model_b"],
        }
    )

    with pytest.raises(ValueError, match="disagree"):
        compare_on_common_population(predictions)


def test_common_population_rejects_duplicate_model_rows():
    predictions = pd.DataFrame(
        {
            "road_id": ["R1", "R1"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:30:00",
                    "2025-01-01 00:30:00",
                ]
            ),
            "actual": [100, 100],
            "predicted": [110, 111],
            "model": ["model_a", "model_a"],
        }
    )

    with pytest.raises(ValueError, match="Duplicate"):
        compare_on_common_population(predictions)
