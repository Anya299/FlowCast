import pandas as pd
import pytest

from src.day13_forecasting_strategies import (
    ForecastEvaluator,
    ForecastStrategy,
    NaiveForecastStrategy,
    SeasonalNaiveForecastStrategy,
    generate_reference_predictions,
)


@pytest.fixture
def history():
    return pd.DataFrame({
        "road_id": ["R1", "R1", "R1"],
        "timestamp": pd.to_datetime([
            "2025-01-01 09:00:00",
            "2025-01-01 09:30:00",
            "2025-01-01 10:00:00",
        ]),
        "traffic_volume": [100, 120, 140],
    })


@pytest.fixture
def validation():
    return pd.DataFrame({
        "road_id": ["R1", "R1"],
        "timestamp": pd.to_datetime([
            "2025-01-01 10:30:00",
            "2025-01-01 11:00:00",
        ]),
        "traffic_volume": [150, 160],
    })


def test_strategies_inherit_from_abstract_base_class():
    assert issubclass(
        NaiveForecastStrategy,
        ForecastStrategy,
    )

    assert issubclass(
        SeasonalNaiveForecastStrategy,
        ForecastStrategy,
    )


def test_base_strategy_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ForecastStrategy(
            name="base",
            reference_minutes=30,
        )


def test_naive_strategy_uses_30_minute_reference(
    history,
    validation,
):
    strategy = NaiveForecastStrategy()

    result = strategy.predict(
        history,
        validation,
    )

    assert result["model"].nunique() == 1
    assert result["model"].iloc[0] == "naive_30min"

    # 10:30 references 10:00, whose traffic is 140.
    first_prediction = result.iloc[0]

    assert first_prediction["predicted"] == 140


def test_seasonal_strategy_uses_24_hour_reference(
    history,
    validation,
):
    strategy = SeasonalNaiveForecastStrategy()

    assert strategy.reference_minutes == 1440

    result = strategy.predict(
        history,
        validation,
    )

    assert result["model"].iloc[0] == "seasonal_naive_24h"


def test_missing_reference_produces_missing_prediction(
    validation,
):
    history = pd.DataFrame({
        "road_id": ["R1"],
        "timestamp": pd.to_datetime([
            "2025-01-01 09:00:00",
        ]),
        "traffic_volume": [100],
    })

    result = generate_reference_predictions(
        history=history,
        validation=validation.iloc[[0]],
        strategy_name="naive_30min",
        reference_minutes=30,
    )

    assert pd.isna(
        result["predicted"].iloc[0]
    )


def test_duplicate_history_keys_are_rejected(
    validation,
):
    duplicate_history = pd.DataFrame({
        "road_id": ["R1", "R1"],
        "timestamp": pd.to_datetime([
            "2025-01-01 10:00:00",
            "2025-01-01 10:00:00",
        ]),
        "traffic_volume": [100, 110],
    })

    with pytest.raises(ValueError):
        generate_reference_predictions(
            history=duplicate_history,
            validation=validation.iloc[[0]],
            strategy_name="naive_30min",
            reference_minutes=30,
        )


def test_evaluator_calculates_metrics():
    predictions = pd.DataFrame({
        "road_id": ["R1", "R1", "R1"],
        "timestamp": pd.to_datetime([
            "2025-01-01 10:30:00",
            "2025-01-01 11:00:00",
            "2025-01-01 11:30:00",
        ]),
        "actual": [100.0, 120.0, 150.0],
        "predicted": [110.0, 100.0, float("nan")],
        "reference_timestamp": pd.to_datetime([
            "2025-01-01 10:00:00",
            "2025-01-01 10:30:00",
            "2025-01-01 11:00:00",
        ]),
        "model": ["naive_30min"] * 3,
    })

    metrics = ForecastEvaluator().evaluate(
        predictions
    )

    assert metrics["validation_rows"] == 3
    assert metrics["valid_predictions"] == 2
    assert metrics["missing_predictions"] == 1
    assert metrics["prediction_coverage_pct"] == 66.6667
    assert metrics["mae"] == 15.0


def test_evaluator_rejects_empty_predictions():
    empty = pd.DataFrame({
        "road_id": pd.Series(dtype="str"),
        "timestamp": pd.Series(dtype="datetime64[ns]"),
        "actual": pd.Series(dtype="float"),
        "predicted": pd.Series(dtype="float"),
        "reference_timestamp": pd.Series(
            dtype="datetime64[ns]"
        ),
        "model": pd.Series(dtype="str"),
    })

    with pytest.raises(ValueError):
        ForecastEvaluator().evaluate(empty)


def test_evaluator_rejects_multiple_models():
    predictions = pd.DataFrame({
        "road_id": ["R1", "R1"],
        "timestamp": pd.to_datetime([
            "2025-01-01 10:30:00",
            "2025-01-01 11:00:00",
        ]),
        "actual": [100.0, 120.0],
        "predicted": [110.0, 115.0],
        "reference_timestamp": pd.to_datetime([
            "2025-01-01 10:00:00",
            "2025-01-01 10:30:00",
        ]),
        "model": [
            "naive_30min",
            "seasonal_naive_24h",
        ],
    })

    with pytest.raises(ValueError):
        ForecastEvaluator().evaluate(predictions)