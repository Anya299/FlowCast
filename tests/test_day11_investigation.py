import pandas as pd

from src.analysis.day11_missing_prediction_investigation import (
    build_model_summary,
    build_reason_summary,
    create_markdown_report,
    investigate_missing_predictions,
)


def test_model_summary_counts_missing_predictions():
    predictions = pd.DataFrame({
        "model": ["naive_30min"] * 3,
        "predicted": [100.0, float("nan"), 120.0],
    })

    missing_cases = pd.DataFrame({
        "missing_reason": ["historical_reference_row_absent"],
    })

    result = build_model_summary(predictions, missing_cases)

    assert result["validation_rows"] == 3
    assert result["valid_predictions"] == 2
    assert result["missing_predictions"] == 1
    assert result["missing_cases_classified"] == 1
    assert result["prediction_coverage_pct"] == 66.6667


def test_reason_summary_counts_categories():
    cases = pd.DataFrame({
        "model": [
            "naive_30min",
            "naive_30min",
            "seasonal_naive_24h",
        ],
        "missing_reason": [
            "historical_reference_row_absent",
            "historical_reference_row_absent",
            "historical_reference_row_absent",
        ],
    })

    result = build_reason_summary(cases)

    counts = dict(zip(result["model"], result["missing_cases"]))

    assert counts["naive_30min"] == 2
    assert counts["seasonal_naive_24h"] == 1


def test_missing_prediction_is_classified_as_absent_reference():
    timestamp = pd.Timestamp("2025-04-07 00:00:00")
    reference_timestamp = timestamp - pd.Timedelta(minutes=30)

    predictions = pd.DataFrame({
        "road_id": ["NL-001"],
        "timestamp": [timestamp],
        "actual": [144.0],
        "predicted": [float("nan")],
        "reference_timestamp": [reference_timestamp],
        "model": ["naive_30min"],
    })

    train = pd.DataFrame({
        "road_id": ["NL-001"],
        "timestamp": [pd.Timestamp("2025-04-06 23:00:00")],
        "traffic_volume": [140.0],
    })

    validation = pd.DataFrame({
        "road_id": ["NL-001"],
        "timestamp": [timestamp],
        "traffic_volume": [144.0],
    })

    result = investigate_missing_predictions(
        predictions,
        train,
        validation,
    )

    assert len(result) == 1
    assert (
        result.iloc[0]["missing_reason"]
        == "historical_reference_row_absent"
    )


def test_markdown_report_contains_tables():
    summary = pd.DataFrame({
        "model": ["naive_30min"],
        "validation_rows": [3],
        "valid_predictions": [2],
        "missing_predictions": [1],
        "prediction_coverage_pct": [66.6667],
        "missing_cases_classified": [1],
    })

    reasons = pd.DataFrame({
        "model": ["naive_30min"],
        "missing_reason": ["historical_reference_row_absent"],
        "missing_cases": [1],
    })

    by_road = pd.DataFrame({
        "model": ["naive_30min"],
        "road_id": ["NL-001"],
        "validation_rows": [3],
        "valid_predictions": [2],
        "missing_predictions": [1],
        "coverage_pct": [66.6667],
    })

    report = create_markdown_report(summary, reasons, by_road)

    assert "| model |" in report
    assert "| --- |" in report
    assert "historical_reference_row_absent" in report
    assert "NL-001" in report

