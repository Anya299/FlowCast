import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from src.day18.release_gate import (
    ReleaseGateConfig,
    ReleaseGateError,
    calculate_model_metrics,
    compare_common_population,
    get_valid_predictions,
    load_predictions,
    run_release_gate,
)


def make_predictions():
    rows = []

    records = [
        (
            "R1",
            "2025-01-01 00:30:00",
            100,
            110,
            "2025-01-01 00:00:00",
            30,
            "naive_30min",
        ),
        (
            "R1",
            "2025-01-01 01:00:00",
            120,
            100,
            "2025-01-01 00:30:00",
            30,
            "naive_30min",
        ),
        (
            "R1",
            "2025-01-01 00:30:00",
            100,
            130,
            "2024-12-31 00:30:00",
            1440,
            "seasonal_naive_24h",
        ),
        (
            "R1",
            "2025-01-01 01:00:00",
            120,
            110,
            "2024-12-31 01:00:00",
            1440,
            "seasonal_naive_24h",
        ),
    ]

    for (
        road,
        timestamp,
        actual,
        predicted,
        reference,
        horizon,
        model,
    ) in records:
        rows.append(
            {
                "road_id": road,
                "timestamp": timestamp,
                "actual": actual,
                "predicted": predicted,
                "reference_timestamp": reference,
                "horizon_minutes": horizon,
                "model": model,
                "prediction_status": "valid",
            }
        )

    return pd.DataFrame(rows)


def test_valid_predictions_are_selected():
    df = make_predictions()

    result = get_valid_predictions(df)

    assert len(result) == 4


def test_missing_predictions_are_excluded_from_metrics():
    df = make_predictions()

    df.loc[0, "predicted"] = np.nan

    model_df = df.loc[
        df["model"] == "naive_30min"
    ]

    metrics = calculate_model_metrics(model_df)

    assert metrics["total_predictions"] == 2
    assert metrics["valid_predictions"] == 1
    assert metrics["coverage_pct"] == 50.0


def test_metrics_calculate_mae_rmse_and_bias():
    df = make_predictions()

    model_df = df.loc[
        df["model"] == "naive_30min"
    ]

    metrics = calculate_model_metrics(model_df)

    assert metrics["mae"] == 15.0
    assert metrics["rmse"] == pytest.approx(
        np.sqrt(250)
    )
    assert metrics["bias"] == -5.0


def test_common_population_contains_both_models():
    df = make_predictions()

    common, comparison = compare_common_population(df)

    assert len(common) == 4
    assert set(comparison["model"]) == {
        "naive_30min",
        "seasonal_naive_24h",
    }


def test_common_population_rejects_duplicate_valid_keys():
    df = make_predictions()

    duplicate = df.iloc[[0]].copy()

    df = pd.concat(
        [df, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        ReleaseGateError,
        match="Duplicate valid prediction keys",
    ):
        compare_common_population(df)


def test_common_population_rejects_disagreeing_actuals():
    df = make_predictions()

    df.loc[
        df["model"] == "seasonal_naive_24h",
        "actual",
    ] = 999

    with pytest.raises(
        ReleaseGateError,
        match="Actual values disagree",
    ):
        compare_common_population(df)


def test_missing_required_columns_are_rejected(tmp_path):
    path = tmp_path / "invalid.csv"

    pd.DataFrame(
        {"road_id": ["R1"]}
    ).to_csv(path, index=False)

    with pytest.raises(
        ReleaseGateError,
        match="Missing required prediction columns",
    ):
        load_predictions(path)


def test_missing_file_is_rejected(tmp_path):
    path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_predictions(path)


def test_coverage_threshold_must_be_valid(tmp_path):
    config = ReleaseGateConfig(
        predictions_path=tmp_path / "predictions.csv",
        output_dir=tmp_path / "output",
        minimum_coverage_pct=101,
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 100",
    ):
        run_release_gate(config)


def test_release_gate_generates_reports(tmp_path):
    predictions_path = tmp_path / "predictions.csv"

    make_predictions().to_csv(
        predictions_path,
        index=False,
    )

    config = ReleaseGateConfig(
        predictions_path=predictions_path,
        output_dir=tmp_path / "reports",
        minimum_coverage_pct=80,
    )

    result = run_release_gate(config)

    assert result["release_status"] == "PASS"

    assert result["checks_passed"] > 0

    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["checks"]).exists()
    assert Path(result["artifacts"]["metrics"]).exists()
    assert Path(result["artifacts"]["common_population"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()


def test_release_gate_fails_when_coverage_is_too_low(tmp_path):
    df = make_predictions()

    df.loc[
        df["model"] == "naive_30min",
        "predicted",
    ] = np.nan

    predictions_path = tmp_path / "predictions.csv"

    df.to_csv(
        predictions_path,
        index=False,
    )

    config = ReleaseGateConfig(
        predictions_path=predictions_path,
        output_dir=tmp_path / "reports",
        minimum_coverage_pct=80,
    )

    result = run_release_gate(config)

    assert result["release_status"] == "FAIL"


def test_release_gate_rejects_wrong_forecast_horizon(tmp_path):
    df = make_predictions()

    df.loc[0, "horizon_minutes"] = 1440

    predictions_path = tmp_path / "predictions.csv"

    df.to_csv(
        predictions_path,
        index=False,
    )

    config = ReleaseGateConfig(
        predictions_path=predictions_path,
        output_dir=tmp_path / "reports",
    )

    result = run_release_gate(config)

    assert result["release_status"] == "FAIL"


def test_unexpected_model_is_flagged(tmp_path):
    df = make_predictions()

    df.loc[0, "model"] = "unexpected_model"

    predictions_path = tmp_path / "predictions.csv"

    df.to_csv(
        predictions_path,
        index=False,
    )

    config = ReleaseGateConfig(
        predictions_path=predictions_path,
        output_dir=tmp_path / "reports",
    )

    result = run_release_gate(config)

    failed_names = {
        item["check"]
        for item in result["checks"]
        if not item["passed"]
    }

    assert "recognized_models" in failed_names
    assert result["release_status"] == "FAIL"