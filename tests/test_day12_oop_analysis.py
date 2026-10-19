import pandas as pd
import pytest

from src.day12_oop_analysis import FlowCastCoverageAnalyzer


@pytest.fixture
def analyzer(tmp_path):
    """
    Create an analyzer using temporary directories.
    """

    return FlowCastCoverageAnalyzer(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        coverage_threshold=95.0,
    )


def test_analyzer_initializes_with_threshold(analyzer):
    assert analyzer.coverage_threshold == 95.0


def test_invalid_coverage_threshold_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        FlowCastCoverageAnalyzer(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            coverage_threshold=101.0,
        )


def test_baseline_comparison_selects_best_coverage(analyzer):
    summary = pd.DataFrame({
        "model": [
            "naive_30min",
            "seasonal_naive_24h",
        ],
        "validation_rows": [100, 100],
        "valid_predictions": [96, 94],
        "missing_predictions": [4, 6],
        "prediction_coverage_pct": [96.0, 94.0],
    })

    result = analyzer.compare_baselines(summary)

    best = result[result["is_best_coverage"]]

    assert len(best) == 1
    assert best.iloc[0]["model"] == "naive_30min"
    assert best.iloc[0]["coverage_gap_pp"] == 0.0


def test_low_coverage_roads_are_flagged(analyzer):
    by_road = pd.DataFrame({
        "model": [
            "naive_30min",
            "naive_30min",
            "naive_30min",
        ],
        "road_id": [
            "NL-001",
            "NL-002",
            "NL-003",
        ],
        "missing_predictions": [2, 8, 5],
        "coverage_pct": [98.0, 92.0, 95.0],
    })

    result = analyzer.flag_low_coverage_roads(
        by_road
    )

    flags = dict(
        zip(
            result["road_id"],
            result["coverage_flag"],
        )
    )

    assert flags["NL-001"] == "ACCEPTABLE_COVERAGE"
    assert flags["NL-002"] == "LOW_COVERAGE"

    # Exactly 95% meets the configured threshold.
    assert flags["NL-003"] == "ACCEPTABLE_COVERAGE"


def test_duplicate_baseline_names_are_rejected(analyzer):
    summary = pd.DataFrame({
        "model": [
            "naive_30min",
            "naive_30min",
        ],
        "validation_rows": [100, 100],
        "valid_predictions": [96, 95],
        "missing_predictions": [4, 5],
        "prediction_coverage_pct": [96.0, 95.0],
    })

    with pytest.raises(ValueError):
        analyzer.compare_baselines(summary)


def test_empty_baseline_summary_is_rejected(analyzer):
    summary = pd.DataFrame({
        "model": pd.Series(dtype="str"),
        "validation_rows": pd.Series(dtype="int"),
        "valid_predictions": pd.Series(dtype="int"),
        "missing_predictions": pd.Series(dtype="int"),
        "prediction_coverage_pct": pd.Series(dtype="float"),
    })

    with pytest.raises(ValueError):
        analyzer.compare_baselines(summary)


def test_markdown_report_contains_findings(analyzer):
    comparison = pd.DataFrame({
        "model": ["naive_30min"],
        "validation_rows": [100],
        "valid_predictions": [96],
        "missing_predictions": [4],
        "prediction_coverage_pct": [96.0],
        "best_coverage_pct": [96.0],
        "coverage_gap_pp": [0.0],
        "is_best_coverage": [True],
        "coverage_status": ["MEETS_THRESHOLD"],
    })

    road_flags = pd.DataFrame({
        "model": ["naive_30min"],
        "road_id": ["NL-001"],
        "missing_predictions": [8],
        "coverage_pct": [92.0],
        "coverage_threshold_pct": [95.0],
        "coverage_flag": ["LOW_COVERAGE"],
    })

    report = analyzer.build_report(
        comparison,
        road_flags,
    )

    assert "FlowCast Day 12" in report
    assert "naive_30min" in report
    assert "NL-001" in report
    assert "LOW_COVERAGE" in report