from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREDICTIONS = PROJECT_ROOT / "reports" / "day17"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "day18"

ROAD_COLUMN = "road_id"
TIME_COLUMN = "timestamp"
ACTUAL_COLUMN = "actual"
PREDICTED_COLUMN = "predicted"
REFERENCE_COLUMN = "reference_timestamp"
HORIZON_COLUMN = "horizon_minutes"
MODEL_COLUMN = "model"
STATUS_COLUMN = "prediction_status"

REQUIRED_COLUMNS = {
    ROAD_COLUMN,
    TIME_COLUMN,
    ACTUAL_COLUMN,
    PREDICTED_COLUMN,
    REFERENCE_COLUMN,
    HORIZON_COLUMN,
    MODEL_COLUMN,
    STATUS_COLUMN,
}

EXPECTED_MODELS = {
    "naive_30min",
    "seasonal_naive_24h",
}

MODEL_HORIZONS = {
    "naive_30min": 30,
    "seasonal_naive_24h": 1440,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("flowcast.day18")


@dataclass(frozen=True)
class ReleaseGateConfig:
    predictions_path: Path
    output_dir: Path
    minimum_coverage_pct: float = 80.0
    require_both_models: bool = True


@dataclass
class CheckResult:
    check: str
    passed: bool
    severity: str
    details: str


class ReleaseGateError(ValueError):
    """Raised when the release gate cannot safely evaluate its inputs."""


def add_check(
    checks: list[CheckResult],
    name: str,
    passed: bool,
    details: str,
    severity: str = "error",
) -> None:
    checks.append(
        CheckResult(
            check=name,
            passed=bool(passed),
            severity=severity,
            details=details,
        )
    )


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate the basic prediction file structure."""

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file does not exist: {path}"
        )

    if not path.is_file():
        raise ReleaseGateError(
            f"Prediction path is not a file: {path}"
        )

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ReleaseGateError(
            "Prediction file contains no records."
        )

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ReleaseGateError(
            f"Missing required prediction columns: {sorted(missing)}"
        )

    df = df.copy()

    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        errors="coerce",
    )

    df[REFERENCE_COLUMN] = pd.to_datetime(
        df[REFERENCE_COLUMN],
        errors="coerce",
    )

    df[ACTUAL_COLUMN] = pd.to_numeric(
        df[ACTUAL_COLUMN],
        errors="coerce",
    )

    df[PREDICTED_COLUMN] = pd.to_numeric(
        df[PREDICTED_COLUMN],
        errors="coerce",
    )

    df[HORIZON_COLUMN] = pd.to_numeric(
        df[HORIZON_COLUMN],
        errors="coerce",
    )

    df[MODEL_COLUMN] = df[MODEL_COLUMN].astype("string")

    df[STATUS_COLUMN] = df[STATUS_COLUMN].astype("string")

    return df


def get_valid_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with finite actual and predicted values."""

    actual = pd.to_numeric(
        df[ACTUAL_COLUMN],
        errors="coerce",
    )

    predicted = pd.to_numeric(
        df[PREDICTED_COLUMN],
        errors="coerce",
    )

    mask = (
        actual.notna()
        & predicted.notna()
        & np.isfinite(actual)
        & np.isfinite(predicted)
    )

    return df.loc[mask].copy()


def calculate_model_metrics(
    model_df: pd.DataFrame,
) -> dict:
    """Calculate accuracy and coverage for one model."""

    total = len(model_df)

    valid = get_valid_predictions(model_df)

    valid_count = len(valid)

    if total:
        coverage_pct = valid_count / total * 100.0
    else:
        coverage_pct = 0.0

    if valid_count:
        errors = (
            valid[PREDICTED_COLUMN].to_numpy(dtype=float)
            - valid[ACTUAL_COLUMN].to_numpy(dtype=float)
        )

        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(errors))

    else:
        mae = None
        rmse = None
        bias = None

    return {
        "model": str(model_df[MODEL_COLUMN].iloc[0])
        if total
        else "unknown",
        "total_predictions": int(total),
        "valid_predictions": int(valid_count),
        "missing_or_invalid_predictions": int(total - valid_count),
        "coverage_pct": float(coverage_pct),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }


def compare_common_population(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare models only on road/timestamp pairs for which
    every evaluated model has a valid prediction.
    """

    valid = get_valid_predictions(df)

    if valid.empty:
        return valid.copy(), pd.DataFrame()

    model_names = sorted(
        valid[MODEL_COLUMN].dropna().astype(str).unique()
    )

    if not model_names:
        return valid.iloc[0:0].copy(), pd.DataFrame()

    keys = [ROAD_COLUMN, TIME_COLUMN]

    actual_counts = (
        valid.groupby(keys, dropna=False)[ACTUAL_COLUMN]
        .nunique()
    )

    if (actual_counts > 1).any():
        raise ReleaseGateError(
            "Actual values disagree for the same road/timestamp."
        )

    duplicate_model_keys = valid.duplicated(
        keys + [MODEL_COLUMN],
        keep=False,
    )

    if duplicate_model_keys.any():
        raise ReleaseGateError(
            "Duplicate valid prediction keys found."
        )

    counts = valid.groupby(keys, dropna=False)[MODEL_COLUMN].nunique()

    shared_keys = counts[counts == len(model_names)].index

    shared_index = pd.MultiIndex.from_frame(valid[keys])

    common = valid.loc[
        shared_index.isin(shared_keys)
    ].copy()

    if common.empty:
        return common, pd.DataFrame()

    comparison_rows = []

    for model_name in model_names:
        subset = common.loc[
            common[MODEL_COLUMN].astype(str) == model_name
        ]

        actual = subset[ACTUAL_COLUMN].to_numpy(dtype=float)
        predicted = subset[PREDICTED_COLUMN].to_numpy(dtype=float)

        errors = predicted - actual

        comparison_rows.append(
            {
                "model": model_name,
                "common_observations": int(len(subset)),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "bias": float(np.mean(errors)),
            }
        )

    comparison = pd.DataFrame(comparison_rows)

    return common, comparison


def run_release_gate(
    config: ReleaseGateConfig,
) -> dict:
    """Run all release checks and create audit artifacts."""

    if not 0 <= config.minimum_coverage_pct <= 100:
        raise ValueError(
            "minimum_coverage_pct must be between 0 and 100."
        )

    logger.info("Loading predictions: %s", config.predictions_path)

    df = load_predictions(config.predictions_path)

    checks: list[CheckResult] = []

    # --------------------------------------------------
    # Check 1: Timestamp and identifier integrity
    # --------------------------------------------------

    timestamps_valid = bool(df[TIME_COLUMN].notna().all())

    road_ids_valid = bool(
        df[ROAD_COLUMN].notna().all()
        and df[ROAD_COLUMN].astype(str).str.strip().ne("").all()
    )

    add_check(
        checks,
        "valid_timestamps",
        timestamps_valid,
        f"Invalid timestamps: {int(df[TIME_COLUMN].isna().sum())}",
    )

    add_check(
        checks,
        "valid_road_identifiers",
        road_ids_valid,
        "Every prediction must have a non-empty road identifier.",
    )

    # --------------------------------------------------
    # Check 2: Duplicate records
    # --------------------------------------------------

    duplicate_mask = df.duplicated(
        [ROAD_COLUMN, TIME_COLUMN, MODEL_COLUMN],
        keep=False,
    )

    add_check(
        checks,
        "unique_prediction_keys",
        not duplicate_mask.any(),
        f"Duplicate road/timestamp/model rows: "
        f"{int(duplicate_mask.sum())}",
    )

    # --------------------------------------------------
    # Check 3: Actual values
    # --------------------------------------------------

    actual = df[ACTUAL_COLUMN]

    actual_valid = bool(
        actual.notna().all()
        and np.isfinite(actual.to_numpy(dtype=float)).all()
    )

    add_check(
        checks,
        "finite_actual_values",
        actual_valid,
        f"Missing or non-finite actual values: "
        f"{int((~np.isfinite(actual.to_numpy(dtype=float))).sum())}",
    )

    # --------------------------------------------------
    # Check 4: Model names
    # --------------------------------------------------

    observed_models = set(
        df[MODEL_COLUMN].dropna().astype(str).unique()
    )

    unknown_models = observed_models - EXPECTED_MODELS

    add_check(
        checks,
        "recognized_models",
        not unknown_models,
        f"Observed models: {sorted(observed_models)}; "
        f"unknown models: {sorted(unknown_models)}",
    )

    if config.require_both_models:
        missing_models = EXPECTED_MODELS - observed_models

        add_check(
            checks,
            "both_baselines_present",
            not missing_models,
            f"Missing expected models: {sorted(missing_models)}",
        )

    # --------------------------------------------------
    # Check 5: Forecast horizon alignment
    # --------------------------------------------------

    valid_horizons = True
    horizon_details = []

    for model_name, expected_horizon in MODEL_HORIZONS.items():
        model_df = df.loc[
            df[MODEL_COLUMN].astype(str) == model_name
        ]

        if model_df.empty:
            continue

        horizons = model_df[HORIZON_COLUMN]

        bad_horizon = horizons.isna() | (
            horizons != expected_horizon
        )

        valid_horizons = (
            valid_horizons and not bool(bad_horizon.any())
        )

        reference_expected = (
            model_df[TIME_COLUMN]
            - pd.to_timedelta(expected_horizon, unit="m")
        )

        reference_matches = (
            model_df[REFERENCE_COLUMN] == reference_expected
        )

        bad_reference = (
            model_df[PREDICTED_COLUMN].notna()
            & ~reference_matches
        )

        valid_horizons = (
            valid_horizons and not bool(bad_reference.any())
        )

        horizon_details.append(
            f"{model_name}: incorrect horizons={int(bad_horizon.sum())}, "
            f"incorrect non-missing references={int(bad_reference.sum())}"
        )

    add_check(
        checks,
        "forecast_horizon_alignment",
        valid_horizons,
        "; ".join(horizon_details)
        if horizon_details
        else "No recognized model horizons were available to check.",
    )

    # --------------------------------------------------
    # Check 6: Prediction numerical validity
    # --------------------------------------------------

    prediction_values = df[PREDICTED_COLUMN]

    valid_prediction_values = (
        prediction_values.notna()
        & np.isfinite(prediction_values.to_numpy(dtype=float))
    )

    non_finite_predictions = int(
        (~valid_prediction_values).sum()
    )

    add_check(
        checks,
        "prediction_values_are_finite_or_missing",
        non_finite_predictions == 0,
        f"Missing predictions are allowed; missing or non-finite "
        f"prediction values: {non_finite_predictions}",
        severity="warning",
    )

    # --------------------------------------------------
    # Check 7: Coverage and individual model metrics
    # --------------------------------------------------

    metrics_rows = []

    for model_name in sorted(observed_models):
        model_df = df.loc[
            df[MODEL_COLUMN].astype(str) == model_name
        ]

        metrics = calculate_model_metrics(model_df)

        metrics_rows.append(metrics)

        coverage_pass = (
            metrics["coverage_pct"]
            >= config.minimum_coverage_pct
        )

        add_check(
            checks,
            f"coverage_{model_name}",
            coverage_pass,
            f"Coverage={metrics['coverage_pct']:.2f}%; "
            f"required minimum={config.minimum_coverage_pct:.2f}%.",
        )

    metrics_df = pd.DataFrame(metrics_rows)

    # --------------------------------------------------
    # Check 8: Common-population comparison
    # --------------------------------------------------

    common, comparison = compare_common_population(df)

    common_population_exists = not comparison.empty

    add_check(
        checks,
        "common_population_available",
        common_population_exists,
        f"Common valid prediction records: {len(common)}; "
        f"common comparison rows: {len(comparison)}.",
    )

    # --------------------------------------------------
    # Check 9: Compare baseline performance
    # --------------------------------------------------

    if common_population_exists:
        comparison_models = set(
            comparison["model"].astype(str)
        )

        both_models_available = EXPECTED_MODELS.issubset(
            comparison_models
        )

        if both_models_available:
            comparison_lookup = comparison.set_index("model")

            naive_mae = float(
                comparison_lookup.loc["naive_30min", "mae"]
            )

            seasonal_mae = float(
                comparison_lookup.loc["seasonal_naive_24h", "mae"]
            )

            naive_not_worse = naive_mae <= seasonal_mae

            add_check(
                checks,
                "naive_baseline_comparison",
                naive_not_worse,
                f"Common-population MAE: naive={naive_mae:.4f}; "
                f"seasonal={seasonal_mae:.4f}.",
                severity="warning",
            )

        else:
            add_check(
                checks,
                "naive_baseline_comparison",
                False,
                "Both expected models must be present in the "
                "common-population comparison.",
                severity="warning",
            )

    # --------------------------------------------------
    # Final release decision
    # --------------------------------------------------

    failed_errors = [
        item
        for item in checks
        if not item.passed and item.severity == "error"
    ]

    release_passed = len(failed_errors) == 0

    report = {
        "project": "FlowCast",
        "day": 18,
        "audit_name": "Forecasting Release Quality Gate",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_file": str(config.predictions_path),
        "minimum_coverage_pct": config.minimum_coverage_pct,
        "total_prediction_rows": int(len(df)),
        "observed_models": sorted(observed_models),
        "release_status": "PASS" if release_passed else "FAIL",
        "checks_passed": int(sum(item.passed for item in checks)),
        "checks_failed": int(sum(not item.passed for item in checks)),
        "checks": [asdict(item) for item in checks],
        "model_metrics": metrics_rows,
        "common_population_metrics": comparison.to_dict(
            orient="records"
        ),
    }

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    json_path = (
        config.output_dir
        / f"day18_release_audit_{timestamp}.json"
    )

    checks_path = (
        config.output_dir
        / f"day18_quality_checks_{timestamp}.csv"
    )

    metrics_path = (
        config.output_dir
        / f"day18_model_metrics_{timestamp}.csv"
    )

    comparison_path = (
        config.output_dir
        / f"day18_common_population_{timestamp}.csv"
    )

    markdown_path = (
        config.output_dir
        / f"day18_release_report_{timestamp}.md"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False,
        )

    checks_df = pd.DataFrame(
        [asdict(item) for item in checks]
    )

    checks_df.to_csv(
        checks_path,
        index=False,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    markdown_lines = [
        "# FlowCast Day 18 — Release Quality Gate",
        "",
        f"- **Release status:** {report['release_status']}",
        f"- **Created UTC:** {report['created_at_utc']}",
        f"- **Input rows:** {len(df)}",
        f"- **Models found:** {', '.join(sorted(observed_models))}",
        f"- **Minimum coverage:** {config.minimum_coverage_pct:.2f}%",
        f"- **Checks passed:** {report['checks_passed']}",
        f"- **Checks failed:** {report['checks_failed']}",
        "",
        "## Quality checks",
        "",
        "| Check | Result | Severity | Details |",
        "|---|---|---|---|",
    ]

    for item in checks:
        result = "PASS" if item.passed else "FAIL"

        details = item.details.replace("|", "\\|")

        markdown_lines.append(
            f"| {item.check} | {result} | "
            f"{item.severity} | {details} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Individual model metrics",
            "",
        ]
    )

    if metrics_df.empty:
        markdown_lines.append("No model metrics were available.")
    else:
        markdown_lines.append(
            metrics_df.to_string(index=False)
        )

    markdown_lines.extend(
        [
            "",
            "## Common-population comparison",
            "",
        ]
    )

    if comparison.empty:
        markdown_lines.append(
            "No common valid prediction population was available."
        )
    else:
        markdown_lines.append(
            comparison.to_string(index=False)
        )

    markdown_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The release gate checks structural validity, prediction "
            "alignment, coverage, and baseline comparison.",
            "",
            "A PASS means the configured structural and coverage "
            "checks passed. It does not guarantee real-world "
            "forecasting accuracy or deployment readiness.",
            "",
            "A baseline comparison failure is a warning rather than "
            "an automatic release failure. Baseline performance "
            "must be evaluated on the same observations.",
            "",
        ]
    )

    markdown_path.write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    logger.info(
        "Release status: %s",
        report["release_status"],
    )

    logger.info(
        "Checks passed: %s; failed: %s",
        report["checks_passed"],
        report["checks_failed"],
    )

    logger.info("JSON audit: %s", json_path)
    logger.info("Markdown report: %s", markdown_path)

    report["artifacts"] = {
        "json": str(json_path),
        "checks": str(checks_path),
        "metrics": str(metrics_path),
        "common_population": str(comparison_path),
        "markdown": str(markdown_path),
    }

    return report


def find_latest_predictions(directory: Path) -> Path:
    """Find the latest timestamped Day 17 predictions file."""

    candidates = list(
        directory.glob("day17_predictions_*.csv")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No Day 17 predictions CSV files found in {directory}"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run FlowCast's Day 18 forecasting release quality gate."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to a Day 17 predictions CSV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for Day 18 reports.",
    )

    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=80.0,
        help="Minimum required prediction coverage percentage.",
    )

    args = parser.parse_args()

    predictions_path = (
        args.predictions
        if args.predictions is not None
        else find_latest_predictions(DEFAULT_PREDICTIONS)
    )

    config = ReleaseGateConfig(
        predictions_path=predictions_path,
        output_dir=args.output,
        minimum_coverage_pct=args.minimum_coverage,
    )

    result = run_release_gate(config)

    print()
    print("=" * 60)
    print("FLOWCAST DAY 18 RELEASE QUALITY GATE")
    print("=" * 60)
    print(f"Status       : {result['release_status']}")
    print(f"Rows checked : {result['total_prediction_rows']}")
    print(f"Checks passed: {result['checks_passed']}")
    print(f"Checks failed: {result['checks_failed']}")
    print()
    print("Generated reports:")

    for name, path in result["artifacts"].items():
        print(f"{name:18}: {path}")

    if result["release_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()