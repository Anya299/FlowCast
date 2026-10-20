from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "day9" / "flowcast_train.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "day9" / "flowcast_validation.csv"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "day17"

ROAD_COLUMN = "road_id"
TIME_COLUMN = "timestamp"
TARGET_COLUMN = "traffic_volume"

NAIVE_MODEL = "naive_30min"
SEASONAL_MODEL = "seasonal_naive_24h"

NAIVE_HORIZON_MINUTES = 30
SEASONAL_HORIZON_MINUTES = 24 * 60

REQUIRED_COLUMNS = {
    ROAD_COLUMN,
    TIME_COLUMN,
    TARGET_COLUMN,
}


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("flowcast.day17")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PipelineConfig:
    train_path: Path = DEFAULT_TRAIN_PATH
    validation_path: Path = DEFAULT_VALIDATION_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR

    naive_horizon_minutes: int = NAIVE_HORIZON_MINUTES
    seasonal_horizon_minutes: int = SEASONAL_HORIZON_MINUTES

    @classmethod
    def from_arguments(
        cls,
        train_path: str | None = None,
        validation_path: str | None = None,
        output_dir: str | None = None,
    ) -> "PipelineConfig":

        return cls(
            train_path=Path(train_path) if train_path else DEFAULT_TRAIN_PATH,
            validation_path=(
                Path(validation_path)
                if validation_path
                else DEFAULT_VALIDATION_PATH
            ),
            output_dir=Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR,
        )


# ============================================================
# DATA VALIDATION
# ============================================================

def load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Load a CSV and validate its basic structure.

    The original dataset is not modified on disk.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{dataset_name} dataset is empty.")

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = df.copy()

    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        errors="raise",
    )

    if df[ROAD_COLUMN].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing road identifiers."
        )

    if df[TIME_COLUMN].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing timestamps."
        )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="raise",
    )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing target values."
        )

    if not np.isfinite(df[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError(
            f"{dataset_name} contains infinite target values."
        )

    duplicate_mask = df.duplicated(
        subset=[ROAD_COLUMN, TIME_COLUMN],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        raise ValueError(
            f"{dataset_name} contains {duplicate_count} rows "
            "participating in duplicate road-timestamp keys."
        )

    return df.sort_values(
        [ROAD_COLUMN, TIME_COLUMN]
    ).reset_index(drop=True)


def validate_chronological_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, Any]:
    """
    Ensure the validation period starts after the training period ends.

    This is a global temporal split check.
    """

    train_end = train[TIME_COLUMN].max()
    validation_start = validation[TIME_COLUMN].min()

    if train_end >= validation_start:
        raise ValueError(
            "Chronological split failed: the training period must end "
            "before the validation period begins. "
            f"Train end={train_end}; "
            f"validation start={validation_start}"
        )

    return {
        "train_start": train[TIME_COLUMN].min().isoformat(),
        "train_end": train_end.isoformat(),
        "validation_start": validation_start.isoformat(),
        "validation_end": validation[TIME_COLUMN].max().isoformat(),
        "chronological_split_passed": True,
    }


# ============================================================
# FORECAST GENERATION
# ============================================================

def generate_reference_predictions(
    history: pd.DataFrame,
    forecast_rows: pd.DataFrame,
    horizon_minutes: int,
    model_name: str,
) -> pd.DataFrame:
    """
    Generate historical-reference forecasts.

    The target for each forecast is looked up at:

        forecast timestamp - horizon

    The lookup uses only the supplied history dataframe.

    For the 24-hour seasonal baseline, the 24-hour historical
    reference must exist in history. It is not replaced by a
    different observation when missing.
    """

    if horizon_minutes <= 0:
        raise ValueError("Forecast horizon must be positive.")

    history_lookup = history[
        [ROAD_COLUMN, TIME_COLUMN, TARGET_COLUMN]
    ].copy()

    if history_lookup.duplicated(
        [ROAD_COLUMN, TIME_COLUMN]
    ).any():
        raise ValueError(
            "History contains duplicate road-timestamp keys."
        )

    reference = forecast_rows[
        [ROAD_COLUMN, TIME_COLUMN, TARGET_COLUMN]
    ].copy()

    reference = reference.rename(
        columns={
            TIME_COLUMN: "timestamp",
            TARGET_COLUMN: "actual",
        }
    )

    reference["reference_timestamp"] = (
        reference["timestamp"]
        - pd.Timedelta(minutes=horizon_minutes)
    )

    history_lookup = history_lookup.rename(
        columns={
            TIME_COLUMN: "reference_timestamp",
            TARGET_COLUMN: "predicted",
        }
    )

    predictions = reference.merge(
        history_lookup,
        on=[ROAD_COLUMN, "reference_timestamp"],
        how="left",
        validate="many_to_one",
    )

    predictions["model"] = model_name
    predictions["horizon_minutes"] = horizon_minutes

    predictions["prediction_status"] = np.where(
        predictions["predicted"].notna(),
        "valid",
        "missing_historical_reference",
    )

    return predictions[
        [
            ROAD_COLUMN,
            "timestamp",
            "actual",
            "predicted",
            "reference_timestamp",
            "horizon_minutes",
            "model",
            "prediction_status",
        ]
    ]


# ============================================================
# PREDICTION AUDITS
# ============================================================

def audit_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """
    Check basic prediction integrity.

    Missing predictions are allowed because a historical reference
    may not exist. Missing actual targets are not allowed.
    """

    required = {
        ROAD_COLUMN,
        "timestamp",
        "actual",
        "predicted",
        "reference_timestamp",
        "horizon_minutes",
        "model",
    }

    missing = required.difference(predictions.columns)

    if missing:
        raise ValueError(
            f"Prediction data is missing columns: {sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError("Prediction dataset is empty.")

    if predictions.duplicated(
        [ROAD_COLUMN, "timestamp", "model"]
    ).any():
        raise ValueError(
            "Duplicate road-timestamp-model prediction records found."
        )

    if predictions["actual"].isna().any():
        raise ValueError("Actual target values must not be missing.")

    actual_values = predictions["actual"].to_numpy(dtype=float)

    if not np.isfinite(actual_values).all():
        raise ValueError("Actual target values must be finite.")

    valid_predictions = predictions["predicted"].dropna().to_numpy(
        dtype=float
    )

    if not np.isfinite(valid_predictions).all():
        raise ValueError("Non-missing predictions must be finite.")

    valid_reference = predictions["predicted"].notna()

    if valid_reference.any():
        expected_reference = (
            predictions.loc[valid_reference, "timestamp"]
            - pd.to_timedelta(
                predictions.loc[valid_reference, "horizon_minutes"],
                unit="m",
            )
        )

        actual_reference = predictions.loc[
            valid_reference,
            "reference_timestamp",
        ]

        if not (
            expected_reference.reset_index(drop=True)
            == actual_reference.reset_index(drop=True)
        ).all():
            raise ValueError(
                "Prediction reference timestamps do not match "
                "their configured forecast horizons."
            )

    return {
        "passed": True,
        "prediction_rows": int(len(predictions)),
        "unique_models": sorted(
            predictions["model"].unique().tolist()
        ),
        "duplicate_prediction_keys": 0,
        "missing_prediction_count": int(
            predictions["predicted"].isna().sum()
        ),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate metrics on valid predictions only.

    Error is defined as predicted - actual.
    Positive bias means overprediction.
    """

    total = len(predictions)

    valid = predictions.dropna(
        subset=["actual", "predicted"]
    ).copy()

    valid_count = len(valid)

    coverage = (
        valid_count / total
        if total
        else 0.0
    )

    if valid_count == 0:
        return {
            "model": (
                str(predictions["model"].iloc[0])
                if total
                else "unknown"
            ),
            "total_observations": int(total),
            "valid_predictions": 0,
            "missing_predictions": int(total),
            "coverage": 0.0,
            "coverage_percent": 0.0,
            "mae": None,
            "rmse": None,
            "bias": None,
        }

    errors = (
        valid["predicted"].to_numpy(dtype=float)
        - valid["actual"].to_numpy(dtype=float)
    )

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    return {
        "model": str(valid["model"].iloc[0]),
        "total_observations": int(total),
        "valid_predictions": int(valid_count),
        "missing_predictions": int(total - valid_count),
        "coverage": float(coverage),
        "coverage_percent": float(coverage * 100),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }


def compare_on_common_population(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare models on the same valid road-timestamp observations.

    The common population includes only keys where every model has
    a valid prediction. Actual values must agree between models.
    """

    required = {
        ROAD_COLUMN,
        "timestamp",
        "actual",
        "predicted",
        "model",
    }

    missing = required.difference(predictions.columns)

    if missing:
        raise ValueError(
            f"Comparison data is missing columns: {sorted(missing)}"
        )

    if predictions.duplicated(
        [ROAD_COLUMN, "timestamp", "model"]
    ).any():
        raise ValueError(
            "Duplicate road-timestamp-model rows prevent comparison."
        )

    actual_counts = predictions.groupby(
        [ROAD_COLUMN, "timestamp"]
    )["actual"].nunique(dropna=False)

    if (actual_counts > 1).any():
        raise ValueError(
            "Models disagree on actual values for shared observations."
        )

    model_names = sorted(predictions["model"].unique().tolist())

    valid = predictions.dropna(
        subset=["actual", "predicted"]
    ).copy()

    counts = valid.groupby(
        [ROAD_COLUMN, "timestamp"]
    )["model"].nunique()

    shared_keys = counts[counts == len(model_names)].index

    shared_index = pd.MultiIndex.from_frame(
        valid[[ROAD_COLUMN, "timestamp"]]
    )

    common = valid.loc[
        shared_index.isin(shared_keys)
    ].copy()

    comparison_rows = []

    for model_name in model_names:
        model_rows = common.loc[
            common["model"] == model_name
        ]

        comparison_rows.append(
            calculate_metrics(model_rows)
        )

    comparison = pd.DataFrame(comparison_rows)

    return common, comparison


# ============================================================
# REPORTING
# ============================================================

def json_safe(value: Any) -> Any:
    """
    Convert NumPy and pandas values into JSON-compatible values.
    """

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            json_safe(data),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def build_markdown_report(
    split_summary: dict[str, Any],
    prediction_audit: dict[str, Any],
    metrics: pd.DataFrame,
    common_comparison: pd.DataFrame,
    common_observation_count: int,
) -> str:

    lines = [
        "# Day 17 — End-to-End Forecasting Pipeline",
        "",
        "## 1. Objective",
        "",
        "Integrate historical-reference forecasting strategies, "
        "prediction validation, metric calculation, and paired "
        "model comparison into a reproducible pipeline.",
        "",
        "## 2. Dataset Split",
        "",
        f"- Training start: `{split_summary['train_start']}`",
        f"- Training end: `{split_summary['train_end']}`",
        f"- Validation start: `{split_summary['validation_start']}`",
        f"- Validation end: `{split_summary['validation_end']}`",
        f"- Chronological split passed: "
        f"`{split_summary['chronological_split_passed']}`",
        "",
        "## 3. Prediction Integrity",
        "",
        f"- Audit passed: `{prediction_audit['passed']}`",
        f"- Prediction rows: `{prediction_audit['prediction_rows']}`",
        f"- Missing predictions: "
        f"`{prediction_audit['missing_prediction_count']}`",
        "",
        "## 4. Per-Model Metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 5. Common-Population Comparison",
        "",
        f"Shared road-timestamp observations: `{common_observation_count}`",
        "",
        common_comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 6. Metric Interpretation",
        "",
        "- MAE: average absolute prediction error.",
        "- RMSE: square-root mean squared prediction error.",
        "- Bias: mean of predicted minus actual values.",
        "- Coverage: proportion of observations with valid predictions.",
        "",
        "Comparisons should be interpreted together with coverage. "
        "Per-model metrics can use different valid populations; "
        "the common-population table uses shared valid observations.",
        "",
        "## 7. Limitations",
        "",
        "This pipeline evaluates historical-reference baselines. "
        "It does not yet train a machine-learning model.",
        "",
        "The evaluation audit does not prove complete leakage prevention. "
        "Feature lineage, prediction-time availability, preprocessing "
        "fitting, and test-set isolation require additional review.",
        "",
        "## 8. Conclusion",
        "",
        "The pipeline executed its configured baseline strategies and "
        "generated reproducible prediction and evaluation artifacts. "
        "See the output files and metadata for the exact run results.",
        "",
    ]

    return "\n".join(lines)


# ============================================================
# END-TO-END EXECUTION
# ============================================================

def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    configure_logging()

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Loading training dataset.")
    train = load_dataset(
        config.train_path,
        "Training",
    )

    logger.info("Loading validation dataset.")
    validation = load_dataset(
        config.validation_path,
        "Validation",
    )

    logger.info("Validating chronological separation.")
    split_summary = validate_chronological_split(
        train,
        validation,
    )

    logger.info("Generating naive 30-minute predictions.")
    naive_predictions = generate_reference_predictions(
        history=train,
        forecast_rows=validation,
        horizon_minutes=config.naive_horizon_minutes,
        model_name=NAIVE_MODEL,
    )

    logger.info("Generating seasonal naive 24-hour predictions.")
    seasonal_predictions = generate_reference_predictions(
        history=train,
        forecast_rows=validation,
        horizon_minutes=config.seasonal_horizon_minutes,
        model_name=SEASONAL_MODEL,
    )

    predictions = pd.concat(
        [naive_predictions, seasonal_predictions],
        ignore_index=True,
    )

    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"]
    )

    predictions["reference_timestamp"] = pd.to_datetime(
        predictions["reference_timestamp"]
    )

    logger.info("Auditing generated predictions.")
    prediction_audit = audit_predictions(
        predictions
    )

    logger.info("Calculating individual model metrics.")
    metrics_rows = [
        calculate_metrics(naive_predictions),
        calculate_metrics(seasonal_predictions),
    ]

    metrics = pd.DataFrame(metrics_rows)

    logger.info("Building common evaluation population.")
    common, common_comparison = compare_on_common_population(
        predictions
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    predictions_path = (
        config.output_dir / f"day17_predictions_{timestamp}.csv"
    )

    metrics_path = (
        config.output_dir / f"day17_metrics_{timestamp}.csv"
    )

    common_path = (
        config.output_dir / f"day17_common_population_{timestamp}.csv"
    )

    common_comparison_path = (
        config.output_dir
        / f"day17_common_population_metrics_{timestamp}.csv"
    )

    metadata_path = (
        config.output_dir / f"day17_metadata_{timestamp}.json"
    )

    report_path = (
        config.output_dir / f"day17_pipeline_report_{timestamp}.md"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    common.to_csv(
        common_path,
        index=False,
    )

    common_comparison.to_csv(
        common_comparison_path,
        index=False,
    )

    report = build_markdown_report(
        split_summary=split_summary,
        prediction_audit=prediction_audit,
        metrics=metrics,
        common_comparison=common_comparison,
        common_observation_count=len(
            common[
                [ROAD_COLUMN, "timestamp"]
            ].drop_duplicates()
        ),
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    metadata = {
        "pipeline": "FlowCast Day 17",
        "run_timestamp_utc": timestamp,
        "training_dataset": str(config.train_path),
        "validation_dataset": str(config.validation_path),
        "output_directory": str(config.output_dir),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "split_summary": split_summary,
        "prediction_audit": prediction_audit,
        "models": [
            NAIVE_MODEL,
            SEASONAL_MODEL,
        ],
        "metrics": metrics.to_dict(orient="records"),
        "common_population_metrics": (
            common_comparison.to_dict(orient="records")
        ),
        "artifacts": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "common_population": str(common_path),
            "common_population_metrics": str(
                common_comparison_path
            ),
            "report": str(report_path),
        },
    }

    write_json(
        metadata_path,
        metadata,
    )

    logger.info("Pipeline completed successfully.")
    logger.info("Predictions: %s", predictions_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("Report: %s", report_path)

    return {
        "metadata": metadata,
        "predictions_path": predictions_path,
        "metrics_path": metrics_path,
        "common_path": common_path,
        "common_comparison_path": common_comparison_path,
        "metadata_path": metadata_path,
        "report_path": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FlowCast Day 17 forecasting pipeline."
    )

    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Optional training CSV path.",
    )

    parser.add_argument(
        "--validation",
        type=str,
        default=None,
        help="Optional validation CSV path.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output directory.",
    )

    args = parser.parse_args()

    config = PipelineConfig.from_arguments(
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output,
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()