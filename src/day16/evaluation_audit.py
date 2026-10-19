"""
FlowCast Day 16
Leakage-Safe Forecast Evaluation Audit

Validates:
- Chronological train/validation separation
- Duplicate road/timestamp keys
- Feature and target separation
- Historical reference timestamps
- Prediction/actual alignment
- Consistent evaluation populations

Important:
Structural checks cannot independently prove that every
feature was available in production at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


ROAD_COLUMN = "road_id"
TIME_COLUMN = "timestamp"
TARGET_COLUMN = "traffic_volume"

PREDICTION_REQUIRED_COLUMNS = {
    ROAD_COLUMN,
    TIME_COLUMN,
    "actual",
    "predicted",
}


@dataclass(frozen=True)
class AuditResult:
    """Structured result from a validation check."""

    check_name: str
    passed: bool
    message: str


def _require_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    dataset_name: str,
) -> None:
    """Raise a clear error when required columns are absent."""

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing}"
        )


def _parse_timestamps(
    df: pd.DataFrame,
    column: str = TIME_COLUMN,
) -> pd.Series:
    """Parse timestamps and reject invalid values."""

    parsed = pd.to_datetime(
        df[column],
        errors="coerce",
    )

    if parsed.isna().any():
        raise ValueError(
            f"Column '{column}' contains missing or invalid timestamps."
        )

    return parsed


def audit_chronological_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    time_column: str = TIME_COLUMN,
) -> AuditResult:
    """
    Ensure the latest training timestamp precedes the earliest
    validation timestamp.

    This checks timestamp separation, not feature availability.
    """

    _require_columns(train, [time_column], "Training dataset")
    _require_columns(
        validation,
        [time_column],
        "Validation dataset",
    )

    if train.empty or validation.empty:
        raise ValueError(
            "Training and validation datasets must both be non-empty."
        )

    train_time = _parse_timestamps(train, time_column)
    validation_time = _parse_timestamps(
        validation,
        time_column,
    )

    train_end = train_time.max()
    validation_start = validation_time.min()

    if train_end >= validation_start:
        return AuditResult(
            check_name="chronological_split",
            passed=False,
            message=(
                f"Training ends at {train_end}; validation begins "
                f"at {validation_start}. Training must end before "
                "validation begins."
            ),
        )

    return AuditResult(
        check_name="chronological_split",
        passed=True,
        message=(
            f"Training ends at {train_end}; validation begins "
            f"at {validation_start}. No timestamp overlap detected."
        ),
    )


def audit_unique_road_timestamps(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
) -> AuditResult:
    """Reject duplicate road/timestamp observation keys."""

    _require_columns(
        df,
        [ROAD_COLUMN, TIME_COLUMN],
        dataset_name,
    )

    parsed_time = _parse_timestamps(df)

    keys = pd.DataFrame(
        {
            ROAD_COLUMN: df[ROAD_COLUMN],
            TIME_COLUMN: parsed_time,
        },
        index=df.index,
    )

    duplicate_mask = keys.duplicated(
        subset=[ROAD_COLUMN, TIME_COLUMN],
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        return AuditResult(
            check_name="unique_road_timestamps",
            passed=False,
            message=(
                f"{dataset_name} contains {duplicate_count} rows "
                "participating in duplicate road/timestamp keys."
            ),
        )

    return AuditResult(
        check_name="unique_road_timestamps",
        passed=True,
        message=f"{dataset_name} has unique road/timestamp keys.",
    )


def audit_feature_target_separation(
    feature_columns: Sequence[str],
    target_column: str = TARGET_COLUMN,
    forbidden_columns: Sequence[str] = (),
) -> AuditResult:
    """
    Check that the target and explicitly forbidden columns are
    not included among model inputs.

    Pass the actual model feature list, not every column in a CSV.
    """

    features = list(feature_columns)

    if len(features) != len(set(features)):
        return AuditResult(
            check_name="feature_target_separation",
            passed=False,
            message="The model feature list contains duplicate names.",
        )

    forbidden = set(forbidden_columns)
    forbidden.add(target_column)

    leakage_columns = sorted(
        set(features) & forbidden
    )

    if leakage_columns:
        return AuditResult(
            check_name="feature_target_separation",
            passed=False,
            message=(
                "Forbidden columns appear in model inputs: "
                f"{leakage_columns}"
            ),
        )

    return AuditResult(
        check_name="feature_target_separation",
        passed=True,
        message=(
            "The supplied feature list excludes the target and "
            "explicitly forbidden columns."
        ),
    )


def audit_prediction_alignment(
    predictions: pd.DataFrame,
    forecast_horizon_minutes: int | None = None,
) -> AuditResult:
    """
    Validate prediction rows and optional historical reference times.

    Expected optional column:
        reference_timestamp

    If a horizon is supplied, each reference timestamp must match
    forecast timestamp minus that horizon exactly.
    """

    _require_columns(
        predictions,
        list(PREDICTION_REQUIRED_COLUMNS),
        "Prediction dataset",
    )

    if predictions.empty:
        raise ValueError(
            "Prediction dataset must not be empty."
        )

    if predictions[
        [ROAD_COLUMN, TIME_COLUMN]
    ].isna().any().any():
        raise ValueError(
            "Prediction road IDs and timestamps cannot be missing."
        )

    if predictions.duplicated(
        subset=[ROAD_COLUMN, TIME_COLUMN]
    ).any():
        return AuditResult(
            check_name="prediction_alignment",
            passed=False,
            message=(
                "Duplicate road/timestamp prediction rows were found."
            ),
        )

    forecast_time = _parse_timestamps(predictions)

    actual = pd.to_numeric(
        predictions["actual"],
        errors="coerce",
    )

    predicted = pd.to_numeric(
        predictions["predicted"],
        errors="coerce",
    )

    invalid_values = (
        actual.isna()
        | predicted.isna()
        | ~actual.map(lambda value: pd.notna(value))
        | ~predicted.map(lambda value: pd.notna(value))
    )

    if invalid_values.any():
        return AuditResult(
            check_name="prediction_alignment",
            passed=False,
            message=(
                f"{int(invalid_values.sum())} rows contain missing "
                "or non-numeric actual/predicted values."
            ),
        )

    if (
        forecast_horizon_minutes is not None
        and forecast_horizon_minutes <= 0
    ):
        raise ValueError(
            "Forecast horizon must be a positive number of minutes."
        )

    if forecast_horizon_minutes is not None:
        if "reference_timestamp" not in predictions.columns:
            raise ValueError(
                "A reference_timestamp column is required when "
                "validating a forecast horizon."
            )

        reference_time = _parse_timestamps(
            predictions,
            "reference_timestamp",
        )

        expected_reference = (
            forecast_time
            - pd.Timedelta(
                minutes=forecast_horizon_minutes
            )
        )

        mismatch = (
            reference_time.reset_index(drop=True)
            != expected_reference.reset_index(drop=True)
        )

        if mismatch.any():
            return AuditResult(
                check_name="prediction_alignment",
                passed=False,
                message=(
                    f"{int(mismatch.sum())} reference timestamps "
                    "do not match the requested historical horizon."
                ),
            )

    return AuditResult(
        check_name="prediction_alignment",
        passed=True,
        message=(
            "Prediction rows have valid values, unique keys, "
            "and satisfy the requested alignment checks."
        ),
    )


def build_common_evaluation_population(
    predictions: pd.DataFrame,
    model_column: str = "model",
) -> pd.DataFrame:
    """
    Return observations for which every model has both an actual
    and a prediction.

    Each model must have at most one row per road/timestamp key.
    Actual values must agree across models for a given key.
    """

    required = [
        ROAD_COLUMN,
        TIME_COLUMN,
        model_column,
        "actual",
        "predicted",
    ]

    _require_columns(
        predictions,
        required,
        "Predictions",
    )

    if predictions.empty:
        raise ValueError(
            "Predictions must not be empty."
        )

    df = predictions.copy()

    if df[
        [ROAD_COLUMN, TIME_COLUMN, model_column]
    ].isna().any().any():
        raise ValueError(
            "Road IDs, timestamps, and model names cannot be missing."
        )

    df[TIME_COLUMN] = _parse_timestamps(df)

    if df.duplicated(
        subset=[ROAD_COLUMN, TIME_COLUMN, model_column]
    ).any():
        raise ValueError(
            "Duplicate road/timestamp/model rows were found."
        )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    df["predicted"] = pd.to_numeric(
        df["predicted"],
        errors="coerce",
    )

    valid = df[
        df["actual"].notna()
        & df["predicted"].notna()
    ].copy()

    if valid.empty:
        return valid

    actual_counts = valid.groupby(
        [ROAD_COLUMN, TIME_COLUMN]
    )["actual"].nunique()

    if (actual_counts > 1).any():
        raise ValueError(
            "Models disagree on actual values for the same "
            "road/timestamp key."
        )

    model_count = valid[model_column].nunique()

    population_counts = valid.groupby(
        [ROAD_COLUMN, TIME_COLUMN]
    )[model_column].nunique()

    common_keys = population_counts[
        population_counts == model_count
    ].index

    common_index = pd.MultiIndex.from_frame(
        valid[[ROAD_COLUMN, TIME_COLUMN]]
    )

    result = valid.loc[
        common_index.isin(common_keys)
    ].copy()

    return result.sort_values(
        [ROAD_COLUMN, TIME_COLUMN, model_column]
    ).reset_index(drop=True)