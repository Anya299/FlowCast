"""
FLOWCAST — DAY 10
Baseline Forecasting and Evaluation

Purpose:
1. Validate Day 9 datasets.
2. Check temporal integrity.
3. Evaluate historical baselines.
4. Compare actual and predicted traffic.
5. Save reproducible experiment reports.

This script does not train a machine-learning model.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "day9"

REPORT_DIR = PROJECT_ROOT / "reports" / "day10"

PLOT_DIR = REPORT_DIR / "plots"

TRAIN_FILE = DATA_DIR / "flowcast_train.csv"
VALIDATION_FILE = DATA_DIR / "flowcast_validation.csv"
TEST_FILE = DATA_DIR / "flowcast_test.csv"

TARGET = "traffic_volume"
TIMESTAMP = "timestamp"
ROAD_ID = "road_id"

# Half-hourly data normally has 48 observations per day.
HALF_HOURS_PER_DAY = 48


# ============================================================
# 2. DIRECTORY SETUP
# ============================================================

def create_directories():
    """Create output directories."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Output directories ready.")


# ============================================================
# 3. LOAD DATASETS
# ============================================================

def load_dataset(path):
    """Load a dataset and validate required columns."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        TARGET,
        TIMESTAMP,
        ROAD_ID,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df[TIMESTAMP] = pd.to_datetime(
        df[TIMESTAMP],
        errors="coerce",
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    if df[TIMESTAMP].isna().any():
        raise ValueError(
            f"{path.name} contains invalid timestamps."
        )

    if df[ROAD_ID].isna().any():
        raise ValueError(
            f"{path.name} contains missing road IDs."
        )

    # Sort the rows deterministically.
    df = df.sort_values(
        [ROAD_ID, TIMESTAMP]
    ).reset_index(drop=True)

    return df


# ============================================================
# 4. DATASET INTEGRITY CHECKS
# ============================================================

def inspect_dataset(name, df):
    """Collect essential integrity statistics."""

    duplicate_keys = int(
        df.duplicated(
            subset=[ROAD_ID, TIMESTAMP]
        ).sum()
    )

    invalid_target_count = int(
        df[TARGET].isna().sum()
    )

    negative_target_count = int(
        (df[TARGET] < 0).sum()
    )

    infinite_target_count = int(
        np.isinf(df[TARGET].to_numpy(dtype=float)).sum()
    )

    rows_per_road = (
        df.groupby(ROAD_ID)
        .size()
        .to_dict()
    )

    timestamp_gaps = {}

    for road, group in df.groupby(ROAD_ID):
        timestamps = group[TIMESTAMP].sort_values()

        differences = timestamps.diff()

        gaps = int(
            (
                differences
                > pd.Timedelta(minutes=30)
            ).sum()
        )

        timestamp_gaps[str(road)] = gaps

    return {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "unique_roads": int(df[ROAD_ID].nunique()),
        "start_timestamp": str(df[TIMESTAMP].min()),
        "end_timestamp": str(df[TIMESTAMP].max()),
        "duplicate_road_timestamp_keys": duplicate_keys,
        "missing_targets": invalid_target_count,
        "negative_targets": negative_target_count,
        "infinite_targets": infinite_target_count,
        "rows_per_road": {
            str(k): int(v)
            for k, v in rows_per_road.items()
        },
        "gaps_over_30_minutes_by_road": timestamp_gaps,
    }


# ============================================================
# 5. PREPARE TARGET-ELIGIBLE ROWS
# ============================================================

def eligible_rows(df):
    """
    Keep records with finite, nonnegative target values.

    This is a conservative evaluation policy, not a claim
    that every removed observation is necessarily erroneous.
    """

    target_values = df[TARGET].to_numpy(dtype=float)

    valid_mask = (
        np.isfinite(target_values)
        & (target_values >= 0)
    )

    return df.loc[valid_mask].copy()


# ============================================================
# 6. BUILD HISTORICAL REFERENCE VALUES
# ============================================================

def build_historical_reference(
    history,
    future,
    lag,
    reference_name,
):
    """
    Match a historical observation by road and timestamp.

    lag=30 minutes:
        Find traffic at prediction timestamp - 30 minutes.

    lag=24 hours:
        Find traffic at prediction timestamp - 24 hours.

    Matching by timestamp prevents incorrect assumptions
    caused by missing rows or inconsistent row ordering.
    """

    history = history[
        [ROAD_ID, TIMESTAMP, TARGET]
    ].copy()

    future = future.copy()

    history = history.rename(
        columns={TARGET: reference_name}
    )

    future["reference_timestamp"] = (
        future[TIMESTAMP]
        - pd.Timedelta(minutes=lag)
    )

    merged = future.merge(
        history,
        left_on=[ROAD_ID, "reference_timestamp"],
        right_on=[ROAD_ID, TIMESTAMP],
        how="left",
        validate="many_to_one",
    )

    merged = merged.drop(
        columns=["reference_timestamp", "timestamp_y"],
        errors="ignore",
    )

    # Restore a consistent timestamp column after the merge.
    if "timestamp_x" in merged.columns:
        merged = merged.rename(
            columns={"timestamp_x": TIMESTAMP}
        )

    return merged


# ============================================================
# 7. METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    """Calculate MAE and RMSE using valid prediction pairs."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "evaluated_rows": 0,
            "mae": None,
            "rmse": None,
        }

    errors = actual - predicted

    return {
        "evaluated_rows": int(len(actual)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(
            np.sqrt(np.mean(errors ** 2))
        ),
    }


# ============================================================
# 8. PREDICTION TABLE
# ============================================================

def evaluate_baseline(
    history,
    evaluation_df,
    lag_minutes,
    model_name,
):
    """
    Evaluate one historical baseline.

    Historical values must come from records that occurred
    before the evaluation timestamp.

    For validation, history contains training observations.
    Validation observations are added to history one at a time
    so later predictions may use earlier observed traffic.

    This represents rolling one-step-ahead evaluation.
    """

    history = history[
        [ROAD_ID, TIMESTAMP, TARGET]
    ].copy()

    history = history[
        np.isfinite(history[TARGET])
        & (history[TARGET] >= 0)
    ]

    evaluation_df = evaluation_df.copy()

    evaluation_df = evaluation_df.sort_values(
        [TIMESTAMP, ROAD_ID]
    )

    predictions = []

    # Store known historical targets by road and timestamp.
    lookup = {}

    for row in history.itertuples(index=False):
        lookup[(str(row.road_id), row.timestamp)] = (
            getattr(row, TARGET)
        )

    for row in evaluation_df.itertuples(index=False):
        road = str(getattr(row, ROAD_ID))
        timestamp = getattr(row, TIMESTAMP)
        actual = getattr(row, TARGET)

        reference_timestamp = (
            timestamp
            - pd.Timedelta(minutes=lag_minutes)
        )

        prediction = lookup.get(
            (road, reference_timestamp),
            np.nan,
        )

        predictions.append(
            {
                ROAD_ID: road,
                TIMESTAMP: timestamp,
                "actual": actual,
                "predicted": prediction,
                "model": model_name,
                "reference_timestamp": reference_timestamp,
            }
        )

        # Rolling one-step-ahead protocol:
        # The actual observation becomes available after
        # the prediction for this timestamp is made.
        if (
            np.isfinite(actual)
            and actual >= 0
        ):
            lookup[(road, timestamp)] = actual

    return pd.DataFrame(predictions)


# ============================================================
# 9. METRIC SUMMARY
# ============================================================

def summarise_predictions(predictions):
    """Return overall and per-road evaluation results."""

    overall = calculate_metrics(
        predictions["actual"],
        predictions["predicted"],
    )

    by_road = []

    for road, group in predictions.groupby(ROAD_ID):
        result = calculate_metrics(
            group["actual"],
            group["predicted"],
        )

        by_road.append(
            {
                ROAD_ID: str(road),
                **result,
            }
        )

    return overall, pd.DataFrame(by_road)


# ============================================================
# 10. PLOT ACTUAL VS PREDICTED
# ============================================================

def plot_predictions(predictions, model_name):
    """Save an actual-versus-predicted plot."""

    valid = predictions[
        np.isfinite(predictions["actual"])
        & np.isfinite(predictions["predicted"])
    ].copy()

    if valid.empty:
        print(
            f"Skipping plot for {model_name}: "
            "no valid prediction pairs."
        )
        return

    # Plot a single road to keep the figure readable.
    selected_road = valid[ROAD_ID].iloc[0]

    sample = valid[
        valid[ROAD_ID] == selected_road
    ].sort_values(TIMESTAMP).head(200)

    plt.figure(figsize=(12, 5))

    plt.plot(
        sample[TIMESTAMP],
        sample["actual"],
        label="Actual traffic",
    )

    plt.plot(
        sample[TIMESTAMP],
        sample["predicted"],
        label="Predicted traffic",
    )

    plt.title(
        f"{model_name}: Actual vs Predicted "
        f"({selected_road})"
    )

    plt.xlabel("Timestamp")
    plt.ylabel("Traffic volume")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_actual_vs_predicted.png"
    )

    plt.savefig(
        PLOT_DIR / filename,
        dpi=160,
    )

    plt.close()


# ============================================================
# 11. MAIN EXPERIMENT
# ============================================================

def main():
    """Run Day 10 baseline evaluation."""

    print("=" * 65)
    print("FLOWCAST DAY 10 — BASELINE EVALUATION")
    print("=" * 65)

    create_directories()

    print("\nLoading datasets...")

    train = load_dataset(TRAIN_FILE)
    validation = load_dataset(VALIDATION_FILE)
    test = load_dataset(TEST_FILE)

    print(f"Training rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows:       {len(test):,}")

    # --------------------------------------------------------
    # Dataset integrity report
    # --------------------------------------------------------

    integrity = {
        "train": inspect_dataset("train", train),
        "validation": inspect_dataset(
            "validation",
            validation,
        ),
        "test": inspect_dataset("test", test),
    }

    with open(
        REPORT_DIR / "dataset_integrity.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            integrity,
            file,
            indent=2,
        )

    # Check chronological boundaries.
    if train[TIMESTAMP].max() >= validation[TIMESTAMP].min():
        raise ValueError(
            "Training and validation timestamps overlap."
        )

    if validation[TIMESTAMP].max() >= test[TIMESTAMP].min():
        raise ValueError(
            "Validation and test timestamps overlap."
        )

    # Check duplicate road/timestamp keys.
    for name, df in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        duplicates = df.duplicated(
            subset=[ROAD_ID, TIMESTAMP]
        ).sum()

        if duplicates:
            raise ValueError(
                f"{name} has {duplicates} duplicate "
                "road/timestamp keys."
            )

    print("\nDataset integrity checks passed.")

    # --------------------------------------------------------
    # Prepare usable target rows.
    # --------------------------------------------------------

    train_valid = eligible_rows(train)
    validation_valid = eligible_rows(validation)
    test_valid = eligible_rows(test)

    print("\nRows with valid targets:")
    print(f"Train:      {len(train_valid):,}")
    print(f"Validation: {len(validation_valid):,}")
    print(f"Test:       {len(test_valid):,}")

    # --------------------------------------------------------
    # Baseline 1: most recent half-hour observation
    # --------------------------------------------------------

    naive_validation = evaluate_baseline(
        history=train_valid,
        evaluation_df=validation_valid,
        lag_minutes=30,
        model_name="Naive 30-minute",
    )

    # --------------------------------------------------------
    # Baseline 2: same time on the previous day
    # --------------------------------------------------------

    seasonal_validation = evaluate_baseline(
        history=train_valid,
        evaluation_df=validation_valid,
        lag_minutes=24 * 60,
        model_name="Seasonal naive 24-hour",
    )

    # --------------------------------------------------------
    # Summarise validation metrics.
    # --------------------------------------------------------

    naive_metrics, naive_by_road = (
        summarise_predictions(naive_validation)
    )

    seasonal_metrics, seasonal_by_road = (
        summarise_predictions(seasonal_validation)
    )

    summary = pd.DataFrame(
        [
            {
                "model": "Naive 30-minute",
                **naive_metrics,
            },
            {
                "model": "Seasonal naive 24-hour",
                **seasonal_metrics,
            },
        ]
    )

    summary.to_csv(
        REPORT_DIR / "validation_baseline_metrics.csv",
        index=False,
    )

    naive_by_road.to_csv(
        REPORT_DIR / "naive_metrics_by_road.csv",
        index=False,
    )

    seasonal_by_road.to_csv(
        REPORT_DIR / "seasonal_metrics_by_road.csv",
        index=False,
    )

    naive_validation.to_csv(
        REPORT_DIR / "naive_validation_predictions.csv",
        index=False,
    )

    seasonal_validation.to_csv(
        REPORT_DIR / "seasonal_validation_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_predictions(
        naive_validation,
        "Naive 30-minute",
    )

    plot_predictions(
        seasonal_validation,
        "Seasonal naive 24-hour",
    )

    # --------------------------------------------------------
    # Markdown report
    # --------------------------------------------------------

    report = f"""# FlowCast Day 10 — Baseline Evaluation

## Objective

Establish reference forecasting performance before training
machine-learning models.

## Dataset

- Training rows: {len(train):,}
- Validation rows: {len(validation):,}
- Test rows: {len(test):,}
- Target: `{TARGET}`

## Validation results

{summary.to_string(index=False)}

## Evaluation protocol

The evaluation uses rolling one-step-ahead predictions.
At each timestamp, the model uses historical target values
from the same road and the required earlier timestamp.

The actual observation is added to history after its
prediction is made. This assumes the latest observed traffic
is available before the next prediction.

## Important limitations

- These are simple historical baselines, not trained ML models.
- Results depend on the availability of the required historical
  observation.
- The validation period is used for baseline comparison.
- The test period has not been used for model selection.
- Feature-generation logic must be audited separately before
  declaring the full feature pipeline leakage-free.
- The evaluation assumes rolling one-step-ahead forecasting,
  not a full-day forecast made in advance.

## Next step

Inspect validation metrics, compare errors by road, verify the
feature-generation implementation, and establish a documented
baseline before fitting a learned model.
"""

    (REPORT_DIR / "day10_baseline_report.md").write_text(
        report,
        encoding="utf-8",
    )

    print("\nVALIDATION BASELINE RESULTS")
    print("=" * 65)
    print(summary.to_string(index=False))

    print("\nSaved reports to:")
    print(REPORT_DIR)

    print("\nDAY 10 BASELINE EVALUATION COMPLETED.")


if __name__ == "__main__":
    main()