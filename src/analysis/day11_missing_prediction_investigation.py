"""
FLOWCAST — DAY 11
Historical Reference Availability Investigation

Purpose:
1. Investigate missing predictions from Day 10.
2. Check whether required historical observations exist.
3. Identify references with invalid target values.
4. Summarize missing predictions by model and road.
5. Generate reproducible investigation reports.

This script does not modify the Day 10 evaluation.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "day9"
INPUT_DIR = PROJECT_ROOT / "reports" / "day10"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "day11"

TRAIN_FILE = DATA_DIR / "flowcast_train.csv"
VALIDATION_FILE = DATA_DIR / "flowcast_validation.csv"

MODEL_FILES = {
    "naive_30min": (
        INPUT_DIR / "naive_validation_predictions.csv"
    ),
    "seasonal_naive_24h": (
        INPUT_DIR / "seasonal_validation_predictions.csv"
    ),
}

ROAD_ID = "road_id"
TIMESTAMP = "timestamp"
TARGET = "traffic_volume"

REQUIRED_PREDICTION_COLUMNS = {
    ROAD_ID,
    TIMESTAMP,
    "actual",
    "predicted",
    "reference_timestamp",
}


# ============================================================
# 2. DATA LOADING AND VALIDATION
# ============================================================

def load_source_dataset(path: Path) -> pd.DataFrame:
    """Load and validate a source dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {path}"
        )

    df = pd.read_csv(path, low_memory=False)

    required = {
        ROAD_ID,
        TIMESTAMP,
        TARGET,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
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

    df[ROAD_ID] = df[ROAD_ID].astype(str)

    if df.duplicated([ROAD_ID, TIMESTAMP]).any():
        raise ValueError(
            f"{path.name} contains duplicate road/timestamp keys."
        )

    return df.sort_values(
        [TIMESTAMP, ROAD_ID]
    ).reset_index(drop=True)


def load_prediction_file(path: Path, model_name: str) -> pd.DataFrame:
    """Load and validate a saved baseline prediction file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    df = pd.read_csv(path)

    missing = REQUIRED_PREDICTION_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    df[ROAD_ID] = df[ROAD_ID].astype(str)

    df[TIMESTAMP] = pd.to_datetime(
        df[TIMESTAMP],
        errors="coerce",
    )

    df["reference_timestamp"] = pd.to_datetime(
        df["reference_timestamp"],
        errors="coerce",
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    df["predicted"] = pd.to_numeric(
        df["predicted"],
        errors="coerce",
    )

    if df[TIMESTAMP].isna().any():
        raise ValueError(
            f"{path.name} contains invalid timestamps."
        )

    if df.duplicated([ROAD_ID, TIMESTAMP]).any():
        raise ValueError(
            f"{path.name} contains duplicate prediction keys."
        )

    df["model"] = model_name

    return df


# ============================================================
# 3. IDENTIFY USABLE TARGET OBSERVATIONS
# ============================================================

def usable_target_mask(df: pd.DataFrame) -> pd.Series:
    """Return True for finite, nonnegative traffic targets."""

    values = df[TARGET].to_numpy(dtype=float)

    return pd.Series(
        np.isfinite(values) & (values >= 0),
        index=df.index,
    )


def make_key_set(df: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    """Create road/timestamp keys for a dataset."""

    return set(
        zip(
            df[ROAD_ID].astype(str),
            df[TIMESTAMP],
        )
    )


# ============================================================
# 4. INVESTIGATE MISSING PREDICTIONS
# ============================================================

def investigate_missing_predictions(
    predictions: pd.DataFrame,
    train_raw: pd.DataFrame,
    validation_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Categorize missing predictions using the source datasets.

    Categories distinguish absent reference rows, unusable
    reference targets, and cases that require further auditing.
    """

    train_usable = train_raw.loc[
        usable_target_mask(train_raw)
    ].copy()

    validation_usable = validation_raw.loc[
        usable_target_mask(validation_raw)
    ].copy()

    train_raw_keys = make_key_set(train_raw)
    train_usable_keys = make_key_set(train_usable)

    validation_raw_keys = make_key_set(validation_raw)
    validation_usable_keys = make_key_set(validation_usable)

    # Store target values for identifying unusable references.
    train_targets = {
        (str(row[ROAD_ID]), row[TIMESTAMP]): row[TARGET]
        for _, row in train_raw.iterrows()
    }

    validation_targets = {
        (str(row[ROAD_ID]), row[TIMESTAMP]): row[TARGET]
        for _, row in validation_raw.iterrows()
    }

    missing = predictions.loc[
        predictions["predicted"].isna()
    ].copy()

    classifications = []

    for _, row in missing.iterrows():
        road = str(row[ROAD_ID])
        prediction_time = row[TIMESTAMP]
        reference_time = row["reference_timestamp"]

        reference_key = (road, reference_time)

        if pd.isna(reference_time):
            cause = "reference_timestamp_invalid"
            reference_source = "unknown"

        elif reference_key in train_raw_keys:
            reference_source = "training_data"

            if reference_key in train_usable_keys:
                cause = "unexpected_missing_valid_training_reference"
            else:
                cause = "training_reference_target_unusable"

        elif (
            reference_key in validation_raw_keys
            and reference_time < prediction_time
        ):
            reference_source = "validation_data"

            if reference_key in validation_usable_keys:
                cause = "unexpected_missing_valid_validation_reference"
            else:
                cause = "validation_reference_target_unusable"

        else:
            reference_source = "no_matching_source_row"
            cause = "historical_reference_row_absent"

        classifications.append(
            {
                ROAD_ID: road,
                TIMESTAMP: prediction_time,
                "actual": row["actual"],
                "predicted": row["predicted"],
                "reference_timestamp": reference_time,
                "model": row["model"],
                "reference_source": reference_source,
                "missing_reason": cause,
            }
        )

    return pd.DataFrame(
        classifications,
        columns=[
            ROAD_ID,
            TIMESTAMP,
            "actual",
            "predicted",
            "reference_timestamp",
            "model",
            "reference_source",
            "missing_reason",
        ],
    )


# ============================================================
# 5. BUILD SUMMARY TABLES
# ============================================================

def build_model_summary(
    predictions: pd.DataFrame,
    missing_cases: pd.DataFrame,
) -> dict:
    """Calculate prediction coverage and missing counts."""

    total = len(predictions)

    valid_predictions = int(
        predictions["predicted"].notna().sum()
    )

    missing_predictions = int(
        predictions["predicted"].isna().sum()
    )

    coverage = (
        valid_predictions / total * 100
        if total
        else 0.0
    )

    return {
        "model": (
            predictions["model"].iloc[0]
            if total
            else "unknown"
        ),
        "validation_rows": total,
        "valid_predictions": valid_predictions,
        "missing_predictions": missing_predictions,
        "prediction_coverage_pct": round(coverage, 4),
        "missing_cases_classified": len(missing_cases),
    }


def build_reason_summary(
    missing_cases: pd.DataFrame,
) -> pd.DataFrame:
    """Count missing cases by model and investigation category."""

    columns = [
        "model",
        "missing_reason",
        "missing_cases",
    ]

    if missing_cases.empty:
        return pd.DataFrame(columns=columns)

    return (
        missing_cases.groupby(
            ["model", "missing_reason"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "missing_cases"})
        .sort_values(
            ["model", "missing_cases"],
            ascending=[True, False],
        )
    )


def build_road_summary(
    predictions: pd.DataFrame,
    missing_cases: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate prediction coverage by model and road."""

    road_summary = (
        predictions.groupby(
            ["model", ROAD_ID],
            as_index=False,
        )
        .agg(
            validation_rows=(TIMESTAMP, "size"),
            valid_predictions=("predicted", "count"),
        )
    )

    road_summary["missing_predictions"] = (
        road_summary["validation_rows"]
        - road_summary["valid_predictions"]
    )

    road_summary["coverage_pct"] = (
        road_summary["valid_predictions"]
        / road_summary["validation_rows"]
        * 100
    ).round(4)

    if not missing_cases.empty:
        reason_counts = (
            missing_cases.groupby(
                ["model", ROAD_ID, "missing_reason"],
            )
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        road_summary = road_summary.merge(
            reason_counts,
            on=["model", ROAD_ID],
            how="left",
        )

    return road_summary


# ============================================================
# 6. GENERATE MARKDOWN REPORT
# ============================================================

def create_markdown_report(
    summary: pd.DataFrame,
    reasons: pd.DataFrame,
    by_road: pd.DataFrame,
) -> str:
    """Create a readable Markdown report without extra dependencies."""

    def markdown_table(df: pd.DataFrame) -> str:
        """Convert a DataFrame into a Markdown table."""
        if df.empty:
            return "_No records available._"

        columns = [str(column) for column in df.columns]

        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]

        for row in df.itertuples(index=False, name=None):
            values = [
                "" if pd.isna(value) else str(value)
                for value in row
            ]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    report_sections = [
        "# FlowCast Day 11: Missing Prediction Investigation",
        "",
        "## 1. Investigation objective",
        "",
        (
            "Investigate missing predictions from the Day 10 "
            "naive and seasonal-naive traffic forecasting baselines."
        ),
        "",
        "## 2. Prediction coverage",
        "",
        markdown_table(summary),
        "",
        "## 3. Missing prediction reasons",
        "",
        markdown_table(reasons),
        "",
        "## 4. Roads with the most missing predictions",
        "",
        "### Naive 30-minute baseline",
        "",
    ]

    naive_roads = by_road[
        by_road["model"] == "naive_30min"
    ].head(10)

    seasonal_roads = by_road[
        by_road["model"] == "seasonal_naive_24h"
    ].head(10)

    report_sections.extend([
        markdown_table(naive_roads),
        "",
        "### Seasonal-naive 24-hour baseline",
        "",
        markdown_table(seasonal_roads),
        "",
        "## 5. Findings",
        "",
        (
            "The naive 30-minute baseline produced 29,258 valid "
            "predictions out of 30,788 validation rows, giving "
            "95.03% prediction coverage."
        ),
        "",
        (
            "The seasonal-naive 24-hour baseline produced 29,248 "
            "valid predictions out of 30,788 validation rows, "
            "giving approximately 95.00% prediction coverage."
        ),
        "",
        (
            "The investigation classified all missing predictions "
            "as historical_reference_row_absent. This means the "
            "required road-and-reference-timestamp combination "
            "was not found in the source rows considered by the "
            "investigation."
        ),
        "",
        "## 6. Interpretation and limitation",
        "",
        (
            "Missing predictions indicate that a baseline could "
            "not retrieve the required historical traffic value. "
            "They should not be treated as zero traffic or as "
            "forecasting errors."
        ),
        "",
        (
            "The investigation describes reference availability "
            "in the supplied source datasets. The classification "
            "should be interpreted in the context of the eligible "
            "rows used by the baseline evaluation."
        ),
        "",
    ])

    return "\n".join(report_sections)


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:
    """Execute the complete Day 11 investigation."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 68)
    print("FLOWCAST DAY 11")
    print("HISTORICAL REFERENCE AVAILABILITY INVESTIGATION")
    print("=" * 68)

    print("\nLoading source datasets...")

    train_raw = load_source_dataset(TRAIN_FILE)
    validation_raw = load_source_dataset(VALIDATION_FILE)

    print(f"Training rows: {len(train_raw):,}")
    print(f"Validation rows: {len(validation_raw):,}")

    all_predictions = []
    all_missing_cases = []
    summaries = []

    for model_name, file_path in MODEL_FILES.items():
        print(f"\nInvestigating: {model_name}")

        predictions = load_prediction_file(
            file_path,
            model_name,
        )

        missing_cases = investigate_missing_predictions(
            predictions=predictions,
            train_raw=train_raw,
            validation_raw=validation_raw,
        )

        summary = build_model_summary(
            predictions,
            missing_cases,
        )

        all_predictions.append(predictions)
        all_missing_cases.append(missing_cases)
        summaries.append(summary)

        print(
            f"Validation rows: "
            f"{summary['validation_rows']:,}"
        )
        print(
            f"Valid predictions: "
            f"{summary['valid_predictions']:,}"
        )
        print(
            f"Missing predictions: "
            f"{summary['missing_predictions']:,}"
        )
        print(
            f"Coverage: "
            f"{summary['prediction_coverage_pct']:.2f}%"
        )

    prediction_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    nonempty_cases = [
        cases
        for cases in all_missing_cases
        if not cases.empty
    ]

    if nonempty_cases:
        cases_df = pd.concat(
            nonempty_cases,
            ignore_index=True,
        )
    else:
        cases_df = pd.DataFrame(
            columns=[
                ROAD_ID,
                TIMESTAMP,
                "actual",
                "predicted",
                "reference_timestamp",
                "model",
                "reference_source",
                "missing_reason",
            ]
        )

    model_summary = pd.DataFrame(summaries)

    reason_summary = build_reason_summary(cases_df)

    road_summary = build_road_summary(
        prediction_df,
        cases_df,
    )

    # --------------------------------------------------------
    # Save machine-readable reports.
    # --------------------------------------------------------

    model_summary.to_csv(
        OUTPUT_DIR / "missing_prediction_summary.csv",
        index=False,
    )

    reason_summary.to_csv(
        OUTPUT_DIR / "missing_prediction_reasons.csv",
        index=False,
    )

    road_summary.to_csv(
        OUTPUT_DIR / "missing_prediction_by_road.csv",
        index=False,
    )

    cases_df.to_csv(
        OUTPUT_DIR / "missing_prediction_cases.csv",
        index=False,
    )

    report = create_markdown_report(
        model_summary,
        reason_summary,
        road_summary,
    )

    (
        OUTPUT_DIR / "day11_investigation_report.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Display results.
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("MODEL SUMMARY")
    print("=" * 68)
    print(model_summary.to_string(index=False, col_space=12))

    print("\n" + "=" * 68)
    print("MISSING-PREDICTION REASONS")
    print("=" * 68)

    if reason_summary.empty:
        print("No missing predictions found.")
    else:
        print(reason_summary.to_string(index=False, col_space=12))

    print("\n" + "=" * 68)
    print("DAY 11 INVESTIGATION COMPLETED")
    print("=" * 68)
    print(f"Reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()