"""
FlowCast — Day 7
Data Cleaning and Quality Control Pipeline

Input:
    data/interim/flowcast_integrated.csv

Outputs:
    data/processed/flowcast_cleaned.csv
    reports/day7/cleaning_summary.json
    reports/day7/cleaning_report.md
    reports/day7/rows_requiring_review.csv
    reports/day7/duplicate_records.csv

The original integrated dataset is never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "interim" / "flowcast_integrated.csv"

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "day7"

CLEANED_FILE = PROCESSED_DIR / "flowcast_cleaned.csv"
SUMMARY_FILE = REPORT_DIR / "cleaning_summary.json"
REPORT_FILE = REPORT_DIR / "cleaning_report.md"
REVIEW_FILE = REPORT_DIR / "rows_requiring_review.csv"
DUPLICATES_FILE = REPORT_DIR / "duplicate_records.csv"


# ============================================================
# 2. REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "road_id",
    "date",
    "time",
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "road_capacity",
    "congestion_level",
]


# ============================================================
# 3. LOAD AND VALIDATE THE INPUT
# ============================================================

def load_data() -> pd.DataFrame:
    """Load the integrated dataset and validate its schema."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # Preserve the original row position for traceability.
    df.insert(
        0,
        "source_row_number",
        range(1, len(df) + 1),
    )

    print("=" * 65)
    print("FLOWCAST DAY 7 — DATA CLEANING PIPELINE")
    print("=" * 65)
    print(f"Input file: {INPUT_FILE}")
    print(f"Input rows: {len(df):,}")
    print(f"Input columns: {len(df.columns):,}")

    return df


# ============================================================
# 4. NORMALIZE TIMESTAMPS
# ============================================================

def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Create a timestamp from the date and time columns."""

    df = df.copy()

    timestamp_text = (
        df["date"].astype("string").str.strip()
        + " "
        + df["time"].astype("string").str.strip()
    )

    df["timestamp"] = pd.to_datetime(
        timestamp_text,
        errors="coerce",
    )

    invalid_count = int(df["timestamp"].isna().sum())

    print(f"Invalid timestamps: {invalid_count:,}")

    return df


# ============================================================
# 5. IDENTIFY AND REMOVE DUPLICATE ROAD-TIME RECORDS
# ============================================================

def remove_duplicate_records(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify duplicate road-time keys.

    Keep the first record for each road_id + timestamp.
    Preserve the extra records in a separate audit file.
    """

    duplicate_key = df.duplicated(
        subset=["road_id", "timestamp"],
        keep="first",
    )

    duplicates = df.loc[duplicate_key].copy()

    cleaned = df.loc[~duplicate_key].copy()

    print(f"Duplicate records identified: {len(duplicates):,}")
    print(f"Rows remaining after deduplication: {len(cleaned):,}")

    return cleaned, duplicates


# ============================================================
# 6. CONVERT MEASUREMENTS TO NUMERIC VALUES
# ============================================================

def convert_numeric_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Convert expected numeric fields without hiding failures."""

    df = df.copy()

    numeric_columns = [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "accident_count",
        "signal_timing",
        "road_capacity",
        "temperature",
        "rainfall",
        "visibility",
        "latitude",
        "longitude",
    ]

    numeric_columns = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    parsing_issues = {}

    for column in numeric_columns:
        original = df[column]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        # Count nonblank values that could not be converted.
        original_text = original.astype("string").str.strip()

        nonblank = (
            original.notna()
            & original_text.ne("")
        )

        failed = nonblank & converted.isna()

        parsing_issues[column] = int(failed.sum())

        df[column] = converted

    total_failures = sum(parsing_issues.values())

    print(f"Numeric parsing failures: {total_failures:,}")

    return df, parsing_issues


# ============================================================
# 7. STANDARDIZE CATEGORICAL LABELS
# ============================================================

def standardize_categories(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize known congestion and weather categories."""

    df = df.copy()

    # Standardize congestion labels.
    if "congestion_level" in df.columns:

        congestion_mapping = {
            "free-flow": "Free-flow",
            "free flow": "Free-flow",
            "freeflow": "Free-flow",
            "moderate": "Moderate",
            "heavy": "Heavy",
            "severe": "Severe",
        }

        normalized = (
            df["congestion_level"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        df["congestion_level"] = normalized.map(
            congestion_mapping
        )

    # Standardize weather labels.
    if "weather_condition" in df.columns:

        weather_mapping = {
            "clear": "Clear",
            "sunny": "Clear",
            "cloudy": "Cloudy",
            "overcast": "Cloudy",
            "rain": "Rain",
            "rainy": "Rain",
            "fog": "Fog",
            "foggy": "Fog",
        }

        normalized_weather = (
            df["weather_condition"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        df["weather_condition"] = normalized_weather.map(
            weather_mapping
        )

    return df


# ============================================================
# 8. IDENTIFY INVALID MEASUREMENTS
# ============================================================

def validate_measurements(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mark invalid measurements as missing.

    Preserve affected rows and record why they need review.
    """

    df = df.copy()

    # Each row can have more than one issue.
    issues = pd.Series(
        "",
        index=df.index,
        dtype="string",
    )

    def record_issue(mask: pd.Series, message: str) -> None:
        """Append an issue description to affected rows."""

        mask = mask.fillna(False)

        existing = issues.loc[mask]

        issues.loc[mask] = existing.map(
            lambda text: (
                f"{text}; {message}"
                if text
                else message
            )
        )

    # Traffic volume cannot be negative.
    if "traffic_volume" in df.columns:

        invalid = df["traffic_volume"] < 0

        record_issue(
            invalid,
            "negative_traffic_volume",
        )

        df.loc[invalid, "traffic_volume"] = pd.NA

    # Speed must be between 0 and 200 km/h.
    if "avg_speed" in df.columns:

        invalid = (
            (df["avg_speed"] < 0)
            | (df["avg_speed"] > 200)
        )

        record_issue(
            invalid,
            "speed_outside_0_to_200_kmh",
        )

        df.loc[invalid, "avg_speed"] = pd.NA

    # Occupancy must be between 0% and 100%.
    if "occupancy" in df.columns:

        invalid = (
            (df["occupancy"] < 0)
            | (df["occupancy"] > 100)
        )

        record_issue(
            invalid,
            "occupancy_outside_0_to_100_percent",
        )

        df.loc[invalid, "occupancy"] = pd.NA

    # Road capacity must be positive.
    if "road_capacity" in df.columns:

        invalid = df["road_capacity"] <= 0

        record_issue(
            invalid,
            "nonpositive_road_capacity",
        )

        df.loc[invalid, "road_capacity"] = pd.NA

    # Counts cannot be negative.
    for column in ["vehicle_count", "accident_count"]:

        if column not in df.columns:
            continue

        invalid = df[column] < 0

        record_issue(
            invalid,
            f"negative_{column}",
        )

        df.loc[invalid, column] = pd.NA

    # Rainfall, visibility, and travel time cannot be negative.
    for column in [
        "rainfall",
        "visibility",
        "travel_time",
    ]:

        if column not in df.columns:
            continue

        invalid = df[column] < 0

        record_issue(
            invalid,
            f"negative_{column}",
        )

        df.loc[invalid, column] = pd.NA

    review_mask = issues.ne("")

    review_records = df.loc[review_mask].copy()

    review_records["quality_issues"] = issues.loc[
        review_mask
    ]

    print(
        "Rows requiring measurement review: "
        f"{len(review_records):,}"
    )

    return df, review_records


# ============================================================
# 9. DERIVE MISSING CONGESTION LABELS
# ============================================================

def derive_congestion_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Derive missing congestion labels using the project data
    dictionary's volume-to-capacity ratio.

    V/C = traffic_volume / (road_capacity / 2)

    Only derive labels when both required inputs are valid.
    """

    df = df.copy()

    volume = df["traffic_volume"]
    capacity = df["road_capacity"]

    valid_inputs = (
        volume.notna()
        & capacity.notna()
        & capacity.gt(0)
    )

    ratio = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Float64",
    )

    ratio.loc[valid_inputs] = (
        volume.loc[valid_inputs]
        / (capacity.loc[valid_inputs] / 2)
    )

    df["volume_capacity_ratio"] = ratio

    missing_label = df["congestion_level"].isna()

    eligible = missing_label & valid_inputs

    derived_count = int(eligible.sum())

    df.loc[
        eligible & df["volume_capacity_ratio"].lt(0.50),
        "congestion_level",
    ] = "Free-flow"

    df.loc[
        eligible
        & df["volume_capacity_ratio"].ge(0.50)
        & df["volume_capacity_ratio"].lt(0.80),
        "congestion_level",
    ] = "Moderate"

    df.loc[
        eligible
        & df["volume_capacity_ratio"].ge(0.80)
        & df["volume_capacity_ratio"].lt(1.00),
        "congestion_level",
    ] = "Heavy"

    df.loc[
        eligible & df["volume_capacity_ratio"].ge(1.00),
        "congestion_level",
    ] = "Severe"

    print(f"Congestion labels derived: {derived_count:,}")

    return df, derived_count


# ============================================================
# 10. BUILD QUALITY SUMMARY
# ============================================================

def build_summary(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    duplicates: pd.DataFrame,
    review_records: pd.DataFrame,
    parsing_issues: dict,
    derived_count: int,
) -> dict:
    """Create a machine-readable data-quality summary."""

    missing_counts = (
        cleaned.isna()
        .sum()
        .astype(int)
        .to_dict()
    )

    summary = {
        "project": "FlowCast",
        "day": 7,
        "stage": "Data Cleaning and Quality Control",
        "input_file": str(
            INPUT_FILE.relative_to(ROOT)
        ),
        "output_file": str(
            CLEANED_FILE.relative_to(ROOT)
        ),
        "original_dataset_modified": False,
        "input_rows": int(len(original)),
        "cleaned_rows": int(len(cleaned)),
        "duplicate_rows_separated": int(len(duplicates)),
        "rows_requiring_review": int(len(review_records)),
        "numeric_parsing_failures": parsing_issues,
        "derived_congestion_labels": int(derived_count),
        "missing_values_after_cleaning": missing_counts,
        "remaining_missing_target_rows": int(
            cleaned["traffic_volume"].isna().sum()
        ),
        "remaining_missing_congestion_labels": int(
            cleaned["congestion_level"].isna().sum()
        ),
        "unique_roads": int(
            cleaned["road_id"].nunique()
        ),
        "invalid_timestamps": int(
            cleaned["timestamp"].isna().sum()
        ),
    }

    return summary


# ============================================================
# 11. WRITE THE MARKDOWN REPORT
# ============================================================

def write_markdown_report(summary: dict) -> None:
    """Write a readable summary of the cleaning decisions."""

    missing_values = summary["missing_values_after_cleaning"]

    lines = [
        "# FlowCast — Day 7 Data Cleaning Report",
        "",
        "## 1. Dataset overview",
        "",
        f"- Input rows: {summary['input_rows']:,}",
        f"- Output rows: {summary['cleaned_rows']:,}",
        f"- Duplicate rows separated: "
        f"{summary['duplicate_rows_separated']:,}",
        f"- Rows requiring review: "
        f"{summary['rows_requiring_review']:,}",
        f"- Unique road segments: {summary['unique_roads']:,}",
        f"- Invalid timestamps: {summary['invalid_timestamps']:,}",
        "",
        "## 2. Cleaning decisions",
        "",
        "- Negative traffic volumes are converted to missing values.",
        "- Speeds outside 0–200 km/h are converted to missing values.",
        "- Occupancy outside 0–100% is converted to missing values.",
        "- Nonpositive road capacity is invalid for volume-capacity calculations.",
        "- Negative count, rainfall, visibility, and travel-time values "
        "are converted to missing values.",
        "- Duplicate road-time records are separated into an audit file.",
        "- Known congestion and weather labels are standardized.",
        "- Missing congestion labels are derived only when valid "
        "traffic volume and road capacity are available.",
        "- Rows with missing traffic targets are retained for traceability.",
        "- The original integrated dataset is not modified.",
        "",
        "## 3. Congestion derivation",
        "",
        "V/C = traffic_volume / (road_capacity / 2)",
        "",
        "| V/C ratio | Congestion label |",
        "|---|---|",
        "| < 0.50 | Free-flow |",
        "| 0.50 to < 0.80 | Moderate |",
        "| 0.80 to < 1.00 | Heavy |",
        "| >= 1.00 | Severe |",
        "",
        f"Labels derived: {summary['derived_congestion_labels']:,}",
        "",
        "## 4. Missing values after cleaning",
        "",
        "| Column | Missing values |",
        "|---|---:|",
    ]

    for column, count in sorted(
        missing_values.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"| {column} | {count:,} |")

    lines.extend(
        [
            "",
            "## 5. Numeric parsing failures",
            "",
            "| Column | Failed conversions |",
            "|---|---:|",
        ]
    )

    for column, count in sorted(
        summary["numeric_parsing_failures"].items()
    ):
        lines.append(f"| {column} | {count:,} |")

    lines.extend(
        [
            "",
            "## 6. Output artifacts",
            "",
            "- `data/processed/flowcast_cleaned.csv`",
            "- `reports/day7/cleaning_summary.json`",
            "- `reports/day7/rows_requiring_review.csv`",
            "- `reports/day7/duplicate_records.csv`",
            "- `reports/day7/cleaning_report.md`",
            "",
            "## 7. Limitations and next steps",
            "",
            "This pipeline does not automatically impute missing traffic "
            "targets or sensor measurements. Imputation decisions require "
            "time-aware analysis and must avoid information leakage.",
            "",
            "The review file identifies records with rule violations; "
            "it does not establish the true value of a faulty sensor reading.",
            "",
            "Before modelling, verify feature availability at prediction "
            "time and ensure all validation splits respect chronology.",
            "",
        ]
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# 12. MAIN PIPELINE
# ============================================================

def main() -> None:
    """Execute the complete Day 7 pipeline."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load source data.
    original = load_data()

    # Normalize timestamps.
    df = normalize_timestamps(original)

    # Separate duplicate road-time records.
    df, duplicates = remove_duplicate_records(df)

    # Convert measurement columns to numeric types.
    df, parsing_issues = convert_numeric_columns(df)

    # Standardize known categories.
    df = standardize_categories(df)

    # Apply documented measurement validation rules.
    df, review_records = validate_measurements(df)

    # Derive missing congestion labels where possible.
    df, derived_count = derive_congestion_labels(df)

    # Save the cleaned dataset.
    df.to_csv(
        CLEANED_FILE,
        index=False,
    )

    # Save the audit files.
    duplicates.to_csv(
        DUPLICATES_FILE,
        index=False,
    )

    review_records.to_csv(
        REVIEW_FILE,
        index=False,
    )

    # Build and save the summary.
    summary = build_summary(
        original=original,
        cleaned=df,
        duplicates=duplicates,
        review_records=review_records,
        parsing_issues=parsing_issues,
        derived_count=derived_count,
    )

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=4,
            default=str,
        ),
        encoding="utf-8",
    )

    # Generate the human-readable report.
    write_markdown_report(summary)

    print("\n" + "=" * 65)
    print("DAY 7 PIPELINE COMPLETED")
    print("=" * 65)
    print(f"Cleaned dataset: {CLEANED_FILE}")
    print(f"Quality summary: {SUMMARY_FILE}")
    print(f"Review records: {REVIEW_FILE}")
    print(f"Duplicate records: {DUPLICATES_FILE}")
    print(f"Markdown report: {REPORT_FILE}")
    print(f"Output rows: {len(df):,}")
    print(f"Review rows: {len(review_records):,}")
    print(f"Derived congestion labels: {derived_count:,}")
    print("\nOriginal integrated data was not modified.")


if __name__ == "__main__":
    main()