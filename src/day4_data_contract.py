"""
FlowCast Day 4
Reusable Data Contract and Validation Pipeline.

Responsibilities:
1. Validate the integrated dataset's schema.
2. Check measurement types and ranges.
3. Record missing values.
4. Identify duplicate road-time keys.
5. Quarantine records with blocking errors.
6. Preserve every original input row in the validated output.
7. Produce auditable reports.

Input:
    data/interim/flowcast_integrated.csv

Outputs:
    data/processed/flowcast_validated.csv
    data/processed/quarantined_records.csv
    reports/day4/validation_summary.json
    reports/day4/validation_issues.csv
    reports/day4/missingness_report.csv
    reports/day4/duplicate_records.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "flowcast_integrated.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REPORT_DIR = PROJECT_ROOT / "reports" / "day4"


# ============================================================
# 2. DATA CONTRACT
# ============================================================

# Columns required to identify a traffic observation.
REQUIRED_COLUMNS = [
    "road_id",
    "date",
    "time",
]

# Columns expected in the integrated FlowCast dataset.
EXPECTED_COLUMNS = [
    "road_id",
    "road_name",
    "latitude",
    "longitude",
    "weather_station_id",
    "date",
    "time",
    "traffic_volume",
    "vehicle_count",
    "vehicle_type_dist",
    "avg_speed",
    "occupancy",
    "congestion_level",
    "travel_time",
    "accident_count",
    "signal_timing",
    "road_capacity",
    "speed_above_200_flag",
    "occupancy_above_100_flag",
    "hour",
    "weather_condition",
    "temperature",
    "rainfall",
    "visibility",
    "public_holiday",
    "holiday_name",
    "event_flag",
    "event_name",
    "roadwork_flag",
]

NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "traffic_volume",
    "vehicle_count",
    "avg_speed",
    "occupancy",
    "travel_time",
    "accident_count",
    "signal_timing",
    "road_capacity",
    "hour",
    "temperature",
    "rainfall",
    "visibility",
    "public_holiday",
    "event_flag",
    "roadwork_flag",
    "speed_above_200_flag",
    "occupancy_above_100_flag",
]

NON_NEGATIVE_COLUMNS = [
    "traffic_volume",
    "vehicle_count",
    "avg_speed",
    "occupancy",
    "travel_time",
    "accident_count",
    "signal_timing",
    "road_capacity",
    "rainfall",
    "visibility",
]

# These are screening rules.
# They identify suspicious values but do not prove that
# a measurement is erroneous.
RANGE_RULES = {
    "avg_speed": {
        "minimum": 0,
        "maximum": 200,
    },
    "occupancy": {
        "minimum": 0,
        "maximum": 100,
    },
}

# Missingness in these columns is not automatically a
# blocking error. Missing targets and features require
# model-specific decisions later.
TARGET_COLUMNS = [
    "traffic_volume",
    "congestion_level",
]

DUPLICATE_KEY = [
    "road_id",
    "date",
    "time",
]


# ============================================================
# 3. HELPER: RECORD AN ISSUE
# ============================================================

def record_issue(
    issues: list[dict[str, Any]],
    source_row_number: int | str,
    road_id: Any,
    issue_type: str,
    column: str,
    observed_value: Any,
    severity: str,
    explanation: str,
) -> None:
    """
    Append one auditable validation finding.

    ERROR:
        Blocking issue; the row will be quarantined.

    REVIEW:
        Suspicious value requiring investigation.

    INFO:
        Informational finding.
    """

    issues.append(
        {
            "source_row_number": source_row_number,
            "road_id": (
                road_id
                if pd.notna(road_id)
                else ""
            ),
            "issue_type": issue_type,
            "column": column,
            "observed_value": str(observed_value),
            "severity": severity,
            "explanation": explanation,
        }
    )


# ============================================================
# 4. VALIDATE THE INPUT SCHEMA
# ============================================================

def validate_schema(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Return missing expected columns and unexpected columns.
    """

    missing_expected = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    unexpected_columns = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    missing_required = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_required:
        raise ValueError(
            "Required identity columns are missing: "
            + ", ".join(missing_required)
        )

    return missing_expected, unexpected_columns


# ============================================================
# 5. REUSABLE VALIDATION ENGINE
# ============================================================

def inspect_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Inspect a DataFrame and return:

    1. All original rows with validation metadata.
    2. A row-level issue table.
    3. A summary dictionary.

    This function does not delete original records.
    """

    if df.empty:
        raise ValueError(
            "The input dataset contains no data rows."
        )

    # Keep a separate working copy.
    # Never modify the caller's DataFrame.
    output = df.copy(deep=True)

    source_row_numbers = pd.Series(
        range(2, len(output) + 2),
        index=output.index,
    )

    issues: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # A. Schema
    # --------------------------------------------------------

    missing_expected, unexpected_columns = (
        validate_schema(output)
    )

    for column in missing_expected:
        record_issue(
            issues=issues,
            source_row_number="",
            road_id="",
            issue_type="missing_expected_column",
            column=column,
            observed_value="COLUMN_ABSENT",
            severity="ERROR",
            explanation=(
                "An expected column is absent from the dataset."
            ),
        )

    for column in unexpected_columns:
        record_issue(
            issues=issues,
            source_row_number="",
            road_id="",
            issue_type="unexpected_column",
            column=column,
            observed_value="COLUMN_PRESENT",
            severity="INFO",
            explanation=(
                "The input contains a column not listed in "
                "the current data contract."
            ),
        )

    # --------------------------------------------------------
    # B. Required identifiers
    # --------------------------------------------------------

    for column in REQUIRED_COLUMNS:

        missing_mask = (
            output[column].isna()
            | output[column].astype("string").str.strip().eq("")
        )

        for index in output.index[missing_mask]:

            record_issue(
                issues,
                int(source_row_numbers.loc[index]),
                output.at[index, "road_id"],
                "missing_required_value",
                column,
                "MISSING",
                "ERROR",
                "Required road identity or timestamp field is missing.",
            )

    # --------------------------------------------------------
    # C. Date and time parsing
    # --------------------------------------------------------

    parsed_dates = pd.to_datetime(
        output["date"],
        errors="coerce",
    )

    for index in output.index:

        original_date = output.at[index, "date"]

        if (
            pd.notna(original_date)
            and pd.isna(parsed_dates.loc[index])
        ):
            record_issue(
                issues,
                int(source_row_numbers.loc[index]),
                output.at[index, "road_id"],
                "invalid_date",
                "date",
                original_date,
                "ERROR",
                "The date could not be parsed.",
            )

    def parse_time(value: Any) -> Any:
        """
        Parse common HH:MM and HH:MM:SS time strings.

        Returns None when the value cannot be parsed.
        """

        if pd.isna(value):
            return None

        text = str(value).strip()

        for time_format in ("%H:%M", "%H:%M:%S"):

            try:
                from datetime import datetime

                return datetime.strptime(
                    text,
                    time_format,
                ).time()

            except ValueError:
                continue

        return None

    parsed_times = output["time"].map(parse_time)

    for index in output.index:

        original_time = output.at[index, "time"]

        if (
            pd.notna(original_time)
            and parsed_times.loc[index] is None
        ):
            record_issue(
                issues,
                int(source_row_numbers.loc[index]),
                output.at[index, "road_id"],
                "invalid_time",
                "time",
                original_time,
                "ERROR",
                "The time could not be parsed as HH:MM or HH:MM:SS.",
            )

    # --------------------------------------------------------
    # D. Numeric type validation
    # --------------------------------------------------------

    numeric_values: dict[str, pd.Series] = {}

    for column in NUMERIC_COLUMNS:

        if column not in output.columns:
            continue

        converted = pd.to_numeric(
            output[column],
            errors="coerce",
        )

        numeric_values[column] = converted

        nonnumeric_mask = (
            output[column].notna()
            & converted.isna()
        )

        for index in output.index[nonnumeric_mask]:

            record_issue(
                issues,
                int(source_row_numbers.loc[index]),
                output.at[index, "road_id"],
                "nonnumeric_measurement",
                column,
                output.at[index, column],
                "ERROR",
                "The value cannot be interpreted as numeric.",
            )

    # --------------------------------------------------------
    # E. Negative measurements
    # --------------------------------------------------------

    for column in NON_NEGATIVE_COLUMNS:

        if column not in numeric_values:
            continue

        values = numeric_values[column]

        negative_mask = values < 0

        for index in output.index[
            negative_mask.fillna(False)
        ]:

            record_issue(
                issues,
                int(source_row_numbers.loc[index]),
                output.at[index, "road_id"],
                "negative_measurement",
                column,
                values.loc[index],
                "ERROR",
                "The measurement is below zero.",
            )

    # --------------------------------------------------------
    # F. Suspicious measurement ranges
    # --------------------------------------------------------

    for column, limits in RANGE_RULES.items():

        if column not in numeric_values:
            continue

        values = numeric_values[column]

        outside_range = (
            (values < limits["minimum"])
            | (values > limits["maximum"])
        )

        for index in output.index[
            outside_range.fillna(False)
        ]:

            record_issue(
                issues,
                int(source_row_numbers.loc[index]),
                output.at[index, "road_id"],
                "outside_screening_range",
                column,
                values.loc[index],
                "REVIEW",
                (
                    "Value falls outside the screening range "
                    f"[{limits['minimum']}, {limits['maximum']}]. "
                    "Investigate before correcting or deleting it."
                ),
            )

    # --------------------------------------------------------
    # G. Binary flag validation
    # --------------------------------------------------------

    binary_columns = [
        "public_holiday",
        "event_flag",
        "roadwork_flag",
        "speed_above_200_flag",
        "occupancy_above_100_flag",
    ]

    for column in binary_columns:

        if column not in numeric_values:
            continue

        values = numeric_values[column]

        invalid_binary = (
            values.notna()
            & ~values.isin([0, 1])
        )

        for index in output.index[invalid_binary]:

            record_issue(
                issues,
                int(source_row_numbers.loc[index]),
                output.at[index, "road_id"],
                "invalid_binary_flag",
                column,
                values.loc[index],
                "ERROR",
                "Expected a binary flag of 0 or 1.",
            )

    # --------------------------------------------------------
    # H. Hour validation
    # --------------------------------------------------------

    if "hour" in numeric_values:

        hour_values = numeric_values["hour"]

        invalid_hour = (
            hour_values.notna()
            & (
                (hour_values < 0)
                | (hour_values > 23)
                | (hour_values % 1 != 0)
            )
        )

        for index in output.index[invalid_hour]:

            record_issue(
                issues,
                int(source_row_numbers.loc[index]),
                output.at[index, "road_id"],
                "invalid_hour",
                "hour",
                hour_values.loc[index],
                "ERROR",
                "Hour must be an integer between 0 and 23.",
            )

    # --------------------------------------------------------
    # I. Duplicate road-time keys
    # --------------------------------------------------------

    complete_key_mask = output[
        DUPLICATE_KEY
    ].notna().all(axis=1)

    duplicate_mask = pd.Series(
        False,
        index=output.index,
    )

    duplicate_mask.loc[complete_key_mask] = (
        output.loc[complete_key_mask]
        .duplicated(
            subset=DUPLICATE_KEY,
            keep=False,
        )
    )

    for index in output.index[duplicate_mask]:

        record_issue(
            issues,
            int(source_row_numbers.loc[index]),
            output.at[index, "road_id"],
            "duplicate_road_time_key",
            ",".join(DUPLICATE_KEY),
            (
                f"{output.at[index, 'road_id']} | "
                f"{output.at[index, 'date']} | "
                f"{output.at[index, 'time']}"
            ),
            "REVIEW",
            (
                "Multiple records share a road-time key. "
                "Investigate before deduplication."
            ),
        )

    # --------------------------------------------------------
    # J. Build row-level status
    # --------------------------------------------------------

    issue_df = pd.DataFrame(
        issues,
        columns=[
            "source_row_number",
            "road_id",
            "issue_type",
            "column",
            "observed_value",
            "severity",
            "explanation",
        ],
    )

    output.insert(
        0,
        "source_row_number",
        source_row_numbers.values,
    )

    output["quality_status"] = "VALID"
    output["quality_issue_count"] = 0

    if not issue_df.empty:

        row_issues = issue_df[
            issue_df["source_row_number"].astype(str).str.isdigit()
        ].copy()

        if not row_issues.empty:

            row_issues["source_row_number"] = (
                row_issues["source_row_number"].astype(int)
            )

            counts = row_issues.groupby(
                "source_row_number"
            ).size()

            error_rows = set(
                row_issues.loc[
                    row_issues["severity"] == "ERROR",
                    "source_row_number",
                ]
            )

            review_rows = set(
                row_issues.loc[
                    row_issues["severity"] == "REVIEW",
                    "source_row_number",
                ]
            )

            for index in output.index:

                source_number = int(
                    output.at[index, "source_row_number"]
                )

                output.at[
                    index,
                    "quality_issue_count",
                ] = int(counts.get(source_number, 0))

                if source_number in error_rows:
                    output.at[index, "quality_status"] = "QUARANTINE"

                elif source_number in review_rows:
                    output.at[index, "quality_status"] = "REVIEW"

    # --------------------------------------------------------
    # K. Missingness report
    # --------------------------------------------------------

    missingness = pd.DataFrame(
        {
            "column": df.columns,
            "total_rows": len(df),
            "missing_count": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
            "missing_percentage": [
                round(
                    float(df[column].isna().mean() * 100),
                    4,
                )
                for column in df.columns
            ],
        }
    ).sort_values(
        "missing_percentage",
        ascending=False,
    )

    # --------------------------------------------------------
    # L. Duplicate report
    # --------------------------------------------------------

    duplicate_records = df.loc[
        duplicate_mask
    ].copy()

    duplicate_records.insert(
        0,
        "source_row_number",
        source_row_numbers.loc[
            duplicate_records.index
        ].values,
    )

    # --------------------------------------------------------
    # M. Summary
    # --------------------------------------------------------

    error_count = int(
        (issue_df["severity"] == "ERROR").sum()
    ) if not issue_df.empty else 0

    review_count = int(
        (issue_df["severity"] == "REVIEW").sum()
    ) if not issue_df.empty else 0

    info_count = int(
        (issue_df["severity"] == "INFO").sum()
    ) if not issue_df.empty else 0

    summary = {
        "project": "FlowCast",
        "day": 4,
        "input_rows": int(len(df)),
        "input_columns": int(len(df.columns)),
        "expected_columns_missing": missing_expected,
        "unexpected_columns": unexpected_columns,
        "total_validation_findings": int(len(issue_df)),
        "error_findings": error_count,
        "review_findings": review_count,
        "info_findings": info_count,
        "missing_cells": int(df.isna().sum().sum()),
        "columns_with_missing_values": int(
            (df.isna().sum() > 0).sum()
        ),
        "duplicate_rows_flagged": int(duplicate_mask.sum()),
        "valid_rows": int(
            (output["quality_status"] == "VALID").sum()
        ),
        "review_rows": int(
            (output["quality_status"] == "REVIEW").sum()
        ),
        "quarantined_rows": int(
            (output["quality_status"] == "QUARANTINE").sum()
        ),
        "all_input_rows_preserved": (
            len(output) == len(df)
        ),
        "automatic_row_deletion": False,
    }

    return output, issue_df, missingness, duplicate_records, summary


# ============================================================
# 6. FILE-BASED PIPELINE
# ============================================================

def run_pipeline() -> None:
    """
    Run validation and persist all outputs.
    """

    print("=" * 65)
    print("FLOWCAST DAY 4 — DATA CONTRACT")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}\n\n"
            "Make sure the Day 1 integration pipeline has run."
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[1/5] Loading integrated dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input: {INPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n[2/5] Running data contract checks...")

    (
        validated,
        issues,
        missingness,
        duplicates,
        summary,
    ) = inspect_dataframe(df)

    print("\n[3/5] Saving validated data...")

    validated.to_csv(
        PROCESSED_DIR / "flowcast_validated.csv",
        index=False,
    )

    quarantined = validated.loc[
        validated["quality_status"] == "QUARANTINE"
    ]

    quarantined.to_csv(
        PROCESSED_DIR / "quarantined_records.csv",
        index=False,
    )

    print("\n[4/5] Saving audit reports...")

    issues.to_csv(
        REPORT_DIR / "validation_issues.csv",
        index=False,
    )

    missingness.to_csv(
        REPORT_DIR / "missingness_report.csv",
        index=False,
    )

    duplicates.to_csv(
        REPORT_DIR / "duplicate_records.csv",
        index=False,
    )

    with open(
        REPORT_DIR / "validation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n[5/5] Verifying row preservation...")

    saved = pd.read_csv(
        PROCESSED_DIR / "flowcast_validated.csv"
    )

    if len(saved) != len(df):
        raise RuntimeError(
            "Row preservation check failed. "
            "The validated output does not contain "
            "the same number of rows as the input."
        )

    print("\n" + "=" * 65)
    print("DAY 4 VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Input rows: {summary['input_rows']:,}")
    print(f"Output rows: {len(saved):,}")

    print(
        "Missing expected columns:",
        len(summary["expected_columns_missing"]),
    )

    print(
        "Validation findings:",
        summary["total_validation_findings"],
    )

    print(
        "ERROR findings:",
        summary["error_findings"],
    )

    print(
        "REVIEW findings:",
        summary["review_findings"],
    )

    print(
        "INFO findings:",
        summary["info_findings"],
    )

    print(
        "Rows flagged for quarantine:",
        summary["quarantined_rows"],
    )

    print(
        "Rows flagged for review:",
        summary["review_rows"],
    )

    print(
        "Duplicate rows flagged:",
        summary["duplicate_rows_flagged"],
    )

    print(
        "All input rows preserved:",
        summary["all_input_rows_preserved"],
    )

    print("\nProcessed dataset:")
    print(PROCESSED_DIR / "flowcast_validated.csv")

    print("\nQuarantined records:")
    print(PROCESSED_DIR / "quarantined_records.csv")

    print("\nReports:")
    print(REPORT_DIR)

    print("\nOriginal integrated dataset was not modified.")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()