"""
FLOWCAST — DAY 3
Data Quality Policy and Automated Validation

Purpose:
1. Validate the integrated traffic dataset.
2. Identify missing values.
3. Detect invalid measurements.
4. Flag suspicious measurements for investigation.
5. Detect invalid timestamps and duplicate road-time records.
6. Generate machine-readable reports.
7. Preserve original data without automatically deleting records.

Input:
    data/interim/flowcast_integrated.csv

Outputs:
    reports/data_quality_policy.md
    reports/day3/validation_summary.json
    reports/day3/validation_issues.csv
    reports/day3/missingness_report.csv
    reports/day3/duplicate_records.csv
"""

from pathlib import Path
import json

import pandas as pd


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "flowcast_integrated.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "day3"

POLICY_FILE = PROJECT_ROOT / "reports" / "data_quality_policy.md"


# ==========================================================
# 2. EXPECTED SCHEMA
# ==========================================================

REQUIRED_COLUMNS = [
    "road_id",
    "date",
    "time",
]

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

# Measurements that should not be negative.
NON_NEGATIVE_COLUMNS = [
    "traffic_volume",
    "vehicle_count",
    "travel_time",
    "accident_count",
    "rainfall",
]

# Measurements that will be screened using broad ranges.
# These are investigation thresholds, not proof of bad data.
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


# ==========================================================
# 3. REPORTING HELPERS
# ==========================================================

def add_issue(
    issues,
    row_number,
    road_id,
    issue_type,
    column,
    value,
    severity,
    explanation,
):
    """Add one validation finding to the issue list."""

    issues.append(
        {
            "source_row_number": row_number,
            "road_id": road_id,
            "issue_type": issue_type,
            "column": column,
            "observed_value": str(value),
            "severity": severity,
            "explanation": explanation,
        }
    )


def write_policy():
    """Write the documented data-quality policy."""

    POLICY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy_text = """# FlowCast Data Quality Policy

## 1. Purpose

This policy defines how FlowCast identifies missing, invalid,
suspicious, and duplicate traffic observations.

The objective is to protect data integrity before feature
engineering, model training, and traffic prediction.

## 2. Data preservation

- Original source datasets must remain unchanged.
- Validation identifies issues; it does not silently repair them.
- Suspicious observations must not be automatically deleted.
- Every reported issue should be traceable to its source row.
- Cleaning decisions must be documented and justified.

## 3. Schema and identity rules

The integrated dataset must contain:

- `road_id`
- `date`
- `time`

The validation script also checks the expected project schema.
Missing expected columns are reported as schema problems.

Rows without a road identifier, date, or time are considered
unusable for reliable road-time analysis until investigated.

Invalid timestamps are reported as errors.

## 4. Missing-value policy

Missingness is measured separately for every column.

- Missing road identifiers and timestamps require investigation.
- Missing traffic targets must not be silently imputed.
- Missing features may require training-time imputation,
  depending on the intended model and feature availability.
- Missing congestion labels affect classification training.
- Sparse `holiday_name` and `event_name` values may be expected
  when no holiday or named event applies.
- A missing value must not automatically be interpreted as zero.

Imputation decisions must be made separately for each prediction
task and must be fitted using training data only.

## 5. Measurement rules

The following values are considered invalid candidates:

- Negative traffic volume.
- Negative vehicle count.
- Negative travel time.
- Negative accident count.
- Negative rainfall.

These findings must be investigated before the observations
are used in training.

## 6. Suspicious-range screening

The following broad screening thresholds are used:

- Average speed below 0 or above 200.
- Occupancy below 0 or above 100.

These are screening rules, not proof that every flagged value
is erroneous.

Flagged records must be reviewed against the source data,
measurement definitions, and sensor documentation.

## 7. Duplicate records

Records sharing the same `road_id`, `date`, and `time` are
reported for investigation.

A duplicate key is not automatically proof of an erroneous
record. The intended observation frequency and source-system
semantics must be checked before any record is removed.

## 8. Validation severity

- ERROR: a structural or measurement problem requiring attention.
- REVIEW: a suspicious observation requiring investigation.
- INFO: a diagnostic finding that does not independently prove
  that the record is invalid.

## 9. Model-specific usability

A row can be usable for one task and unusable for another.

For example:

- Missing traffic volume prevents supervised training when
  traffic volume is the target.
- Missing congestion labels prevent supervised classification
  training for that row.
- Missing a feature may be manageable if the training pipeline
  implements an appropriate, leakage-safe imputation strategy.

Do not discard a row globally just because one column is missing.

## 10. Output and auditability

The validation process generates:

- A JSON summary.
- A CSV of validation findings.
- A CSV of missingness statistics.
- A CSV of duplicate road-time records.

This policy and its validation rules must be reviewed whenever
the dataset schema or sensor definitions change.
"""

    POLICY_FILE.write_text(
        policy_text,
        encoding="utf-8",
    )


# ==========================================================
# 4. MAIN VALIDATION PIPELINE
# ==========================================================

def main():

    print("=" * 65)
    print("FLOWCAST — DAY 3 DATA QUALITY VALIDATION")
    print("=" * 65)

    # ------------------------------------------------------
    # A. Check input
    # ------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nIntegrated dataset not found:\n{INPUT_FILE}\n\n"
            "Run the existing FlowCast integration pipeline first."
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # B. Load the dataset
    # ------------------------------------------------------

    print("\n[1/8] Loading integrated dataset...")

    df = pd.read_csv(INPUT_FILE)

    original_row_count = len(df)

    print(f"Input file: {INPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Use the existing DataFrame index to trace findings.
    # The reported row number is the CSV data-row number,
    # where the first data row is row 2 because row 1 is
    # the header.
    source_row_numbers = pd.Series(
        range(2, len(df) + 2),
        index=df.index,
    )

    issues = []

    # ------------------------------------------------------
    # C. Schema validation
    # ------------------------------------------------------

    print("\n[2/8] Checking the expected schema...")

    missing_required = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

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

    for column in missing_expected:
        issues.append(
            {
                "source_row_number": "",
                "road_id": "",
                "issue_type": "missing_expected_column",
                "column": column,
                "observed_value": "",
                "severity": "ERROR",
                "explanation": (
                    "Expected column is absent from the dataset."
                ),
            }
        )

    for column in unexpected_columns:
        issues.append(
            {
                "source_row_number": "",
                "road_id": "",
                "issue_type": "unexpected_column",
                "column": column,
                "observed_value": "",
                "severity": "INFO",
                "explanation": (
                    "Column is not listed in the current expected schema. "
                    "Review whether the schema should be updated."
                ),
            }
        )

    print(f"Expected columns missing: {len(missing_expected)}")
    print(f"Unexpected columns: {len(unexpected_columns)}")

    # Do not proceed if key columns are absent.
    if missing_required:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing_required)
        )

    # ------------------------------------------------------
    # D. Missingness analysis
    # ------------------------------------------------------

    print("\n[3/8] Calculating missingness...")

    missing_count = df.isna().sum()

    missingness_report = pd.DataFrame(
        {
            "column": df.columns,
            "total_rows": len(df),
            "missing_count": [
                int(missing_count[column])
                for column in df.columns
            ],
            "missing_percentage": [
                round(
                    float(missing_count[column])
                    / len(df)
                    * 100,
                    4,
                )
                if len(df) > 0
                else 0.0
                for column in df.columns
            ],
        }
    )

    missingness_report = missingness_report.sort_values(
        by="missing_percentage",
        ascending=False,
    )

    missingness_report.to_csv(
        REPORT_DIR / "missingness_report.csv",
        index=False,
    )

    for column in REQUIRED_COLUMNS:
        missing_mask = df[column].isna()

        for index in df.index[missing_mask]:
            add_issue(
                issues=issues,
                row_number=int(source_row_numbers.loc[index]),
                road_id=(
                    df.at[index, "road_id"]
                    if pd.notna(df.at[index, "road_id"])
                    else ""
                ),
                issue_type="missing_required_value",
                column=column,
                value="MISSING",
                severity="ERROR",
                explanation=(
                    "Required identity or timestamp field is missing."
                ),
            )

    # ------------------------------------------------------
    # E. Timestamp validation
    # ------------------------------------------------------

    print("\n[4/8] Validating dates and times...")

    date_values = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    time_values = pd.to_datetime(
        df["time"].astype("string"),
        format="mixed",
        errors="coerce",
    )

    invalid_date_mask = (
        df["date"].notna()
        & date_values.isna()
    )

    invalid_time_mask = (
        df["time"].notna()
        & time_values.isna()
    )

    for index in df.index[invalid_date_mask]:
        add_issue(
            issues,
            int(source_row_numbers.loc[index]),
            df.at[index, "road_id"],
            "invalid_date",
            "date",
            df.at[index, "date"],
            "ERROR",
            "Date could not be parsed.",
        )

    for index in df.index[invalid_time_mask]:
        add_issue(
            issues,
            int(source_row_numbers.loc[index]),
            df.at[index, "road_id"],
            "invalid_time",
            "time",
            df.at[index, "time"],
            "ERROR",
            "Time could not be parsed.",
        )

    # ------------------------------------------------------
    # F. Numeric measurement validation
    # ------------------------------------------------------

    print("\n[5/8] Checking measurement values...")

    numeric_columns = list(
        dict.fromkeys(
            NON_NEGATIVE_COLUMNS
            + list(RANGE_RULES.keys())
        )
    )

    numeric_columns = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    for column in numeric_columns:

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        # Distinguish missing values from nonnumeric values.
        nonnumeric_mask = (
            df[column].notna()
            & numeric_values.isna()
        )

        for index in df.index[nonnumeric_mask]:
            add_issue(
                issues,
                int(source_row_numbers.loc[index]),
                df.at[index, "road_id"],
                "nonnumeric_measurement",
                column,
                df.at[index, column],
                "ERROR",
                "Measurement could not be converted to a number.",
            )

        # Negative measurements that should not be negative.
        if column in NON_NEGATIVE_COLUMNS:

            negative_mask = numeric_values < 0

            for index in df.index[negative_mask.fillna(False)]:
                add_issue(
                    issues,
                    int(source_row_numbers.loc[index]),
                    df.at[index, "road_id"],
                    "negative_measurement",
                    column,
                    numeric_values.loc[index],
                    "ERROR",
                    "Measurement is below zero.",
                )

        # Suspicious range values require investigation.
        if column in RANGE_RULES:

            minimum = RANGE_RULES[column]["minimum"]
            maximum = RANGE_RULES[column]["maximum"]

            outside_range_mask = (
                (numeric_values < minimum)
                | (numeric_values > maximum)
            )

            for index in df.index[
                outside_range_mask.fillna(False)
            ]:
                add_issue(
                    issues,
                    int(source_row_numbers.loc[index]),
                    df.at[index, "road_id"],
                    "outside_screening_range",
                    column,
                    numeric_values.loc[index],
                    "REVIEW",
                    (
                        f"Value is outside the screening range "
                        f"[{minimum}, {maximum}]. "
                        "Investigate before deciding whether it is invalid."
                    ),
                )

    # ------------------------------------------------------
    # G. Duplicate road-time records
    # ------------------------------------------------------

    print("\n[6/8] Checking duplicate road-time keys...")

    key_columns = [
        "road_id",
        "date",
        "time",
    ]

    # Check duplicates only when all key components are present.
    complete_key_mask = df[key_columns].notna().all(axis=1)

    duplicate_mask = pd.Series(
        False,
        index=df.index,
    )

    duplicate_mask.loc[complete_key_mask] = (
        df.loc[complete_key_mask]
        .duplicated(
            subset=key_columns,
            keep=False,
        )
    )

    duplicate_records = df.loc[duplicate_mask].copy()

    duplicate_records.insert(
        0,
        "source_row_number",
        source_row_numbers.loc[duplicate_records.index].values,
    )

    duplicate_records.to_csv(
        REPORT_DIR / "duplicate_records.csv",
        index=False,
    )

    for index in df.index[duplicate_mask]:
        add_issue(
            issues,
            int(source_row_numbers.loc[index]),
            df.at[index, "road_id"],
            "duplicate_road_time_key",
            "road_id,date,time",
            (
                f"{df.at[index, 'road_id']} | "
                f"{df.at[index, 'date']} | "
                f"{df.at[index, 'time']}"
            ),
            "REVIEW",
            (
                "Multiple rows share the same road, date, and time. "
                "Review source semantics before removing records."
            ),
        )

    # ------------------------------------------------------
    # H. Save validation reports
    # ------------------------------------------------------

    print("\n[7/8] Saving validation reports...")

    issues_df = pd.DataFrame(
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

    issues_df.to_csv(
        REPORT_DIR / "validation_issues.csv",
        index=False,
    )

    error_count = int(
        (issues_df["severity"] == "ERROR").sum()
    )

    review_count = int(
        (issues_df["severity"] == "REVIEW").sum()
    )

    info_count = int(
        (issues_df["severity"] == "INFO").sum()
    )

    missing_cells = int(df.isna().sum().sum())

    variables_with_missing_values = int(
        (df.isna().sum() > 0).sum()
    )

    summary = {
        "project": "FlowCast",
        "day": 3,
        "validation_status": (
            "REVIEW_REQUIRED"
            if error_count > 0 or review_count > 0
            else "NO_FLAGGED_ISSUES"
        ),
        "input_file": str(
            INPUT_FILE.relative_to(PROJECT_ROOT)
        ),
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "expected_columns_missing": missing_expected,
        "unexpected_columns": unexpected_columns,
        "required_columns": REQUIRED_COLUMNS,
        "required_columns_missing": missing_required,
        "total_missing_cells": missing_cells,
        "variables_with_missing_values": (
            variables_with_missing_values
        ),
        "total_validation_findings": len(issues_df),
        "error_findings": error_count,
        "review_findings": review_count,
        "info_findings": info_count,
        "rows_with_duplicate_road_time_keys": int(
            duplicate_mask.sum()
        ),
        "missingness_report": "missingness_report.csv",
        "issues_report": "validation_issues.csv",
        "duplicate_report": "duplicate_records.csv",
        "automatic_row_deletion": False,
        "original_data_modified": False,
    }

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

    write_policy()

    # ------------------------------------------------------
    # Final output
    # ------------------------------------------------------

    print("\n[8/8] Validation completed.")
    print("=" * 65)
    print("FLOWCAST DAY 3 SUMMARY")
    print("=" * 65)

    print(f"Rows inspected: {original_row_count:,}")
    print(f"Columns inspected: {len(df.columns)}")
    print(f"Missing cells: {missing_cells:,}")
    print(
        "Variables containing missing values:",
        variables_with_missing_values,
    )
    print(f"ERROR findings: {error_count:,}")
    print(f"REVIEW findings: {review_count:,}")
    print(f"INFO findings: {info_count:,}")
    print(
        "Rows sharing duplicate road-time keys:",
        int(duplicate_mask.sum()),
    )

    print("\nReports saved to:")
    print(REPORT_DIR)

    print("\nPolicy saved to:")
    print(POLICY_FILE)

    print("\nOriginal dataset was not modified.")
    print("Suspicious records were not automatically deleted.")
    print("=" * 65)


if __name__ == "__main__":
    main()
