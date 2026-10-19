"""
FLOWCAST — DAY 5
Continuous Timeline Reconstruction and Missing Interval Audit

Purpose
-------
1. Load the integrated traffic dataset.
2. Validate date, time, and road identifiers.
3. Reconstruct the expected 30-minute timeline.
4. Identify missing road-time observations.
5. Measure coverage by road and date.
6. Group consecutive missing intervals into gap events.
7. Detect timestamps outside the expected 30-minute grid.
8. Export machine-readable and human-readable reports.

Important
---------
This script does not modify the source dataset.
It does not invent missing traffic observations.
It does not automatically remove duplicates.
It treats a missing sensor interval as an observation gap,
not as proof that a sensor malfunctioned.
"""

from __future__ import annotations

import json
import logging
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "day5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_INTERVAL_MINUTES = 30

EXPECTED_ROAD_COUNT = 25

REQUIRED_COLUMNS = [
    "road_id",
    "date",
    "time",
]

# If the source contains a complete 30-minute time series,
# the expected number of intervals per full day is 48.
INTERVALS_PER_DAY = (
    24 * 60 // EXPECTED_INTERVAL_MINUTES
)


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger("flowcast.day5")


# ============================================================
# 3. LOAD INPUT DATA
# ============================================================

def load_data(path: Path) -> pd.DataFrame:
    """
    Load the integrated dataset without modifying the source file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Input dataset was not found: {path}"
        )

    logger.info("Loading integrated dataset...")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "The input dataset contains no rows."
        )

    # Remove accidental whitespace from column names.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            f"{missing_columns}"
        )

    logger.info(
        "Loaded %s rows and %s columns.",
        f"{len(df):,}",
        len(df.columns),
    )

    return df


# ============================================================
# 4. NORMALIZE IDENTIFIERS AND PARSE TIMESTAMPS
# ============================================================

def prepare_timestamps(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare road identifiers and construct timestamps.

    This creates an audit copy. The source CSV is not changed.
    Invalid timestamps remain identifiable in the audit.
    """

    audit = df.copy()

    # Stable string identifiers prevent mismatches caused by
    # road IDs being interpreted as different data types.
    audit["road_id"] = (
        audit["road_id"]
        .astype("string")
        .str.strip()
    )

    # Treat empty identifiers as missing.
    audit["road_id"] = audit["road_id"].replace(
        "",
        pd.NA,
    )

    date_text = (
        audit["date"]
        .astype("string")
        .str.strip()
    )

    time_text = (
        audit["time"]
        .astype("string")
        .str.strip()
    )

    # Combine date and time into a single timestamp.
    # Invalid combinations become NaT and are reported later.
    audit["timestamp"] = pd.to_datetime(
        date_text + " " + time_text,
        errors="coerce",
    )

    audit["timestamp_valid"] = (
        audit["timestamp"].notna()
    )

    audit["road_id_valid"] = (
        audit["road_id"].notna()
    )

    logger.info(
        "Valid timestamps: %s",
        f"{audit['timestamp_valid'].sum():,}",
    )

    logger.info(
        "Invalid timestamps: %s",
        f"{(~audit['timestamp_valid']).sum():,}",
    )

    logger.info(
        "Missing road identifiers: %s",
        f"{(~audit['road_id_valid']).sum():,}",
    )

    return audit


# ============================================================
# 5. INVESTIGATE TIMESTAMP ALIGNMENT
# ============================================================

def identify_off_grid_timestamps(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find observations that do not align with a 30-minute grid.

    Examples of expected grid-aligned times:
        09:00
        09:30
        10:00
        10:30

    A timestamp such as 09:17 is off-grid.

    Off-grid timestamps are flagged for investigation.
    They are not automatically rounded or deleted.
    """

    valid = audit.loc[
        audit["timestamp_valid"]
        & audit["road_id_valid"]
    ].copy()

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "source_row_number",
                "road_id",
                "timestamp",
                "nearest_floor_interval",
                "minutes_from_floor",
            ]
        )

    floor_timestamp = valid["timestamp"].dt.floor(
        f"{EXPECTED_INTERVAL_MINUTES}min"
    )

    off_grid_mask = (
        valid["timestamp"] != floor_timestamp
    )

    off_grid = valid.loc[
        off_grid_mask,
        ["road_id", "timestamp"],
    ].copy()

    off_grid["nearest_floor_interval"] = (
        off_grid["timestamp"].dt.floor(
            f"{EXPECTED_INTERVAL_MINUTES}min"
        )
    )

    off_grid["minutes_from_floor"] = (
        (
            off_grid["timestamp"]
            - off_grid["nearest_floor_interval"]
        ).dt.total_seconds()
        / 60
    )

    # CSV row number: header is line 1, first data row is line 2.
    off_grid["source_row_number"] = (
        off_grid.index + 2
    )

    return off_grid[
        [
            "source_row_number",
            "road_id",
            "timestamp",
            "nearest_floor_interval",
            "minutes_from_floor",
        ]
    ].reset_index(drop=True)


# ============================================================
# 6. BUILD THE EXPECTED TIMELINE
# ============================================================

def build_expected_timeline(
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Build the expected 30-minute grid for all observed roads.

    The grid spans the earliest and latest valid timestamps
    found in the dataset, rounded outward to 30-minute
    boundaries.

    All identified roads receive the same expected time grid.

    This is an audit assumption, not proof that every road
    sensor was operational throughout the entire period.
    """

    valid = audit.loc[
        audit["timestamp_valid"]
        & audit["road_id_valid"]
    ].copy()

    if valid.empty:
        raise ValueError(
            "No valid road identifiers and timestamps are "
            "available to reconstruct the timeline."
        )

    roads = sorted(
        valid["road_id"].dropna().unique().tolist()
    )

    earliest_timestamp = (
        valid["timestamp"].min().floor(
            f"{EXPECTED_INTERVAL_MINUTES}min"
        )
    )

    latest_timestamp = (
        valid["timestamp"].max().ceil(
            f"{EXPECTED_INTERVAL_MINUTES}min"
        )
    )

    expected_timestamps = pd.date_range(
        start=earliest_timestamp,
        end=latest_timestamp,
        freq=f"{EXPECTED_INTERVAL_MINUTES}min",
    )

    # Construct every expected road + timestamp combination.
    expected = pd.MultiIndex.from_product(
        [
            roads,
            expected_timestamps,
        ],
        names=[
            "road_id",
            "timestamp",
        ],
    ).to_frame(index=False)

    expected["expected_interval"] = True

    logger.info(
        "Roads found: %s",
        len(roads),
    )

    logger.info(
        "Expected timestamps per road: %s",
        f"{len(expected_timestamps):,}",
    )

    logger.info(
        "Expected road-time combinations: %s",
        f"{len(expected):,}",
    )

    logger.info(
        "Expected road count in project specification: %s",
        EXPECTED_ROAD_COUNT,
    )

    if len(roads) != EXPECTED_ROAD_COUNT:
        logger.warning(
            "The dataset contains %s unique roads, "
            "while the project specification expects %s. "
            "Investigate before interpreting coverage.",
            len(roads),
            EXPECTED_ROAD_COUNT,
        )

    return expected, expected_timestamps


# ============================================================
# 7. IDENTIFY MISSING ROAD-TIME INTERVALS
# ============================================================

def find_missing_intervals(
    audit: pd.DataFrame,
    expected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare expected road-time combinations with observed data.

    Returns:
        missing_intervals:
            Expected road-time combinations not observed.

        duplicate_keys:
            Road-time combinations appearing more than once.
    """

    valid = audit.loc[
        audit["timestamp_valid"]
        & audit["road_id_valid"]
    ].copy()

    # Count observations for every exact road-time key.
    observation_counts = (
        valid.groupby(
            ["road_id", "timestamp"],
            dropna=False,
        )
        .size()
        .reset_index(name="observed_row_count")
    )

    # Duplicate keys can represent repeated records.
    duplicate_keys = observation_counts.loc[
        observation_counts["observed_row_count"] > 1
    ].copy()

    # Count at least one valid record as an observed interval.
    observed_keys = observation_counts[
        ["road_id", "timestamp"]
    ].drop_duplicates()

    comparison = expected.merge(
        observed_keys.assign(observed=True),
        on=["road_id", "timestamp"],
        how="left",
    )

    missing_intervals = comparison.loc[
        comparison["observed"].isna()
    ].copy()

    missing_intervals = missing_intervals[
        [
            "road_id",
            "timestamp",
            "expected_interval",
        ]
    ]

    missing_intervals = missing_intervals.sort_values(
        ["road_id", "timestamp"]
    ).reset_index(drop=True)

    duplicate_keys = duplicate_keys.sort_values(
        ["road_id", "timestamp"]
    ).reset_index(drop=True)

    logger.info(
        "Missing road-time intervals: %s",
        f"{len(missing_intervals):,}",
    )

    logger.info(
        "Road-time keys with duplicate observations: %s",
        f"{len(duplicate_keys):,}",
    )

    return missing_intervals, duplicate_keys


# ============================================================
# 8. CALCULATE COVERAGE BY ROAD
# ============================================================

def calculate_road_coverage(
    expected: pd.DataFrame,
    missing_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate expected, observed, and missing intervals by road.
    """

    expected_by_road = (
        expected.groupby("road_id")
        .size()
        .rename("expected_intervals")
    )

    missing_by_road = (
        missing_intervals.groupby("road_id")
        .size()
        .rename("missing_intervals")
    )

    coverage = pd.concat(
        [
            expected_by_road,
            missing_by_road,
        ],
        axis=1,
    ).fillna(0)

    coverage["missing_intervals"] = (
        coverage["missing_intervals"].astype(int)
    )

    coverage["observed_intervals"] = (
        coverage["expected_intervals"]
        - coverage["missing_intervals"]
    )

    coverage["coverage_percent"] = (
        coverage["observed_intervals"]
        / coverage["expected_intervals"]
        * 100
    )

    coverage["missing_percent"] = (
        coverage["missing_intervals"]
        / coverage["expected_intervals"]
        * 100
    )

    coverage = coverage.reset_index()

    return coverage.sort_values(
        "coverage_percent",
        ascending=True,
    ).reset_index(drop=True)


# ============================================================
# 9. CALCULATE DAILY COVERAGE
# ============================================================

def calculate_daily_coverage(
    expected: pd.DataFrame,
    missing_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily coverage across all identified roads.

    A date with 25 roads and 48 intervals per road has
    1,200 expected road-time combinations.
    """

    expected_daily = expected.copy()

    expected_daily["date"] = (
        expected_daily["timestamp"].dt.date
    )

    expected_counts = (
        expected_daily.groupby("date")
        .size()
        .rename("expected_intervals")
    )

    missing_daily = missing_intervals.copy()

    missing_daily["date"] = (
        missing_daily["timestamp"].dt.date
    )

    missing_counts = (
        missing_daily.groupby("date")
        .size()
        .rename("missing_intervals")
    )

    daily = pd.concat(
        [
            expected_counts,
            missing_counts,
        ],
        axis=1,
    ).fillna(0)

    daily["missing_intervals"] = (
        daily["missing_intervals"].astype(int)
    )

    daily["observed_intervals"] = (
        daily["expected_intervals"]
        - daily["missing_intervals"]
    )

    daily["coverage_percent"] = (
        daily["observed_intervals"]
        / daily["expected_intervals"]
        * 100
    )

    daily["missing_percent"] = (
        daily["missing_intervals"]
        / daily["expected_intervals"]
        * 100
    )

    return daily.reset_index().sort_values(
        "date"
    ).reset_index(drop=True)


# ============================================================
# 10. GROUP CONSECUTIVE MISSING INTERVALS
# ============================================================

def group_missing_gaps(
    missing_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Group consecutive missing intervals into gap events.

    Example:
        09:00 missing
        09:30 missing
        10:00 missing

    These become one three-interval gap event.

    Gap duration here means the number of missing intervals
    multiplied by 30 minutes. It is not proof of sensor failure.
    """

    columns = [
        "road_id",
        "gap_start",
        "gap_end",
        "missing_interval_count",
        "gap_duration_minutes",
    ]

    if missing_intervals.empty:
        return pd.DataFrame(columns=columns)

    gaps = missing_intervals.sort_values(
        ["road_id", "timestamp"]
    ).copy()

    gaps["previous_timestamp"] = (
        gaps.groupby("road_id")["timestamp"].shift(1)
    )

    gaps["time_difference_minutes"] = (
        (
            gaps["timestamp"]
            - gaps["previous_timestamp"]
        ).dt.total_seconds()
        / 60
    )

    # Start a new group if:
    # 1. This is the first missing interval for a road, or
    # 2. The gap is not exactly 30 minutes after the previous one.
    gaps["new_gap"] = (
        gaps["previous_timestamp"].isna()
        | (
            gaps["time_difference_minutes"]
            != EXPECTED_INTERVAL_MINUTES
        )
    )

    gaps["gap_group"] = (
        gaps.groupby("road_id")["new_gap"]
        .cumsum()
    )

    grouped = (
        gaps.groupby(
            ["road_id", "gap_group"],
            as_index=False,
        )
        .agg(
            gap_start=("timestamp", "min"),
            gap_end=("timestamp", "max"),
            missing_interval_count=("timestamp", "size"),
        )
    )

    grouped["gap_duration_minutes"] = (
        grouped["missing_interval_count"]
        * EXPECTED_INTERVAL_MINUTES
    )

    return grouped[columns].sort_values(
        ["road_id", "gap_start"]
    ).reset_index(drop=True)


# ============================================================
# 11. BUILD SOURCE-ROW QUALITY REPORT
# ============================================================

def build_timestamp_quality_report(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve source-row traceability for timestamp problems.
    """

    report = audit.loc[
        (~audit["timestamp_valid"])
        | (~audit["road_id_valid"])
    ].copy()

    if report.empty:
        return pd.DataFrame(
            columns=[
                "source_row_number",
                "road_id",
                "date",
                "time",
                "timestamp",
                "timestamp_valid",
                "road_id_valid",
                "issue",
            ]
        )

    report["source_row_number"] = (
        report.index + 2
    )

    def describe_issue(row: pd.Series) -> str:
        problems = []

        if not row["road_id_valid"]:
            problems.append("MISSING_ROAD_ID")

        if not row["timestamp_valid"]:
            problems.append("INVALID_TIMESTAMP")

        return ";".join(problems)

    report["issue"] = report.apply(
        describe_issue,
        axis=1,
    )

    columns = [
        "source_row_number",
        "road_id",
        "date",
        "time",
        "timestamp",
        "timestamp_valid",
        "road_id_valid",
        "issue",
    ]

    return report[columns].reset_index(drop=True)


# ============================================================
# 12. BUILD MACHINE-READABLE SUMMARY
# ============================================================

def build_summary(
    original: pd.DataFrame,
    audit: pd.DataFrame,
    expected: pd.DataFrame,
    expected_timestamps: pd.DatetimeIndex,
    missing_intervals: pd.DataFrame,
    duplicate_keys: pd.DataFrame,
    off_grid: pd.DataFrame,
    timestamp_quality: pd.DataFrame,
    road_coverage: pd.DataFrame,
    daily_coverage: pd.DataFrame,
    gap_events: pd.DataFrame,
) -> dict[str, Any]:
    """
    Create a JSON-compatible summary of the audit.
    """

    expected_count = len(expected)
    missing_count = len(missing_intervals)

    observed_count = expected_count - missing_count

    overall_coverage = (
        observed_count / expected_count * 100
        if expected_count
        else 0.0
    )

    valid_timestamps = audit.loc[
        audit["timestamp_valid"],
        "timestamp",
    ]

    if not valid_timestamps.empty:
        earliest = str(valid_timestamps.min())
        latest = str(valid_timestamps.max())
    else:
        earliest = None
        latest = None

    worst_road = None

    if not road_coverage.empty:
        worst = road_coverage.iloc[0]

        worst_road = {
            "road_id": str(worst["road_id"]),
            "coverage_percent": round(
                float(worst["coverage_percent"]),
                3,
            ),
            "missing_intervals": int(
                worst["missing_intervals"]
            ),
        }

    lowest_coverage_day = None

    if not daily_coverage.empty:
        worst_day = daily_coverage.loc[
            daily_coverage["coverage_percent"].idxmin()
        ]

        lowest_coverage_day = {
            "date": str(worst_day["date"]),
            "coverage_percent": round(
                float(worst_day["coverage_percent"]),
                3,
            ),
            "missing_intervals": int(
                worst_day["missing_intervals"]
            ),
        }

    summary = {
        "project": "FlowCast",
        "audit_day": 5,
        "audit_name": (
            "Continuous Timeline Reconstruction "
            "and Missing Interval Audit"
        ),
        "input_file": str(
            INPUT_FILE.relative_to(PROJECT_ROOT)
        ),
        "source_dataset_modified": False,
        "missing_readings_imputed": False,
        "duplicate_records_deleted": False,
        "interval_minutes": EXPECTED_INTERVAL_MINUTES,
        "expected_intervals_per_full_day_per_road": (
            INTERVALS_PER_DAY
        ),
        "expected_road_count_from_specification": (
            EXPECTED_ROAD_COUNT
        ),
        "source_rows": int(len(original)),
        "source_columns": int(len(original.columns)),
        "valid_timestamps": int(
            audit["timestamp_valid"].sum()
        ),
        "invalid_timestamps": int(
            (~audit["timestamp_valid"]).sum()
        ),
        "missing_road_identifiers": int(
            (~audit["road_id_valid"]).sum()
        ),
        "unique_roads_found": int(
            audit.loc[
                audit["road_id_valid"],
                "road_id",
            ].nunique()
        ),
        "first_valid_timestamp": earliest,
        "last_valid_timestamp": latest,
        "expected_timestamp_count": int(
            len(expected_timestamps)
        ),
        "expected_road_time_combinations": int(
            expected_count
        ),
        "observed_expected_intervals": int(
            observed_count
        ),
        "missing_intervals": int(
            missing_count
        ),
        "overall_timeline_coverage_percent": round(
            overall_coverage,
            3,
        ),
        "road_time_keys_with_duplicates": int(
            len(duplicate_keys)
        ),
        "off_grid_timestamp_rows": int(
            len(off_grid)
        ),
        "rows_with_timestamp_or_road_id_issues": int(
            len(timestamp_quality)
        ),
        "consecutive_gap_events": int(
            len(gap_events)
        ),
        "worst_coverage_road": worst_road,
        "lowest_coverage_day": lowest_coverage_day,
        "interpretation_warning": (
            "A missing road-time interval indicates that no "
            "valid timestamped row was observed for that road "
            "and interval. It does not independently prove "
            "sensor failure. The common timeline assumes all "
            "identified roads were expected to report across "
            "the full observed time range."
        ),
    }

    return summary


# ============================================================
# 13. EXPORT CSV AND JSON REPORTS
# ============================================================

def save_reports(
    original: pd.DataFrame,
    audit: pd.DataFrame,
    expected: pd.DataFrame,
    expected_timestamps: pd.DatetimeIndex,
    missing_intervals: pd.DataFrame,
    duplicate_keys: pd.DataFrame,
    off_grid: pd.DataFrame,
    timestamp_quality: pd.DataFrame,
    road_coverage: pd.DataFrame,
    daily_coverage: pd.DataFrame,
    gap_events: pd.DataFrame,
) -> dict[str, Any]:
    """
    Save all Day 5 deliverables.
    """

    summary = build_summary(
        original=original,
        audit=audit,
        expected=expected,
        expected_timestamps=expected_timestamps,
        missing_intervals=missing_intervals,
        duplicate_keys=duplicate_keys,
        off_grid=off_grid,
        timestamp_quality=timestamp_quality,
        road_coverage=road_coverage,
        daily_coverage=daily_coverage,
        gap_events=gap_events,
    )

    missing_intervals.to_csv(
        OUTPUT_DIR / "missing_timeline_slots.csv",
        index=False,
    )

    duplicate_keys.to_csv(
        OUTPUT_DIR / "duplicate_road_time_keys.csv",
        index=False,
    )

    off_grid.to_csv(
        OUTPUT_DIR / "off_grid_timestamps.csv",
        index=False,
    )

    timestamp_quality.to_csv(
        OUTPUT_DIR / "timestamp_quality_issues.csv",
        index=False,
    )

    road_coverage.to_csv(
        OUTPUT_DIR / "road_coverage.csv",
        index=False,
    )

    daily_coverage.to_csv(
        OUTPUT_DIR / "daily_coverage.csv",
        index=False,
    )

    gap_events.to_csv(
        OUTPUT_DIR / "gap_events.csv",
        index=False,
    )

    summary_path = (
        OUTPUT_DIR / "timeline_audit_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "Saved reports to %s",
        OUTPUT_DIR,
    )

    return summary


# ============================================================
# 14. GENERATE ENGINEERING REPORT
# ============================================================

def write_markdown_report(
    summary: dict[str, Any],
    road_coverage: pd.DataFrame,
    daily_coverage: pd.DataFrame,
    gap_events: pd.DataFrame,
) -> None:
    """
    Generate a report suitable for project documentation.
    """

    worst_roads = road_coverage.head(5)

    road_lines = [
        "| Road ID | Expected | Observed | Missing | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in worst_roads.iterrows():
        road_lines.append(
            f"| {row['road_id']} "
            f"| {int(row['expected_intervals'])} "
            f"| {int(row['observed_intervals'])} "
            f"| {int(row['missing_intervals'])} "
            f"| {row['coverage_percent']:.2f}% |"
        )

    worst_days = daily_coverage.sort_values(
        "coverage_percent"
    ).head(5)

    day_lines = [
        "| Date | Expected | Observed | Missing | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in worst_days.iterrows():
        day_lines.append(
            f"| {row['date']} "
            f"| {int(row['expected_intervals'])} "
            f"| {int(row['observed_intervals'])} "
            f"| {int(row['missing_intervals'])} "
            f"| {row['coverage_percent']:.2f}% |"
        )

    lines = [
        "# FlowCast Day 5 — Timeline Audit Report",
        "",
        "## 1. Objective",
        "",
        "Reconstruct the expected 30-minute road-sensor "
        "timeline and identify missing observations, duplicate "
        "road-time keys, timestamp alignment problems, and "
        "coverage gaps.",
        "",
        "## 2. Dataset overview",
        "",
        f"- Source rows: {summary['source_rows']:,}",
        f"- Source columns: {summary['source_columns']:,}",
        f"- Unique roads found: {summary['unique_roads_found']}",
        f"- First valid timestamp: "
        f"{summary['first_valid_timestamp']}",
        f"- Last valid timestamp: "
        f"{summary['last_valid_timestamp']}",
        f"- Expected interval: "
        f"{summary['interval_minutes']} minutes",
        f"- Expected road-time combinations: "
        f"{summary['expected_road_time_combinations']:,}",
        "",
        "## 3. Timeline completeness",
        "",
        f"- Observed expected intervals: "
        f"{summary['observed_expected_intervals']:,}",
        f"- Missing intervals: "
        f"{summary['missing_intervals']:,}",
        f"- Overall timeline coverage: "
        f"{summary['overall_timeline_coverage_percent']:.3f}%",
        "",
        "Coverage is calculated from the expected road-time "
        "grid. It is not the same as the percentage of "
        "non-missing traffic-volume values.",
        "",
        "## 4. Data quality findings",
        "",
        f"- Invalid timestamps: "
        f"{summary['invalid_timestamps']:,}",
        f"- Missing road identifiers: "
        f"{summary['missing_road_identifiers']:,}",
        f"- Off-grid timestamp rows: "
        f"{summary['off_grid_timestamp_rows']:,}",
        f"- Road-time keys with duplicates: "
        f"{summary['road_time_keys_with_duplicates']:,}",
        f"- Consecutive missing-gap events: "
        f"{summary['consecutive_gap_events']:,}",
        "",
        "## 5. Five roads with the lowest coverage",
        "",
        *road_lines,
        "",
        "## 6. Five dates with the lowest coverage",
        "",
        *day_lines,
        "",
        "## 7. Engineering decisions",
        "",
        "1. The integrated source dataset is not modified.",
        "2. Missing intervals are reported, not imputed.",
        "3. Duplicate observations are identified, not deleted.",
        "4. Off-grid timestamps are flagged, not automatically "
        "rounded.",
        "5. Missing intervals are not automatically interpreted "
        "as sensor failures.",
        "6. All roads are compared against a common time grid. "
        "This assumption must be checked against sensor "
        "deployment and operational records.",
        "",
        "## 8. Output artifacts",
        "",
        "- `missing_timeline_slots.csv`",
        "- `duplicate_road_time_keys.csv`",
        "- `off_grid_timestamps.csv`",
        "- `timestamp_quality_issues.csv`",
        "- `road_coverage.csv`",
        "- `daily_coverage.csv`",
        "- `gap_events.csv`",
        "- `timeline_audit_summary.json`",
        "- `day5_timeline_audit_report.md`",
        "",
        "## 9. Interpretation and limitations",
        "",
        summary["interpretation_warning"],
        "",
        "A row absent from the timeline can indicate missing "
        "data, a sensor reporting problem, a deployment "
        "boundary, or an incomplete extraction. These "
        "possibilities require further investigation.",
        "",
        "Timeline coverage measures whether a road-time key "
        "exists. It does not establish that the associated "
        "traffic measurements are valid or complete.",
        "",
        "## 10. Reproducibility",
        "",
        "Run from the FlowCast project root:",
        "",
        "```powershell",
        "python src/day5_timeline_audit.py",
        "```",
        "",
    ]

    report_path = (
        OUTPUT_DIR / "day5_timeline_audit_report.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logger.info(
        "Saved engineering report: %s",
        report_path,
    )


# ============================================================
# 15. MAIN PIPELINE
# ============================================================

def main() -> None:
    """
    Execute the complete Day 5 audit.
    """

    logger.info("=" * 60)
    logger.info("FLOWCAST — DAY 5 TIMELINE AUDIT")
    logger.info("=" * 60)

    # Load and prepare the data.
    original = load_data(INPUT_FILE)

    audit = prepare_timestamps(original)

    # Investigate invalid and off-grid timestamps.
    off_grid = identify_off_grid_timestamps(audit)

    timestamp_quality = (
        build_timestamp_quality_report(audit)
    )

    # Reconstruct expected road-time combinations.
    expected, expected_timestamps = (
        build_expected_timeline(audit)
    )

    # Compare expected intervals with actual observations.
    missing_intervals, duplicate_keys = (
        find_missing_intervals(
            audit=audit,
            expected=expected,
        )
    )

    # Measure coverage.
    road_coverage = calculate_road_coverage(
        expected=expected,
        missing_intervals=missing_intervals,
    )

    daily_coverage = calculate_daily_coverage(
        expected=expected,
        missing_intervals=missing_intervals,
    )

    # Group consecutive missing intervals.
    gap_events = group_missing_gaps(
        missing_intervals
    )

    # Save machine-readable reports.
    summary = save_reports(
        original=original,
        audit=audit,
        expected=expected,
        expected_timestamps=expected_timestamps,
        missing_intervals=missing_intervals,
        duplicate_keys=duplicate_keys,
        off_grid=off_grid,
        timestamp_quality=timestamp_quality,
        road_coverage=road_coverage,
        daily_coverage=daily_coverage,
        gap_events=gap_events,
    )

    # Save the human-readable engineering report.
    write_markdown_report(
        summary=summary,
        road_coverage=road_coverage,
        daily_coverage=daily_coverage,
        gap_events=gap_events,
    )

    # Print a concise summary.
    print()
    print("=" * 60)
    print("FLOWCAST DAY 5 — AUDIT COMPLETE")
    print("=" * 60)

    print(
        f"Source rows: "
        f"{summary['source_rows']:,}"
    )

    print(
        f"Unique roads: "
        f"{summary['unique_roads_found']}"
    )

    print(
        f"Expected road-time combinations: "
        f"{summary['expected_road_time_combinations']:,}"
    )

    print(
        f"Missing intervals: "
        f"{summary['missing_intervals']:,}"
    )

    print(
        f"Timeline coverage: "
        f"{summary['overall_timeline_coverage_percent']:.3f}%"
    )

    print(
        f"Duplicate road-time keys: "
        f"{summary['road_time_keys_with_duplicates']:,}"
    )

    print(
        f"Off-grid timestamp rows: "
        f"{summary['off_grid_timestamp_rows']:,}"
    )

    print(
        f"Invalid timestamps: "
        f"{summary['invalid_timestamps']:,}"
    )

    print()
    print(f"Reports saved in: {OUTPUT_DIR}")
    print("=" * 60)


# ============================================================
# 16. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()