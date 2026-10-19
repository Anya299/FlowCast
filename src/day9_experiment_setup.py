# FlowCast — Day 9
# Time-Aware Dataset Preparation and ML Experiment Setup

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flowcast_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "day9"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "day9"

TARGET = "traffic_volume"

TRAIN_END = pd.Timestamp("2025-04-06 23:59:59")
VALIDATION_END = pd.Timestamp("2025-05-03 23:59:59")

BASELINE_FEATURES = [
    "traffic_lag_30m",
    "traffic_lag_1h",
    "traffic_lag_90m",
    "traffic_lag_24h",
    "traffic_lag_7d",
    "traffic_rolling_mean_3h",
    "traffic_rolling_mean_6h",
    "traffic_rolling_mean_24h",
    "traffic_rolling_std_24h",
    "hour_of_day",
    "minute_of_hour",
    "day_of_week_number",
    "day_of_month",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "road_id_feature",
    "latitude",
    "longitude",
    "road_capacity",
    "public_holiday",
    "event_flag",
    "roadwork_flag",
]


# ============================================================
# 2. DIRECTORY SETUP
# ============================================================

def create_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD AND CHECK INPUT DATA
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "road_id",
        "timestamp",
        TARGET,
        *BASELINE_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing_columns)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamps = int(
        df["timestamp"].isna().sum()
    )

    if invalid_timestamps:
        raise ValueError(
            f"Found {invalid_timestamps} invalid timestamps. "
            "Resolve timestamp quality before splitting."
        )

    duplicate_count = int(
        df.duplicated(
            subset=["road_id", "timestamp"],
            keep=False,
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} rows belonging to "
            "duplicate road/timestamp keys. Resolve these "
            "before experiment setup."
        )

    df = df.sort_values(
        ["timestamp", "road_id"]
    ).reset_index(drop=True)

    print(f"Input file: {INPUT_FILE}")
    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")
    print(f"Unique roads: {df['road_id'].nunique()}")
    print(
        "Time range:",
        df["timestamp"].min(),
        "to",
        df["timestamp"].max(),
    )

    return df


# ============================================================
# 4. PREPARE THE SUPERVISED-LEARNING TARGET
# ============================================================

def prepare_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    result = df.copy()

    result[TARGET] = pd.to_numeric(
        result[TARGET],
        errors="coerce",
    )

    negative_targets = int(
        (result[TARGET] < 0).fillna(False).sum()
    )

    # Negative traffic volume is physically invalid.
    result.loc[result[TARGET] < 0, TARGET] = float("nan")

    missing_target_count = int(
        result[TARGET].isna().sum()
    )

    summary = {
        "target": TARGET,
        "negative_target_values_invalidated": negative_targets,
        "missing_target_rows": missing_target_count,
        "rows_before_target_filter": len(result),
        "target_policy": (
            "Missing and negative traffic-volume targets are "
            "excluded from supervised-learning splits. "
            "The original Day 8 feature dataset is not modified."
        ),
    }

    return result, summary


# ============================================================
# 5. CREATE CHRONOLOGICAL SPLITS
# ============================================================

def create_splits(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:

    train_mask = df["timestamp"] <= TRAIN_END

    validation_mask = (
        (df["timestamp"] > TRAIN_END)
        & (df["timestamp"] <= VALIDATION_END)
    )

    test_mask = df["timestamp"] > VALIDATION_END

    split_masks = {
        "train": train_mask,
        "validation": validation_mask,
        "test": test_mask,
    }

    result = {}

    for split_name, mask in split_masks.items():
        partition = df.loc[mask].copy()

        # Supervised learning requires a valid target.
        partition = partition.loc[
            partition[TARGET].notna()
        ].copy()

        result[split_name] = partition.reset_index(
            drop=True
        )

    return result


# ============================================================
# 6. FEATURE QUALITY AND LEAKAGE CHECKS
# ============================================================

def audit_features(
    splits: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:

    feature_audit_rows = []
    split_quality = {}

    forbidden_features = {
        TARGET,
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "accident_count",
        "congestion_level",
        "timestamp",
        "date",
        "time",
    }

    leaked_features = sorted(
        set(BASELINE_FEATURES) & forbidden_features
    )

    if leaked_features:
        raise ValueError(
            "Potential target leakage detected in baseline "
            "features: "
            + ", ".join(leaked_features)
        )

    for split_name, partition in splits.items():

        missing_feature_cells = int(
            partition[BASELINE_FEATURES]
            .isna()
            .sum()
            .sum()
        )

        rows_with_missing_features = int(
            partition[BASELINE_FEATURES]
            .isna()
            .any(axis=1)
            .sum()
        )

        split_quality[split_name] = {
            "rows": len(partition),
            "missing_target_rows": int(
                partition[TARGET].isna().sum()
            ),
            "rows_with_any_missing_feature": (
                rows_with_missing_features
            ),
            "missing_feature_cells": missing_feature_cells,
            "first_timestamp": (
                str(partition["timestamp"].min())
                if not partition.empty
                else None
            ),
            "last_timestamp": (
                str(partition["timestamp"].max())
                if not partition.empty
                else None
            ),
        }

        for feature in BASELINE_FEATURES:
            feature_audit_rows.append({
                "split": split_name,
                "feature": feature,
                "rows": len(partition),
                "missing_values": int(
                    partition[feature].isna().sum()
                ),
                "missing_percentage": (
                    round(
                        partition[feature].isna().mean() * 100,
                        4,
                    )
                    if len(partition)
                    else 0.0
                ),
            })

    feature_audit = pd.DataFrame(feature_audit_rows)

    audit_summary = {
        "baseline_feature_count": len(BASELINE_FEATURES),
        "target_in_feature_list": (
            TARGET in BASELINE_FEATURES
        ),
        "forbidden_feature_overlap": leaked_features,
        "splits": split_quality,
        "missing_feature_policy": (
            "Missing lag and rolling features remain missing "
            "at this stage. No forward filling or future-based "
            "imputation is performed."
        ),
    }

    return feature_audit, audit_summary


# ============================================================
# 7. SAVE DATASETS AND REPORTS
# ============================================================

def save_outputs(
    splits: dict[str, pd.DataFrame],
    target_summary: dict,
    audit_summary: dict,
    feature_audit: pd.DataFrame,
    original_rows: int,
) -> None:

    for split_name, partition in splits.items():
        output_path = (
            OUTPUT_DIR
            / f"flowcast_{split_name}.csv"
        )

        partition.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Saved {split_name}: "
            f"{len(partition):,} rows -> {output_path}"
        )

    feature_audit.to_csv(
        REPORT_DIR / "feature_missingness_by_split.csv",
        index=False,
    )

    split_summary = {
        "project": "FlowCast",
        "day": 9,
        "stage": (
            "Chronological supervised-learning "
            "experiment preparation"
        ),
        "input_file": str(
            INPUT_FILE.relative_to(PROJECT_ROOT)
        ),
        "source_rows": original_rows,
        "target": TARGET,
        "baseline_features": BASELINE_FEATURES,
        "split_policy": "Chronological; no random shuffling",
        "boundaries": {
            "training": (
                "2025-01-01 through 2025-04-06"
            ),
            "validation": (
                "2025-04-07 through 2025-05-03"
            ),
            "test": (
                "2025-05-04 through 2025-05-31"
            ),
        },
        "target_quality": target_summary,
        "audit": audit_summary,
        "model_training_performed": False,
    }

    with open(
        REPORT_DIR / "experiment_setup_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_summary,
            file,
            indent=4,
        )

    report_lines = [
        "# FlowCast — Day 9 Experiment Setup Report",
        "",
        "## Objective",
        "",
        "Prepare chronological datasets for supervised "
        "traffic-volume prediction.",
        "",
        "## Prediction target",
        "",
        f"- Target: `{TARGET}`",
        f"- Baseline features: {len(BASELINE_FEATURES)}",
        "- Random shuffling: not used",
        "- Model training: not performed",
        "",
        "## Chronological boundaries",
        "",
        "| Split | Period | Rows |",
        "|---|---|---:|",
    ]

    display_periods = {
        "train": "Jan 1 – Apr 6, 2025",
        "validation": "Apr 7 – May 3, 2025",
        "test": "May 4 – May 31, 2025",
    }

    for split_name, partition in splits.items():
        report_lines.append(
            f"| {split_name.title()} | "
            f"{display_periods[split_name]} | "
            f"{len(partition):,} |"
        )

    report_lines.extend([
        "",
        "## Target quality",
        "",
        f"- Negative targets invalidated: "
        f"{target_summary['negative_target_values_invalidated']}",
        f"- Missing target rows in source: "
        f"{target_summary['missing_target_rows']:,}",
        "",
        "## Missing feature policy",
        "",
        "Missing lag and rolling features are preserved. "
        "No forward filling or future-based imputation is used.",
        "",
        "Rows with missing target values are excluded from "
        "supervised-learning splits.",
        "",
        "## Interpretation",
        "",
        "This stage creates experiment-ready partitions but "
        "does not train or evaluate a model. Model selection "
        "must use the training and validation periods. The test "
        "period should remain untouched until final evaluation.",
        "",
        "Feature availability at the real prediction cutoff "
        "must be verified before production use.",
        "",
    ])

    with open(
        REPORT_DIR / "experiment_setup_report.md",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(report_lines))

    print("\nReports saved successfully.")


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def main() -> None:

    print("=" * 65)
    print("FLOWCAST DAY 9 — EXPERIMENT SETUP")
    print("=" * 65)

    create_directories()

    df = load_dataset()
    original_rows = len(df)

    prepared_df, target_summary = prepare_target(df)
    splits = create_splits(prepared_df)

    # Verify the chronological boundaries.
    if not (
        splits["train"]["timestamp"].empty
        or splits["train"]["timestamp"].max() <= TRAIN_END
    ):
        raise AssertionError(
            "Training split extends beyond its boundary."
        )

    if not splits["validation"]["timestamp"].empty:
        if not (
            (splits["validation"]["timestamp"] > TRAIN_END).all()
            and (
                splits["validation"]["timestamp"]
                <= VALIDATION_END
            ).all()
        ):
            raise AssertionError(
                "Validation split violates its time boundaries."
            )

    if not splits["test"]["timestamp"].empty:
        if not (
            (splits["test"]["timestamp"] > VALIDATION_END).all()
        ):
            raise AssertionError(
                "Test split starts before its assigned period."
            )

    feature_audit, audit_summary = audit_features(splits)

    # The prediction target must not be a model feature.
    assert TARGET not in BASELINE_FEATURES

    # Verify that the splits account for all valid-target rows.
    expected_rows = int(
        prepared_df[TARGET].notna().sum()
    )

    actual_rows = sum(
        len(partition)
        for partition in splits.values()
    )

    if expected_rows != actual_rows:
        raise AssertionError(
            f"Split row accounting mismatch: expected "
            f"{expected_rows}, got {actual_rows}."
        )

    save_outputs(
        splits=splits,
        target_summary=target_summary,
        audit_summary=audit_summary,
        feature_audit=feature_audit,
        original_rows=original_rows,
    )

    print("\n" + "=" * 65)
    print("DAY 9 COMPLETED SUCCESSFULLY")
    print("=" * 65)

    for split_name, partition in splits.items():
        print(
            f"{split_name.title():12} rows: "
            f"{len(partition):,}"
        )

    print(
        f"\nBaseline features: {len(BASELINE_FEATURES)}"
    )
    print(f"Target: {TARGET}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Report directory: {REPORT_DIR}")
    print("\nNo model has been trained in this stage.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nDAY 9 FAILED: {exc}",
            file=sys.stderr,
        )
        raise

