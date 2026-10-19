"""
FLOWCAST — DAY 8
Leakage-Aware Traffic Feature Engineering

Input:
    data/processed/flowcast_cleaned.csv

Outputs:
    data/processed/flowcast_features.csv
    data/processed/flowcast_model_features.csv
    reports/day8/feature_catalog.csv
    reports/day8/feature_engineering_summary.json
    reports/day8/feature_engineering_report.md
    reports/day8/leakage_control_report.md

Prediction target:
    traffic_volume

Time interval:
    30 minutes

Important:
    Historical lag features are matched using exact timestamps.
    Rolling statistics use only earlier observations.
    Current traffic_volume is never included as a model input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "flowcast_cleaned.csv"

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "day8"

ALL_FEATURES_FILE = PROCESSED_DIR / "flowcast_features.csv"
MODEL_FEATURES_FILE = (
    PROCESSED_DIR / "flowcast_model_features.csv"
)

CATALOG_FILE = REPORT_DIR / "feature_catalog.csv"
SUMMARY_FILE = REPORT_DIR / "feature_engineering_summary.json"
REPORT_FILE = REPORT_DIR / "feature_engineering_report.md"
LEAKAGE_FILE = REPORT_DIR / "leakage_control_report.md"


# ============================================================
# 2. CONFIGURATION
# ============================================================

TARGET = "traffic_volume"

# Each road's historical observations are matched by exact time.
LAG_INTERVALS = {
    "traffic_lag_30m": pd.Timedelta(minutes=30),
    "traffic_lag_1h": pd.Timedelta(hours=1),
    "traffic_lag_90m": pd.Timedelta(minutes=90),
    "traffic_lag_24h": pd.Timedelta(hours=24),
    "traffic_lag_7d": pd.Timedelta(days=7),
}

# Rolling windows exclude the current timestamp.
ROLLING_WINDOWS = {
    "traffic_rolling_mean_3h": "3h",
    "traffic_rolling_mean_6h": "6h",
    "traffic_rolling_mean_24h": "24h",
    "traffic_rolling_std_24h": "24h",
}


# ============================================================
# 3. LOAD AND CHECK THE CLEANED DATASET
# ============================================================

def load_data() -> pd.DataFrame:
    """Load the cleaned dataset and validate required columns."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned input not found: {INPUT_FILE}\n"
            "Run the Day 7 cleaning pipeline first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "road_id",
        "timestamp",
        "traffic_volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["traffic_volume"] = pd.to_numeric(
        df["traffic_volume"],
        errors="coerce",
    )

    # Invalid keys cannot be used to construct temporal features.
    invalid_key = (
        df["road_id"].isna()
        | df["timestamp"].isna()
    )

    invalid_key_count = int(invalid_key.sum())

    if invalid_key_count:
        raise ValueError(
            f"{invalid_key_count} rows have missing road IDs "
            "or invalid timestamps. Resolve them before "
            "feature engineering."
        )

    # A unique road-time key is essential for reliable joins.
    duplicate_keys = df.duplicated(
        subset=["road_id", "timestamp"],
        keep=False,
    )

    if duplicate_keys.any():
        raise ValueError(
            f"{int(duplicate_keys.sum())} rows have duplicate "
            "road_id + timestamp keys. Resolve duplicates first."
        )

    df = df.sort_values(
        ["road_id", "timestamp"]
    ).reset_index(drop=True)

    print("=" * 65)
    print("FLOWCAST DAY 8 — FEATURE ENGINEERING")
    print("=" * 65)
    print(f"Input rows: {len(df):,}")
    print(f"Input columns: {len(df.columns):,}")
    print(f"Unique roads: {df['road_id'].nunique():,}")
    print(
        "Time range: "
        f"{df['timestamp'].min()} to "
        f"{df['timestamp'].max()}"
    )

    return df


# ============================================================
# 4. CREATE TIME FEATURES
# ============================================================

def create_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create calendar and cyclical time features."""

    df = df.copy()

    timestamp = df["timestamp"]

    # Calendar features.
    df["hour_of_day"] = timestamp.dt.hour

    df["minute_of_hour"] = timestamp.dt.minute

    df["day_of_week"] = timestamp.dt.day_name()

    df["day_of_week_number"] = (
        timestamp.dt.dayofweek
    )

    df["day_of_month"] = timestamp.dt.day

    df["month"] = timestamp.dt.month

    df["is_weekend"] = (
        timestamp.dt.dayofweek >= 5
    ).astype("int8")

    # Cyclical encodings preserve periodicity.
    decimal_hour = (
        timestamp.dt.hour
        + timestamp.dt.minute / 60.0
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * decimal_hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * decimal_hour / 24
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * timestamp.dt.dayofweek / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * timestamp.dt.dayofweek / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * (timestamp.dt.month - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (timestamp.dt.month - 1) / 12
    )

    print("Created calendar and cyclical time features.")

    return df


# ============================================================
# 5. CREATE EXACT-TIMESTAMP HISTORICAL LAGS
# ============================================================

def create_lag_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Match historical traffic measurements for the same road.

    Exact timestamp matching prevents a missing half-hourly
    record from silently becoming the previous observation.
    """

    df = df.copy()

    lag_features = []

    # The lookup table contains only the historical target
    # and the keys needed for the join.
    history = df[
        [
            "road_id",
            "timestamp",
            TARGET,
        ]
    ].copy()

    for feature_name, interval in LAG_INTERVALS.items():

        lookup = history.copy()

        # Shift the historical timestamp forward so that
        # each historical value joins to its future timestamp.
        lookup["timestamp"] = (
            lookup["timestamp"] + interval
        )

        lookup = lookup.rename(
            columns={
                TARGET: feature_name,
            }
        )

        df = df.merge(
            lookup,
            on=["road_id", "timestamp"],
            how="left",
            validate="one_to_one",
        )

        lag_features.append(feature_name)

        print(f"Created {feature_name}")

    return df, lag_features


# ============================================================
# 6. CREATE PAST-ONLY ROLLING STATISTICS
# ============================================================

def rolling_statistic_for_road(
    road_df: pd.DataFrame,
    window: str,
    statistic: str,
) -> pd.Series:
    """
    Calculate a time-based rolling statistic.

    closed='left' excludes the current timestamp.
    Therefore, the current traffic target is not used.
    """

    ordered = road_df.sort_values(
        "timestamp"
    ).copy()

    indexed_target = ordered.set_index(
        "timestamp"
    )[TARGET]

    rolling = indexed_target.rolling(
        window=window,
        closed="left",
        min_periods=1,
    )

    if statistic == "mean":
        result = rolling.mean()

    elif statistic == "std":
        result = rolling.std()

    else:
        raise ValueError(
            f"Unsupported rolling statistic: {statistic}"
        )

    result.index = ordered.index

    return result


def create_rolling_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Create historical rolling means and standard deviation."""

    df = df.copy()

    rolling_features = []

    for feature_name, window in ROLLING_WINDOWS.items():

        if feature_name.endswith("_std_24h"):
            statistic = "std"
        else:
            statistic = "mean"

        parts = []

        for _, road_df in df.groupby(
            "road_id",
            sort=False,
        ):
            values = rolling_statistic_for_road(
                road_df=road_df,
                window=window,
                statistic=statistic,
            )

            parts.append(values)

        if parts:
            feature_values = pd.concat(parts)

            df[feature_name] = feature_values.reindex(
                df.index
            )

        else:
            df[feature_name] = np.nan

        rolling_features.append(feature_name)

        print(f"Created {feature_name}")

    return df, rolling_features


# ============================================================
# 7. CREATE ROAD AND WEATHER CONTEXT
# ============================================================

def create_context_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select context features and identify availability concerns.

    Static road attributes and calendar features can often
    be known ahead of time.

    Observed weather may not be known at prediction time.
    Weather is therefore marked as conditional.
    """

    df = df.copy()

    context_features = []

    # Road identity is categorical, not a continuous quantity.
    if "road_id" in df.columns:
        df["road_id_feature"] = df["road_id"].astype(
            "string"
        )

        context_features.append(
            "road_id_feature"
        )

    # Include road attributes when available.
    for column in [
        "latitude",
        "longitude",
        "road_capacity",
    ]:
        if column in df.columns:
            context_features.append(column)

    # Calendar/event indicators may be known ahead of time,
    # but the operational availability of event information
    # must be verified for deployment.
    for column in [
        "public_holiday",
        "event_flag",
        "roadwork_flag",
    ]:
        if column in df.columns:
            context_features.append(column)

    # Weather observations are useful for analysis, but a
    # future forecast must use only weather information
    # available at the actual prediction cutoff.
    for column in [
        "temperature",
        "rainfall",
        "visibility",
    ]:
        if column in df.columns:
            context_features.append(column)

    print(
        "Created/selectable context features: "
        f"{len(context_features)}"
    )

    return df, context_features


# ============================================================
# 8. DEFINE THE BASELINE MODEL FEATURE SET
# ============================================================

def define_model_features(
    lag_features: list[str],
    rolling_features: list[str],
    context_features: list[str],
) -> tuple[list[str], list[str]]:
    """
    Define a conservative feature list.

    Observed weather is excluded from the baseline because
    future availability has not been established.

    Contemporaneous speed, occupancy, vehicle_count, travel
    time and congestion labels are also excluded. They may
    contain information measured at the prediction timestamp.
    """

    calendar_features = [
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
    ]

    allowed_context = [
        "road_id_feature",
        "latitude",
        "longitude",
        "road_capacity",
        "public_holiday",
        "event_flag",
        "roadwork_flag",
    ]

    selected_context = [
        feature
        for feature in context_features
        if feature in allowed_context
    ]

    model_features = (
        lag_features
        + rolling_features
        + calendar_features
        + selected_context
    )

    # Preserve order while removing duplicates.
    model_features = list(
        dict.fromkeys(model_features)
    )

    excluded_features = [
        TARGET,
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "accident_count",
        "congestion_level",
        "temperature",
        "rainfall",
        "visibility",
        "timestamp",
        "date",
        "time",
    ]

    return model_features, excluded_features


# ============================================================
# 9. BUILD THE FEATURE CATALOG
# ============================================================

def build_feature_catalog(
    lag_features: list[str],
    rolling_features: list[str],
    context_features: list[str],
    model_features: list[str],
    excluded_features: list[str],
) -> pd.DataFrame:
    """Document the meaning and status of each feature."""

    descriptions = {
        "traffic_lag_30m": (
            "Traffic volume exactly 30 minutes before "
            "the current timestamp."
        ),
        "traffic_lag_1h": (
            "Traffic volume exactly one hour before "
            "the current timestamp."
        ),
        "traffic_lag_90m": (
            "Traffic volume exactly 90 minutes before "
            "the current timestamp."
        ),
        "traffic_lag_24h": (
            "Traffic volume at the same time on the "
            "previous day."
        ),
        "traffic_lag_7d": (
            "Traffic volume at the same time seven days earlier."
        ),
        "traffic_rolling_mean_3h": (
            "Mean of available traffic observations in the "
            "preceding three-hour window."
        ),
        "traffic_rolling_mean_6h": (
            "Mean of available traffic observations in the "
            "preceding six-hour window."
        ),
        "traffic_rolling_mean_24h": (
            "Mean of available traffic observations in the "
            "preceding 24-hour window."
        ),
        "traffic_rolling_std_24h": (
            "Standard deviation of available traffic "
            "observations in the preceding 24-hour window."
        ),
        "hour_sin": (
            "Sine encoding of the time of day."
        ),
        "hour_cos": (
            "Cosine encoding of the time of day."
        ),
        "day_of_week_sin": (
            "Sine encoding of the day of the week."
        ),
        "day_of_week_cos": (
            "Cosine encoding of the day of the week."
        ),
        "month_sin": (
            "Sine encoding of the month."
        ),
        "month_cos": (
            "Cosine encoding of the month."
        ),
        "is_weekend": (
            "Indicates Saturday or Sunday."
        ),
        "road_id_feature": (
            "Categorical road identifier."
        ),
    }

    rows = []

    for feature in model_features:
        rows.append(
            {
                "feature": feature,
                "description": descriptions.get(
                    feature,
                    "Calendar, road, or event context feature.",
                ),
                "role": "baseline_model_input",
                "availability": (
                    "Historical or planned information; "
                    "verify operational availability."
                ),
                "included_in_baseline": True,
            }
        )

    for feature in context_features:
        if feature not in model_features:
            rows.append(
                {
                    "feature": feature,
                    "description": descriptions.get(
                        feature,
                        "Contextual feature.",
                    ),
                    "role": "conditional_context",
                    "availability": (
                        "Must be checked for availability "
                        "at prediction time."
                    ),
                    "included_in_baseline": False,
                }
            )

    for feature in excluded_features:
        rows.append(
            {
                "feature": feature,
                "description": (
                    "Excluded from baseline model inputs "
                    "to reduce target leakage risk."
                ),
                "role": "target_or_excluded_field",
                "availability": "Not used in baseline.",
                "included_in_baseline": False,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 10. BUILD THE MODEL-INPUT DATASET
# ============================================================

def create_model_dataset(
    df: pd.DataFrame,
    model_features: list[str],
) -> pd.DataFrame:
    """
    Keep the target and approved baseline features.

    Missing historical features remain missing; they are not
    filled with zero or future observations.
    """

    required_columns = [
        "road_id",
        "timestamp",
        TARGET,
    ]

    selected_columns = (
        required_columns
        + model_features
    )

    selected_columns = list(
        dict.fromkeys(selected_columns)
    )

    missing_columns = [
        column
        for column in selected_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Expected model columns missing: {missing_columns}"
        )

    model_df = df[selected_columns].copy()

    # Explicitly verify that the target is not a feature.
    if TARGET in model_features:
        raise AssertionError(
            "Leakage detected: target included in model inputs."
        )

    return model_df


# ============================================================
# 11. WRITE LEAKAGE CONTROL REPORT
# ============================================================

def write_leakage_report(
    model_features: list[str],
    excluded_features: list[str],
) -> None:
    """Document baseline feature inclusion/exclusion decisions."""

    lines = [
        "# FlowCast — Day 8 Leakage Control Report",
        "",
        "## Prediction target",
        "",
        "`traffic_volume`",
        "",
        "## Included baseline features",
        "",
    ]

    lines.extend(
        f"- `{feature}`"
        for feature in model_features
    )

    lines.extend(
        [
            "",
            "## Excluded fields",
            "",
        ]
    )

    lines.extend(
        f"- `{feature}`"
        for feature in excluded_features
    )

    lines.extend(
        [
            "",
            "## Leakage safeguards",
            "",
            "- Historical lag features are matched using the same road "
            "and an exact earlier timestamp.",
            "- Missing timestamps are not silently replaced by the "
            "previous available row.",
            "- Rolling statistics exclude the current timestamp.",
            "- The current traffic target is never included as a feature.",
            "- Contemporaneous speed, occupancy, travel time, and "
            "vehicle counts are excluded from the baseline.",
            "- Observed weather is excluded from the baseline until "
            "future-weather availability is established.",
            "- Missing lag values are preserved instead of being "
            "filled with future information.",
            "",
            "## Remaining considerations",
            "",
            "A feature can still leak information if it is unavailable "
            "at the actual prediction cutoff or was computed from "
            "future records. The training and validation pipeline must "
            "use chronological splits.",
            "",
            "The event and roadwork indicators must be checked against "
            "their real operational availability before deployment.",
            "",
            "The seven-day lag is useful only when sufficient historical "
            "observations exist for that road and timestamp.",
            "",
        ]
    )

    LEAKAGE_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# 12. WRITE SUMMARY AND ENGINEERING REPORT
# ============================================================

def write_reports(
    original: pd.DataFrame,
    featured: pd.DataFrame,
    model_df: pd.DataFrame,
    model_features: list[str],
    lag_features: list[str],
    rolling_features: list[str],
    catalog: pd.DataFrame,
) -> None:
    """Generate machine-readable and human-readable reports."""

    lag_missingness = {
        feature: int(
            featured[feature].isna().sum()
        )
        for feature in lag_features
    }

    rolling_missingness = {
        feature: int(
            featured[feature].isna().sum()
        )
        for feature in rolling_features
    }

    summary = {
        "project": "FlowCast",
        "day": 8,
        "stage": "Leakage-aware temporal feature engineering",
        "input_file": str(
            INPUT_FILE.relative_to(ROOT)
        ),
        "output_file": str(
            ALL_FEATURES_FILE.relative_to(ROOT)
        ),
        "model_dataset_file": str(
            MODEL_FEATURES_FILE.relative_to(ROOT)
        ),
        "original_rows": int(len(original)),
        "feature_dataset_rows": int(len(featured)),
        "model_dataset_rows": int(len(model_df)),
        "input_feature_columns": int(len(original.columns)),
        "engineered_dataset_columns": int(len(featured.columns)),
        "baseline_model_features": model_features,
        "number_of_baseline_features": len(model_features),
        "lag_feature_missingness": lag_missingness,
        "rolling_feature_missingness": rolling_missingness,
        "target_missing_rows": int(
            model_df[TARGET].isna().sum()
        ),
        "rows_with_all_lags_available": int(
            featured[lag_features].notna().all(axis=1).sum()
        ),
        "target_in_feature_list": TARGET in model_features,
        "timestamp_duplicates": int(
            featured.duplicated(
                subset=["road_id", "timestamp"]
            ).sum()
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=4,
        ),
        encoding="utf-8",
    )

    catalog.to_csv(
        CATALOG_FILE,
        index=False,
    )

    lines = [
        "# FlowCast — Day 8 Feature Engineering Report",
        "",
        "## 1. Dataset overview",
        "",
        f"- Input rows: {len(original):,}",
        f"- Feature dataset rows: {len(featured):,}",
        f"- Model dataset rows: {len(model_df):,}",
        f"- Engineered columns: {len(featured.columns):,}",
        f"- Baseline model features: {len(model_features):,}",
        "",
        "## 2. Historical lag features",
        "",
    ]

    lines.extend(
        f"- `{feature}`"
        for feature in lag_features
    )

    lines.extend(
        [
            "",
            "## 3. Rolling statistics",
            "",
        ]
    )

    lines.extend(
        f"- `{feature}`"
        for feature in rolling_features
    )

    lines.extend(
        [
            "",
            "## 4. Missing historical information",
            "",
            "Missing lag and rolling values are retained. They can occur "
            "at the beginning of the dataset, when observations are "
            "missing, or when a road lacks the required historical record.",
            "",
            "| Feature | Missing rows |",
            "|---|---:|",
        ]
    )

    for feature, count in {
        **lag_missingness,
        **rolling_missingness,
    }.items():
        lines.append(
            f"| {feature} | {count:,} |"
        )

    lines.extend(
        [
            "",
            "## 5. Leakage safeguards",
            "",
            "- Historical lags use exact timestamps.",
            "- Rolling windows exclude the current timestamp.",
            "- The target is excluded from the baseline feature list.",
            "- Contemporaneous sensor measurements are excluded from "
            "baseline model inputs.",
            "- Observed weather is not included in the baseline until "
            "future availability is confirmed.",
            "- Rows are preserved even when lag features are missing.",
            "",
            "## 6. Output artifacts",
            "",
            "- `data/processed/flowcast_features.csv`",
            "- `data/processed/flowcast_model_features.csv`",
            "- `reports/day8/feature_catalog.csv`",
            "- `reports/day8/feature_engineering_summary.json`",
            "- `reports/day8/feature_engineering_report.md`",
            "- `reports/day8/leakage_control_report.md`",
            "",
            "## 7. Important limitation",
            "",
            "This stage creates features but does not train a model or "
            "demonstrate forecasting accuracy. Predictive performance "
            "must be evaluated with chronological validation and "
            "appropriate baselines.",
            "",
        ]
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nReports generated successfully.")


# ============================================================
# 13. MAIN
# ============================================================

def main() -> None:
    """Execute the Day 8 feature-engineering workflow."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load cleaned observations.
    df = load_data()

    original_rows = len(df)

    # Preserve the target and all existing fields.
    featured = create_time_features(df)

    featured, lag_features = create_lag_features(
        featured
    )

    featured, rolling_features = create_rolling_features(
        featured
    )

    featured, context_features = create_context_features(
        featured
    )

    model_features, excluded_features = define_model_features(
        lag_features=lag_features,
        rolling_features=rolling_features,
        context_features=context_features,
    )

    # Build the feature catalog.
    catalog = build_feature_catalog(
        lag_features=lag_features,
        rolling_features=rolling_features,
        context_features=context_features,
        model_features=model_features,
        excluded_features=excluded_features,
    )

    # Construct a minimal modelling dataset.
    model_df = create_model_dataset(
        df=featured,
        model_features=model_features,
    )

    # Verify the fundamental invariants.
    assert len(featured) == original_rows, (
        "Feature engineering unexpectedly changed the row count."
    )

    assert not featured.duplicated(
        subset=["road_id", "timestamp"]
    ).any(), (
        "Duplicate road-time records found in output."
    )

    assert TARGET not in model_features, (
        "The target was included in the feature list."
    )

    # Save the full feature dataset and model dataset.
    featured.to_csv(
        ALL_FEATURES_FILE,
        index=False,
    )

    model_df.to_csv(
        MODEL_FEATURES_FILE,
        index=False,
    )

    # Save documentation and audit reports.
    write_leakage_report(
        model_features=model_features,
        excluded_features=excluded_features,
    )

    write_reports(
        original=df,
        featured=featured,
        model_df=model_df,
        model_features=model_features,
        lag_features=lag_features,
        rolling_features=rolling_features,
        catalog=catalog,
    )

    print("\n" + "=" * 65)
    print("DAY 8 COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"Rows preserved: {len(featured):,}")
    print(f"New model features: {len(model_features):,}")
    print(f"Full dataset: {ALL_FEATURES_FILE}")
    print(f"Model dataset: {MODEL_FEATURES_FILE}")
    print(f"Feature catalog: {CATALOG_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"Leakage audit: {LEAKAGE_FILE}")
    print("\nNo model has been trained in this stage.")


if __name__ == "__main__":
    main()

