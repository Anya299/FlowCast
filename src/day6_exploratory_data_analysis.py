"""
FLOWCAST — DAY 6
Exploratory Data Analysis and Traffic Behaviour Intelligence

Input:
    data/interim/flowcast_integrated.csv

Outputs:
    reports/day6/

Objectives:
    1. Profile the integrated dataset.
    2. Analyze missingness and numeric parsing issues.
    3. Analyze traffic-volume distributions.
    4. Investigate hourly traffic patterns.
    5. Compare traffic across roads.
    6. Analyze congestion categories when available.
    7. Investigate weather relationships when available.
    8. Generate correlation analysis.
    9. Produce reproducible CSV, JSON, Markdown, and PNG outputs.

Data integrity:
    - The original dataset is never overwritten.
    - Missing values are not automatically imputed.
    - Outliers are not automatically deleted.
    - Correlations are not interpreted as causal relationships.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

# Use a non-interactive backend so the script can run
# reliably from PowerShell without opening chart windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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
    / "day6"
)

FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROJECT_NAME = "FlowCast"

RANDOM_STATE = 42

# We do not automatically remove extreme values.
# This percentile is used only to report possible extremes.
EXTREME_PERCENTILE = 0.99


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger("flowcast.day6")


# ============================================================
# 3. LOAD THE DATASET
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the integrated dataset and verify essential columns.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    logger.info("Loading integrated dataset...")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "The integrated dataset contains no rows."
        )

    # Remove whitespace from column names.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    required = [
        "road_id",
        "date",
        "time",
    ]

    missing_required = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing_required:
        raise ValueError(
            "Required columns are missing: "
            f"{missing_required}"
        )

    logger.info(
        "Rows loaded: %s",
        f"{len(df):,}",
    )

    logger.info(
        "Columns loaded: %s",
        len(df.columns),
    )

    return df


# ============================================================
# 4. PREPARE AN ANALYSIS COPY
# ============================================================

def prepare_data(
    original: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create an analysis copy.

    Numeric conversion failures are tracked rather than
    silently ignored.

    The source dataset remains unchanged.
    """

    df = original.copy()

    # Construct a timestamp from the existing date and time.
    df["timestamp"] = pd.to_datetime(
        df["date"].astype("string").str.strip()
        + " "
        + df["time"].astype("string").str.strip(),
        errors="coerce",
    )

    df["hour"] = df["timestamp"].dt.hour

    df["date_only"] = (
        df["timestamp"].dt.date
    )

    df["day_of_week"] = (
        df["timestamp"].dt.day_name()
    )

    df["day_of_week_number"] = (
        df["timestamp"].dt.dayofweek
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    # Numeric candidates.
    #
    # Numeric columns are discovered using the CSV's inferred
    # data types, plus a list of known FlowCast measurements.
    known_numeric_candidates = [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "road_capacity",
        "temperature",
        "rainfall",
        "visibility",
        "humidity",
        "wind_speed",
        "pressure",
    ]

    numeric_columns = []

    for column in df.columns:
        if column in {
            "timestamp",
            "date_only",
            "day_of_week",
        }:
            continue

        if (
            pd.api.types.is_numeric_dtype(df[column])
            or column in known_numeric_candidates
        ):
            numeric_columns.append(column)

    # Remove duplicates while preserving column order.
    numeric_columns = list(
        dict.fromkeys(numeric_columns)
    )

    parsing_issues = []

    for column in numeric_columns:
        original_values = df[column]

        converted = pd.to_numeric(
            original_values,
            errors="coerce",
        )

        # A parsing failure occurs when a non-missing original
        # value becomes missing during numeric conversion.
        failure_mask = (
            original_values.notna()
            & converted.isna()
        )

        failure_count = int(
            failure_mask.sum()
        )

        if failure_count > 0:
            parsing_issues.append(
                {
                    "column": column,
                    "numeric_parsing_failures": (
                        failure_count
                    ),
                }
            )

        df[column] = converted

    if parsing_issues:
        pd.DataFrame(
            parsing_issues
        ).to_csv(
            OUTPUT_DIR
            / "numeric_parsing_issues.csv",
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "column",
                "numeric_parsing_failures",
            ]
        ).to_csv(
            OUTPUT_DIR
            / "numeric_parsing_issues.csv",
            index=False,
        )

    logger.info(
        "Numeric columns identified: %s",
        len(numeric_columns),
    )

    logger.info(
        "Timestamp parsing failures: %s",
        f"{df['timestamp'].isna().sum():,}",
    )

    return df, numeric_columns


# ============================================================
# 5. DATASET PROFILE
# ============================================================

def create_dataset_profile(
    original: pd.DataFrame,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Summarize dataset size, schema, date range, and road count.
    """

    valid_timestamps = df["timestamp"].dropna()

    if not valid_timestamps.empty:
        first_timestamp = str(
            valid_timestamps.min()
        )

        last_timestamp = str(
            valid_timestamps.max()
        )
    else:
        first_timestamp = None
        last_timestamp = None

    profile = {
        "project": PROJECT_NAME,
        "day": 6,
        "analysis": (
            "Exploratory Data Analysis "
            "and Traffic Behaviour Intelligence"
        ),
        "input_file": str(
            INPUT_FILE.relative_to(PROJECT_ROOT)
        ),
        "source_dataset_modified": False,
        "rows": int(len(original)),
        "columns": int(len(original.columns)),
        "column_names": [
            str(column)
            for column in original.columns
        ],
        "unique_roads": int(
            original["road_id"].nunique(
                dropna=True
            )
        ),
        "missing_road_ids": int(
            original["road_id"].isna().sum()
        ),
        "invalid_timestamps": int(
            df["timestamp"].isna().sum()
        ),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "missing_cells": int(
            original.isna().sum().sum()
        ),
        "columns_with_missing_values": int(
            (
                original.isna()
                .sum()
                .gt(0)
            ).sum()
        ),
        "numeric_parsing_issues_file": (
            "numeric_parsing_issues.csv"
        ),
    }

    return profile


# ============================================================
# 6. MISSINGNESS ANALYSIS
# ============================================================

def analyze_missingness(
    original: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate missing cells and percentages per column.
    """

    missing_count = (
        original.isna().sum()
    )

    missing_percent = (
        missing_count
        / len(original)
        * 100
    )

    report = pd.DataFrame(
        {
            "column": original.columns,
            "data_type": [
                str(dtype)
                for dtype in original.dtypes
            ],
            "total_rows": len(original),
            "missing_count": [
                int(missing_count[column])
                for column in original.columns
            ],
            "missing_percent": [
                float(missing_percent[column])
                for column in original.columns
            ],
            "non_missing_count": [
                int(
                    original[column].notna().sum()
                )
                for column in original.columns
            ],
        }
    )

    report = report.sort_values(
        "missing_percent",
        ascending=False,
    ).reset_index(drop=True)

    report.to_csv(
        OUTPUT_DIR / "missingness_analysis.csv",
        index=False,
    )

    # Chart: missingness for columns containing missing values.
    missing_only = report.loc[
        report["missing_count"] > 0
    ].copy()

    if not missing_only.empty:
        plot_data = missing_only.sort_values(
            "missing_percent",
            ascending=True,
        )

        plt.figure(figsize=(11, 7))

        plt.barh(
            plot_data["column"],
            plot_data["missing_percent"],
        )

        plt.xlabel("Missing values (%)")
        plt.ylabel("Column")
        plt.title(
            "FlowCast — Missingness by Column"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR
            / "missingness_by_column.png",
            dpi=180,
        )

        plt.close()

    return report


# ============================================================
# 7. NUMERIC DESCRIPTIVE STATISTICS
# ============================================================

def calculate_numeric_statistics(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """
    Generate descriptive statistics for numeric measurements.
    """

    available_columns = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    if not available_columns:
        return pd.DataFrame()

    statistics = (
        df[available_columns]
        .describe()
        .T
        .reset_index()
        .rename(
            columns={
                "index": "column",
            }
        )
    )

    statistics["missing_count"] = [
        int(df[column].isna().sum())
        for column in statistics["column"]
    ]

    statistics["missing_percent"] = [
        float(
            df[column].isna().mean() * 100
        )
        for column in statistics["column"]
    ]

    statistics.to_csv(
        OUTPUT_DIR
        / "numeric_descriptive_statistics.csv",
        index=False,
    )

    return statistics


# ============================================================
# 8. TRAFFIC-VOLUME DISTRIBUTION
# ============================================================

def analyze_traffic_volume(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analyze the distribution of traffic_volume.
    """

    column = "traffic_volume"

    if column not in df.columns:
        logger.warning(
            "traffic_volume is unavailable. "
            "Skipping target distribution analysis."
        )

        return {
            "available": False,
            "reason": (
                "traffic_volume column not found"
            ),
        }

    values = df[column].dropna()

    if values.empty:
        return {
            "available": False,
            "reason": (
                "traffic_volume has no numeric observations"
            ),
        }

    stats = {
        "available": True,
        "observations": int(len(values)),
        "missing_values": int(
            df[column].isna().sum()
        ),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "standard_deviation": float(
            values.std()
        ) if len(values) > 1 else None,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "q90": float(values.quantile(0.90)),
        "q99": float(
            values.quantile(
                EXTREME_PERCENTILE
            )
        ),
        "skewness": float(
            values.skew()
        ) if len(values) > 2 else None,
        "negative_values": int(
            (values < 0).sum()
        ),
    }

    # Distribution chart.
    plt.figure(figsize=(10, 6))

    plt.hist(
        values,
        bins=50,
        edgecolor="black",
    )

    plt.xlabel("Traffic volume")
    plt.ylabel("Observation count")
    plt.title(
        "FlowCast — Traffic Volume Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "traffic_volume_distribution.png",
        dpi=180,
    )

    plt.close()

    # Box plot to reveal extreme values.
    plt.figure(figsize=(10, 4))

    plt.boxplot(
        values,
        vert=False,
    )

    plt.xlabel("Traffic volume")
    plt.title(
        "FlowCast — Traffic Volume Box Plot"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "traffic_volume_boxplot.png",
        dpi=180,
    )

    plt.close()

    return stats


# ============================================================
# 9. HOURLY TRAFFIC PATTERNS
# ============================================================

def analyze_hourly_traffic(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate average and median traffic volume by hour.
    """

    if (
        "traffic_volume" not in df.columns
        or "hour" not in df.columns
    ):
        return pd.DataFrame()

    valid = df.loc[
        df["timestamp"].notna()
        & df["traffic_volume"].notna()
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    hourly = (
        valid.groupby("hour")
        .agg(
            observation_count=(
                "traffic_volume",
                "count",
            ),
            mean_traffic_volume=(
                "traffic_volume",
                "mean",
            ),
            median_traffic_volume=(
                "traffic_volume",
                "median",
            ),
            standard_deviation=(
                "traffic_volume",
                "std",
            ),
        )
        .reset_index()
    )

    hourly.to_csv(
        OUTPUT_DIR
        / "hourly_traffic_analysis.csv",
        index=False,
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        hourly["hour"],
        hourly["mean_traffic_volume"],
        marker="o",
        label="Mean",
    )

    plt.plot(
        hourly["hour"],
        hourly["median_traffic_volume"],
        marker="s",
        label="Median",
    )

    plt.xticks(
        range(0, 24)
    )

    plt.xlabel("Hour of day")
    plt.ylabel("Traffic volume")
    plt.title(
        "FlowCast — Traffic Volume by Hour"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "traffic_volume_by_hour.png",
        dpi=180,
    )

    plt.close()

    return hourly


# ============================================================
# 10. ROAD-LEVEL TRAFFIC ANALYSIS
# ============================================================

def analyze_roads(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare traffic volume and measurement completeness
    across roads.
    """

    if "traffic_volume" not in df.columns:
        return pd.DataFrame()

    road_stats = (
        df.groupby("road_id", dropna=False)
        .agg(
            total_rows=(
                "traffic_volume",
                "size",
            ),
            valid_traffic_volume=(
                "traffic_volume",
                "count",
            ),
            mean_traffic_volume=(
                "traffic_volume",
                "mean",
            ),
            median_traffic_volume=(
                "traffic_volume",
                "median",
            ),
            maximum_traffic_volume=(
                "traffic_volume",
                "max",
            ),
            traffic_volume_std=(
                "traffic_volume",
                "std",
            ),
        )
        .reset_index()
    )

    road_stats["missing_traffic_volume"] = (
        road_stats["total_rows"]
        - road_stats["valid_traffic_volume"]
    )

    road_stats["traffic_volume_missing_percent"] = (
        road_stats["missing_traffic_volume"]
        / road_stats["total_rows"]
        * 100
    )

    road_stats = road_stats.sort_values(
        "mean_traffic_volume",
        ascending=False,
    ).reset_index(drop=True)

    road_stats.to_csv(
        OUTPUT_DIR
        / "road_level_traffic_analysis.csv",
        index=False,
    )

    # Chart: highest mean traffic volume roads.
    top_roads = road_stats.dropna(
        subset=["mean_traffic_volume"]
    ).head(15)

    if not top_roads.empty:
        top_roads = top_roads.sort_values(
            "mean_traffic_volume",
            ascending=True,
        )

        plt.figure(figsize=(11, 7))

        plt.barh(
            top_roads["road_id"].astype(str),
            top_roads["mean_traffic_volume"],
        )

        plt.xlabel("Mean traffic volume")
        plt.ylabel("Road ID")
        plt.title(
            "FlowCast — Roads with Highest Mean Traffic Volume"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR
            / "top_roads_by_mean_traffic.png",
            dpi=180,
        )

        plt.close()

    return road_stats


# ============================================================
# 11. CONGESTION CATEGORY ANALYSIS
# ============================================================

def analyze_congestion(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Examine existing congestion labels.

    Labels are not automatically replaced or recalculated.
    """

    column = "congestion_level"

    if column not in df.columns:
        return {
            "available": False,
            "reason": (
                "congestion_level column not found"
            ),
        }

    labels = (
        df[column]
        .astype("string")
        .str.strip()
    )

    missing_count = int(
        labels.isna().sum()
        + labels.eq("").sum()
    )

    normalized = labels.str.lower()

    counts = (
        normalized.dropna()
        .loc[lambda series: series.ne("")]
        .value_counts()
        .rename_axis("congestion_level")
        .reset_index(name="observation_count")
    )

    total_non_missing = int(
        counts["observation_count"].sum()
    )

    if total_non_missing:
        counts["percentage"] = (
            counts["observation_count"]
            / total_non_missing
            * 100
        )
    else:
        counts["percentage"] = pd.Series(
            dtype="float64"
        )

    counts.to_csv(
        OUTPUT_DIR
        / "congestion_category_distribution.csv",
        index=False,
    )

    if not counts.empty:
        plt.figure(figsize=(10, 6))

        plt.bar(
            counts["congestion_level"],
            counts["observation_count"],
        )

        plt.xlabel("Congestion category")
        plt.ylabel("Observation count")
        plt.title(
            "FlowCast — Congestion Category Distribution"
        )

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR
            / "congestion_category_distribution.png",
            dpi=180,
        )

        plt.close()

    return {
        "available": True,
        "missing_labels": missing_count,
        "non_missing_labels": total_non_missing,
        "category_count": int(len(counts)),
        "categories": (
            counts.to_dict(orient="records")
        ),
    }


# ============================================================
# 12. WEATHER RELATIONSHIPS
# ============================================================

def analyze_weather_relationships(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Investigate correlations between traffic volume
    and available numeric weather variables.
    """

    if "traffic_volume" not in df.columns:
        return pd.DataFrame()

    weather_candidates = [
        "temperature",
        "rainfall",
        "visibility",
        "humidity",
        "wind_speed",
        "pressure",
    ]

    available = [
        column
        for column in weather_candidates
        if column in df.columns
    ]

    if not available:
        return pd.DataFrame()

    rows = []

    for column in available:
        pair = df[
            ["traffic_volume", column]
        ].dropna()

        if len(pair) < 3:
            continue

        correlation = pair[
            "traffic_volume"
        ].corr(
            pair[column],
            method="pearson",
        )

        rows.append(
            {
                "weather_variable": column,
                "paired_observations": int(
                    len(pair)
                ),
                "pearson_correlation_with_traffic": (
                    float(correlation)
                    if pd.notna(correlation)
                    else None
                ),
            }
        )

        # Scatter plots help identify non-linear relationships
        # and extreme values that correlation alone may hide.
        if len(pair) > 20_000:
            pair = pair.sample(
                n=20_000,
                random_state=RANDOM_STATE,
            )

        plt.figure(figsize=(9, 6))

        plt.scatter(
            pair[column],
            pair["traffic_volume"],
            alpha=0.25,
        )

        plt.xlabel(column)
        plt.ylabel("Traffic volume")

        plt.title(
            "FlowCast — Traffic Volume vs "
            f"{column.replace('_', ' ').title()}"
        )

        plt.tight_layout()

        filename = (
            "traffic_vs_"
            + column.lower().replace(" ", "_")
            + ".png"
        )

        plt.savefig(
            FIGURES_DIR / filename,
            dpi=180,
        )

        plt.close()

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT_DIR
        / "weather_traffic_relationships.csv",
        index=False,
    )

    return result


# ============================================================
# 13. NUMERIC CORRELATION ANALYSIS
# ============================================================

def analyze_correlations(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """
    Generate numeric correlation results and a heatmap.
    """

    available = [
        column
        for column in numeric_columns
        if column in df.columns
        and df[column].notna().any()
    ]

    if len(available) < 2:
        logger.warning(
            "Not enough numeric columns for correlation analysis."
        )

        return pd.DataFrame()

    # Avoid correlation calculations on constant columns.
    available = [
        column
        for column in available
        if df[column].nunique(
            dropna=True
        ) > 1
    ]

    if len(available) < 2:
        return pd.DataFrame()

    correlation = df[
        available
    ].corr(
        method="pearson",
        numeric_only=True,
    )

    correlation.to_csv(
        OUTPUT_DIR
        / "numeric_correlation_matrix.csv"
    )

    # Save a heatmap without seaborn.
    matrix = correlation.to_numpy()

    fig_width = max(
        9,
        min(18, len(available) * 0.8),
    )

    fig_height = max(
        7,
        min(16, len(available) * 0.7),
    )

    plt.figure(
        figsize=(fig_width, fig_height)
    )

    image = plt.imshow(
        matrix,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Pearson correlation",
    )

    plt.xticks(
        range(len(available)),
        available,
        rotation=90,
    )

    plt.yticks(
        range(len(available)),
        available,
    )

    plt.title(
        "FlowCast — Numeric Feature Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "numeric_correlation_heatmap.png",
        dpi=180,
    )

    plt.close()

    # Convert the upper triangle into a compact table.
    rows = []

    for i, first in enumerate(available):
        for j in range(i + 1, len(available)):
            second = available[j]

            value = correlation.loc[
                first,
                second,
            ]

            if pd.isna(value):
                continue

            rows.append(
                {
                    "feature_1": first,
                    "feature_2": second,
                    "pearson_correlation": float(
                        value
                    ),
                }
            )

    pairs = pd.DataFrame(rows)

    if not pairs.empty:
        pairs["absolute_correlation"] = (
            pairs["pearson_correlation"].abs()
        )

        pairs = pairs.sort_values(
            "absolute_correlation",
            ascending=False,
        )

        pairs.to_csv(
            OUTPUT_DIR
            / "numeric_correlation_pairs.csv",
            index=False,
        )

    return pairs


# ============================================================
# 14. DAY-OF-WEEK TRAFFIC ANALYSIS
# ============================================================

def analyze_weekday_patterns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare traffic volume by weekday.
    """

    if (
        "traffic_volume" not in df.columns
        or "day_of_week" not in df.columns
    ):
        return pd.DataFrame()

    valid = df.loc[
        df["day_of_week"].notna()
        & df["traffic_volume"].notna()
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday = (
        valid.groupby("day_of_week")
        .agg(
            observation_count=(
                "traffic_volume",
                "count",
            ),
            mean_traffic_volume=(
                "traffic_volume",
                "mean",
            ),
            median_traffic_volume=(
                "traffic_volume",
                "median",
            ),
        )
        .reindex(weekday_order)
        .reset_index()
    )

    weekday.to_csv(
        OUTPUT_DIR
        / "weekday_traffic_analysis.csv",
        index=False,
    )

    plot_data = weekday.dropna(
        subset=["mean_traffic_volume"]
    )

    if not plot_data.empty:
        plt.figure(figsize=(10, 6))

        plt.bar(
            plot_data["day_of_week"],
            plot_data["mean_traffic_volume"],
        )

        plt.xlabel("Day of week")
        plt.ylabel("Mean traffic volume")

        plt.title(
            "FlowCast — Traffic Volume by Day of Week"
        )

        plt.xticks(
            rotation=30,
            ha="right",
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR
            / "traffic_volume_by_weekday.png",
            dpi=180,
        )

        plt.close()

    return weekday


# ============================================================
# 15. WRITE THE FINAL EDA REPORT
# ============================================================

def write_eda_report(
    profile: dict[str, Any],
    traffic_stats: dict[str, Any],
    hourly: pd.DataFrame,
    roads: pd.DataFrame,
    congestion: dict[str, Any],
    weather: pd.DataFrame,
    correlations: pd.DataFrame,
    weekday: pd.DataFrame,
    missingness: pd.DataFrame,
) -> None:
    """
    Write a Markdown report based on actual calculated results.
    """

    lines = [
        "# FlowCast — Day 6 Exploratory Data Analysis",
        "",
        "## 1. Executive summary",
        "",
        f"- Source rows: {profile['rows']:,}",
        f"- Source columns: {profile['columns']}",
        f"- Unique roads: {profile['unique_roads']}",
        f"- Invalid timestamps: "
        f"{profile['invalid_timestamps']:,}",
        f"- Missing cells: "
        f"{profile['missing_cells']:,}",
        f"- Columns containing missing values: "
        f"{profile['columns_with_missing_values']}",
        f"- First timestamp: "
        f"{profile['first_timestamp']}",
        f"- Last timestamp: "
        f"{profile['last_timestamp']}",
        "",
        "## 2. Traffic-volume distribution",
        "",
    ]

    if traffic_stats.get("available"):
        lines.extend(
            [
                f"- Valid numeric observations: "
                f"{traffic_stats['observations']:,}",
                f"- Missing values: "
                f"{traffic_stats['missing_values']:,}",
                f"- Mean: "
                f"{traffic_stats['mean']:.3f}",
                f"- Median: "
                f"{traffic_stats['median']:.3f}",
                f"- Standard deviation: "
                f"{traffic_stats['standard_deviation']}",
                f"- Minimum: "
                f"{traffic_stats['minimum']:.3f}",
                f"- Maximum: "
                f"{traffic_stats['maximum']:.3f}",
                f"- 90th percentile: "
                f"{traffic_stats['q90']:.3f}",
                f"- 99th percentile: "
                f"{traffic_stats['q99']:.3f}",
                f"- Skewness: "
                f"{traffic_stats['skewness']}",
                f"- Negative observations: "
                f"{traffic_stats['negative_values']}",
            ]
        )
    else:
        lines.append(
            "Traffic-volume statistics could not be calculated."
        )

    lines.extend(
        [
            "",
            "## 3. Hourly traffic patterns",
            "",
        ]
    )

    if not hourly.empty:
        busiest = hourly.loc[
            hourly["mean_traffic_volume"].idxmax()
        ]

        quietest = hourly.loc[
            hourly["mean_traffic_volume"].idxmin()
        ]

        lines.extend(
            [
                f"- Highest mean traffic hour: "
                f"{int(busiest['hour']):02d}:00",
                f"- Highest mean traffic volume: "
                f"{busiest['mean_traffic_volume']:.3f}",
                f"- Lowest mean traffic hour: "
                f"{int(quietest['hour']):02d}:00",
                f"- Lowest mean traffic volume: "
                f"{quietest['mean_traffic_volume']:.3f}",
                "",
                "These findings describe the aggregate data. "
                "Road-specific patterns may differ.",
            ]
        )
    else:
        lines.append(
            "Hourly traffic statistics were unavailable."
        )

    lines.extend(
        [
            "",
            "## 4. Road-level comparison",
            "",
        ]
    )

    if not roads.empty:
        lines.extend(
            [
                "| Road ID | Mean traffic | Median traffic | "
                "Maximum traffic | Missing traffic % |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        for _, row in roads.head(10).iterrows():
            lines.append(
                f"| {row['road_id']} "
                f"| {row['mean_traffic_volume']:.3f} "
                f"| {row['median_traffic_volume']:.3f} "
                f"| {row['maximum_traffic_volume']:.3f} "
                f"| "
                f"{row['traffic_volume_missing_percent']:.2f}% |"
            )
    else:
        lines.append(
            "Road-level traffic statistics were unavailable."
        )

    lines.extend(
        [
            "",
            "## 5. Congestion distribution",
            "",
        ]
    )

    if congestion.get("available"):
        lines.extend(
            [
                f"- Missing labels: "
                f"{congestion['missing_labels']:,}",
                f"- Non-missing labels: "
                f"{congestion['non_missing_labels']:,}",
                f"- Distinct categories: "
                f"{congestion['category_count']}",
                "",
                "| Category | Count | Percentage |",
                "|---|---:|---:|",
            ]
        )

        for category in congestion["categories"]:
            lines.append(
                f"| {category['congestion_level']} "
                f"| {category['observation_count']:,} "
                f"| {category['percentage']:.2f}% |"
            )
    else:
        lines.append(
            "Congestion labels were unavailable."
        )

    lines.extend(
        [
            "",
            "## 6. Weather and traffic relationships",
            "",
        ]
    )

    if not weather.empty:
        lines.extend(
            [
                "| Weather variable | Paired observations | "
                "Pearson correlation |",
                "|---|---:|---:|",
            ]
        )

        for _, row in weather.iterrows():
            value = row[
                "pearson_correlation_with_traffic"
            ]

            correlation_text = (
                f"{value:.4f}"
                if pd.notna(value)
                else "Unavailable"
            )

            lines.append(
                f"| {row['weather_variable']} "
                f"| {row['paired_observations']:,} "
                f"| {correlation_text} |"
            )
    else:
        lines.append(
            "No suitable numeric weather variables "
            "were available for this analysis."
        )

    lines.extend(
        [
            "",
            "Pearson correlation measures linear association. "
            "It does not establish causation. Road differences, "
            "time-of-day patterns, missing values, and other "
            "confounding factors may influence the results.",
            "",
            "## 7. Numeric feature relationships",
            "",
        ]
    )

    if not correlations.empty:
        lines.extend(
            [
                "| Feature 1 | Feature 2 | Correlation |",
                "|---|---|---:|",
            ]
        )

        for _, row in correlations.head(15).iterrows():
            lines.append(
                f"| {row['feature_1']} "
                f"| {row['feature_2']} "
                f"| {row['pearson_correlation']:.4f} |"
            )
    else:
        lines.append(
            "No numeric correlation pairs were available."
        )

    lines.extend(
        [
            "",
            "## 8. Missingness",
            "",
            "| Column | Missing count | Missing % |",
            "|---|---:|---:|",
        ]
    )

    for _, row in missingness.head(15).iterrows():
        lines.append(
            f"| {row['column']} "
            f"| {row['missing_count']:,} "
            f"| {row['missing_percent']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## 9. Generated visualizations",
            "",
            "- `figures/missingness_by_column.png`",
            "- `figures/traffic_volume_distribution.png`",
            "- `figures/traffic_volume_boxplot.png`",
            "- `figures/traffic_volume_by_hour.png`",
            "- `figures/top_roads_by_mean_traffic.png`",
            "- `figures/congestion_category_distribution.png`",
            "- `figures/numeric_correlation_heatmap.png`",
            "- `figures/traffic_volume_by_weekday.png`",
            "",
            "Additional weather scatter plots are generated "
            "when suitable columns are available.",
            "",
            "## 10. Engineering limitations",
            "",
            "1. The source dataset was not overwritten.",
            "2. Missing measurements were not imputed.",
            "3. Extreme values were not automatically deleted.",
            "4. Correlation was not interpreted as causation.",
            "5. Aggregate statistics may hide road-specific "
            "or temporal differences.",
            "6. The current EDA is descriptive. It does not "
            "measure forecasting performance.",
            "7. Any numeric parsing failures require "
            "investigation before model training.",
            "",
            "## 11. Next steps",
            "",
            "Use the findings to plan leakage-safe feature "
            "engineering, investigate data-quality concerns, "
            "define forecasting targets, and establish a "
            "time-aware train/validation/test strategy.",
            "",
            "## 12. Reproducibility",
            "",
            "Run:",
            "",
            "```powershell",
            "python src/day6_exploratory_data_analysis.py",
            "```",
            "",
        ]
    )

    report_path = (
        OUTPUT_DIR / "day6_eda_report.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logger.info(
        "Engineering report saved: %s",
        report_path,
    )


# ============================================================
# 16. MAIN EXECUTION PIPELINE
# ============================================================

def main() -> None:
    """
    Run the complete FlowCast Day 6 EDA workflow.
    """

    logger.info("=" * 65)
    logger.info(
        "FLOWCAST — DAY 6 EXPLORATORY DATA ANALYSIS"
    )
    logger.info("=" * 65)

    # Step 1: Load source data.
    original = load_dataset(INPUT_FILE)

    # Step 2: Create an analysis copy.
    df, numeric_columns = prepare_data(original)

    # Step 3: Profile the dataset.
    profile = create_dataset_profile(
        original=original,
        df=df,
    )

    # Step 4: Analyze missingness.
    missingness = analyze_missingness(
        original
    )

    # Step 5: Calculate numeric statistics.
    numeric_stats = calculate_numeric_statistics(
        df,
        numeric_columns,
    )

    # Step 6: Investigate traffic volume.
    traffic_stats = analyze_traffic_volume(
        df
    )

    # Step 7: Investigate hourly patterns.
    hourly = analyze_hourly_traffic(
        df
    )

    # Step 8: Compare roads.
    roads = analyze_roads(
        df
    )

    # Step 9: Analyze congestion labels.
    congestion = analyze_congestion(
        df
    )

    # Step 10: Investigate weather relationships.
    weather = analyze_weather_relationships(
        df
    )

    # Step 11: Generate numeric correlation analysis.
    correlations = analyze_correlations(
        df,
        numeric_columns,
    )

    # Step 12: Compare weekdays.
    weekday = analyze_weekday_patterns(
        df
    )

    # Step 13: Save the profile as JSON.
    profile_path = (
        OUTPUT_DIR / "eda_dataset_profile.json"
    )

    with profile_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            profile,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # Step 14: Save congestion summary.
    with (
        OUTPUT_DIR
        / "congestion_analysis_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            congestion,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # Step 15: Write the report.
    write_eda_report(
        profile=profile,
        traffic_stats=traffic_stats,
        hourly=hourly,
        roads=roads,
        congestion=congestion,
        weather=weather,
        correlations=correlations,
        weekday=weekday,
        missingness=missingness,
    )

    # Step 16: Print final results.
    logger.info("=" * 65)
    logger.info(
        "FLOWCAST DAY 6 — EDA COMPLETED"
    )
    logger.info("=" * 65)

    print()
    print("=" * 65)
    print("FLOWCAST DAY 6 — SUMMARY")
    print("=" * 65)

    print(
        f"Source rows: {profile['rows']:,}"
    )

    print(
        f"Source columns: {profile['columns']}"
    )

    print(
        f"Unique roads: {profile['unique_roads']}"
    )

    print(
        f"Missing cells: {profile['missing_cells']:,}"
    )

    print(
        f"Columns with missing values: "
        f"{profile['columns_with_missing_values']}"
    )

    print(
        f"Invalid timestamps: "
        f"{profile['invalid_timestamps']:,}"
    )

    if traffic_stats.get("available"):
        print(
            f"Mean traffic volume: "
            f"{traffic_stats['mean']:.3f}"
        )

        print(
            f"Median traffic volume: "
            f"{traffic_stats['median']:.3f}"
        )

        print(
            f"Maximum traffic volume: "
            f"{traffic_stats['maximum']:.3f}"
        )

    print()
    print(f"Reports: {OUTPUT_DIR}")
    print(f"Charts: {FIGURES_DIR}")
    print("=" * 65)


# ============================================================
# 17. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()