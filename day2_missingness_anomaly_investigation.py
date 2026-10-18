# ============================================================
# FLOWCAST
# DAY 2 — MISSINGNESS & ANOMALY INVESTIGATION
# ============================================================
#
# Objective:
# Investigate missing measurements and suspicious values by
# variable, road, date, and hour.
#
# Input:
# data/interim/flowcast_integrated.csv
#
# Outputs:
# reports/day2/
#
# IMPORTANT:
# This script investigates data quality.
# It does not automatically delete suspicious observations.
# ============================================================

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "flowcast_integrated.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "day2"

FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

print("=" * 65)
print("FLOWCAST — DAY 2 INVESTIGATION")
print("=" * 65)

print(f"\nProject root: {PROJECT_ROOT}")
print(f"Input file:   {INPUT_FILE}")
print(f"Output path:  {OUTPUT_DIR}")


# ============================================================
# 2. LOAD AND VALIDATE THE INTEGRATED DATASET
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Integrated dataset not found: {INPUT_FILE}\n"
        "Verify that the Day 1 pipeline completed successfully."
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("\n[DATASET OVERVIEW]")
print("-" * 65)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns):,}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes.astype(str).to_string())

print("\nFirst five records:")
print(df.head().to_string(index=False))


# ============================================================
# 3. IDENTIFY AVAILABLE INVESTIGATION VARIABLES
# ============================================================

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

identifier_columns = [
    column
    for column in [
        "road_id",
        "road_name",
        "latitude",
        "longitude",
        "date",
        "time",
        "hour",
    ]
    if column in df.columns
]

measurement_candidates = [
    "traffic_volume",
    "vehicle_count",
    "avg_speed",
    "occupancy",
    "travel_time",
    "temperature",
    "rainfall",
    "visibility",
]

measurement_columns = [
    column
    for column in measurement_candidates
    if column in df.columns
]

print("\n[INVESTIGATION VARIABLES]")
print("-" * 65)

print("Numeric columns:", numeric_columns)
print("Measurement columns:", measurement_columns)
print("Available identifiers:", identifier_columns)


# ============================================================
# 4. INVESTIGATE MISSINGNESS BY VARIABLE
# ============================================================

missing_count = df.isna().sum()

missingness = pd.DataFrame({
    "variable": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "total_rows": len(df),
    "missing_count": missing_count.values,
    "observed_count": (len(df) - missing_count).values,
    "missing_percentage": (
        missing_count.values / max(len(df), 1) * 100
    ),
})

missingness = missingness.sort_values(
    "missing_percentage",
    ascending=False,
)

missingness.to_csv(
    OUTPUT_DIR / "missingness_by_variable.csv",
    index=False,
)

print("\n[MISSINGNESS BY VARIABLE]")
print("-" * 65)

print(
    missingness[
        [
            "variable",
            "missing_count",
            "missing_percentage",
        ]
    ].to_string(index=False)
)


# ============================================================
# 5. MISSINGNESS PATTERNS BY ROAD
# ============================================================

if "road_id" in df.columns:

    road_missingness = df.groupby(
        "road_id",
        dropna=False,
    ).agg(
        records=("road_id", "size")
    )

    if "road_name" in df.columns:
        road_names = df.groupby(
            "road_id",
            dropna=False,
        )["road_name"].first()

        road_missingness["road_name"] = road_names

    for column in measurement_columns:

        grouped = df.groupby(
            "road_id",
            dropna=False,
        )[column].apply(
            lambda values: values.isna().mean() * 100
        )

        road_missingness[
            f"{column}_missing_percentage"
        ] = grouped

    road_missingness = road_missingness.reset_index()

    road_missingness.to_csv(
        OUTPUT_DIR / "missingness_by_road.csv",
        index=False,
    )

    print("\n[MISSINGNESS BY ROAD]")
    print("-" * 65)

    print(
        road_missingness.head(10).to_string(index=False)
    )

else:
    print(
        "\nRoad-level analysis skipped: "
        "'road_id' is unavailable."
    )


# ============================================================
# 6. INVESTIGATE MISSINGNESS BY DATE
# ============================================================

if "date" in df.columns:

    df["investigation_date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.date

    daily_records = df.groupby(
        "investigation_date",
        dropna=False,
    ).size().rename("record_count")

    daily_missingness = pd.DataFrame(
        index=daily_records.index
    )

    daily_missingness["record_count"] = daily_records

    for column in measurement_columns:

        daily_missingness[
            f"{column}_missing_percentage"
        ] = (
            df.groupby(
                "investigation_date",
                dropna=False,
            )[column].apply(
                lambda values: values.isna().mean() * 100
            )
        )

    daily_missingness = daily_missingness.reset_index()

    daily_missingness.to_csv(
        OUTPUT_DIR / "missingness_by_date.csv",
        index=False,
    )

    print("\n[MISSINGNESS BY DATE]")
    print("-" * 65)

    print(daily_missingness.head(10).to_string(index=False))

else:
    print(
        "\nDate-level analysis skipped: "
        "'date' is unavailable."
    )


# ============================================================
# 7. INVESTIGATE MISSINGNESS BY HOUR
# ============================================================

if "hour" in df.columns:

    hour_missingness = pd.DataFrame(
        index=sorted(df["hour"].dropna().unique())
    )

    for column in measurement_columns:

        hour_missingness[
            f"{column}_missing_percentage"
        ] = (
            df.groupby("hour")[column]
            .apply(lambda values: values.isna().mean() * 100)
        )

    hour_missingness.index.name = "hour"

    hour_missingness = hour_missingness.reset_index()

    hour_missingness.to_csv(
        OUTPUT_DIR / "missingness_by_hour.csv",
        index=False,
    )

    print("\n[MISSINGNESS BY HOUR]")
    print("-" * 65)

    print(hour_missingness.to_string(index=False))

else:
    print(
        "\nHour-level analysis skipped: "
        "'hour' is unavailable."
    )


# ============================================================
# 8. FLAG SUSPICIOUS MEASUREMENTS
# ============================================================
#
# These are screening rules, not universal physical laws.
# Every flagged record requires contextual investigation.
# ============================================================

anomaly_rules = {
    "traffic_volume": lambda s: s < 0,
    "vehicle_count": lambda s: s < 0,
    "avg_speed": lambda s: (s < 0) | (s > 200),
    "occupancy": lambda s: (s < 0) | (s > 100),
    "travel_time": lambda s: s < 0,
    "rainfall": lambda s: s < 0,
    "visibility": lambda s: s < 0,
}

anomaly_frames = []
anomaly_summary_rows = []

for column, rule in anomaly_rules.items():

    if column not in df.columns:
        continue

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    flags = rule(values).fillna(False)

    count = int(flags.sum())

    anomaly_summary_rows.append({
        "variable": column,
        "screening_rule": {
            "traffic_volume": "value < 0",
            "vehicle_count": "value < 0",
            "avg_speed": "value < 0 or value > 200",
            "occupancy": "value < 0 or value > 100",
            "travel_time": "value < 0",
            "rainfall": "value < 0",
            "visibility": "value < 0",
        }[column],
        "flagged_count": count,
        "percentage_of_all_records": (
            count / max(len(df), 1) * 100
        ),
    })

    if count == 0:
        continue

    flagged = df.loc[flags].copy()

    flagged["anomaly_variable"] = column
    flagged["anomaly_value"] = values.loc[flags].values

    flagged["anomaly_reason"] = {
        "traffic_volume": "Negative traffic volume",
        "vehicle_count": "Negative vehicle count",
        "avg_speed": "Speed outside screening range [0, 200]",
        "occupancy": "Occupancy outside screening range [0, 100]",
        "travel_time": "Negative travel time",
        "rainfall": "Negative rainfall",
        "visibility": "Negative visibility",
    }[column]

    anomaly_frames.append(flagged)

    print(
        f"Flagged {count:,} suspicious records "
        f"for {column}"
    )


if anomaly_summary_rows:

    anomaly_summary = pd.DataFrame(
        anomaly_summary_rows
    )

else:

    anomaly_summary = pd.DataFrame(
        columns=[
            "variable",
            "screening_rule",
            "flagged_count",
            "percentage_of_all_records",
        ]
    )


anomaly_summary.to_csv(
    OUTPUT_DIR / "anomaly_summary.csv",
    index=False,
)


if anomaly_frames:

    anomaly_records = pd.concat(
        anomaly_frames,
        ignore_index=True,
    )

else:

    anomaly_records = pd.DataFrame(
        columns=list(df.columns) + [
            "anomaly_variable",
            "anomaly_value",
            "anomaly_reason",
        ]
    )


anomaly_records.to_csv(
    OUTPUT_DIR / "flagged_anomaly_records.csv",
    index=False,
)

print("\n[ANOMALY SUMMARY]")
print("-" * 65)
print(anomaly_summary.to_string(index=False))


# ============================================================
# 9. INVESTIGATE SUSPICIOUS VALUES BY ROAD
# ============================================================

if not anomaly_records.empty and "road_id" in df.columns:

    anomaly_by_road = (
        anomaly_records.groupby(
            ["anomaly_variable", "road_id"],
            dropna=False,
        )
        .size()
        .reset_index(name="flagged_records")
    )

    anomaly_by_road.to_csv(
        OUTPUT_DIR / "anomalies_by_road.csv",
        index=False,
    )

    print("\n[ANOMALIES BY ROAD]")
    print("-" * 65)

    print(
        anomaly_by_road.sort_values(
            "flagged_records",
            ascending=False,
        ).head(15).to_string(index=False)
    )


# ============================================================
# 10. VISUALISE MISSINGNESS BY VARIABLE
# ============================================================

plot_data = missingness.loc[
    missingness["missing_percentage"] > 0
].sort_values("missing_percentage")

if not plot_data.empty:

    plt.figure(figsize=(12, max(6, len(plot_data) * 0.35)))

    plt.barh(
        plot_data["variable"],
        plot_data["missing_percentage"],
    )

    plt.xlabel("Missing values (%)")
    plt.ylabel("Variable")

    plt.title(
        "FlowCast | Missingness by Variable"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "missingness_by_variable.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

else:
    print("\nNo missing values found in the dataset.")


# ============================================================
# 11. VISUALISE MISSINGNESS BY ROAD
# ============================================================

road_plot_columns = [
    f"{column}_missing_percentage"
    for column in measurement_columns
]

if (
    "road_id" in df.columns
    and "road_missingness" in locals()
    and road_plot_columns
):

    available_plot_columns = [
        column
        for column in road_plot_columns
        if column in road_missingness.columns
    ]

    if available_plot_columns:

        heatmap_data = road_missingness.set_index(
            "road_id"
        )[available_plot_columns]

        plt.figure(
            figsize=(14, max(6, len(heatmap_data) * 0.35))
        )

        sns.heatmap(
            heatmap_data,
            annot=False,
            cmap="magma",
            linewidths=0.2,
            cbar_kws={
                "label": "Missing values (%)"
            },
        )

        plt.title(
            "FlowCast | Road-Level Missingness Heatmap"
        )

        plt.xlabel("Measurement variable")
        plt.ylabel("Road ID")

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "road_missingness_heatmap.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# 12. VISUALISE MISSINGNESS OVER TIME
# ============================================================

if (
    "investigation_date" in df.columns
    and "date" in df.columns
):

    date_plot = df.copy()

    date_plot["investigation_date"] = pd.to_datetime(
        date_plot["investigation_date"],
        errors="coerce",
    )

    date_plot = date_plot.dropna(
        subset=["investigation_date"]
    )

    if not date_plot.empty:

        daily_rates = (
            date_plot.groupby("investigation_date")[
                measurement_columns
            ]
            .apply(lambda group: group.isna().mean() * 100)
        )

        if not daily_rates.empty:

            plt.figure(figsize=(14, 6))

            for column in measurement_columns:

                if column in daily_rates.columns:

                    plt.plot(
                        daily_rates.index,
                        daily_rates[column],
                        label=column,
                        linewidth=1.5,
                    )

            plt.title(
                "FlowCast | Measurement Missingness Over Time"
            )

            plt.xlabel("Date")
            plt.ylabel("Missing values (%)")

            plt.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
            )

            plt.tight_layout()

            plt.savefig(
                FIGURE_DIR / "missingness_over_time.png",
                dpi=200,
                bbox_inches="tight",
            )

            plt.close()


# ============================================================
# 13. GENERATE A MACHINE-READABLE SUMMARY
# ============================================================

summary = {
    "project": "FlowCast",
    "investigation": "Day 2 — Missingness and Anomalies",
    "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
    "row_count": int(len(df)),
    "column_count": int(len(df.columns)),
    "numeric_column_count": int(len(numeric_columns)),
    "measurement_columns_investigated": measurement_columns,
    "variables_with_missing_values": int(
        (missing_count > 0).sum()
    ),
    "total_missing_cells": int(
        missing_count.sum()
    ),
    "total_anomaly_flags": int(
        anomaly_summary["flagged_count"].sum()
    ) if not anomaly_summary.empty else 0,
    "anomaly_screening_is_not_automatic_deletion": True,
}

with open(
    OUTPUT_DIR / "day2_summary.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=4,
    )


# ============================================================
# 14. COMPLETION STATUS
# ============================================================

print("\n" + "=" * 65)
print("DAY 2 INVESTIGATION FINISHED")
print("=" * 65)

print("\nSummary:")
print(json.dumps(summary, indent=4))

print("\nGenerated output directory:")
print(OUTPUT_DIR)

print("\nGenerated figures directory:")
print(FIGURE_DIR)

print(
    "\nNext step: inspect the reports and investigate "
    "the most important patterns."
)

