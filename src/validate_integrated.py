from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "interim" / "flowcast_integrated.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, low_memory=False)

required_columns = [
    "road_id", "date", "time", "traffic_volume",
    "avg_speed", "occupancy", "weather_condition",
    "temperature", "rainfall", "visibility",
    "public_holiday", "event_flag", "roadwork_flag",
]

missing_columns = sorted(set(required_columns) - set(df.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

duplicate_keys = int(df.duplicated(["road_id", "date", "time"]).sum())
if duplicate_keys:
    raise ValueError(f"Found {duplicate_keys} duplicate traffic keys.")

if df["date"].isna().any():
    raise ValueError("Missing dates detected.")

if df[["public_holiday", "event_flag", "roadwork_flag"]].isna().any().any():
    raise ValueError("Missing calendar flags detected.")

if len(df) != 176701:
    print(f"WARNING: Expected 176,701 rows, found {len(df):,}")

print("=" * 55)
print("FLOWCAST — FINAL DAY 1 VALIDATION")
print("=" * 55)

print(f"Integrated rows: {len(df):,}")
print(f"Roads: {df['road_id'].nunique():,}")
print(f"Dates: {df['date'].nunique():,}")
print(f"Duplicate traffic keys: {duplicate_keys:,}")
print(f"Missing required columns: {len(missing_columns)}")

print("\nMissing values in important columns:")
important = [
    "traffic_volume", "avg_speed", "occupancy",
    "temperature", "rainfall", "visibility",
    "weather_condition",
]
for col in important:
    print(f"{col}: {df[col].isna().sum():,}")

print("\nTraffic anomaly flags:")
for col in ["speed_above_200_flag", "occupancy_above_100_flag"]:
    if col in df.columns:
        print(f"{col}: {int(df[col].fillna(False).sum()):,}")

print("\nCalendar flag counts:")
for col in ["public_holiday", "event_flag", "roadwork_flag"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).to_string())

report = f"""# FlowCast — Day 1 Data Engineering Report

## Dataset integration
- Integrated rows: {len(df):,}
- Unique roads: {df['road_id'].nunique():,}
- Unique dates: {df['date'].nunique():,}
- Duplicate road/date/time keys: {duplicate_keys:,}
- Missing required columns: {len(missing_columns)}

## Important missing values
"""

for col in important:
    report += f"- {col}: {int(df[col].isna().sum()):,}\\n"

report += """
## Completed work
- Loaded and structurally validated the three raw datasets.
- Investigated and removed exact duplicate traffic records.
- Audited traffic timestamps, missing observations, and numeric anomalies.
- Created a traffic quality-checked copy.
- Audited weather and calendar records.
- Standardized weather labels.
- Joined traffic to weather by station, date, and hour.
- Joined traffic to calendar by date.
- Preserved traffic rows during integration.

## Assumptions and limitations
- Each hourly weather reading represents both half-hour traffic windows
  within the same hour.
- Missing measurements have not been generally imputed.
- Suspicious speed and occupancy measurements remain flagged for review.
- Missing traffic time slots have been audited but not yet reindexed.
- No forecasting features or ML models have been built yet.
- Model evaluation must use chronological splits and leakage-safe features.
"""

report_path = REPORTS / "day1_data_engineering_report.md"
report_path.write_text(report, encoding="utf-8")

print(f"\nDay 1 report saved to: {report_path}")
print("Validation checks completed.")
