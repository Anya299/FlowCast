from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "interim" / "traffic_deduplicated.csv"
OUTPUT_DIR = ROOT / "data" / "interim"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, low_memory=False)
original_rows = len(df)

numeric_columns = [
    "traffic_volume",
    "vehicle_count",
    "avg_speed",
    "occupancy",
    "travel_time",
    "accident_count",
    "signal_timing",
    "road_capacity",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Record invalid values before applying documented rules.
negative_volume = int((df["traffic_volume"] < 0).sum())
speed_above_200 = int((df["avg_speed"] > 200).sum())
occupancy_above_100 = int((df["occupancy"] > 100).sum())

# Preserve suspicious speed/occupancy observations for investigation.
df["speed_above_200_flag"] = df["avg_speed"] > 200
df["occupancy_above_100_flag"] = df["occupancy"] > 100

# Negative traffic volume cannot represent a valid traffic count.
df.loc[df["traffic_volume"] < 0, "traffic_volume"] = pd.NA

# Do not silently replace missing measurements or delete outliers.
output_path = OUTPUT_DIR / "traffic_quality_checked.csv"
df.to_csv(output_path, index=False)

report = f"""# FlowCast — Traffic Quality-Check Report

## Dataset
- Input rows: {original_rows:,}
- Output rows: {len(df):,}
- Rows removed: {original_rows - len(df):,}

## Rules applied
- Converted selected measurement columns to numeric values.
- Replaced negative traffic-volume values with missing values.
- Added a flag for speed values above 200.
- Added a flag for occupancy values above 100.
- Preserved all rows.
- Preserved suspicious speed and occupancy values for investigation.
- Did not impute missing values.

## Counts recorded before correction
- Negative traffic-volume values: {negative_volume:,}
- Speed values above 200: {speed_above_200:,}
- Occupancy values above 100: {occupancy_above_100:,}

## Missing values after quality checks
"""

for column in [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "travel_time",
    "accident_count",
]:
    report += f"- {column}: {int(df[column].isna().sum()):,}\\n"

report += """
## Limitations
This is an initial quality-check stage, not a final modelling dataset.
Outlier flags require further investigation. Missing-value treatment,
feature engineering, temporal alignment, and train/validation/test
splitting remain outstanding.
"""

report_path = REPORT_DIR / "traffic_quality_check_report.md"
report_path.write_text(report, encoding="utf-8")

print("FlowCast quality checks completed.")
print(f"Input rows: {original_rows:,}")
print(f"Output rows: {len(df):,}")
print(f"Negative volumes converted to missing: {negative_volume:,}")
print(f"Speed-above-200 flags: {speed_above_200:,}")
print(f"Occupancy-above-100 flags: {occupancy_above_100:,}")
print(f"Output: {output_path}")
print(f"Report: {report_path}")
