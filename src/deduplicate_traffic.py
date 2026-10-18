from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "traffic_sensor_log.csv"
OUT_DIR = ROOT / "data" / "interim"
REPORT_DIR = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW, low_memory=False)

original_rows = len(df)
exact_duplicates = int(df.duplicated().sum())

cleaned = df.drop_duplicates().copy()

remaining_exact_duplicates = int(cleaned.duplicated().sum())
remaining_duplicate_keys = int(
    cleaned.duplicated(["road_id", "date", "time"]).sum()
)

output_path = OUT_DIR / "traffic_deduplicated.csv"
cleaned.to_csv(output_path, index=False)

report = f"""# FlowCast — Traffic Deduplication Report

## Results

- Original rows: {original_rows:,}
- Exact duplicate rows removed: {exact_duplicates:,}
- Rows after deduplication: {len(cleaned):,}
- Exact duplicate rows remaining: {remaining_exact_duplicates:,}
- Duplicate road/date/time keys remaining: {remaining_duplicate_keys:,}

## Method

Removed exact duplicate rows by comparing all columns using pandas
`drop_duplicates()`. Original source data was not modified.

## Limitations

This step does not repair missing values, validate outliers, parse dates,
or join weather/calendar data. Duplicate keys with non-identical records,
if any, require separate investigation.
"""

report_path = REPORT_DIR / "traffic_deduplication_report.md"
report_path.write_text(report, encoding="utf-8")

print(report)
print(f"\nCleaned copy saved to: {output_path}")
print(f"Report saved to: {report_path}")
