from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "interim" / "traffic_quality_checked.csv"
REPORTS = ROOT / "reports"

df = pd.read_csv(INPUT, low_memory=False)

df["timestamp"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["time"].astype(str),
    errors="coerce"
)

valid = df.dropna(subset=["timestamp"]).copy()

start = valid["timestamp"].min()
end = valid["timestamp"].max()
expected_slots = pd.date_range(start, end, freq="30min")

print("=" * 55)
print("FLOWCAST — SENSOR TIMELINE AUDIT")
print("=" * 55)
print(f"Observed rows: {len(valid):,}")
print(f"Roads: {valid['road_id'].nunique():,}")
print(f"Observed start: {start}")
print(f"Observed end: {end}")
print(f"Expected slots per road over this range: {len(expected_slots):,}")

summary = []

for road_id, group in valid.groupby("road_id"):
    observed = pd.DatetimeIndex(group["timestamp"].drop_duplicates())
    missing_slots = expected_slots.difference(observed)

    summary.append({
        "road_id": road_id,
        "observed_rows": len(group),
        "unique_timestamps": len(observed),
        "missing_time_slots": len(missing_slots),
        "duplicate_timestamps": int(group["timestamp"].duplicated().sum()),
    })

audit = pd.DataFrame(summary).sort_values("road_id")

print("\nPer-road results:")
print(audit.to_string(index=False))

print("\nOverall:")
print(f"Expected road-time observations: {len(expected_slots) * len(audit):,}")
print(f"Actual observations: {len(valid):,}")
print(f"Missing road-time slots: {int(audit['missing_time_slots'].sum()):,}")
print(f"Duplicate timestamps within roads: {int(audit['duplicate_timestamps'].sum()):,}")

REPORTS.mkdir(parents=True, exist_ok=True)
output = REPORTS / "sensor_timeline_audit.csv"
audit.to_csv(output, index=False)

print(f"\nAudit saved to: {output}")
