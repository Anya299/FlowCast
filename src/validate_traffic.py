from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "interim" / "traffic_deduplicated.csv"
REPORTS = ROOT / "reports"

df = pd.read_csv(DATA, low_memory=False)
REPORTS.mkdir(parents=True, exist_ok=True)

# Parse timestamps without changing the source CSV.
df["parsed_date"] = pd.to_datetime(
    df["date"], format="%Y-%m-%d", errors="coerce"
)
df["parsed_time"] = pd.to_datetime(
    df["time"], format="%H:%M", errors="coerce"
)

print("=" * 55)
print("FLOWCAST — TRAFFIC DATA VALIDATION")
print("=" * 55)

print("\n1. DATASET")
print(f"Rows: {len(df):,}")
print(f"Roads: {df['road_id'].nunique()}")
print(f"Date range: {df['parsed_date'].min()} to {df['parsed_date'].max()}")
print(f"Invalid dates: {df['parsed_date'].isna().sum():,}")
print(f"Invalid times: {df['parsed_time'].isna().sum():,}")

print("\n2. TIME INTERVALS")
valid_times = df.loc[df["parsed_time"].notna(), "time"]
print("Most common time values:")
print(valid_times.value_counts().head(10).to_string())

print("\n3. MISSING VALUES")
for col in [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "travel_time",
    "accident_count",
    "road_capacity",
]:
    print(f"{col}: {df[col].isna().sum():,}")

print("\n4. NUMERIC RANGES")
for col in [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "travel_time",
    "accident_count",
    "road_capacity",
]:
    values = pd.to_numeric(df[col], errors="coerce")
    print(
        f"{col}: min={values.min()}, "
        f"median={values.median()}, max={values.max()}"
    )

print("\n5. BASIC ANOMALY COUNTS")
checks = {
    "Negative traffic volume": pd.to_numeric(
        df["traffic_volume"], errors="coerce"
    ) < 0,
    "Non-positive speed": pd.to_numeric(
        df["avg_speed"], errors="coerce"
    ) <= 0,
    "Speed above 200": pd.to_numeric(
        df["avg_speed"], errors="coerce"
    ) > 200,
    "Occupancy below 0": pd.to_numeric(
        df["occupancy"], errors="coerce"
    ) < 0,
    "Occupancy above 100": pd.to_numeric(
        df["occupancy"], errors="coerce"
    ) > 100,
    "Negative accident count": pd.to_numeric(
        df["accident_count"], errors="coerce"
    ) < 0,
    "Non-positive road capacity": pd.to_numeric(
        df["road_capacity"], errors="coerce"
    ) <= 0,
}

for label, mask in checks.items():
    print(f"{label}: {int(mask.fillna(False).sum()):,}")

print("\nValidation complete. No source data was modified.")
