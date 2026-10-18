from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

traffic_path = INTERIM / "traffic_quality_checked.csv"
weather_path = RAW / "weather_observations.csv"
calendar_path = RAW / "calendar_events.csv"

traffic = pd.read_csv(traffic_path, low_memory=False)
weather = pd.read_csv(weather_path, low_memory=False)
calendar = pd.read_csv(calendar_path, low_memory=False)

# Parse traffic timestamps.
traffic["date"] = pd.to_datetime(
    traffic["date"], format="%Y-%m-%d", errors="raise"
)
traffic["time"] = pd.to_datetime(
    traffic["time"], format="%H:%M", errors="raise"
).dt.strftime("%H:%M")

traffic["hour"] = traffic["time"].str.slice(0, 2).astype(int)

# Parse weather timestamps.
weather["weather_date"] = pd.to_datetime(
    weather["date"], format="%d/%m/%Y", errors="raise"
)
weather["weather_time"] = pd.to_datetime(
    weather["time"], format="%H:%M", errors="raise"
).dt.strftime("%H:%M")

weather["hour"] = weather["weather_time"].str.slice(0, 2).astype(int)

weather["weather_condition"] = (
    weather["weather_condition"]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace({
        "rainy": "rain",
        "foggy": "fog",
    })
)

# Verify that each station has at most one weather row per date/hour.
weather_keys = ["station_id", "weather_date", "hour"]
duplicate_weather_hours = int(weather.duplicated(weather_keys).sum())

if duplicate_weather_hours:
    raise ValueError(
        f"Found {duplicate_weather_hours} duplicate station/date/hour "
        "weather keys. Investigate before joining."
    )

# Keep only the required weather fields and rename join columns.
weather_for_join = weather[
    [
        "station_id",
        "weather_date",
        "hour",
        "weather_condition",
        "temperature",
        "rainfall",
        "visibility",
    ]
].copy()

# Preserve original traffic row count and match weather by station/date/hour.
traffic["date"] = pd.to_datetime(traffic["date"])
integrated = traffic.merge(
    weather_for_join,
    left_on=["weather_station_id", "date", "hour"],
    right_on=["station_id", "weather_date", "hour"],
    how="left",
    validate="many_to_one",
    indicator="_weather_match",
)

weather_unmatched = int((integrated["_weather_match"] == "left_only").sum())
integrated = integrated.drop(
    columns=["station_id", "weather_date", "_weather_match"]
)

# Parse calendar date and join one calendar record per date.
calendar["date"] = pd.to_datetime(
    calendar["date"], format="%Y-%m-%d", errors="raise"
)

if calendar["date"].duplicated().any():
    raise ValueError("Calendar contains duplicate dates; refusing to join.")

calendar["holiday_name"] = calendar["holiday_name"].fillna("")
calendar["event_name"] = calendar["event_name"].fillna("")

integrated = integrated.merge(
    calendar,
    on="date",
    how="left",
    validate="many_to_one",
    indicator="_calendar_match",
)

calendar_unmatched = int(
    (integrated["_calendar_match"] == "left_only").sum()
)
integrated = integrated.drop(columns=["_calendar_match"])

# Assert row preservation and required joins.
if len(integrated) != len(traffic):
    raise ValueError(
        f"Row count changed during joins: {len(traffic)} -> {len(integrated)}"
    )

if calendar_unmatched:
    raise ValueError(
        f"{calendar_unmatched} traffic rows have no matching calendar date."
    )

INTERIM.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

output_path = INTERIM / "flowcast_integrated.csv"
integrated.to_csv(output_path, index=False)

print("FLOWCAST — INTEGRATION RESULTS")
print("-" * 45)
print(f"Traffic input rows: {len(traffic):,}")
print(f"Integrated output rows: {len(integrated):,}")
print(f"Weather rows unmatched: {weather_unmatched:,}")
print(f"Calendar rows unmatched: {calendar_unmatched:,}")
print(f"Unique roads: {integrated['road_id'].nunique():,}")
print(f"Unique dates: {integrated['date'].nunique():,}")
print(f"Standardized weather categories:")
print(integrated["weather_condition"].value_counts(dropna=False).to_string())
print(f"\nIntegrated dataset saved to: {output_path}")
print("Original raw datasets were not modified.")
