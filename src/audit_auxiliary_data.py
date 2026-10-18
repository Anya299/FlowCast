from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

weather = pd.read_csv(
    RAW / "weather_observations.csv",
    low_memory=False
)
calendar = pd.read_csv(
    RAW / "calendar_events.csv",
    low_memory=False
)

print("=" * 60)
print("FLOWCAST — WEATHER AND CALENDAR AUDIT")
print("=" * 60)

print("\n1. WEATHER DATASET")
print(f"Rows: {len(weather):,}")
print(f"Columns: {list(weather.columns)}")
print(f"Duplicate complete rows: {weather.duplicated().sum():,}")

weather["parsed_date"] = pd.to_datetime(
    weather["date"].astype(str),
    format="%d/%m/%Y",
    errors="coerce"
)
weather["parsed_time"] = pd.to_datetime(
    weather["time"].astype(str),
    format="%H:%M",
    errors="coerce"
)

print(f"Invalid dates: {weather['parsed_date'].isna().sum():,}")
print(f"Invalid times: {weather['parsed_time'].isna().sum():,}")
print(f"Unique weather stations: {weather['station_id'].nunique():,}")
print(
    f"Date range: {weather['parsed_date'].min()} "
    f"to {weather['parsed_date'].max()}"
)

print("\nWeather missing values:")
print(weather.isna().sum().to_string())

print("\nWeather condition values:")
print(
    weather["weather_condition"]
    .astype("string")
    .str.strip()
    .str.lower()
    .value_counts(dropna=False)
    .to_string()
)

print("\nDuplicate weather station/date/time keys:")
weather_key_duplicates = weather.duplicated(
    ["station_id", "date", "time"]
).sum()
print(f"{weather_key_duplicates:,}")

print("\n2. CALENDAR DATASET")
print(f"Rows: {len(calendar):,}")
print(f"Columns: {list(calendar.columns)}")
print(f"Duplicate complete rows: {calendar.duplicated().sum():,}")

calendar["parsed_date"] = pd.to_datetime(
    calendar["date"].astype(str),
    format="%Y-%m-%d",
    errors="coerce"
)

print(f"Invalid dates: {calendar['parsed_date'].isna().sum():,}")
print(f"Unique dates: {calendar['parsed_date'].nunique():,}")
print(
    f"Date range: {calendar['parsed_date'].min()} "
    f"to {calendar['parsed_date'].max()}"
)

print("\nCalendar missing values:")
print(calendar.isna().sum().to_string())

for col in ["public_holiday", "event_flag", "roadwork_flag"]:
    if col in calendar.columns:
        print(f"\n{col} values:")
        print(calendar[col].value_counts(dropna=False).to_string())

print("\nCalendar duplicate dates:")
print(f"{calendar['parsed_date'].duplicated().sum():,}")

print("\nAudit finished. Source datasets were not modified.")
