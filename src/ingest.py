from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

FILES = {
    "traffic": "traffic_sensor_log.csv",
    "weather": "weather_observations.csv",
    "calendar": "calendar_events.csv",
}

REQUIRED_COLUMNS = {
    "traffic": {
        "road_id", "date", "time", "traffic_volume",
        "avg_speed", "occupancy", "road_capacity",
        "weather_station_id", "congestion_level",
        "accident_count",
    },
    "weather": {
        "station_id", "date", "time", "weather_condition",
        "temperature", "rainfall", "visibility",
    },
    "calendar": {
        "date", "public_holiday", "event_flag",
        "roadwork_flag",
    },
}


def load_and_validate():
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = ["# FlowCast — Initial Data Quality Report", ""]

    datasets = {}

    for name, filename in FILES.items():
        path = RAW / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")

        df = pd.read_csv(path, low_memory=False)
        datasets[name] = df

        missing_columns = REQUIRED_COLUMNS[name] - set(df.columns)

        report.extend([
            f"## {name.title()} dataset",
            "",
            f"- File: `{filename}`",
            f"- Rows: {len(df):,}",
            f"- Columns: {len(df.columns)}",
            f"- Duplicate complete rows: {int(df.duplicated().sum()):,}",
            f"- Missing cells: {int(df.isna().sum().sum()):,}",
            f"- Missing required columns: "
            f"{sorted(missing_columns) if missing_columns else 'None'}",
            "",
            "### Missing values by column",
            "",
        ])

        for column, count in df.isna().sum().items():
            if count:
                report.append(f"- `{column}`: {int(count):,}")

        if not df.isna().any().any():
            report.append("- No pandas-detected missing values.")

        report.append("")

        if missing_columns:
            raise ValueError(
                f"{name} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if df.empty:
            raise ValueError(f"{name} dataset is empty.")

    traffic = datasets["traffic"]
    weather = datasets["weather"]
    calendar = datasets["calendar"]

    report.extend([
        "## Initial consistency checks",
        "",
        f"- Unique road IDs: {traffic['road_id'].nunique():,}",
        f"- Unique traffic dates: {traffic['date'].nunique():,}",
        f"- Unique weather stations: {weather['station_id'].nunique():,}",
        f"- Calendar date rows: {calendar['date'].nunique():,}",
        f"- Duplicate traffic keys "
        f"(road_id + date + time): "
        f"{int(traffic.duplicated(['road_id', 'date', 'time']).sum()):,}",
        "",
        "### Important limitations",
        "",
        "- Missing values, duplicates, and outliers have not yet been repaired.",
        "- Date parsing and weather/calendar joins are not yet performed.",
        "- This report describes raw data; it does not certify modelling readiness.",
        "",
    ])

    report_path = REPORTS / "data_quality_initial.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nFlowCast ingestion checks completed.")
    for name, df in datasets.items():
        print(f"{name.title():<10} {len(df):>10,} rows")

    print(f"\nInitial report saved to: {report_path}")
    return datasets


if __name__ == "__main__":
    load_and_validate()
