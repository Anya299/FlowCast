"""Unit tests for the FlowCast Day 4 data contract."""

import pandas as pd

from src.day4_data_contract import inspect_dataframe


def make_test_data():
    """Return a small valid dataset for controlled tests."""

    return pd.DataFrame(
        {
            "road_id": ["NL-001", "NL-002", "NL-003"],
            "road_name": ["North Road", "Central Road", "South Road"],
            "latitude": [13.0, 13.1, 13.2],
            "longitude": [77.0, 77.1, 77.2],
            "weather_station_id": [
                "WS-NORTH",
                "WS-CENTRAL",
                "WS-SOUTH",
            ],
            "date": [
                "2025-01-01",
                "2025-01-01",
                "2025-01-01",
            ],
            "time": ["00:00", "00:30", "01:00"],
            "traffic_volume": [100, 120, 130],
            "vehicle_count": [100, 120, 130],
            "vehicle_type_dist": ["{}", "{}", "{}"],
            "avg_speed": [40.0, 35.0, 30.0],
            "occupancy": [25.0, 40.0, 50.0],
            "congestion_level": [
                "Free-flow",
                "Moderate",
                "Heavy",
            ],
            "travel_time": [4.0, 5.0, 6.0],
            "accident_count": [0, 0, 1],
            "signal_timing": [30, 30, 35],
            "road_capacity": [1000, 1000, 1200],
            "speed_above_200_flag": [0, 0, 0],
            "occupancy_above_100_flag": [0, 0, 0],
            "hour": [0, 0, 1],
            "weather_condition": ["Clear", "Rain", "Cloudy"],
            "temperature": [22.0, 21.0, 20.0],
            "rainfall": [0.0, 1.0, 0.0],
            "visibility": [1000.0, 700.0, 1200.0],
            "public_holiday": [0, 0, 0],
            "holiday_name": [None, None, None],
            "event_flag": [0, 0, 0],
            "event_name": [None, None, None],
            "roadwork_flag": [0, 0, 0],
        }
    )


def test_valid_data_is_preserved():
    """Valid rows should remain in the output."""

    df = make_test_data()

    validated, issues, missingness, duplicates, summary = (
        inspect_dataframe(df)
    )

    assert len(validated) == len(df)
    assert summary["input_rows"] == len(df)
    assert summary["all_input_rows_preserved"] is True
    assert (validated["quality_status"] == "VALID").all()


def test_negative_traffic_volume_is_quarantined():
    """A negative traffic count should create a blocking finding."""

    df = make_test_data()

    df.loc[0, "traffic_volume"] = -10

    validated, issues, missingness, duplicates, summary = (
        inspect_dataframe(df)
    )

    row = validated.loc[
        validated["road_id"] == "NL-001"
    ].iloc[0]

    assert row["quality_status"] == "QUARANTINE"

    assert (
        (issues["issue_type"] == "negative_measurement")
        & (issues["column"] == "traffic_volume")
    ).any()


def test_suspicious_speed_is_flagged_for_review():
    """A speed over 200 should require review."""

    df = make_test_data()

    df.loc[0, "avg_speed"] = 240

    validated, issues, missingness, duplicates, summary = (
        inspect_dataframe(df)
    )

    row = validated.loc[
        validated["road_id"] == "NL-001"
    ].iloc[0]

    assert row["quality_status"] == "REVIEW"

    assert (
        (issues["issue_type"] == "outside_screening_range")
        & (issues["column"] == "avg_speed")
    ).any()


def test_duplicate_road_time_keys_are_detected():
    """Repeated road-time keys should be reported."""

    df = make_test_data()

    df.loc[1, "road_id"] = df.loc[0, "road_id"]
    df.loc[1, "date"] = df.loc[0, "date"]
    df.loc[1, "time"] = df.loc[0, "time"]

    validated, issues, missingness, duplicates, summary = (
        inspect_dataframe(df)
    )

    assert len(duplicates) == 2

    assert (
        issues["issue_type"] == "duplicate_road_time_key"
    ).sum() == 2


def test_missing_optional_measurement_is_reported():
    """A missing target should be reported without automatic deletion."""

    df = make_test_data()

    df.loc[0, "traffic_volume"] = None

    validated, issues, missingness, duplicates, summary = (
        inspect_dataframe(df)
    )

    assert len(validated) == len(df)

    traffic_row = missingness.loc[
        missingness["column"] == "traffic_volume"
    ].iloc[0]

    assert traffic_row["missing_count"] == 1

    # A missing target is not automatically treated as
    # a blocking row-level error by this contract.
    assert (
        validated.loc[0, "quality_status"] == "VALID"
    )