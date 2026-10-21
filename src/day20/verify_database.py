"""Read-only verification of the FlowCast PostgreSQL database schema."""

from __future__ import annotations

import sys

from sqlalchemy import DateTime, Float, String, inspect, text
from sqlalchemy.engine import Engine

from src.day20.database import build_engine_from_environment


TABLE_NAME = "traffic_observations"

EXPECTED_COLUMNS = {
    "road_id",
    "timestamp",
    "traffic_volume",
    "created_at",
}

EXPECTED_PRIMARY_KEY = ["road_id", "timestamp"]

EXPECTED_CHECK_CONSTRAINTS = {
    "ck_traffic_observations_road_id_nonempty",
    "ck_traffic_observations_volume_nonnegative",
}


def verify_schema(engine: Engine) -> list[str]:
    """Return schema verification failures without modifying the database."""
    failures: list[str] = []
    inspector = inspect(engine)

    if not inspector.has_table(TABLE_NAME):
        return [f"Required table '{TABLE_NAME}' does not exist."]

    columns = {
        column["name"]: column
        for column in inspector.get_columns(TABLE_NAME)
    }

    missing_columns = EXPECTED_COLUMNS - set(columns)
    if missing_columns:
        failures.append(
            f"Missing required columns: {', '.join(sorted(missing_columns))}"
        )
        return failures

    for name in sorted(EXPECTED_COLUMNS):
        if columns[name]["nullable"]:
            failures.append(f"Column '{name}' must be NOT NULL.")

    road_id_type = columns["road_id"]["type"]
    if not isinstance(road_id_type, String) or road_id_type.length != 100:
        failures.append("Column 'road_id' must be VARCHAR(100).")

    timestamp_type = columns["timestamp"]["type"]
    if (
        not isinstance(timestamp_type, DateTime)
        or timestamp_type.timezone
    ):
        failures.append(
            "Column 'timestamp' must be TIMESTAMP WITHOUT TIME ZONE."
        )

    volume_type = columns["traffic_volume"]["type"]
    if not isinstance(volume_type, Float):
        failures.append("Column 'traffic_volume' must be a floating-point type.")

    created_at_type = columns["created_at"]["type"]
    if (
        not isinstance(created_at_type, DateTime)
        or not created_at_type.timezone
    ):
        failures.append(
            "Column 'created_at' must be TIMESTAMP WITH TIME ZONE."
        )

    primary_key = inspector.get_pk_constraint(TABLE_NAME)
    actual_primary_key = primary_key.get("constrained_columns") or []

    if actual_primary_key != EXPECTED_PRIMARY_KEY:
        failures.append(
            "Primary key must consist of (road_id, timestamp), "
            f"in that order; found {actual_primary_key}."
        )

    checks = inspector.get_check_constraints(TABLE_NAME)
    actual_check_names = {
        check.get("name")
        for check in checks
        if check.get("name")
    }

    missing_checks = EXPECTED_CHECK_CONSTRAINTS - actual_check_names
    if missing_checks:
        failures.append(
            "Missing check constraints: "
            + ", ".join(sorted(missing_checks))
        )

    check_expressions = {
        check.get("name"): (check.get("sqltext") or "").lower()
        for check in checks
    }

    road_check = check_expressions.get(
        "ck_traffic_observations_road_id_nonempty", ""
    )
    if road_check and not all(
        token in road_check
        for token in ("road_id", "trim", "length", ">")
    ):
        failures.append(
            "The road_id check constraint does not appear to enforce "
            "a nonempty, trimmed road ID."
        )

    volume_check = check_expressions.get(
        "ck_traffic_observations_volume_nonnegative", ""
    )
    if volume_check and not all(
        token in volume_check
        for token in ("traffic_volume", ">=", "0")
    ):
        failures.append(
            "The traffic_volume check constraint does not appear to "
            "enforce nonnegative values."
        )

    return failures


def main() -> int:
    """Run read-only connection and schema checks."""
    engine = None

    try:
        engine = build_engine_from_environment()

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            database_name = connection.execute(
                text("SELECT current_database()")
            ).scalar_one()
            server_version = connection.execute(
                text("SHOW server_version")
            ).scalar_one()

        print("PASS: PostgreSQL connection established.")
        print(f"PASS: Database: {database_name}")
        print(f"PASS: PostgreSQL server version: {server_version}")

        failures = verify_schema(engine)

        if failures:
            print("\nFAIL: Database schema verification failed.")
            for failure in failures:
                print(f"  - {failure}")
            return 1

        print(f"PASS: Required table '{TABLE_NAME}' exists.")
        print("PASS: Required columns, data types, and nullability.")
        print("PASS: Composite primary key (road_id, timestamp).")
        print("PASS: Required check constraints are present.")
        print("\nRESULT: All database verification checks passed.")
        return 0

    except Exception as exc:
        print(f"\nFAIL: Database verification could not complete: {exc}")
        return 1

    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    sys.exit(main())