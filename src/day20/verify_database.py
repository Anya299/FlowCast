"""Read-only database verification for FlowCast Day 20."""

from sqlalchemy import inspect, text

from src.day20.database import build_engine_from_environment


def main() -> None:
    engine = build_engine_from_environment()

    try:
        with engine.connect() as connection:
            database_name = connection.execute(
                text("SELECT current_database()")
            ).scalar_one()

            server_version = connection.execute(
                text("SHOW server_version")
            ).scalar_one()

            print("PostgreSQL connection: OK")
            print("Database:", database_name)
            print("Server version:", server_version)

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        print("\nTables:")
        for table in sorted(tables):
            print("-", table)

        if "traffic_observations" not in tables:
            raise RuntimeError(
                "traffic_observations is missing. Initialize the schema first."
            )

        columns = inspector.get_columns("traffic_observations")

        print("\ntraffic_observations columns:")
        for column in columns:
            print(f"- {column['name']}: {column['type']}")

        primary_key = inspector.get_pk_constraint("traffic_observations")
        print("\nPrimary key:", primary_key["constrained_columns"])

        checks = inspector.get_check_constraints("traffic_observations")
        print("Check constraints:")
        for check in checks:
            print("-", check["name"], ":", check["sqltext"])

        print("\nDatabase verification passed.")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
