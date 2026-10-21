"""Unit tests for the read-only database schema verifier."""

from sqlalchemy import DateTime, Float, String

from src.day20 import verify_database


class FakeInspector:
    """Simulate SQLAlchemy schema inspection without a database."""

    def __init__(self):
        self.columns = [
            {
                "name": "road_id",
                "type": String(100),
                "nullable": False,
            },
            {
                "name": "timestamp",
                "type": DateTime(timezone=False),
                "nullable": False,
            },
            {
                "name": "traffic_volume",
                "type": Float(),
                "nullable": False,
            },
            {
                "name": "created_at",
                "type": DateTime(timezone=True),
                "nullable": False,
            },
        ]

        self.primary_key = {
            "constrained_columns": ["road_id", "timestamp"],
        }

        self.checks = [
            {
                "name": "ck_traffic_observations_road_id_nonempty",
                "sqltext": "length(TRIM(BOTH FROM road_id)) > 0",
            },
            {
                "name": "ck_traffic_observations_volume_nonnegative",
                "sqltext": "traffic_volume >= 0::double precision",
            },
        ]

    def has_table(self, table_name):
        return table_name == "traffic_observations"

    def get_columns(self, table_name):
        return self.columns

    def get_pk_constraint(self, table_name):
        return self.primary_key

    def get_check_constraints(self, table_name):
        return self.checks


def install_fake_inspector(monkeypatch, inspector):
    """Replace real database inspection with a deterministic test double."""
    monkeypatch.setattr(
        verify_database,
        "inspect",
        lambda engine: inspector,
    )


def test_verify_schema_accepts_valid_schema(monkeypatch):
    inspector = FakeInspector()
    install_fake_inspector(monkeypatch, inspector)

    failures = verify_database.verify_schema(engine=object())

    assert failures == []


def test_verify_schema_detects_missing_column(monkeypatch):
    inspector = FakeInspector()
    inspector.columns = [
        column
        for column in inspector.columns
        if column["name"] != "traffic_volume"
    ]
    install_fake_inspector(monkeypatch, inspector)

    failures = verify_database.verify_schema(engine=object())

    assert any("traffic_volume" in failure for failure in failures)
    assert any("Missing required columns" in failure for failure in failures)


def test_verify_schema_detects_missing_check_constraint(monkeypatch):
    inspector = FakeInspector()
    inspector.checks = [
        check
        for check in inspector.checks
        if check["name"]
        != "ck_traffic_observations_volume_nonnegative"
    ]
    install_fake_inspector(monkeypatch, inspector)

    failures = verify_database.verify_schema(engine=object())

    assert any(
        "ck_traffic_observations_volume_nonnegative" in failure
        for failure in failures
    )


def test_verify_schema_detects_incorrect_primary_key(monkeypatch):
    inspector = FakeInspector()
    inspector.primary_key = {
        "constrained_columns": ["road_id"],
    }
    install_fake_inspector(monkeypatch, inspector)

    failures = verify_database.verify_schema(engine=object())

    assert any("Primary key" in failure for failure in failures)


def test_verify_schema_detects_nullable_column(monkeypatch):
    inspector = FakeInspector()
    inspector.columns[0]["nullable"] = True
    install_fake_inspector(monkeypatch, inspector)

    failures = verify_database.verify_schema(engine=object())

    assert any("road_id" in failure and "NOT NULL" in failure
               for failure in failures)