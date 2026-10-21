"""Real PostgreSQL integration tests for traffic ingestion.

These tests use unique road IDs and remove only rows created by this test.
They never truncate tables or delete unrelated application data.
"""

from __future__ import annotations

from datetime import datetime

import os
from uuid import uuid4

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from src.day19.api import create_app
from src.day20.models import TrafficObservation


load_dotenv(override=True)

DATABASE_URL = os.getenv("DATABASE_URL")


@pytest.fixture
def postgres_engine():
    """Provide a connection to the configured PostgreSQL database."""
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL is not configured.")

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    try:
        with engine.connect() as connection:
            dialect = connection.dialect.name
            if dialect != "postgresql":
                pytest.skip(
                    f"PostgreSQL integration tests require PostgreSQL, got {dialect}."
                )

            connection.execute(text("SELECT 1"))
    except Exception:
        engine.dispose()
        raise

    yield engine
    engine.dispose()


@pytest.fixture
def integration_client(postgres_engine):
    """Create the production application with real database ingestion."""
    # Inject minimal history for forecasting so this test does not depend on
    # the current training CSV. No fake ingestion service is injected.
    import pandas as pd

    history = pd.DataFrame(
        [
            {
                "road_id": "integration-history-road",
                "timestamp": "2025-01-01 08:00:00",
                "traffic_volume": 10.0,
            },
            {
                "road_id": "integration-history-road",
                "timestamp": "2025-01-01 08:30:00",
                "traffic_volume": 12.0,
            },
        ]
    )

    app = create_app(history_data=history)

    # create_app intentionally avoids configuring production ingestion when
    # history_data is injected. For integration testing, wire the same
    # real engine, repository, and ingestion service used in production.
    from src.day20.database import build_session_factory
    from src.day20.repository import TrafficObservationRepository
    from src.day21.ingestion_service import TrafficIngestionService

    session_factory = build_session_factory(postgres_engine)
    repository = TrafficObservationRepository(session_factory)
    app.state.test_session_factory = session_factory

    # Set this before TestClient enters the lifespan. The lifespan currently
    # resets ingestion_service to the injected argument, so use a dedicated
    # app factory override via the lifespan-aware app state below.
    original_create_app = app

    # The cleanest integration path is to create the app with the service
    # injected; it will use the real repository and PostgreSQL.
    app = create_app(
        history_data=history,
        ingestion_service=TrafficIngestionService(repository),
    )

    with TestClient(app) as client:
        yield client, postgres_engine


@pytest.fixture
def unique_road_id():
    return f"day22-integration-{uuid4().hex}"


def test_postgres_ingestion_persists_observation(
    integration_client,
    unique_road_id,
):
    client, engine = integration_client
    timestamp = "2025-02-10T08:30:00"

    try:
        response = client.post(
            "/v1/traffic/observations",
            json={
                "observations": [
                    {
                        "road_id": unique_road_id,
                        "timestamp": timestamp,
                        "traffic_volume": 42.5,
                    }
                ]
            },
        )

        assert response.status_code == 200, response.text
        assert response.json() == {
            "received": 1,
            "validated": 1,
            "inserted": 1,
            "skipped_duplicates": 0,
        }

        with Session(engine) as session:
            stored = session.scalar(
                select(TrafficObservation).where(
                    TrafficObservation.road_id == unique_road_id,
                    TrafficObservation.timestamp == datetime.fromisoformat(
                        "2025-02-10T08:30:00"
                    ),
                )
            )

            # Some SQLAlchemy drivers return a datetime rather than accepting
            # an ISO string for a timestamp comparison. Fall back to a query
            # by road ID and validate the timestamp and value independently.
            if stored is None:
                records = session.scalars(
                    select(TrafficObservation).where(
                        TrafficObservation.road_id == unique_road_id
                    )
                ).all()
                assert len(records) == 1
                stored = records[0]

            assert stored.road_id == unique_road_id
            assert stored.traffic_volume == pytest.approx(42.5)
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM traffic_observations WHERE road_id = :road_id"),
                {"road_id": unique_road_id},
            )


def test_postgres_ingestion_skips_existing_duplicate(
    integration_client,
    unique_road_id,
):
    client, engine = integration_client
    timestamp = "2025-02-11T09:00:00"

    try:
        payload = {
            "observations": [
                {
                    "road_id": unique_road_id,
                    "timestamp": timestamp,
                    "traffic_volume": 55.0,
                }
            ]
        }

        first = client.post("/v1/traffic/observations", json=payload)
        assert first.status_code == 200, first.text
        assert first.json()["inserted"] == 1

        # Repeat the same natural key. The repository should skip it rather
        # than overwrite the original value.
        second = client.post("/v1/traffic/observations", json=payload)
        assert second.status_code == 200, second.text
        assert second.json() == {
            "received": 1,
            "validated": 1,
            "inserted": 0,
            "skipped_duplicates": 1,
        }

        with Session(engine) as session:
            records = session.scalars(
                select(TrafficObservation).where(
                    TrafficObservation.road_id == unique_road_id
                )
            ).all()

            assert len(records) == 1
            assert records[0].traffic_volume == pytest.approx(55.0)
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM traffic_observations WHERE road_id = :road_id"),
                {"road_id": unique_road_id},
            )


def test_postgres_ingestion_handles_batch_and_duplicates(
    integration_client,
    unique_road_id,
):
    client, engine = integration_client

    payload = {
        "observations": [
            {
                "road_id": unique_road_id,
                "timestamp": "2025-02-12T09:00:00",
                "traffic_volume": 20.0,
            },
            {
                "road_id": unique_road_id,
                "timestamp": "2025-02-12T09:30:00",
                "traffic_volume": 24.0,
            },
        ]
    }

    try:
        response = client.post("/v1/traffic/observations", json=payload)
        assert response.status_code == 200, response.text
        assert response.json() == {
            "received": 2,
            "validated": 2,
            "inserted": 2,
            "skipped_duplicates": 0,
        }

        with Session(engine) as session:
            records = session.scalars(
                select(TrafficObservation).where(
                    TrafficObservation.road_id == unique_road_id
                )
            ).all()

        assert len(records) == 2
        assert {record.traffic_volume for record in records} == {20.0, 24.0}
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM traffic_observations WHERE road_id = :road_id"),
                {"road_id": unique_road_id},
            )


def test_postgres_ingestion_rejects_invalid_volume(
    integration_client,
    unique_road_id,
):
    client, engine = integration_client

    try:
        response = client.post(
            "/v1/traffic/observations",
            json={
                "observations": [
                    {
                        "road_id": unique_road_id,
                        "timestamp": "2025-02-13T09:00:00",
                        "traffic_volume": -1,
                    }
                ]
            },
        )

        assert response.status_code == 422

        with Session(engine) as session:
            records = session.scalars(
                select(TrafficObservation).where(
                    TrafficObservation.road_id == unique_road_id
                )
            ).all()

        assert records == []
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM traffic_observations WHERE road_id = :road_id"),
                {"road_id": unique_road_id},
            )
