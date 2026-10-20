"""PostgreSQL integration tests for the FlowCast repository."""

import os
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from dotenv import load_dotenv
from sqlalchemy import delete, func, select
from sqlalchemy.orm import sessionmaker

from src.day20.database import Base, build_engine
from src.day20.models import TrafficObservation
from src.day20.repository import TrafficObservationRepository


@pytest.fixture
def repository():
    """Create a repository connected to the configured PostgreSQL database."""
    load_dotenv()

    database_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")

    if not database_url:
        pytest.fail(
            "No database URL found. Configure TEST_DATABASE_URL or DATABASE_URL."
        )

    if not database_url.startswith(
        ("postgresql://", "postgresql+psycopg://")
    ):
        pytest.fail("Integration tests require a PostgreSQL database URL.")

    engine = build_engine(database_url)

    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")

        Base.metadata.create_all(engine)

        factory = sessionmaker(
            bind=engine,
            autoflush=False,
            expire_on_commit=False,
        )

        repo = TrafficObservationRepository(factory)
        road_id = f"pytest-{uuid4().hex}"

        try:
            yield repo, factory, road_id
        finally:
            with factory.begin() as session:
                session.execute(
                    delete(TrafficObservation).where(
                        TrafficObservation.road_id == road_id
                    )
                )
    finally:
        engine.dispose()


def test_insert_and_retrieve_observation(repository):
    repo, _, road_id = repository
    timestamp = datetime(2026, 1, 1, 8, 0, 0)

    assert repo.add_observation(road_id, timestamp, 125.0) is True

    observation = repo.get_observation(road_id, timestamp)

    assert observation is not None
    assert observation.road_id == road_id
    assert observation.timestamp == timestamp
    assert observation.traffic_volume == 125.0
    assert observation.created_at is not None


def test_duplicate_observation_is_skipped(repository):
    repo, _, road_id = repository
    timestamp = datetime(2026, 1, 1, 8, 0, 0)

    assert repo.add_observation(road_id, timestamp, 100.0) is True
    assert repo.add_observation(road_id, timestamp, 999.0) is False

    observation = repo.get_observation(road_id, timestamp)

    assert observation is not None
    assert observation.traffic_volume == 100.0


def test_history_returns_sorted_half_open_range(repository):
    repo, _, road_id = repository
    start = datetime(2026, 1, 1, 8, 0, 0)

    repo.add_observations(
        [
            {
                "road_id": road_id,
                "timestamp": start,
                "traffic_volume": 100.0,
            },
            {
                "road_id": road_id,
                "timestamp": start + timedelta(hours=1),
                "traffic_volume": 150.0,
            },
            {
                "road_id": road_id,
                "timestamp": start + timedelta(hours=2),
                "traffic_volume": 200.0,
            },
        ]
    )

    history = repo.get_history(
        road_id,
        start,
        start + timedelta(hours=2),
    )

    assert len(history) == 2
    assert [item.traffic_volume for item in history] == [100.0, 150.0]


def test_latest_observation(repository):
    repo, _, road_id = repository
    start = datetime(2026, 1, 1, 8, 0, 0)

    repo.add_observation(road_id, start, 100.0)
    repo.add_observation(road_id, start + timedelta(hours=1), 200.0)

    latest = repo.get_latest(road_id)

    assert latest is not None
    assert latest.timestamp == start + timedelta(hours=1)
    assert latest.traffic_volume == 200.0


@pytest.mark.parametrize(
    "road_id,volume,error",
    [
        ("", 10.0, ValueError),
        ("   ", 10.0, ValueError),
        ("road-1", -1.0, ValueError),
        ("road-1", float("nan"), ValueError),
        ("road-1", float("inf"), ValueError),
        ("road-1", True, TypeError),
        ("road-1", "100", TypeError),
    ],
)
def test_invalid_observations_are_rejected(
    repository,
    road_id,
    volume,
    error,
):
    repo, _, _ = repository

    with pytest.raises(error):
        repo.add_observation(
            road_id,
            datetime(2026, 1, 1, 8, 0, 0),
            volume,
        )


def test_timezone_aware_timestamp_is_rejected(repository):
    repo, _, road_id = repository

    with pytest.raises(ValueError, match="timezone-naive"):
        repo.add_observation(
            road_id,
            datetime.fromisoformat("2026-01-01T08:00:00+05:30"),
            100.0,
        )


def test_batch_insert_and_duplicate_skip(repository):
    repo, _, road_id = repository
    timestamp = datetime(2026, 1, 1, 8, 0, 0)

    observations = [
        {
            "road_id": road_id,
            "timestamp": timestamp,
            "traffic_volume": 100.0,
        },
        {
            "road_id": road_id,
            "timestamp": timestamp + timedelta(hours=1),
            "traffic_volume": 150.0,
        },
    ]

    assert repo.add_observations(observations) == 2
    assert repo.add_observations(observations) == 0

    history = repo.get_history(
        road_id,
        timestamp,
        timestamp + timedelta(hours=2),
    )

    assert len(history) == 2


def test_duplicate_keys_inside_batch_are_rejected(repository):
    repo, _, road_id = repository
    timestamp = datetime(2026, 1, 1, 8, 0, 0)

    observations = [
        {
            "road_id": road_id,
            "timestamp": timestamp,
            "traffic_volume": 100.0,
        },
        {
            "road_id": road_id,
            "timestamp": timestamp,
            "traffic_volume": 200.0,
        },
    ]

    with pytest.raises(ValueError, match="duplicate"):
        repo.add_observations(observations)

    assert repo.get_observation(road_id, timestamp) is None


def test_invalid_batch_does_not_insert_partial_data(repository):
    repo, factory, road_id = repository
    timestamp = datetime(2026, 1, 1, 8, 0, 0)

    observations = [
        {
            "road_id": road_id,
            "timestamp": timestamp,
            "traffic_volume": 100.0,
        },
        {
            "road_id": road_id,
            "timestamp": timestamp + timedelta(hours=1),
            "traffic_volume": -5.0,
        },
    ]

    with pytest.raises(ValueError):
        repo.add_observations(observations)

    with factory() as session:
        count = session.execute(
            select(func.count())
            .select_from(TrafficObservation)
            .where(TrafficObservation.road_id == road_id)
        ).scalar_one()

    assert count == 0


def test_transaction_rolls_back_after_failure(repository):
    _, factory, road_id = repository
    timestamp = datetime(2026, 1, 1, 8, 0, 0)

    with pytest.raises(RuntimeError):
        with factory.begin() as session:
            session.add(
                TrafficObservation(
                    road_id=road_id,
                    timestamp=timestamp,
                    traffic_volume=100.0,
                )
            )

            raise RuntimeError("Simulated transaction failure")

    with factory() as session:
        observation = session.get(
            TrafficObservation,
            (road_id, timestamp),
        )

    assert observation is None
