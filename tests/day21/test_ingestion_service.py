"""Unit tests for the FlowCast Day 21 ingestion service."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from src.day21.ingestion_service import (
    IngestionValidationError,
    TrafficIngestionService,
)


def valid_observation(
    road_id: str = "ROAD_01",
    timestamp: str = "2025-01-01T10:00:00",
    traffic_volume: float = 25.0,
) -> dict[str, object]:
    """Build a deterministic observation for testing."""
    return {
        "road_id": road_id,
        "timestamp": timestamp,
        "traffic_volume": traffic_volume,
    }


@pytest.fixture
def repository():
    """Return a repository mock with deterministic insert behavior."""
    mock = Mock()
    mock.add_observations.return_value = 1
    return mock


@pytest.fixture
def service(repository):
    """Build the ingestion service around the repository mock."""
    return TrafficIngestionService(repository)


def test_ingest_valid_observation(service, repository):
    repository.add_observations.return_value = 1

    result = service.ingest([valid_observation()])

    assert result.received == 1
    assert result.validated == 1
    assert result.inserted == 1
    assert result.skipped_duplicates == 0
    assert result.failed == 0

    repository.add_observations.assert_called_once()


def test_ingest_empty_batch_does_not_call_repository(
    service,
    repository,
):
    result = service.ingest([])

    assert result.received == 0
    assert result.validated == 0
    assert result.inserted == 0
    assert result.skipped_duplicates == 0

    repository.add_observations.assert_not_called()


def test_ingest_multiple_observations(service, repository):
    repository.add_observations.return_value = 3

    records = [
        valid_observation(
            road_id=f"ROAD_{number}",
            traffic_volume=float(number * 10),
        )
        for number in range(1, 4)
    ]

    result = service.ingest(records)

    assert result.received == 3
    assert result.validated == 3
    assert result.inserted == 3
    assert result.skipped_duplicates == 0

    repository.add_observations.assert_called_once()


@pytest.mark.parametrize(
    "record",
    [
        {
            "timestamp": "2025-01-01T10:00:00",
            "traffic_volume": 25.0,
        },
        {
            "road_id": "ROAD_01",
            "timestamp": "2025-01-01T10:00:00",
        },
        {
            "road_id": "ROAD_01",
            "timestamp": "2025-01-01T10:00:00",
            "traffic_volume": 25.0,
            "unexpected": "field",
        },
    ],
)
def test_invalid_fields_rejected_before_database_write(
    service,
    repository,
    record,
):
    with pytest.raises(IngestionValidationError):
        service.ingest([record])

    repository.add_observations.assert_not_called()


@pytest.mark.parametrize(
    "road_id",
    [
        "",
        "   ",
        "x" * 101,
        123,
        None,
    ],
)
def test_invalid_road_id_rejected(service, repository, road_id):
    record = valid_observation()
    record["road_id"] = road_id

    with pytest.raises(IngestionValidationError):
        service.ingest([record])

    repository.add_observations.assert_not_called()


@pytest.mark.parametrize(
    "timestamp",
    [
        "",
        "not-a-timestamp",
        None,
        12345,
    ],
)
def test_invalid_timestamp_rejected(
    service,
    repository,
    timestamp,
):
    record = valid_observation()
    record["timestamp"] = timestamp

    with pytest.raises(IngestionValidationError):
        service.ingest([record])

    repository.add_observations.assert_not_called()


@pytest.mark.parametrize(
    "volume",
    [
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "25",
        None,
    ],
)
def test_invalid_traffic_volume_rejected(
    service,
    repository,
    volume,
):
    record = valid_observation()
    record["traffic_volume"] = volume

    with pytest.raises(IngestionValidationError):
        service.ingest([record])

    repository.add_observations.assert_not_called()


def test_whitespace_around_road_id_is_normalized(
    service,
    repository,
):
    repository.add_observations.return_value = 1

    result = service.ingest(
        [valid_observation(road_id="  ROAD_01  ")]
    )

    assert result.inserted == 1

    rows = repository.add_observations.call_args.args[0]
    assert rows[0]["road_id"] == "ROAD_01"


def test_timezone_aware_timestamp_is_normalized_to_utc(
    service,
    repository,
):
    repository.add_observations.return_value = 1

    record = valid_observation(
        timestamp="2025-01-01T10:00:00+05:30"
    )

    service.ingest([record])

    rows = repository.add_observations.call_args.args[0]

    assert rows[0]["timestamp"] == datetime(
        2025, 1, 1, 4, 30, 0
    )

    assert rows[0]["timestamp"].tzinfo is None


def test_utc_z_suffix_is_supported(service, repository):
    repository.add_observations.return_value = 1

    record = valid_observation(
        timestamp="2025-01-01T10:00:00Z"
    )

    service.ingest([record])

    rows = repository.add_observations.call_args.args[0]

    assert rows[0]["timestamp"] == datetime(
        2025, 1, 1, 10, 0, 0
    )


def test_naive_timestamp_is_preserved(service, repository):
    repository.add_observations.return_value = 1

    record = valid_observation(
        timestamp="2025-01-01T10:00:00"
    )

    service.ingest([record])

    rows = repository.add_observations.call_args.args[0]

    assert rows[0]["timestamp"] == datetime(
        2025, 1, 1, 10, 0, 0
    )


def test_duplicate_keys_inside_batch_are_rejected(
    service,
    repository,
):
    first = valid_observation()
    second = valid_observation(
        traffic_volume=99.0,
    )

    with pytest.raises(
        IngestionValidationError,
        match="duplicate",
    ):
        service.ingest([first, second])

    repository.add_observations.assert_not_called()


def test_invalid_record_rejects_entire_batch(
    service,
    repository,
):
    records = [
        valid_observation(road_id="ROAD_01"),
        valid_observation(
            road_id="ROAD_02",
            traffic_volume=-10,
        ),
    ]

    with pytest.raises(IngestionValidationError):
        service.ingest(records)

    # Critical invariant: the valid first record is not inserted alone.
    repository.add_observations.assert_not_called()


def test_existing_database_duplicates_are_counted(
    service,
    repository,
):
    repository.add_observations.return_value = 1

    records = [
        valid_observation(road_id="ROAD_01"),
        valid_observation(
            road_id="ROAD_02",
            timestamp="2025-01-01T11:00:00",
        ),
        valid_observation(
            road_id="ROAD_03",
            timestamp="2025-01-01T12:00:00",
        ),
    ]

    result = service.ingest(records)

    assert result.received == 3
    assert result.validated == 3
    assert result.inserted == 1
    assert result.skipped_duplicates == 2


def test_repository_failure_is_not_hidden(
    service,
    repository,
):
    repository.add_observations.side_effect = RuntimeError(
        "simulated database failure"
    )

    with pytest.raises(
        RuntimeError,
        match="simulated database failure",
    ):
        service.ingest([valid_observation()])

    repository.add_observations.assert_called_once()


def test_non_list_batch_is_rejected(service, repository):
    with pytest.raises(TypeError):
        service.ingest(())  # type: ignore[arg-type]

    repository.add_observations.assert_not_called()


def test_repository_must_be_provided():
    with pytest.raises(ValueError):
        TrafficIngestionService(None)  # type: ignore[arg-type]