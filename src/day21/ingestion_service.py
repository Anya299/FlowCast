"""Validated, transactional ingestion service for FlowCast."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.day20.repository import TrafficObservationRepository


REQUIRED_FIELDS = frozenset(
    {"road_id", "timestamp", "traffic_volume"}
)

MAX_ROAD_ID_LENGTH = 100


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Immutable summary of a successful ingestion request."""

    received: int
    validated: int
    inserted: int
    skipped_duplicates: int

    @property
    def failed(self) -> int:
        """Number of records rejected during successful ingestion.

        Invalid requests raise an exception before any database write.
        Therefore, a successfully returned result has failed == 0.
        """
        return 0


class IngestionValidationError(ValueError):
    """Raised when an ingestion request violates its data contract."""


def _normalize_timestamp(value: Any) -> datetime:
    """Parse and normalize a timestamp for the current database schema.

    Policy:
    - datetime objects are accepted directly.
    - ISO-8601 strings are accepted.
    - Timezone-aware values are converted to UTC and made naive.
    - Naive values are preserved as supplied.

    Naive timestamps must already follow the source's documented
    canonical time convention. They are not assumed to be UTC.
    """
    if isinstance(value, datetime):
        parsed = value

    elif isinstance(value, str):
        timestamp_text = value.strip()

        if not timestamp_text:
            raise IngestionValidationError(
                "timestamp must not be an empty string."
            )

        # datetime.fromisoformat accepts +00:00. Normalize the
        # common ISO-8601 UTC suffix Z explicitly for compatibility.
        if timestamp_text.endswith(("Z", "z")):
            timestamp_text = timestamp_text[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(timestamp_text)
        except ValueError as exc:
            raise IngestionValidationError(
                "timestamp must be a valid ISO-8601 timestamp."
            ) from exc

    else:
        raise IngestionValidationError(
            "timestamp must be a datetime object or ISO-8601 string."
        )

    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)

    return parsed


def _validate_observation(
    item: Mapping[str, Any],
    index: int,
) -> dict[str, object]:
    """Validate one input record and return a normalized dictionary."""
    if not isinstance(item, Mapping):
        raise IngestionValidationError(
            f"Record {index}: observation must be a mapping."
        )

    actual_fields = set(item)

    if actual_fields != REQUIRED_FIELDS:
        missing = sorted(REQUIRED_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - REQUIRED_FIELDS)

        details: list[str] = []

        if missing:
            details.append(f"missing fields: {missing}")

        if unexpected:
            details.append(f"unexpected fields: {unexpected}")

        raise IngestionValidationError(
            f"Record {index}: invalid fields; "
            + "; ".join(details)
            + "."
        )

    road_id = item["road_id"]

    if not isinstance(road_id, str):
        raise IngestionValidationError(
            f"Record {index}: road_id must be a string."
        )

    road_id = road_id.strip()

    if not road_id:
        raise IngestionValidationError(
            f"Record {index}: road_id must not be empty."
        )

    if len(road_id) > MAX_ROAD_ID_LENGTH:
        raise IngestionValidationError(
            f"Record {index}: road_id must not exceed "
            f"{MAX_ROAD_ID_LENGTH} characters."
        )

    try:
        timestamp = _normalize_timestamp(item["timestamp"])
    except IngestionValidationError as exc:
        raise IngestionValidationError(
            f"Record {index}: {exc}"
        ) from exc

    volume = item["traffic_volume"]

    if isinstance(volume, bool) or not isinstance(volume, (int, float)):
        raise IngestionValidationError(
            f"Record {index}: traffic_volume must be numeric."
        )

    traffic_volume = float(volume)

    if not math.isfinite(traffic_volume):
        raise IngestionValidationError(
            f"Record {index}: traffic_volume must be finite."
        )

    if traffic_volume < 0:
        raise IngestionValidationError(
            f"Record {index}: traffic_volume must not be negative."
        )

    return {
        "road_id": road_id,
        "timestamp": timestamp,
        "traffic_volume": traffic_volume,
    }


class TrafficIngestionService:
    """Validate traffic observations and persist them through a repository."""

    def __init__(
        self,
        repository: TrafficObservationRepository,
    ) -> None:
        if repository is None:
            raise ValueError("repository must not be None.")

        self._repository = repository

    def ingest(
        self,
        observations: list[Mapping[str, Any]],
    ) -> IngestionResult:
        """Validate and atomically ingest a batch of observations.

        All input validation occurs before the repository is called.

        The repository owns database transaction management and skips
        records whose composite primary keys already exist.

        Raises:
            IngestionValidationError: if the request violates the contract.
            TypeError: if the batch is not a list.
            Exception: database/repository exceptions propagate to the caller.
        """
        if not isinstance(observations, list):
            raise TypeError("observations must be a list.")

        received = len(observations)

        if received == 0:
            return IngestionResult(
                received=0,
                validated=0,
                inserted=0,
                skipped_duplicates=0,
            )

        # Validate the complete batch before performing any database write.
        validated_rows = [
            _validate_observation(item, index)
            for index, item in enumerate(observations, start=1)
        ]

        # Reject duplicate keys inside this request. The repository has
        # the same restriction for bulk insertion; checking here gives
        # the caller a service-level error before database access.
        keys = [
            (row["road_id"], row["timestamp"])
            for row in validated_rows
        ]

        if len(keys) != len(set(keys)):
            raise IngestionValidationError(
                "The batch contains duplicate road_id/timestamp keys."
            )

        # One repository call delegates the entire batch to its existing
        # transaction. Do not split this into per-record inserts.
        inserted = self._repository.add_observations(validated_rows)

        # The repository uses ON CONFLICT DO NOTHING. Given unique keys
        # within this batch, the difference represents records already
        # present in the database at insert time.
        skipped_duplicates = received - inserted

        return IngestionResult(
            received=received,
            validated=len(validated_rows),
            inserted=inserted,
            skipped_duplicates=skipped_duplicates,
        )