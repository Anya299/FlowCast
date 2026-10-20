"""Repository operations for FlowCast traffic observations."""

import math
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from src.day20.models import TrafficObservation


def _validate_observation(
    road_id: str,
    timestamp: datetime,
    traffic_volume: float,
) -> tuple[str, datetime, float]:
    """Validate and normalize an observation before database access."""
    if not isinstance(road_id, str):
        raise TypeError("road_id must be a string.")

    road_id = road_id.strip()

    if not road_id:
        raise ValueError("road_id must not be empty.")

    if len(road_id) > 100:
        raise ValueError("road_id must not exceed 100 characters.")

    if not isinstance(timestamp, datetime):
        raise TypeError("timestamp must be a datetime.")

    if timestamp.tzinfo is not None and timestamp.utcoffset() is not None:
        raise ValueError(
            "timestamp must be timezone-naive under the current schema."
        )

    if isinstance(traffic_volume, bool) or not isinstance(
        traffic_volume, (int, float)
    ):
        raise TypeError("traffic_volume must be a numeric value.")

    traffic_volume = float(traffic_volume)

    if not math.isfinite(traffic_volume):
        raise ValueError("traffic_volume must be finite.")

    if traffic_volume < 0:
        raise ValueError("traffic_volume must not be negative.")

    return road_id, timestamp, traffic_volume


class TrafficObservationRepository:
    """Database operations for historical traffic observations."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def add_observation(
        self,
        road_id: str,
        timestamp: datetime,
        traffic_volume: float,
    ) -> bool:
        """Insert one observation.

        Return True if inserted or False if its primary key already exists.
        """
        road_id, timestamp, traffic_volume = _validate_observation(
            road_id,
            timestamp,
            traffic_volume,
        )

        statement = (
            pg_insert(TrafficObservation)
            .values(
                road_id=road_id,
                timestamp=timestamp,
                traffic_volume=traffic_volume,
            )
            .on_conflict_do_nothing(
                index_elements=[
                    TrafficObservation.road_id,
                    TrafficObservation.timestamp,
                ]
            )
            .returning(TrafficObservation.road_id)
        )

        with self._session_factory.begin() as session:
            inserted_road_id = session.execute(statement).scalar_one_or_none()

        return inserted_road_id is not None

    def add_observations(
        self,
        observations: list[dict[str, object]],
    ) -> int:
        """Insert multiple observations in one transaction.

        Existing primary keys are skipped. If validation or insertion fails,
        the transaction is rolled back.

        Return the number of inserted records.
        """
        if not isinstance(observations, list):
            raise TypeError("observations must be a list.")

        if not observations:
            return 0

        validated_rows: list[dict[str, object]] = []

        for item in observations:
            if not isinstance(item, dict):
                raise TypeError("Each observation must be a dictionary.")

            required_fields = {
                "road_id",
                "timestamp",
                "traffic_volume",
            }

            if set(item) != required_fields:
                raise ValueError(
                    "Each observation must contain exactly: "
                    "road_id, timestamp, traffic_volume."
                )

            road_id, timestamp, traffic_volume = _validate_observation(
                item["road_id"],
                item["timestamp"],
                item["traffic_volume"],
            )

            validated_rows.append(
                {
                    "road_id": road_id,
                    "timestamp": timestamp,
                    "traffic_volume": traffic_volume,
                }
            )

        # PostgreSQL cannot insert the same conflict key twice in one
        # ON CONFLICT DO UPDATE statement. We reject duplicate keys within
        # the input batch, making the operation deterministic.
        keys = [
            (row["road_id"], row["timestamp"])
            for row in validated_rows
        ]

        if len(keys) != len(set(keys)):
            raise ValueError(
                "The batch contains duplicate road_id/timestamp keys."
            )

        statement = (
            pg_insert(TrafficObservation)
            .values(validated_rows)
            .on_conflict_do_nothing(
                index_elements=[
                    TrafficObservation.road_id,
                    TrafficObservation.timestamp,
                ]
            )
            .returning(TrafficObservation.road_id)
        )

        with self._session_factory.begin() as session:
            inserted_rows = session.execute(statement).all()

        return len(inserted_rows)

    def get_observation(
        self,
        road_id: str,
        timestamp: datetime,
    ) -> TrafficObservation | None:
        """Retrieve an observation by its composite primary key."""
        if not isinstance(road_id, str) or not road_id.strip():
            raise ValueError("road_id must be a nonempty string.")

        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime.")

        if timestamp.tzinfo is not None and timestamp.utcoffset() is not None:
            raise ValueError(
                "timestamp must be timezone-naive under the current schema."
            )

        statement = select(TrafficObservation).where(
            TrafficObservation.road_id == road_id.strip(),
            TrafficObservation.timestamp == timestamp,
        )

        with self._session_factory() as session:
            return session.execute(statement).scalar_one_or_none()

    def get_history(
        self,
        road_id: str,
        start: datetime,
        end: datetime,
    ) -> list[TrafficObservation]:
        """Retrieve a road's observations in [start, end)."""
        if not isinstance(road_id, str) or not road_id.strip():
            raise ValueError("road_id must be a nonempty string.")

        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise TypeError("start and end must be datetime objects.")

        if start.tzinfo is not None and start.utcoffset() is not None:
            raise ValueError("start must be timezone-naive.")

        if end.tzinfo is not None and end.utcoffset() is not None:
            raise ValueError("end must be timezone-naive.")

        if start >= end:
            raise ValueError("start must be earlier than end.")

        statement = (
            select(TrafficObservation)
            .where(
                TrafficObservation.road_id == road_id.strip(),
                TrafficObservation.timestamp >= start,
                TrafficObservation.timestamp < end,
            )
            .order_by(TrafficObservation.timestamp.asc())
        )

        with self._session_factory() as session:
            return list(session.execute(statement).scalars().all())

    def get_latest(
        self,
        road_id: str,
    ) -> TrafficObservation | None:
        """Retrieve the most recent observation for a road."""
        if not isinstance(road_id, str) or not road_id.strip():
            raise ValueError("road_id must be a nonempty string.")

        statement = (
            select(TrafficObservation)
            .where(TrafficObservation.road_id == road_id.strip())
            .order_by(TrafficObservation.timestamp.desc())
            .limit(1)
        )

        with self._session_factory() as session:
            return session.execute(statement).scalar_one_or_none()
