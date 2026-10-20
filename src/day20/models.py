"""SQLAlchemy ORM models for FlowCast."""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Index,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.day20.database import Base


class TrafficObservation(Base):
    """Historical traffic volume for one road at one timestamp."""

    __tablename__ = "traffic_observations"

    __table_args__ = (
        CheckConstraint(
            "length(trim(road_id)) > 0",
            name="ck_traffic_observations_road_id_nonempty",
        ),
        CheckConstraint(
            "traffic_volume >= 0",
            name="ck_traffic_observations_volume_nonnegative",
        ),
        Index(
            "ix_traffic_observations_timestamp",
            "timestamp",
        ),
    )

    road_id: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        primary_key=True,
    )

    traffic_volume: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
