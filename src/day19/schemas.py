from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ForecastStrategy(str, Enum):
    """Forecasting strategies currently supported by FlowCast."""

    NAIVE_30MIN = "naive_30min"
    SEASONAL_NAIVE_24H = "seasonal_naive_24h"


class ForecastRequest(BaseModel):
    """Validated input for a single traffic forecast."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    road_id: str = Field(
        min_length=1,
        max_length=100,
        description="Identifier of the road being forecast.",
    )

    target_timestamp: datetime = Field(
        description=(
            "Timestamp to forecast. Must use the same "
            "timezone convention as the historical dataset."
        )
    )

    strategy: ForecastStrategy = Field(
        default=ForecastStrategy.NAIVE_30MIN,
        description="Forecasting baseline to use.",
    )

    @field_validator("road_id")
    @classmethod
    def validate_road_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("road_id cannot be blank.")

        return value

    @field_validator("target_timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError(
                "Timezone-aware timestamps are not supported yet. "
                "Provide a timestamp using the dataset's timezone convention."
            )

        return value


class ForecastResponse(BaseModel):
    """Structured output for a traffic forecast."""

    model_config = ConfigDict(
        extra="forbid",
    )

    road_id: str
    target_timestamp: datetime
    reference_timestamp: datetime
    predicted_traffic_volume: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    strategy: ForecastStrategy
    horizon_minutes: int = Field(gt=0)
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    """Application liveness response."""

    status: str


class ReadinessResponse(BaseModel):
    """Application readiness response."""

    status: str
    historical_records: int
    unique_roads: int



MAX_INGESTION_BATCH_SIZE = 1000


class TrafficObservationInput(BaseModel):
    """Validated representation of a single traffic observation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    road_id: str = Field(
        min_length=1,
        max_length=100,
        description="Stable identifier for the road.",
    )

    timestamp: datetime = Field(
        description=(
            "Observation timestamp. Naive timestamps follow the "
            "dataset's canonical timezone convention."
        ),
    )

    traffic_volume: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Nonnegative observed traffic volume.",
    )

    @field_validator("road_id")
    @classmethod
    def validate_road_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("road_id cannot be blank.")

        return value

    @field_validator("traffic_volume", mode="before")
    @classmethod
    def reject_boolean_traffic_volume(cls, value):
        if isinstance(value, bool):
            raise ValueError(
                "traffic_volume must be a numeric value, not a boolean."
            )

        return value


class TrafficObservationBatchRequest(BaseModel):
    """Request containing a bounded batch of traffic observations."""

    model_config = ConfigDict(extra="forbid")

    observations: list[TrafficObservationInput] = Field(
        min_length=1,
        max_length=MAX_INGESTION_BATCH_SIZE,
        description=(
            "Between 1 and 1000 observations per request."
        ),
    )


class TrafficIngestionResponse(BaseModel):
    """Summary of a successfully processed ingestion batch."""

    model_config = ConfigDict(extra="forbid")

    received: int = Field(ge=0)
    validated: int = Field(ge=0)
    inserted: int = Field(ge=0)
    skipped_duplicates: int = Field(ge=0)
