from pathlib import Path

import numpy as np
import pandas as pd

from src.day19.schemas import ForecastStrategy


ROAD_COLUMN = "road_id"
TIME_COLUMN = "timestamp"
TARGET_COLUMN = "traffic_volume"

REQUIRED_COLUMNS = {
    ROAD_COLUMN,
    TIME_COLUMN,
    TARGET_COLUMN,
}

STRATEGY_HORIZONS = {
    ForecastStrategy.NAIVE_30MIN: 30,
    ForecastStrategy.SEASONAL_NAIVE_24H: 1440,
}


class HistoricalDataError(ValueError):
    """Raised when historical data violates the service contract."""


class RoadNotFoundError(LookupError):
    """Raised when the requested road is not present in history."""


class ReferenceUnavailableError(LookupError):
    """Raised when the required historical reference is unavailable."""


def load_history(path: Path) -> pd.DataFrame:
    """Load and validate the historical traffic dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Historical dataset does not exist: {path}"
        )

    if not path.is_file():
        raise HistoricalDataError(
            f"Historical dataset path is not a file: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if df.empty:
        raise HistoricalDataError(
            "Historical dataset contains no records."
        )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise HistoricalDataError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = df[list(REQUIRED_COLUMNS)].copy()

    df[ROAD_COLUMN] = df[ROAD_COLUMN].astype("string").str.strip()

    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        errors="coerce",
    )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    if df[ROAD_COLUMN].isna().any():
        raise HistoricalDataError(
            "Historical data contains missing road identifiers."
        )

    if df[ROAD_COLUMN].eq("").any():
        raise HistoricalDataError(
            "Historical data contains blank road identifiers."
        )

    if df[TIME_COLUMN].isna().any():
        raise HistoricalDataError(
            "Historical data contains invalid timestamps."
        )

    if df[TARGET_COLUMN].isna().any():
        raise HistoricalDataError(
            "Historical data contains missing or non-numeric targets."
        )

    if not np.isfinite(df[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise HistoricalDataError(
            "Historical data contains non-finite traffic values."
        )

    if (df[TARGET_COLUMN] < 0).any():
        raise HistoricalDataError(
            "Historical traffic volume cannot be negative."
        )

    if df[TIME_COLUMN].dt.tz is not None:
        raise HistoricalDataError(
            "Timezone-aware historical timestamps are not supported yet."
        )

    duplicate_mask = df.duplicated(
        [ROAD_COLUMN, TIME_COLUMN],
        keep=False,
    )

    if duplicate_mask.any():
        raise HistoricalDataError(
            "Historical data contains duplicate road/timestamp keys."
        )

    return df.sort_values(
        [ROAD_COLUMN, TIME_COLUMN]
    ).reset_index(drop=True)


class ForecastService:
    """Forecasting logic independent of the HTTP framework."""

    MODEL_VERSION = "flowcast-baseline-v1"

    def __init__(self, history: pd.DataFrame):
        missing_columns = REQUIRED_COLUMNS - set(history.columns)

        if missing_columns:
            raise HistoricalDataError(
                f"Missing historical columns: {sorted(missing_columns)}"
            )

        df = history[
            [ROAD_COLUMN, TIME_COLUMN, TARGET_COLUMN]
        ].copy()

        df[ROAD_COLUMN] = df[ROAD_COLUMN].astype("string").str.strip()

        df[TIME_COLUMN] = pd.to_datetime(
            df[TIME_COLUMN],
            errors="coerce",
        )

        df[TARGET_COLUMN] = pd.to_numeric(
            df[TARGET_COLUMN],
            errors="coerce",
        )

        if df.empty:
            raise HistoricalDataError(
                "Historical data contains no records."
            )

        if df.isna().any().any():
            raise HistoricalDataError(
                "Historical data contains missing required values."
            )

        if not np.isfinite(df[TARGET_COLUMN].to_numpy(dtype=float)).all():
            raise HistoricalDataError(
                "Historical traffic values must be finite."
            )

        if (df[TARGET_COLUMN] < 0).any():
            raise HistoricalDataError(
                "Historical traffic volume cannot be negative."
            )

        if df[TIME_COLUMN].dt.tz is not None:
            raise HistoricalDataError(
                "Timezone-aware timestamps are not supported."
            )

        if df.duplicated(
            [ROAD_COLUMN, TIME_COLUMN]
        ).any():
            raise HistoricalDataError(
                "Duplicate road/timestamp keys are not allowed."
            )

        self._history = df.sort_values(
            [ROAD_COLUMN, TIME_COLUMN]
        ).reset_index(drop=True)

        self._roads = set(
            self._history[ROAD_COLUMN].astype(str)
        )

        self._lookup = {
            (
                str(row.road_id),
                pd.Timestamp(row.timestamp),
            ): float(row.traffic_volume)
            for row in self._history.itertuples(index=False)
        }

    @property
    def historical_records(self) -> int:
        return len(self._history)

    @property
    def unique_roads(self) -> int:
        return len(self._roads)

    def forecast(
        self,
        road_id: str,
        target_timestamp,
        strategy: ForecastStrategy,
        request_id: str,
    ) -> dict:
        """Generate a forecast using an exact historical reference."""

        if road_id not in self._roads:
            raise RoadNotFoundError(
                f"Road '{road_id}' is not present in historical data."
            )

        horizon_minutes = STRATEGY_HORIZONS[strategy]

        target = pd.Timestamp(target_timestamp)

        reference = target - pd.Timedelta(
            minutes=horizon_minutes
        )

        lookup_key = (
            road_id,
            reference,
        )

        if lookup_key not in self._lookup:
            raise ReferenceUnavailableError(
                f"No historical reference exists for road '{road_id}' "
                f"at {reference.isoformat()}."
            )

        prediction = self._lookup[lookup_key]

        return {
            "road_id": road_id,
            "target_timestamp": target.to_pydatetime(),
            "reference_timestamp": reference.to_pydatetime(),
            "predicted_traffic_volume": prediction,
            "strategy": strategy,
            "horizon_minutes": horizon_minutes,
            "model_version": self.MODEL_VERSION,
            "request_id": request_id,
        }