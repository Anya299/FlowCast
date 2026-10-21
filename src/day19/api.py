"""HTTP API for FlowCast forecasting and traffic ingestion."""

import logging
import os

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.day19.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ReadinessResponse,
    TrafficIngestionResponse,
    TrafficObservationBatchRequest,
)
from src.day19.service import (
    ForecastService,
    HistoricalDataError,
    ReferenceUnavailableError,
    RoadNotFoundError,
    load_history,
)
from src.day20.database import (
    build_engine_from_environment,
    build_session_factory,
)
from src.day20.repository import TrafficObservationRepository
from src.day21.ingestion_service import (
    IngestionResult,
    IngestionValidationError,
    TrafficIngestionService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "day9"
    / "flowcast_train.csv"
)

HISTORY_PATH = Path(
    os.environ.get(
        "FLOWCAST_HISTORY_PATH",
        str(DEFAULT_HISTORY_PATH),
    )
)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("flowcast.api")


def create_app(
    history_path: Path | None = None,
    history_data=None,
    ingestion_service: TrafficIngestionService | None = None,
) -> FastAPI:
    """Create the FlowCast application.

    Production startup loads forecasting history and configures
    PostgreSQL-backed ingestion.

    Tests can inject historical data and an ingestion service,
    avoiding a dependency on a live database.
    """

    resolved_path = history_path or HISTORY_PATH

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting FlowCast API.")

        engine: Engine | None = None

        app.state.forecast_service = None
        app.state.ingestion_service = ingestion_service
        app.state.database_engine = None

        try:
            if history_data is not None:
                history = history_data.copy()
            else:
                history = load_history(resolved_path)

            app.state.forecast_service = ForecastService(history)

            logger.info(
                "Historical data loaded: %s records, %s roads.",
                app.state.forecast_service.historical_records,
                app.state.forecast_service.unique_roads,
            )

            # Injected test services do not require a database.
            # Production startup configures PostgreSQL ingestion.
            if ingestion_service is None and history_data is None:
                engine = build_engine_from_environment()

                session_factory = build_session_factory(engine)

                repository = TrafficObservationRepository(
                    session_factory
                )

                app.state.ingestion_service = TrafficIngestionService(
                    repository
                )

                app.state.database_engine = engine

                logger.info(
                    "PostgreSQL ingestion service configured."
                )

            yield

        except Exception:
            logger.exception(
                "FlowCast API startup or runtime lifecycle failed."
            )
            raise

        finally:
            app.state.forecast_service = None
            app.state.ingestion_service = None
            app.state.database_engine = None

            if engine is not None:
                engine.dispose()

            logger.info("FlowCast API stopped.")

    app = FastAPI(
        title="FlowCast Traffic Intelligence API",
        description=(
            "Versioned API for historical-reference traffic "
            "forecasting and validated traffic observation ingestion."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "request_id": getattr(
                    request.state,
                    "request_id",
                    None,
                ),
            },
            headers=exc.headers,
        )

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        tags=["Operations"],
    )
    async def health():
        """Liveness check: the application process is responding."""

        return HealthResponse(status="ok")

    @app.get(
        "/readyz",
        response_model=ReadinessResponse,
        tags=["Operations"],
    )
    async def readiness(request: Request):
        """Readiness check for the forecasting service."""

        service = getattr(
            request.app.state,
            "forecast_service",
            None,
        )

        if service is None:
            raise HTTPException(
                status_code=503,
                detail="Forecasting service is not ready.",
            )

        return ReadinessResponse(
            status="ready",
            historical_records=service.historical_records,
            unique_roads=service.unique_roads,
        )

    @app.post(
        "/v1/forecast",
        response_model=ForecastResponse,
        tags=["Forecasting"],
    )
    async def forecast(
        payload: ForecastRequest,
        request: Request,
    ):
        """Generate a single historical-reference forecast."""

        service = getattr(
            request.app.state,
            "forecast_service",
            None,
        )

        if service is None:
            raise HTTPException(
                status_code=503,
                detail="Forecasting service is not ready.",
            )

        request_id = request.state.request_id

        try:
            result = service.forecast(
                road_id=payload.road_id,
                target_timestamp=payload.target_timestamp,
                strategy=payload.strategy,
                request_id=request_id,
            )

            logger.info(
                "Forecast generated. request_id=%s road_id=%s "
                "strategy=%s horizon_minutes=%s",
                request_id,
                payload.road_id,
                payload.strategy.value,
                result["horizon_minutes"],
            )

            return ForecastResponse(**result)

        except RoadNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except ReferenceUnavailableError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except HistoricalDataError as exc:
            logger.exception(
                "Historical data contract failure. request_id=%s",
                request_id,
            )

            raise HTTPException(
                status_code=500,
                detail="Historical data failed validation.",
            ) from exc

    @app.post(
        "/v1/traffic/observations",
        response_model=TrafficIngestionResponse,
        status_code=200,
        tags=["Traffic Ingestion"],
    )
    async def ingest_observations(
        payload: TrafficObservationBatchRequest,
        request: Request,
    ):
        """Validate and persist a batch of traffic observations."""

        service = getattr(
            request.app.state,
            "ingestion_service",
            None,
        )

        request_id = request.state.request_id

        if service is None:
            raise HTTPException(
                status_code=503,
                detail="Traffic ingestion service is not configured.",
            )

        observations = [
            observation.model_dump()
            for observation in payload.observations
        ]

        try:
            result: IngestionResult = service.ingest(observations)

        except IngestionValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except SQLAlchemyError as exc:
            logger.exception(
                "Traffic ingestion database failure. request_id=%s",
                request_id,
            )

            raise HTTPException(
                status_code=503,
                detail="Traffic ingestion is temporarily unavailable.",
            ) from exc

        logger.info(
            "Traffic batch processed. request_id=%s "
            "received=%s validated=%s inserted=%s "
            "skipped_duplicates=%s",
            request_id,
            result.received,
            result.validated,
            result.inserted,
            result.skipped_duplicates,
        )

        return TrafficIngestionResponse(
            received=result.received,
            validated=result.validated,
            inserted=result.inserted,
            skipped_duplicates=result.skipped_duplicates,
        )

    return app


app = create_app()