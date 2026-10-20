import logging
import os

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.day19.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ReadinessResponse,
)
from src.day19.service import (
    ForecastService,
    HistoricalDataError,
    ReferenceUnavailableError,
    RoadNotFoundError,
    load_history,
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
) -> FastAPI:
    """
    Application factory.

    Production use loads history during startup.
    Tests can inject a small deterministic dataset.
    """

    resolved_path = history_path or HISTORY_PATH

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting FlowCast API.")

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

            yield

        except Exception:
            logger.exception(
                "FlowCast API startup or runtime lifecycle failed."
            )
            raise

        finally:
            app.state.forecast_service = None
            logger.info("FlowCast API stopped.")

    app = FastAPI(
        title="FlowCast Forecasting API",
        description=(
            "Versioned API for traffic forecasting using "
            "historical-reference baseline strategies."
        ),
        version="0.1.0",
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

        return HealthResponse(
            status="ok",
        )

    @app.get(
        "/readyz",
        response_model=ReadinessResponse,
        tags=["Operations"],
    )
    async def readiness(request: Request):
        """Readiness check: forecasting data is available."""

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

    return app


app = create_app()