import pandas as pd
import pytest

from fastapi.testclient import TestClient

from src.day19.api import create_app
from src.day19.service import (
    ForecastService,
    HistoricalDataError,
)


@pytest.fixture
def history():
    return pd.DataFrame(
        [
            {
                "road_id": "R1",
                "timestamp": "2025-01-01 00:00:00",
                "traffic_volume": 100,
            },
            {
                "road_id": "R1",
                "timestamp": "2025-01-01 00:30:00",
                "traffic_volume": 120,
            },
            {
                "road_id": "R1",
                "timestamp": "2025-01-02 00:00:00",
                "traffic_volume": 150,
            },
        ]
    )


@pytest.fixture
def client(history):
    app = create_app(history_data=history)

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint(client):
    response = client.get("/readyz")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ready"
    assert body["historical_records"] == 3
    assert body["unique_roads"] == 1


def test_naive_forecast_uses_30_minute_reference(client):
    response = client.post(
        "/v1/forecast",
        json={
            "road_id": "R1",
            "target_timestamp": "2025-01-02T00:30:00",
            "strategy": "naive_30min",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_traffic_volume"] == 150
    assert body["horizon_minutes"] == 30
    assert body["reference_timestamp"] == "2025-01-02T00:00:00"
    assert body["strategy"] == "naive_30min"


def test_seasonal_forecast_uses_24_hour_reference(client):
    response = client.post(
        "/v1/forecast",
        json={
            "road_id": "R1",
            "target_timestamp": "2025-01-02T00:30:00",
            "strategy": "seasonal_naive_24h",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_traffic_volume"] == 120
    assert body["horizon_minutes"] == 1440
    assert body["reference_timestamp"] == "2025-01-01T00:30:00"


def test_unknown_road_returns_404(client):
    response = client.post(
        "/v1/forecast",
        json={
            "road_id": "UNKNOWN",
            "target_timestamp": "2025-01-02T00:30:00",
            "strategy": "naive_30min",
        },
    )

    assert response.status_code == 404


def test_missing_reference_returns_404(client):
    response = client.post(
        "/v1/forecast",
        json={
            "road_id": "R1",
            "target_timestamp": "2025-01-03T00:30:00",
            "strategy": "naive_30min",
        },
    )

    assert response.status_code == 404


def test_unknown_strategy_returns_422(client):
    response = client.post(
        "/v1/forecast",
        json={
            "road_id": "R1",
            "target_timestamp": "2025-01-02T00:30:00",
            "strategy": "unsupported_model",
        },
    )

    assert response.status_code == 422


def test_extra_request_fields_are_rejected(client):
    response = client.post(
        "/v1/forecast",
        json={
            "road_id": "R1",
            "target_timestamp": "2025-01-02T00:30:00",
            "strategy": "naive_30min",
            "unexpected_field": "not_allowed",
        },
    )

    assert response.status_code == 422


def test_timezone_aware_timestamp_is_rejected(client):
    response = client.post(
        "/v1/forecast",
        json={
            "road_id": "R1",
            "target_timestamp": "2025-01-02T00:30:00+05:30",
            "strategy": "naive_30min",
        },
    )

    assert response.status_code == 422


def test_forecast_response_has_request_id(client):
    response = client.post(
        "/v1/forecast",
        json={
            "road_id": "R1",
            "target_timestamp": "2025-01-02T00:30:00",
            "strategy": "naive_30min",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")

    body = response.json()

    assert body["request_id"]
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_service_rejects_duplicate_history_keys(history):
    duplicate_history = pd.concat(
        [history, history.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        HistoricalDataError,
        match="Duplicate road/timestamp",
    ):
        ForecastService(duplicate_history)


def test_service_rejects_negative_traffic_volume(history):
    invalid_history = history.copy()

    invalid_history.loc[0, "traffic_volume"] = -10

    with pytest.raises(
        HistoricalDataError,
        match="cannot be negative",
    ):
        ForecastService(invalid_history)