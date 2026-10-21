"""API tests for FlowCast batch traffic ingestion."""

import pandas as pd
import pytest

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from src.day19.api import create_app
from src.day21.ingestion_service import (
    IngestionResult,
    IngestionValidationError,
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


class FakeIngestionService:
    """Deterministic test double for the ingestion service."""

    def __init__(self):
        self.calls = []
        self.result = IngestionResult(
            received=2,
            validated=2,
            inserted=2,
            skipped_duplicates=0,
        )
        self.error = None

    def ingest(self, observations):
        self.calls.append(observations)

        if self.error is not None:
            raise self.error

        return self.result


@pytest.fixture
def ingestion_service():
    return FakeIngestionService()


@pytest.fixture
def client(history, ingestion_service):
    app = create_app(
        history_data=history,
        ingestion_service=ingestion_service,
    )

    with TestClient(app) as test_client:
        yield test_client


def valid_payload():
    return {
        "observations": [
            {
                "road_id": "R1",
                "timestamp": "2025-01-01T00:00:00",
                "traffic_volume": 100,
            },
            {
                "road_id": "R1",
                "timestamp": "2025-01-01T00:30:00",
                "traffic_volume": 120,
            },
        ]
    }


def test_ingestion_endpoint_returns_batch_summary(
    client,
    ingestion_service,
):
    response = client.post(
        "/v1/traffic/observations",
        json=valid_payload(),
    )

    assert response.status_code == 200

    assert response.json() == {
        "received": 2,
        "validated": 2,
        "inserted": 2,
        "skipped_duplicates": 0,
    }

    assert len(ingestion_service.calls) == 1


def test_ingestion_endpoint_passes_parsed_observations(
    client,
    ingestion_service,
):
    response = client.post(
        "/v1/traffic/observations",
        json=valid_payload(),
    )

    assert response.status_code == 200

    observations = ingestion_service.calls[0]

    assert len(observations) == 2
    assert observations[0]["road_id"] == "R1"
    assert observations[0]["traffic_volume"] == 100
    assert observations[0]["timestamp"].year == 2025


def test_ingestion_response_includes_request_id(client):
    response = client.post(
        "/v1/traffic/observations",
        json=valid_payload(),
    )

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")


def test_ingestion_rejects_empty_batch(
    client,
    ingestion_service,
):
    response = client.post(
        "/v1/traffic/observations",
        json={"observations": []},
    )

    assert response.status_code == 422
    assert ingestion_service.calls == []


def test_ingestion_rejects_negative_volume(
    client,
    ingestion_service,
):
    payload = valid_payload()
    payload["observations"][0]["traffic_volume"] = -1

    response = client.post(
        "/v1/traffic/observations",
        json=payload,
    )

    assert response.status_code == 422
    assert ingestion_service.calls == []


def test_ingestion_rejects_boolean_volume(
    client,
    ingestion_service,
):
    payload = valid_payload()
    payload["observations"][0]["traffic_volume"] = True

    response = client.post(
        "/v1/traffic/observations",
        json=payload,
    )

    assert response.status_code == 422
    assert ingestion_service.calls == []


def test_ingestion_rejects_unexpected_fields(
    client,
    ingestion_service,
):
    payload = valid_payload()
    payload["unexpected"] = "not allowed"

    response = client.post(
        "/v1/traffic/observations",
        json=payload,
    )

    assert response.status_code == 422
    assert ingestion_service.calls == []


def test_ingestion_rejects_extra_observation_fields(
    client,
    ingestion_service,
):
    payload = valid_payload()
    payload["observations"][0]["unexpected"] = "not allowed"

    response = client.post(
        "/v1/traffic/observations",
        json=payload,
    )

    assert response.status_code == 422
    assert ingestion_service.calls == []


def test_ingestion_rejects_batch_over_limit(
    client,
    ingestion_service,
):
    payload = {
        "observations": [
            {
                "road_id": "R1",
                "timestamp": "2025-01-01T00:00:00",
                "traffic_volume": 10,
            }
        ]
        * 1001
    }

    response = client.post(
        "/v1/traffic/observations",
        json=payload,
    )

    assert response.status_code == 422
    assert ingestion_service.calls == []


def test_ingestion_validation_error_returns_422(
    client,
    ingestion_service,
):
    ingestion_service.error = IngestionValidationError(
        "The batch contains duplicate road/timestamp keys."
    )

    response = client.post(
        "/v1/traffic/observations",
        json=valid_payload(),
    )

    assert response.status_code == 422
    assert "duplicate" in response.json()["detail"].lower()


def test_database_failure_returns_safe_503(
    client,
    ingestion_service,
):
    ingestion_service.error = SQLAlchemyError(
        "sensitive database internals"
    )

    response = client.post(
        "/v1/traffic/observations",
        json=valid_payload(),
    )

    assert response.status_code == 503

    body = response.json()

    assert body["detail"] == (
        "Traffic ingestion is temporarily unavailable."
    )

    assert "sensitive database internals" not in str(body)


def test_missing_ingestion_service_returns_503(history):
    app = create_app(history_data=history)

    with TestClient(app) as client:
        response = client.post(
            "/v1/traffic/observations",
            json=valid_payload(),
        )

    assert response.status_code == 503


def test_existing_forecast_endpoint_still_works(
    client,
):
    response = client.post(
        "/v1/forecast",
        json={
            "road_id": "R1",
            "target_timestamp": "2025-01-02T00:30:00",
            "strategy": "naive_30min",
        },
    )

    assert response.status_code == 200
    assert response.json()["predicted_traffic_volume"] == 150


def test_health_endpoint_still_works(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint_still_works(client):
    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"