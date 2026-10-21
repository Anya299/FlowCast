"""Day 23: FlowCast API contract and reliability tests."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.day19.api import create_app


@pytest.fixture
def history_data():
    """Provide deterministic history for both forecasting strategies."""
    timestamps = pd.date_range(
        "2025-01-01 08:00:00",
        periods=57,
        freq="30min",
    )

    return pd.DataFrame(
        {
            "road_id": ["TEST-ROAD-01"] * len(timestamps),
            "timestamp": timestamps,
            "traffic_volume": [
                20 + 5 * index for index in range(len(timestamps))
            ],
        }
    )


@pytest.fixture
def client(history_data):
    """Create an isolated application with injected history."""
    app = create_app(history_data=history_data)

    with TestClient(app) as test_client:
        yield test_client


class TestLivenessContract:
    def test_healthz_returns_success(self, client):
        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_healthz_includes_request_id_header(self, client):
        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.headers.get("X-Request-ID")

    def test_each_request_receives_a_unique_request_id(self, client):
        first = client.get("/healthz")
        second = client.get("/healthz")

        first_id = first.headers["X-Request-ID"]
        second_id = second.headers["X-Request-ID"]

        assert first_id
        assert second_id
        assert first_id != second_id


class TestReadinessContract:
    def test_readyz_reports_historical_dataset_metrics(self, client):
        response = client.get("/readyz")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "historical_records": 57,
            "unique_roads": 1,
        }

    def test_readyz_fails_when_forecast_service_is_missing(self, client):
        client.app.state.forecast_service = None

        response = client.get("/readyz")

        assert response.status_code == 503
        assert response.json()["detail"] == (
            "Forecasting service is not ready."
        )
        assert response.json()["request_id"]
        assert response.headers.get("X-Request-ID")


class TestForecastContract:
    def test_valid_forecast_returns_documented_fields(self, client):
        response = client.post(
            "/v1/forecast",
            json={
                "road_id": "TEST-ROAD-01",
                "target_timestamp": "2025-01-01T12:00:00",
                "strategy": "naive_30min",
            },
        )

        assert response.status_code == 200

        body = response.json()

        expected_fields = {
            "road_id",
            "target_timestamp",
            "reference_timestamp",
            "predicted_traffic_volume",
            "strategy",
            "horizon_minutes",
            "model_version",
            "request_id",
        }

        assert set(body) == expected_fields
        assert body["road_id"] == "TEST-ROAD-01"
        assert body["strategy"] == "naive_30min"
        assert body["predicted_traffic_volume"] >= 0
        assert body["horizon_minutes"] == 30
        assert body["reference_timestamp"] == "2025-01-01T11:30:00"
        assert body["request_id"]
        assert response.headers.get("X-Request-ID")

    def test_forecast_supports_seasonal_strategy(self, client):
        response = client.post(
            "/v1/forecast",
            json={
                "road_id": "TEST-ROAD-01",
                "target_timestamp": "2025-01-02T12:00:00",
                "strategy": "seasonal_naive_24h",
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert body["strategy"] == "seasonal_naive_24h"
        assert body["reference_timestamp"] == "2025-01-01T12:00:00"
        assert body["horizon_minutes"] == 1440
        assert body["predicted_traffic_volume"] == 60.0

    def test_unknown_road_returns_not_found(self, client):
        response = client.post(
            "/v1/forecast",
            json={
                "road_id": "UNKNOWN-ROAD",
                "target_timestamp": "2025-01-01T12:00:00",
            },
        )

        assert response.status_code == 404
        assert response.json()["detail"]
        assert response.json()["request_id"]

    def test_missing_historical_reference_returns_not_found(self, client):
        response = client.post(
            "/v1/forecast",
            json={
                "road_id": "TEST-ROAD-01",
                "target_timestamp": "2025-01-01T08:00:00",
                "strategy": "naive_30min",
            },
        )

        assert response.status_code == 404
        assert "historical reference" in response.json()["detail"].lower()
        assert response.json()["request_id"]

    def test_invalid_strategy_is_rejected(self, client):
        response = client.post(
            "/v1/forecast",
            json={
                "road_id": "TEST-ROAD-01",
                "target_timestamp": "2025-01-01T12:00:00",
                "strategy": "invented_strategy",
            },
        )

        assert response.status_code == 422

    def test_timezone_aware_timestamp_is_rejected(self, client):
        response = client.post(
            "/v1/forecast",
            json={
                "road_id": "TEST-ROAD-01",
                "target_timestamp": "2025-01-01T12:00:00+05:30",
            },
        )

        assert response.status_code == 422

    def test_blank_road_id_is_rejected(self, client):
        response = client.post(
            "/v1/forecast",
            json={
                "road_id": "   ",
                "target_timestamp": "2025-01-01T12:00:00",
            },
        )

        assert response.status_code == 422

    def test_unexpected_request_fields_are_rejected(self, client):
        response = client.post(
            "/v1/forecast",
            json={
                "road_id": "TEST-ROAD-01",
                "target_timestamp": "2025-01-01T12:00:00",
                "unexpected_field": "not-allowed",
            },
        )

        assert response.status_code == 422

    def test_missing_required_timestamp_is_rejected(self, client):
        response = client.post(
            "/v1/forecast",
            json={"road_id": "TEST-ROAD-01"},
        )

        assert response.status_code == 422


class TestIngestionRequestContract:
    def test_ingestion_returns_service_unavailable_without_service(
        self,
        client,
    ):
        response = client.post(
            "/v1/traffic/observations",
            json={
                "observations": [
                    {
                        "road_id": "CONTRACT-ROAD",
                        "timestamp": "2025-02-01T08:00:00",
                        "traffic_volume": 12,
                    }
                ]
            },
        )

        assert response.status_code == 503
        assert response.json()["detail"] == (
            "Traffic ingestion service is not configured."
        )
        assert response.json()["request_id"]

    def test_empty_observation_batch_is_rejected(self, client):
        response = client.post(
            "/v1/traffic/observations",
            json={"observations": []},
        )

        assert response.status_code == 422

    def test_negative_traffic_volume_is_rejected(self, client):
        response = client.post(
            "/v1/traffic/observations",
            json={
                "observations": [
                    {
                        "road_id": "CONTRACT-ROAD",
                        "timestamp": "2025-02-01T08:00:00",
                        "traffic_volume": -1,
                    }
                ]
            },
        )

        assert response.status_code == 422

    def test_boolean_traffic_volume_is_rejected(self, client):
        response = client.post(
            "/v1/traffic/observations",
            json={
                "observations": [
                    {
                        "road_id": "CONTRACT-ROAD",
                        "timestamp": "2025-02-01T08:00:00",
                        "traffic_volume": True,
                    }
                ]
            },
        )

        assert response.status_code == 422

    def test_extra_observation_field_is_rejected(self, client):
        response = client.post(
            "/v1/traffic/observations",
            json={
                "observations": [
                    {
                        "road_id": "CONTRACT-ROAD",
                        "timestamp": "2025-02-01T08:00:00",
                        "traffic_volume": 12,
                        "unapproved_field": "unexpected",
                    }
                ]
            },
        )

        assert response.status_code == 422

    def test_batch_above_maximum_size_is_rejected(self, client):
        observations = [
            {
                "road_id": f"CONTRACT-ROAD-{index}",
                "timestamp": "2025-02-01T08:00:00",
                "traffic_volume": 12,
            }
            for index in range(1001)
        ]

        response = client.post(
            "/v1/traffic/observations",
            json={"observations": observations},
        )

        assert response.status_code == 422


class TestOpenAPIContract:
    def test_openapi_documents_expected_routes(self, client):
        response = client.get("/openapi.json")

        assert response.status_code == 200

        paths = response.json()["paths"]

        assert "/healthz" in paths
        assert "/readyz" in paths
        assert "/v1/forecast" in paths
        assert "/v1/traffic/observations" in paths

    def test_forecast_route_documents_post_method(self, client):
        response = client.get("/openapi.json")

        forecast_path = response.json()["paths"]["/v1/forecast"]

        assert "post" in forecast_path

    def test_ingestion_route_documents_post_method(self, client):
        response = client.get("/openapi.json")

        ingestion_path = response.json()["paths"][
            "/v1/traffic/observations"
        ]

        assert "post" in ingestion_path
