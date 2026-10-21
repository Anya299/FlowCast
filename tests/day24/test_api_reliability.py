"""Day 24: API reliability and observability tests."""

import logging

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.day19.api import create_app


@pytest.fixture
def history_data():
    """Create deterministic history for isolated API tests."""
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


class TestRequestTracing:
    def test_request_id_is_a_valid_uuid(self, client):
        from uuid import UUID

        response = client.get("/healthz")

        assert response.status_code == 200

        request_id = response.headers.get("X-Request-ID")
        assert request_id

        # Raises ValueError if the request ID is not a valid UUID.
        assert str(UUID(request_id)) == request_id

    def test_each_request_has_a_different_request_id(self, client):
        first = client.get("/healthz")
        second = client.get("/readyz")

        assert first.headers["X-Request-ID"]
        assert second.headers["X-Request-ID"]
        assert (
            first.headers["X-Request-ID"]
            != second.headers["X-Request-ID"]
        )

    def test_http_errors_include_matching_request_id(self, client):
        response = client.post(
            "/v1/forecast",
            json={
                "road_id": "UNKNOWN-ROAD",
                "target_timestamp": "2025-01-01T12:00:00",
            },
        )

        assert response.status_code == 404

        request_id = response.headers["X-Request-ID"]
        assert response.json()["request_id"] == request_id


class TestRequestObservability:
    def test_successful_request_is_logged(
        self,
        client,
        caplog,
    ):
        with caplog.at_level(logging.INFO, logger="flowcast.api"):
            response = client.get("/healthz")

        assert response.status_code == 200

        messages = [record.getMessage() for record in caplog.records]

        matching = [
            message
            for message in messages
            if "request_completed" in message
        ]

        assert matching, "Expected a request completion log."

        message = matching[-1]

        assert response.headers["X-Request-ID"] in message
        assert "method=GET" in message
        assert "path=/healthz" in message
        assert "status_code=200" in message
        assert "duration_ms=" in message

    def test_http_error_is_logged_with_status_code(
        self,
        client,
        caplog,
    ):
        with caplog.at_level(logging.INFO, logger="flowcast.api"):
            response = client.get("/not-a-real-route")

        assert response.status_code == 404

        messages = [record.getMessage() for record in caplog.records]

        matching = [
            message
            for message in messages
            if "request_completed" in message
        ]

        assert matching
        assert "status_code=404" in matching[-1]

    def test_invalid_request_is_logged_as_422(
        self,
        client,
        caplog,
    ):
        with caplog.at_level(logging.INFO, logger="flowcast.api"):
            response = client.post(
                "/v1/forecast",
                json={"road_id": "TEST-ROAD-01"},
            )

        assert response.status_code == 422

        messages = [record.getMessage() for record in caplog.records]

        matching = [
            message
            for message in messages
            if "request_completed" in message
        ]

        assert matching
        assert "status_code=422" in matching[-1]


class TestUnexpectedExceptions:
    def test_unexpected_exception_returns_safe_500_response(
        self,
        history_data,
    ):
        app = create_app(history_data=history_data)

        @app.get("/test-unexpected-failure")
        async def unexpected_failure():
            raise RuntimeError("private-internal-error-marker")

        # Starlette may re-raise server exceptions after generating
        # a 500 response. Disable re-raising for this HTTP contract test.
        with TestClient(
            app,
            raise_server_exceptions=False,
        ) as test_client:
            response = test_client.get("/test-unexpected-failure")

        assert response.status_code == 500

        body = response.json()

        assert body["detail"] == "Internal server error."
        assert body["request_id"]
        assert (
            "private-internal-error-marker"
            not in response.text
        )
        assert response.headers["X-Request-ID"] == body["request_id"]

    def test_unexpected_exception_is_logged(
        self,
        history_data,
        caplog,
    ):
        app = create_app(history_data=history_data)

        @app.get("/test-unexpected-failure")
        async def unexpected_failure():
            raise RuntimeError("private-internal-error-marker")

        with caplog.at_level(logging.ERROR, logger="flowcast.api"):
            with TestClient(
                app,
                raise_server_exceptions=False,
            ) as test_client:
                response = test_client.get(
                    "/test-unexpected-failure"
                )

        assert response.status_code == 500

        messages = [record.getMessage() for record in caplog.records]

        assert any(
            "Unhandled API exception" in message
            for message in messages
        )

        # The exception details may appear in trusted server logs,
        # but must not appear in the client response.
        assert (
            "private-internal-error-marker"
            not in response.text
        )
