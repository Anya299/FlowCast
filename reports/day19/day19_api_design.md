# FlowCast Day 19 — Forecasting API

## Objective

Expose FlowCast's existing historical-reference forecasting
strategies through a versioned FastAPI service.

## Architecture

- `schemas.py`: request and response contracts.
- `service.py`: historical-data validation and forecast logic.
- `api.py`: HTTP routes, application lifecycle, request IDs,
  and HTTP error handling.
- `tests/test_day19_api.py`: deterministic API and service tests.

## Supported strategies

- `naive_30min`: uses the observation 30 minutes before the
  target timestamp.
- `seasonal_naive_24h`: uses the observation 24 hours before
  the target timestamp.

## Endpoints

- `GET /healthz`: liveness.
- `GET /readyz`: readiness.
- `POST /v1/forecast`: forecast request.
- `GET /docs`: interactive API documentation.

## Data integrity

The service rejects duplicate road/timestamp keys, invalid
timestamps, missing required fields, non-finite traffic values,
and negative traffic volumes.

A forecast is returned only when the exact historical reference
observation is available.

## Validation

Record the actual test results after running the Day 19 test
suite and the complete project test suite.

Record the API smoke-test outcome and any issues found.

## Limitations

This release serves historical-reference baselines. It is not
a trained machine-learning model, does not implement database
persistence, and does not yet provide authentication, rate
limiting, or production deployment hardening.

## Next steps

Introduce a managed data-access layer, asynchronous database
integration where appropriate, authentication, and stronger
operational monitoring in subsequent engineering milestones.

## Verified Test Results

- Day 19 API tests: 12 passed.
- Complete project test suite: 115 passed.
- Total test failures: 0.
- API health endpoint: passed.
- API readiness endpoint: passed.
- Historical record count reported by readiness: 109460.
- Unique road count reported by readiness: 25.
- Naive 30-minute API smoke test: passed.
- Seasonal strategy manual smoke test: requires a valid historical reference timestamp.
- Warning: Starlette reports a deprecation warning concerning the use of httpx with its test client.

The successful test suite validates the tested application behavior. It does not establish forecasting accuracy, production readiness, or deployment security.
