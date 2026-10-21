# FlowCast — Day 21: Production-Grade Data Ingestion

## Objective

Introduce a validated ingestion service that reuses the existing
PostgreSQL observation repository without changing the Day 20 schema.

## Architecture

The ingestion service validates and normalizes incoming observations.
The existing repository owns PostgreSQL persistence, duplicate handling,
and transaction management.

## Input Contract

Each record contains exactly:
- road_id
- timestamp
- traffic_volume

Road IDs are trimmed and limited to 100 characters.
Traffic volumes must be finite, numeric, and nonnegative.

## Timestamp Policy

Timezone-aware timestamps are normalized to UTC and converted to
timezone-naive values for the existing database column.

Naive timestamps are preserved and must follow the source's documented
canonical time convention.

## Batch Semantics

All records are validated before the repository is called.
Duplicate composite keys within a request are rejected.
Existing database keys are skipped according to repository behavior.
Database exceptions propagate to the caller.

## Testing

Record the actual output of:
- python -m pytest tests/day21 -q
- python -m pytest tests/day20 -q
- python -m pytest -q
- python -m src.day20.verify_database

## Limitations

The service does not yet expose an HTTP endpoint, retry failed database
transactions, or provide durable per-record ingestion audit logs.

The reported skipped-duplicate count is derived from the repository's
insert count, rather than from a separate database audit.

## Result

Record the actual test counts and database verification result here.