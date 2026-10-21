# FlowCast — Day 20: PostgreSQL Database Layer

## Objective

Implement a maintainable database access layer for historical traffic
observations using SQLAlchemy 2.0 and PostgreSQL.

## Architecture

- database.py: engine configuration, session factory, transaction context.
- models.py: ORM schema and database constraints.
- repository.py: validation, inserts, batch operations, and historical queries.
- schema.py: initial table creation for development.
- verify_database.py: read-only database and schema verification.
- tests/day20/test_repository.py: PostgreSQL integration tests.

## Data Model

Table: traffic_observations

Composite primary key:
- road_id
- timestamp

Other fields:
- traffic_volume
- created_at

## Integrity Controls

- Nonempty road identifiers.
- Nonnegative traffic volumes.
- Finite traffic-volume values at the application boundary.
- Duplicate prevention through the composite primary key.
- PostgreSQL conflict handling for repeated inserts.
- Transactional batch insertion.
- Explicit timestamp validation.

## Query Operations

- Retrieve one observation by its composite key.
- Retrieve observations within a half-open historical interval [start, end).
- Retrieve the latest observation for a road.
- Insert one observation or a batch of observations.

## Testing

Run:

    python -m pytest tests/day20/test_repository.py -v

The tests require a PostgreSQL database configured through TEST_DATABASE_URL
or DATABASE_URL. Prefer a dedicated test database.

## Verification

Run:

    python -m src.day20.verify_database

Record the actual result here after executing the commands.

- PostgreSQL connection: PASS - PostgreSQL 17.11; database `flowcast`.
- Schema verification: PASS - `traffic_observations`, composite primary key, and database check constraints verified.
- Integration tests: PASS - 16 tests passed in 2.69 seconds.

## Engineering Limitations

- The initial timestamp schema is timezone-naive.
- create_all() does not migrate existing tables.
- Alembic migrations are required for managed schema evolution.
- Pool sizing must be tuned for deployment topology and database limits.
- Query pagination and retention policies are not implemented in this step.

## Next Step

Add managed schema migrations, repository-level pagination, and integration
with the FlowCast ingestion pipeline after validating this database layer.
