# FlowCast Data Quality Policy

## 1. Purpose

This policy defines how FlowCast identifies missing, invalid,
suspicious, and duplicate traffic observations.

The objective is to protect data integrity before feature
engineering, model training, and traffic prediction.

## 2. Data preservation

- Original source datasets must remain unchanged.
- Validation identifies issues; it does not silently repair them.
- Suspicious observations must not be automatically deleted.
- Every reported issue should be traceable to its source row.
- Cleaning decisions must be documented and justified.

## 3. Schema and identity rules

The integrated dataset must contain:

- `road_id`
- `date`
- `time`

The validation script also checks the expected project schema.
Missing expected columns are reported as schema problems.

Rows without a road identifier, date, or time are considered
unusable for reliable road-time analysis until investigated.

Invalid timestamps are reported as errors.

## 4. Missing-value policy

Missingness is measured separately for every column.

- Missing road identifiers and timestamps require investigation.
- Missing traffic targets must not be silently imputed.
- Missing features may require training-time imputation,
  depending on the intended model and feature availability.
- Missing congestion labels affect classification training.
- Sparse `holiday_name` and `event_name` values may be expected
  when no holiday or named event applies.
- A missing value must not automatically be interpreted as zero.

Imputation decisions must be made separately for each prediction
task and must be fitted using training data only.

## 5. Measurement rules

The following values are considered invalid candidates:

- Negative traffic volume.
- Negative vehicle count.
- Negative travel time.
- Negative accident count.
- Negative rainfall.

These findings must be investigated before the observations
are used in training.

## 6. Suspicious-range screening

The following broad screening thresholds are used:

- Average speed below 0 or above 200.
- Occupancy below 0 or above 100.

These are screening rules, not proof that every flagged value
is erroneous.

Flagged records must be reviewed against the source data,
measurement definitions, and sensor documentation.

## 7. Duplicate records

Records sharing the same `road_id`, `date`, and `time` are
reported for investigation.

A duplicate key is not automatically proof of an erroneous
record. The intended observation frequency and source-system
semantics must be checked before any record is removed.

## 8. Validation severity

- ERROR: a structural or measurement problem requiring attention.
- REVIEW: a suspicious observation requiring investigation.
- INFO: a diagnostic finding that does not independently prove
  that the record is invalid.

## 9. Model-specific usability

A row can be usable for one task and unusable for another.

For example:

- Missing traffic volume prevents supervised training when
  traffic volume is the target.
- Missing congestion labels prevent supervised classification
  training for that row.
- Missing a feature may be manageable if the training pipeline
  implements an appropriate, leakage-safe imputation strategy.

Do not discard a row globally just because one column is missing.

## 10. Output and auditability

The validation process generates:

- A JSON summary.
- A CSV of validation findings.
- A CSV of missingness statistics.
- A CSV of duplicate road-time records.

This policy and its validation rules must be reviewed whenever
the dataset schema or sensor definitions change.
