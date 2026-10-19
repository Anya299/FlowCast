# Day 16 — Forecast Evaluation Integrity Audit

## 1. Objective

The objective of Day 16 is to establish automated safeguards for evaluating traffic forecasting models. The audit checks chronological data separation, observation uniqueness, feature-target separation, prediction alignment, and consistent evaluation populations.

## 2. Dataset and Temporal Separation

The experiment uses chronological partitions:

- Training period: January 1, 2025 – April 6, 2025
- Validation period: April 7, 2025 – May 3, 2025
- Test data: Maintained as a separate dataset for final evaluation.

The chronological audit verifies that the training and validation periods do not overlap. Empty partitions are rejected.

## 3. Observation Uniqueness

Each road and timestamp combination should identify one observation.

The audit detects duplicate road-timestamp records. Prediction records are also checked for duplicate keys so that repeated observations cannot silently distort evaluation metrics.

## 4. Feature-Target Separation

The forecasting target is `traffic_volume`.

The target must not appear in the model feature list. Explicitly forbidden features are also rejected, and duplicate feature names are treated as a configuration error.

These structural checks reduce common leakage risks. However, checking feature names alone cannot prove that every feature was generated using information available at prediction time.

## 5. Forecast Horizon and Reference Alignment

Baseline predictions must use historical reference observations that match their intended forecast horizons.

- Naive 30-minute baseline: reference timestamp equals forecast timestamp minus 30 minutes.
- Seasonal naive 24-hour baseline: reference timestamp equals forecast timestamp minus 24 hours.

The prediction-alignment audit checks the required prediction fields and validates reference-timestamp alignment when a forecast horizon is specified.

Missing historical references must not be silently replaced with future observations.

## 6. Common Evaluation Population

Comparisons between forecasting strategies must use the same valid observations.

The common-population audit:

1. Identifies observations for which all compared models have valid predictions.
2. Verifies that actual traffic values agree across models for each road and timestamp.
3. Rejects duplicate road-timestamp-model records.
4. Produces a consistent population for paired model comparisons.

Coverage must be reported alongside error metrics. A lower error on a smaller subset does not automatically indicate a better forecasting strategy.

## 7. Automated Test Coverage

The Day 16 test module contains 14 tests covering:

- Valid and overlapping chronological splits.
- Empty data partitions.
- Duplicate and unique road-timestamp records.
- Target leakage and forbidden features.
- Duplicate feature names.
- Correct and incorrect forecast-horizon alignment.
- Common evaluation populations.
- Disagreement between actual values.
- Duplicate model prediction records.

Day 16 validation result: 14 tests passed.

Full project validation result: 76 tests passed.

## 8. Additional Leakage Risks Requiring Review

Passing the automated tests does not establish that the entire forecasting pipeline is leakage-free. The following require separate verification:

- Historical lag and rolling-window features use only information available at forecast issuance.
- Weather, calendar, holiday, event, and roadwork features reflect what was actually known at issuance time.
- Imputation, scaling, feature selection, and other learned preprocessing steps are fitted on training data only.
- Target construction does not accidentally include future information in the input features.
- Test data remains isolated from model selection and tuning.
- Feature lineage and data availability are documented and reproducible.

## 9. Acceptance Criteria

Day 16 is accepted at the automated audit level when:

- All 14 Day 16 tests pass.
- The complete existing test suite passes.
- Invalid temporal splits and duplicate prediction keys are detected.
- Target and explicitly forbidden features are rejected.
- Forecast references are checked against their configured horizons.
- Paired model comparisons use a consistent valid observation population.

## 10. Conclusion

Day 16 introduces a dedicated evaluation-audit layer for FlowCast. It strengthens the reliability of forecasting comparisons by validating temporal separation, feature-target boundaries, prediction alignment, and common evaluation populations.

The passing tests demonstrate that the implemented audit rules behave as expected for the tested cases. Further feature-lineage and data-availability reviews remain necessary before claiming complete leakage prevention.
