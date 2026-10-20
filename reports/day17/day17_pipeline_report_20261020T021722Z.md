# Day 17 — End-to-End Forecasting Pipeline

## 1. Objective

Integrate historical-reference forecasting strategies, prediction validation, metric calculation, and paired model comparison into a reproducible pipeline.

## 2. Dataset Split

- Training start: `2025-01-01T00:00:00`
- Training end: `2025-04-06T23:30:00`
- Validation start: `2025-04-07T00:00:00`
- Validation end: `2025-05-03T23:30:00`
- Chronological split passed: `True`

## 3. Prediction Integrity

- Audit passed: `True`
- Prediction rows: `61576`
- Missing predictions: `60472`

## 4. Per-Model Metrics

| model              |   total_observations |   valid_predictions |   missing_predictions |   coverage |   coverage_percent |     mae |    rmse |    bias |
|:-------------------|---------------------:|--------------------:|----------------------:|-----------:|-------------------:|--------:|--------:|--------:|
| naive_30min        |                30788 |                  24 |                 30764 |     0.0008 |             0.0780 | 14.5833 | 19.4743 | -0.5833 |
| seasonal_naive_24h |                30788 |                1080 |                 29708 |     0.0351 |             3.5079 | 42.5722 | 63.0608 | 13.0685 |

## 5. Common-Population Comparison

Shared road-timestamp observations: `22`

| model              |   total_observations |   valid_predictions |   missing_predictions |   coverage |   coverage_percent |     mae |    rmse |   bias |
|:-------------------|---------------------:|--------------------:|----------------------:|-----------:|-------------------:|--------:|--------:|-------:|
| naive_30min        |                   22 |                  22 |                     0 |     1.0000 |           100.0000 | 14.1364 | 19.2602 | 0.2273 |
| seasonal_naive_24h |                   22 |                  22 |                     0 |     1.0000 |           100.0000 | 15.5909 | 19.0919 | 7.5000 |

## 6. Metric Interpretation

- MAE: average absolute prediction error.
- RMSE: square-root mean squared prediction error.
- Bias: mean of predicted minus actual values.
- Coverage: proportion of observations with valid predictions.

Comparisons should be interpreted together with coverage. Per-model metrics can use different valid populations; the common-population table uses shared valid observations.

## 7. Limitations

This pipeline evaluates historical-reference baselines. It does not yet train a machine-learning model.

The evaluation audit does not prove complete leakage prevention. Feature lineage, prediction-time availability, preprocessing fitting, and test-set isolation require additional review.

## 8. Conclusion

The pipeline executed its configured baseline strategies and generated reproducible prediction and evaluation artifacts. See the output files and metadata for the exact run results.
