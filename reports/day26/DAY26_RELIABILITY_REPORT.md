# FlowCast Day 26 — Model Reliability and Error Analysis

## Scope

This report evaluates the fixed Day 25 model on the chronological test partition. No model fitting or hyperparameter tuning is performed.

## Overall test metrics

| Metric | Value |
|---|---:|
| Rows | 31866 |
| MAE | 39.7615 |
| RMSE | 58.1351 |
| R² | 0.95912 |
| MAPE (non-zero targets only) | 9.4055% |
| Mean error (prediction − actual) | 1.9663 |

## Residual diagnostics

- Overprediction rows: 16411
- Underprediction rows: 15455
- 90th percentile absolute error: 95.2837
- 95th percentile absolute error: 126.7015
- 99th percentile absolute error: 198.0846

## Group-level error tables


### Error By Road

| road_id | rows | mae | rmse | r2 | mape_percent_nonzero_targets_only | mean_error_predicted_minus_actual | mean_absolute_percentage_error_denominator_rows_excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NL-019 | 1274 | 52.267 | 75.011 | 0.950 | 9.368 | -1.133 | 0 |
| NL-002 | 1281 | 51.316 | 73.515 | 0.953 | 9.568 | 1.115 | 0 |
| NL-006 | 1268 | 51.727 | 73.469 | 0.950 | 9.162 | -0.726 | 0 |
| NL-004 | 1278 | 48.611 | 68.325 | 0.953 | 9.200 | -1.193 | 0 |
| NL-016 | 1255 | 46.419 | 66.619 | 0.954 | 9.491 | 2.078 | 0 |
| NL-014 | 1270 | 44.003 | 66.091 | 0.960 | 9.223 | 4.560 | 0 |
| NL-015 | 1276 | 45.588 | 64.496 | 0.946 | 9.512 | 4.298 | 0 |
| NL-008 | 1271 | 44.439 | 64.188 | 0.960 | 9.759 | 7.075 | 0 |
| NL-005 | 1277 | 44.867 | 63.522 | 0.955 | 9.053 | 0.866 | 0 |
| NL-023 | 1269 | 44.515 | 63.500 | 0.961 | 9.016 | 2.762 | 0 |

### Error By Hour

| hour_of_day | rows | mae | rmse | r2 | mape_percent_nonzero_targets_only | mean_error_predicted_minus_actual | mean_absolute_percentage_error_denominator_rows_excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1333 | 77.234 | 99.033 | 0.849 | 8.756 | -0.266 | 0 |
| 9 | 1323 | 76.949 | 98.283 | 0.844 | 9.198 | -1.106 | 0 |
| 17 | 1331 | 70.491 | 91.133 | 0.851 | 9.037 | 3.094 | 0 |
| 18 | 1335 | 71.475 | 89.854 | 0.857 | 8.886 | -1.544 | 0 |
| 16 | 1314 | 59.261 | 77.189 | 0.840 | 9.481 | 4.421 | 0 |
| 19 | 1327 | 58.762 | 75.854 | 0.853 | 9.274 | 4.442 | 0 |
| 7 | 1332 | 57.850 | 75.136 | 0.864 | 9.402 | 9.222 | 0 |
| 10 | 1322 | 56.074 | 72.459 | 0.854 | 9.438 | 1.029 | 0 |
| 15 | 1332 | 44.872 | 58.128 | 0.847 | 9.412 | 5.727 | 0 |
| 11 | 1330 | 40.771 | 53.383 | 0.843 | 9.509 | 3.392 | 0 |

### Error By Weekday

| day_of_week_number | rows | mae | rmse | r2 | mape_percent_nonzero_targets_only | mean_error_predicted_minus_actual | mean_absolute_percentage_error_denominator_rows_excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 4547 | 46.887 | 67.961 | 0.958 | 9.249 | 5.558 | 0 |
| 2 | 4566 | 42.550 | 61.691 | 0.955 | 9.184 | 1.791 | 0 |
| 4 | 4532 | 42.769 | 61.611 | 0.957 | 9.334 | 5.095 | 0 |
| 0 | 4555 | 42.800 | 61.551 | 0.955 | 9.243 | -1.410 | 0 |
| 1 | 4575 | 42.452 | 61.220 | 0.956 | 9.120 | 0.542 | 0 |
| 5 | 4561 | 31.262 | 45.061 | 0.954 | 9.497 | 1.618 | 0 |
| 6 | 4530 | 29.575 | 43.194 | 0.955 | 10.215 | 0.592 | 0 |

### Error By Period

| period_of_day | rows | mae | rmse | r2 | mape_percent_nonzero_targets_only | mean_error_predicted_minus_actual | mean_absolute_percentage_error_denominator_rows_excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| morning_peak | 5310 | 67.032 | 87.132 | 0.888 | 9.198 | 2.227 | 0 |
| evening_peak | 5307 | 65.025 | 83.833 | 0.872 | 9.168 | 2.593 | 0 |
| off_peak_or_other | 21249 | 26.637 | 37.628 | 0.941 | 9.516 | 1.745 | 0 |

### Error By Calendar Weekday

| analysis_weekday | rows | mae | rmse | r2 | mape_percent_nonzero_targets_only | mean_error_predicted_minus_actual | mean_absolute_percentage_error_denominator_rows_excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Thursday | 4547 | 46.887 | 67.961 | 0.958 | 9.249 | 5.558 | 0 |
| Wednesday | 4566 | 42.550 | 61.691 | 0.955 | 9.184 | 1.791 | 0 |
| Friday | 4532 | 42.769 | 61.611 | 0.957 | 9.334 | 5.095 | 0 |
| Monday | 4555 | 42.800 | 61.551 | 0.955 | 9.243 | -1.410 | 0 |
| Tuesday | 4575 | 42.452 | 61.220 | 0.956 | 9.120 | 0.542 | 0 |
| Saturday | 4561 | 31.262 | 45.061 | 0.954 | 9.497 | 1.618 | 0 |
| Sunday | 4530 | 29.575 | 43.194 | 0.955 | 10.215 | 0.592 | 0 |

## Feature availability and leakage checklist

Status: **REQUIRES_MANUAL_VERIFICATION**

Presence in a dataset does not prove a feature was available at the prediction cutoff. Verify each source's publication/ingestion time.

### Manual checks

- **public_holiday** — Confirm the holiday calendar was published/known at the forecast cutoff.
- **event_flag** — Confirm event schedules were known at the forecast cutoff and do not encode events inferred from the observed target period.
- **roadwork_flag** — Confirm roadwork information was known at the forecast cutoff.

- **Lag/rolling features:** Every lag/rolling window must use observations strictly before the forecast timestamp; rolling windows must be shifted before aggregation. Confirm this in the Day 8 feature-generation code.

## Interpretation and limitations

- Group-level metrics are descriptive, not causal.
- MAPE excludes zero-target rows and should not be used alone.
- Do not tune against the test set; use validation data for model changes.
- A good retrospective score does not prove inputs were available at forecast time.
- The report is an analysis artifact, not a production monitoring guarantee.

## Generated artifacts

- `day26_reliability_summary.json`
- `test_predictions_with_residuals.csv`
- `top_20_absolute_errors.csv`
- `feature_availability_audit.json`
- Group-level CSV files and diagnostic plots (if matplotlib is installed)
