# FlowCast Day 18 — Release Quality Gate

- **Release status:** FAIL
- **Created UTC:** 2026-10-20T02:32:59.252046+00:00
- **Input rows:** 61576
- **Models found:** naive_30min, seasonal_naive_24h
- **Minimum coverage:** 80.00%
- **Checks passed:** 9
- **Checks failed:** 3

## Quality checks

| Check | Result | Severity | Details |
|---|---|---|---|
| valid_timestamps | PASS | error | Invalid timestamps: 0 |
| valid_road_identifiers | PASS | error | Every prediction must have a non-empty road identifier. |
| unique_prediction_keys | PASS | error | Duplicate road/timestamp/model rows: 0 |
| finite_actual_values | PASS | error | Missing or non-finite actual values: 0 |
| recognized_models | PASS | error | Observed models: ['naive_30min', 'seasonal_naive_24h']; unknown models: [] |
| both_baselines_present | PASS | error | Missing expected models: [] |
| forecast_horizon_alignment | PASS | error | naive_30min: incorrect horizons=0, incorrect non-missing references=0; seasonal_naive_24h: incorrect horizons=0, incorrect non-missing references=0 |
| prediction_values_are_finite_or_missing | FAIL | warning | Missing predictions are allowed; missing or non-finite prediction values: 60472 |
| coverage_naive_30min | FAIL | error | Coverage=0.08%; required minimum=80.00%. |
| coverage_seasonal_naive_24h | FAIL | error | Coverage=3.51%; required minimum=80.00%. |
| common_population_available | PASS | error | Common valid prediction records: 44; common comparison rows: 2. |
| naive_baseline_comparison | PASS | warning | Common-population MAE: naive=14.1364; seasonal=15.5909. |

## Individual model metrics

             model  total_predictions  valid_predictions  missing_or_invalid_predictions  coverage_pct       mae      rmse      bias
       naive_30min              30788                 24                           30764      0.077952 14.583333 19.474342 -0.583333
seasonal_naive_24h              30788               1080                           29708      3.507860 42.572222 63.060817 13.068519

## Common-population comparison

             model  common_observations       mae      rmse     bias
       naive_30min                   22 14.136364 19.260180 0.227273
seasonal_naive_24h                   22 15.590909 19.091883 7.500000

## Interpretation

The release gate checks structural validity, prediction alignment, coverage, and baseline comparison.

A PASS means the configured structural and coverage checks passed. It does not guarantee real-world forecasting accuracy or deployment readiness.

A baseline comparison failure is a warning rather than an automatic release failure. Baseline performance must be evaluated on the same observations.
