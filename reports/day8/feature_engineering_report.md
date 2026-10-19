# FlowCast — Day 8 Feature Engineering Report

## 1. Dataset overview

- Input rows: 176,701
- Feature dataset rows: 176,701
- Model dataset rows: 176,701
- Engineered columns: 55
- Baseline model features: 28

## 2. Historical lag features

- `traffic_lag_30m`
- `traffic_lag_1h`
- `traffic_lag_90m`
- `traffic_lag_24h`
- `traffic_lag_7d`

## 3. Rolling statistics

- `traffic_rolling_mean_3h`
- `traffic_rolling_mean_6h`
- `traffic_rolling_mean_24h`
- `traffic_rolling_std_24h`

## 4. Missing historical information

Missing lag and rolling values are retained. They can occur at the beginning of the dataset, when observations are missing, or when a road lacks the required historical record.

| Feature | Missing rows |
|---|---:|
| traffic_lag_30m | 8,889 |
| traffic_lag_1h | 8,893 |
| traffic_lag_90m | 8,925 |
| traffic_lag_24h | 9,956 |
| traffic_lag_7d | 16,610 |
| traffic_rolling_mean_3h | 28 |
| traffic_rolling_mean_6h | 28 |
| traffic_rolling_mean_24h | 28 |
| traffic_rolling_std_24h | 54 |

## 5. Leakage safeguards

- Historical lags use exact timestamps.
- Rolling windows exclude the current timestamp.
- The target is excluded from the baseline feature list.
- Contemporaneous sensor measurements are excluded from baseline model inputs.
- Observed weather is not included in the baseline until future availability is confirmed.
- Rows are preserved even when lag features are missing.

## 6. Output artifacts

- `data/processed/flowcast_features.csv`
- `data/processed/flowcast_model_features.csv`
- `reports/day8/feature_catalog.csv`
- `reports/day8/feature_engineering_summary.json`
- `reports/day8/feature_engineering_report.md`
- `reports/day8/leakage_control_report.md`

## 7. Important limitation

This stage creates features but does not train a model or demonstrate forecasting accuracy. Predictive performance must be evaluated with chronological validation and appropriate baselines.
