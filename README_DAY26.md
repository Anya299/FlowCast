# FlowCast Day 26 — Model Reliability and Error Analysis

## Goal
Evaluate the fixed Day 25 traffic-volume model by road, hour, weekday, and peak/off-peak period; inspect residuals and large errors; and create an explicit feature-availability audit. This day does not train or tune a model.

## Inputs
- `models/day25/traffic_model.joblib`
- `data/processed/day9/flowcast_test.csv`

## Run
From the FlowCast repository root with `.venv` activated:

```powershell
python -m pytest tests/day26 -q
python src/day26/run_analysis.py
```

## Outputs
Written under `reports/day26/`:
- `day26_reliability_summary.json`
- `DAY26_RELIABILITY_REPORT.md`
- `test_predictions_with_residuals.csv`
- `top_20_absolute_errors.csv`
- `feature_availability_audit.json`
- `error_by_road.csv` (when a road identifier is available)
- `error_by_hour.csv`
- `error_by_weekday.csv`
- `error_by_period.csv`
- `error_by_calendar_weekday.csv` (when timestamps are available)
- PNG diagnostic plots if Matplotlib is installed

## Interpretation rules
- Positive residual (`prediction - actual`) means overprediction; negative means underprediction.
- Group tables are descriptive. Small groups under 30 rows are omitted from the default road ranking.
- MAPE excludes zero-target rows and must not be used alone.
- The test split has already been used for Day 25 evaluation. Do not use it to select features or tune hyperparameters.
- A feature appearing in a historical CSV does not prove it was available at the forecast cutoff. Manually verify event, holiday, roadwork, lag and rolling-window construction.
