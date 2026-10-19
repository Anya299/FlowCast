# FlowCast — Day 7 Data Cleaning Report

## 1. Dataset overview

- Input rows: 176,701
- Output rows: 176,701
- Duplicate rows separated: 0
- Rows requiring review: 465
- Unique road segments: 25
- Invalid timestamps: 0

## 2. Cleaning decisions

- Negative traffic volumes are converted to missing values.
- Speeds outside 0–200 km/h are converted to missing values.
- Occupancy outside 0–100% is converted to missing values.
- Nonpositive road capacity is invalid for volume-capacity calculations.
- Negative count, rainfall, visibility, and travel-time values are converted to missing values.
- Duplicate road-time records are separated into an audit file.
- Known congestion and weather labels are standardized.
- Missing congestion labels are derived only when valid traffic volume and road capacity are available.
- Rows with missing traffic targets are retained for traceability.
- The original integrated dataset is not modified.

## 3. Congestion derivation

V/C = traffic_volume / (road_capacity / 2)

| V/C ratio | Congestion label |
|---|---|
| < 0.50 | Free-flow |
| 0.50 to < 0.80 | Moderate |
| 0.80 to < 1.00 | Heavy |
| >= 1.00 | Severe |

Labels derived: 25,943

## 4. Missing values after cleaning

| Column | Missing values |
|---|---:|
| holiday_name | 169,690 |
| event_name | 169,663 |
| traffic_volume | 4,587 |
| volume_capacity_ratio | 4,587 |
| avg_speed | 4,579 |
| occupancy | 4,573 |
| temperature | 2,708 |
| visibility | 1,806 |
| congestion_level | 681 |
| source_row_number | 0 |
| road_id | 0 |
| road_name | 0 |
| latitude | 0 |
| longitude | 0 |
| weather_station_id | 0 |
| date | 0 |
| time | 0 |
| vehicle_count | 0 |
| vehicle_type_dist | 0 |
| travel_time | 0 |
| accident_count | 0 |
| signal_timing | 0 |
| road_capacity | 0 |
| speed_above_200_flag | 0 |
| occupancy_above_100_flag | 0 |
| hour | 0 |
| weather_condition | 0 |
| rainfall | 0 |
| public_holiday | 0 |
| event_flag | 0 |
| roadwork_flag | 0 |
| timestamp | 0 |

## 5. Numeric parsing failures

| Column | Failed conversions |
|---|---:|
| accident_count | 0 |
| avg_speed | 0 |
| latitude | 0 |
| longitude | 0 |
| occupancy | 0 |
| rainfall | 0 |
| road_capacity | 0 |
| signal_timing | 0 |
| temperature | 0 |
| traffic_volume | 0 |
| travel_time | 0 |
| vehicle_count | 0 |
| visibility | 0 |

## 6. Output artifacts

- `data/processed/flowcast_cleaned.csv`
- `reports/day7/cleaning_summary.json`
- `reports/day7/rows_requiring_review.csv`
- `reports/day7/duplicate_records.csv`
- `reports/day7/cleaning_report.md`

## 7. Limitations and next steps

This pipeline does not automatically impute missing traffic targets or sensor measurements. Imputation decisions require time-aware analysis and must avoid information leakage.

The review file identifies records with rule violations; it does not establish the true value of a faulty sensor reading.

Before modelling, verify feature availability at prediction time and ensure all validation splits respect chronology.
