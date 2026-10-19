# FlowCast — Day 6 Exploratory Data Analysis

## 1. Executive summary

- Source rows: 176,701
- Source columns: 29
- Unique roads: 25
- Invalid timestamps: 0
- Missing cells: 383,765
- Columns containing missing values: 8
- First timestamp: 2025-01-01 00:00:00
- Last timestamp: 2025-05-31 23:30:00

## 2. Traffic-volume distribution

- Valid numeric observations: 172,114
- Missing values: 4,587
- Mean: 431.290
- Median: 362.000
- Standard deviation: 283.4066639965903
- Minimum: 41.000
- Maximum: 2090.000
- 90th percentile: 846.000
- 99th percentile: 1233.000
- Skewness: 1.0028996796143044
- Negative observations: 0

## 3. Hourly traffic patterns

- Highest mean traffic hour: 08:00
- Highest mean traffic volume: 886.152
- Lowest mean traffic hour: 00:00
- Lowest mean traffic volume: 162.233

These findings describe the aggregate data. Road-specific patterns may differ.

## 4. Road-level comparison

| Road ID | Mean traffic | Median traffic | Maximum traffic | Missing traffic % |
|---|---:|---:|---:|---:|
| NL-006 | 564.455 | 517.000 | 2090.000 | 2.51% |
| NL-019 | 562.914 | 510.000 | 1949.000 | 2.63% |
| NL-004 | 538.058 | 489.000 | 1857.000 | 2.45% |
| NL-002 | 536.933 | 480.000 | 2086.000 | 2.57% |
| NL-005 | 511.033 | 468.000 | 1713.000 | 2.80% |
| NL-023 | 504.267 | 442.500 | 1889.000 | 2.56% |
| NL-016 | 491.989 | 431.000 | 1829.000 | 2.78% |
| NL-014 | 488.950 | 411.000 | 2059.000 | 2.78% |
| NL-015 | 484.476 | 444.000 | 1551.000 | 2.44% |
| NL-008 | 477.132 | 405.000 | 1731.000 | 2.45% |

## 5. Congestion distribution

- Missing labels: 26,624
- Non-missing labels: 150,077
- Distinct categories: 4

| Category | Count | Percentage |
|---|---:|---:|
| free-flow | 92,107 | 61.37% |
| moderate | 35,818 | 23.87% |
| heavy | 13,873 | 9.24% |
| severe | 8,279 | 5.52% |

## 6. Weather and traffic relationships

| Weather variable | Paired observations | Pearson correlation |
|---|---:|---:|
| temperature | 169,475 | 0.3863 |
| rainfall | 172,114 | -0.0381 |
| visibility | 170,361 | 0.0729 |

Pearson correlation measures linear association. It does not establish causation. Road differences, time-of-day patterns, missing values, and other confounding factors may influence the results.

## 7. Numeric feature relationships

| Feature 1 | Feature 2 | Correlation |
|---|---|---:|
| traffic_volume | vehicle_count | 1.0000 |
| traffic_volume | occupancy | 0.8998 |
| vehicle_count | occupancy | 0.8985 |
| traffic_volume | signal_timing | 0.7609 |
| vehicle_count | signal_timing | 0.7607 |
| occupancy | signal_timing | 0.7605 |
| rainfall | visibility | -0.7591 |
| traffic_volume | avg_speed | -0.7250 |
| avg_speed | occupancy | -0.7248 |
| vehicle_count | avg_speed | -0.7200 |
| avg_speed | signal_timing | -0.6139 |
| occupancy | travel_time | 0.5947 |
| avg_speed | speed_above_200_flag | 0.5891 |
| avg_speed | travel_time | -0.5403 |
| vehicle_count | travel_time | 0.5364 |

## 8. Missingness

| Column | Missing count | Missing % |
|---|---:|---:|
| holiday_name | 169,690 | 96.03% |
| event_name | 169,663 | 96.02% |
| congestion_level | 26,624 | 15.07% |
| traffic_volume | 4,587 | 2.60% |
| occupancy | 4,344 | 2.46% |
| avg_speed | 4,343 | 2.46% |
| temperature | 2,708 | 1.53% |
| visibility | 1,806 | 1.02% |
| road_name | 0 | 0.00% |
| weather_station_id | 0 | 0.00% |
| longitude | 0 | 0.00% |
| latitude | 0 | 0.00% |
| road_id | 0 | 0.00% |
| vehicle_type_dist | 0 | 0.00% |
| date | 0 | 0.00% |

## 9. Generated visualizations

- `figures/missingness_by_column.png`
- `figures/traffic_volume_distribution.png`
- `figures/traffic_volume_boxplot.png`
- `figures/traffic_volume_by_hour.png`
- `figures/top_roads_by_mean_traffic.png`
- `figures/congestion_category_distribution.png`
- `figures/numeric_correlation_heatmap.png`
- `figures/traffic_volume_by_weekday.png`

Additional weather scatter plots are generated when suitable columns are available.

## 10. Engineering limitations

1. The source dataset was not overwritten.
2. Missing measurements were not imputed.
3. Extreme values were not automatically deleted.
4. Correlation was not interpreted as causation.
5. Aggregate statistics may hide road-specific or temporal differences.
6. The current EDA is descriptive. It does not measure forecasting performance.
7. Any numeric parsing failures require investigation before model training.

## 11. Next steps

Use the findings to plan leakage-safe feature engineering, investigate data-quality concerns, define forecasting targets, and establish a time-aware train/validation/test strategy.

## 12. Reproducibility

Run:

```powershell
python src/day6_exploratory_data_analysis.py
```
