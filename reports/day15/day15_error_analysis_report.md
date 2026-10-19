# FlowCast Day 15 — Forecast Error Analysis

## 1. Objective

Day 15 evaluates the reliability of the Day 14 forecasting
experiments through error analysis, prediction coverage,
road-level diagnostics, and temporal performance analysis.

The purpose is to understand not only which forecasting
strategy has lower error, but also where and when the
strategies perform poorly.

## 2. Evaluation scope

- Total prediction records: **61,576**
- Common valid observations: **27,780**
- Common-population coverage relative to prediction records:
  **45.11%**
- Forecasting strategies: naive 30-minute and seasonal naive
  24-hour.

## 3. Evaluation methodology

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual
and predicted values. Lower values indicate smaller average
forecast errors.

### Root Mean Squared Error (RMSE)

RMSE gives greater weight to larger errors because the
individual errors are squared before averaging.

### Bias

Bias is calculated as:

`mean(predicted - actual)`

Positive values indicate overprediction; negative values
indicate underprediction.

### Prediction coverage

Prediction coverage is the percentage of validation rows
for which a model produced a valid prediction.

### Common-population evaluation

Both forecasting strategies are evaluated on the same
road/timestamp observations for which both models have
valid predictions. This makes their error metrics directly
comparable on that shared subset.

## 4. Common-population comparison

| model              |   total_validation_observations |   common_valid_observations |   common_population_coverage_pct |     mae |     rmse |    bias |
|:-------------------|--------------------------------:|----------------------------:|---------------------------------:|--------:|---------:|--------:|
| naive_30min        |                           30788 |                       27780 |                          90.2300 | 68.6709 |  99.7570 |  0.4062 |
| seasonal_naive_24h |                           30788 |                       27780 |                          90.2300 | 94.4847 | 140.8370 | -0.0850 |

**Lowest common-population MAE:** `naive_30min`

The common-population comparison uses only paired valid
predictions. Its results should not be confused with metrics
calculated independently over each model's full set of valid
predictions.

## 5. Road-level diagnostics

The following table shows up to ten road/model combinations
with the largest MAE.

| road_id   | model       |   validation_rows |   valid_predictions |   missing_predictions |   coverage_pct |     mae |     rmse |    bias |
|:----------|:------------|------------------:|--------------------:|----------------------:|---------------:|--------:|---------:|--------:|
| NL-006    | naive_30min |              1240 |                1187 |                    53 |        95.7258 | 87.2662 | 122.7368 |  0.5013 |
| NL-014    | naive_30min |              1225 |                1157 |                    68 |        94.4490 | 84.4572 | 122.4875 |  3.4183 |
| NL-019    | naive_30min |              1231 |                1169 |                    62 |        94.9634 | 82.5697 | 118.5171 | -0.3097 |
| NL-008    | naive_30min |              1239 |                1183 |                    56 |        95.4802 | 82.1412 | 121.6281 | -0.0533 |
| NL-002    | naive_30min |              1232 |                1170 |                    62 |        94.9675 | 81.9658 | 117.1013 | -1.3043 |
| NL-010    | naive_30min |              1229 |                1165 |                    64 |        94.7925 | 81.2455 | 114.3482 |  0.2343 |
| NL-004    | naive_30min |              1231 |                1171 |                    60 |        95.1259 | 81.2391 | 114.5375 | -0.3903 |
| NL-023    | naive_30min |              1237 |                1180 |                    57 |        95.3921 | 80.6144 | 114.7093 |  1.3347 |
| NL-016    | naive_30min |              1223 |                1153 |                    70 |        94.2764 | 78.1735 | 112.8378 |  0.0711 |
| NL-005    | naive_30min |              1219 |                1144 |                    75 |        93.8474 | 74.2421 | 104.4726 |  0.4904 |

**Highest observed road/model MAE:** NL-006 (seasonal_naive_24h), MAE=128.6913

Road-level metrics help identify locations that may need
additional investigation. High error alone does not establish
the underlying cause.

## 6. Hourly error analysis

Hourly MAE and RMSE help reveal whether forecasting errors
change throughout the day.

| model              |   hour |   observations |   validation_rows |      mae |     rmse |      bias |
|:-------------------|-------:|---------------:|------------------:|---------:|---------:|----------:|
| naive_30min        |      0 |           1219 |              1285 |  27.4495 |  37.2321 |    0.2633 |
| naive_30min        |      1 |           1230 |              1286 |  18.5008 |  24.4224 |    0.1984 |
| naive_30min        |      2 |           1222 |              1289 |  17.7291 |  23.3303 |   -0.2152 |
| naive_30min        |      3 |           1240 |              1292 |  18.0024 |  23.3094 |   -1.8879 |
| naive_30min        |      4 |           1228 |              1288 |  19.3428 |  25.8628 |   -6.0643 |
| naive_30min        |      5 |           1234 |              1287 |  30.8185 |  39.6926 |  -24.0778 |
| naive_30min        |      6 |           1212 |              1275 |  86.3952 | 102.8881 |  -85.5487 |
| naive_30min        |      7 |           1204 |              1275 | 155.0839 | 179.5921 | -152.7716 |
| naive_30min        |      8 |           1190 |              1268 | 129.1017 | 161.5798 | -100.3908 |
| naive_30min        |      9 |           1191 |              1274 | 117.8556 | 149.2136 |   64.6599 |
| naive_30min        |     10 |           1243 |              1294 | 130.5245 | 158.4382 |  123.8858 |
| naive_30min        |     11 |           1209 |              1272 |  74.0736 |  96.0716 |   60.6013 |
| naive_30min        |     12 |           1216 |              1288 |  44.2081 |  58.3156 |   10.2442 |
| naive_30min        |     13 |           1242 |              1289 |  43.7303 |  57.6080 |   -1.9911 |
| naive_30min        |     14 |           1218 |              1283 |  46.0378 |  60.9589 |  -13.1658 |
| naive_30min        |     15 |           1213 |              1283 |  66.4823 |  84.9112 |  -44.1311 |
| naive_30min        |     16 |           1213 |              1280 |  94.3256 | 118.4765 |  -79.6958 |
| naive_30min        |     17 |           1208 |              1277 | 103.4851 | 134.2207 |  -71.1540 |
| naive_30min        |     18 |           1217 |              1282 |  94.6787 | 121.1886 |   17.6138 |
| naive_30min        |     19 |           1206 |              1277 | 117.4809 | 146.3539 |  103.1045 |
| naive_30min        |     20 |           1217 |              1280 | 110.0189 | 129.7987 |  107.6771 |
| naive_30min        |     21 |           1224 |              1290 |  64.8979 |  78.5811 |   62.6038 |
| naive_30min        |     22 |           1235 |              1294 |  28.6016 |  37.1176 |   22.7004 |
| naive_30min        |     23 |           1227 |              1280 |  20.1108 |  26.4016 |    7.3007 |
| seasonal_naive_24h |      0 |           1221 |              1285 |  35.6658 |  46.3037 |    0.0295 |
| seasonal_naive_24h |      1 |           1223 |              1286 |  37.9779 |  48.7328 |    0.4898 |
| seasonal_naive_24h |      2 |           1230 |              1289 |  35.8439 |  47.1727 |    0.2715 |
| seasonal_naive_24h |      3 |           1239 |              1292 |  38.4011 |  49.9331 |   -0.2494 |
| seasonal_naive_24h |      4 |           1228 |              1288 |  39.6726 |  52.5475 |   -0.1743 |
| seasonal_naive_24h |      5 |           1224 |              1287 |  44.8840 |  58.0266 |    0.2337 |
| seasonal_naive_24h |      6 |           1203 |              1275 |  76.8903 | 103.1905 |   -0.6376 |
| seasonal_naive_24h |      7 |           1202 |              1275 | 140.5882 | 184.2166 |   -2.0341 |
| seasonal_naive_24h |      8 |           1194 |              1268 | 195.6139 | 252.7559 |   -0.5101 |
| seasonal_naive_24h |      9 |           1205 |              1274 | 180.6050 | 234.2807 |    1.4548 |
| seasonal_naive_24h |     10 |           1238 |              1294 | 124.9216 | 162.5481 |    0.7246 |
| seasonal_naive_24h |     11 |           1194 |              1272 |  93.6449 | 123.0360 |   -0.6365 |
| seasonal_naive_24h |     12 |           1232 |              1288 |  85.6088 | 112.7295 |   -1.4237 |
| seasonal_naive_24h |     13 |           1230 |              1289 |  86.2033 | 113.7247 |   -0.1122 |
| seasonal_naive_24h |     14 |           1226 |              1283 |  91.5579 | 119.6174 |    0.3279 |
| seasonal_naive_24h |     15 |           1219 |              1283 | 106.6489 | 139.1337 |   -0.3552 |
| seasonal_naive_24h |     16 |           1217 |              1280 | 138.1569 | 180.1632 |   -1.1758 |
| seasonal_naive_24h |     17 |           1202 |              1277 | 177.9160 | 229.5489 |   -0.8461 |
| seasonal_naive_24h |     18 |           1216 |              1282 | 183.0510 | 238.4452 |   -0.4441 |
| seasonal_naive_24h |     19 |           1204 |              1277 | 144.3605 | 187.6470 |    3.0033 |
| seasonal_naive_24h |     20 |           1214 |              1280 |  90.2900 | 117.6827 |    0.4432 |
| seasonal_naive_24h |     21 |           1233 |              1290 |  55.2441 |  72.4833 |   -0.6099 |
| seasonal_naive_24h |     22 |           1243 |              1294 |  41.0708 |  54.0885 |    0.2711 |
| seasonal_naive_24h |     23 |           1211 |              1280 |  36.6474 |  47.3571 |   -0.4657 |

### Highest-error hour by model

- **naive_30min:** hour 07, MAE 155.0839
- **seasonal_naive_24h:** hour 08, MAE 195.6139

Hourly differences may reflect changing traffic patterns,
but additional investigation is needed before attributing
errors to a specific cause.

## 7. Day-of-week analysis

| model              | day_of_week   |   observations |   validation_rows |      mae |     rmse |     bias |
|:-------------------|:--------------|---------------:|------------------:|---------:|---------:|---------:|
| naive_30min        | Monday        |           4295 |              4541 |  73.7700 | 105.5997 |  -0.9006 |
| naive_30min        | Tuesday       |           4371 |              4579 |  78.7646 | 114.4736 |   0.8236 |
| naive_30min        | Wednesday     |           4319 |              4551 |  63.4156 |  93.4738 |   1.9037 |
| naive_30min        | Thursday      |           4358 |              4575 |  71.9674 | 103.6564 |  -0.7205 |
| naive_30min        | Friday        |           4347 |              4567 |  77.8236 | 112.3601 |  -0.3945 |
| naive_30min        | Saturday      |           4322 |              4557 |  56.5398 |  79.8247 |   1.0086 |
| naive_30min        | Sunday        |           3246 |              3418 |  56.1765 |  79.7122 |  -0.3823 |
| seasonal_naive_24h | Monday        |           4308 |              4541 | 112.1112 | 152.6410 | -97.6985 |
| seasonal_naive_24h | Tuesday       |           4336 |              4579 |  76.1310 | 118.8446 | -41.4027 |
| seasonal_naive_24h | Wednesday     |           4341 |              4551 | 117.9065 | 174.4409 |  94.5460 |
| seasonal_naive_24h | Thursday      |           4334 |              4575 |  92.7176 | 146.7853 | -51.9732 |
| seasonal_naive_24h | Friday        |           4354 |              4567 |  75.8135 | 113.8457 | -36.5503 |
| seasonal_naive_24h | Saturday      |           4333 |              4557 | 134.0535 | 171.4079 | 133.5066 |
| seasonal_naive_24h | Sunday        |           3242 |              3418 |  40.7378 |  59.6993 |  -2.1727 |

These results summarize forecasting performance by the
weekday associated with each validation timestamp.

## 8. Figures

Generated figures are stored in `reports/day15/figures/`.

1. `common_population_mae.png`
2. `common_population_rmse.png`
3. `mae_by_hour.png`
4. `prediction_coverage_by_road.png`

## 9. Limitations

- The evaluated models are simple baseline forecasting
  strategies, not advanced machine-learning models.
- Common-population metrics describe only observations
  where both models have valid predictions.
- Missing predictions affect operational coverage and should
  not be hidden when reporting model accuracy.
- Road-level and temporal associations do not establish
  causal explanations for forecast errors.
- Performance on this validation period does not guarantee
  performance on future data.

## 10. Conclusion

Day 15 establishes a repeatable diagnostic workflow for
comparing forecasting strategies, identifying difficult
roads and hours, measuring prediction coverage, and
documenting model errors.

These findings provide an evidence-based baseline for
subsequent feature engineering and forecasting-model
experiments.
