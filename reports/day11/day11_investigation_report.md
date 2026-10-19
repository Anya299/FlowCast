# FlowCast Day 11: Missing Prediction Investigation

## 1. Investigation objective

Investigate missing predictions from the Day 10 naive and seasonal-naive traffic forecasting baselines.

## 2. Prediction coverage

| model | validation_rows | valid_predictions | missing_predictions | prediction_coverage_pct | missing_cases_classified |
| --- | --- | --- | --- | --- | --- |
| naive_30min | 30788 | 29258 | 1530 | 95.0305 | 1530 |
| seasonal_naive_24h | 30788 | 29248 | 1540 | 94.9981 | 1540 |

## 3. Missing prediction reasons

| model | missing_reason | missing_cases |
| --- | --- | --- |
| naive_30min | historical_reference_row_absent | 1530 |
| seasonal_naive_24h | historical_reference_row_absent | 1540 |

## 4. Roads with the most missing predictions

### Naive 30-minute baseline

| model | road_id | validation_rows | valid_predictions | missing_predictions | coverage_pct | historical_reference_row_absent |
| --- | --- | --- | --- | --- | --- | --- |
| naive_30min | NL-001 | 1236 | 1178 | 58 | 95.3074 | 58 |
| naive_30min | NL-002 | 1232 | 1170 | 62 | 94.9675 | 62 |
| naive_30min | NL-003 | 1226 | 1159 | 67 | 94.5351 | 67 |
| naive_30min | NL-004 | 1231 | 1171 | 60 | 95.1259 | 60 |
| naive_30min | NL-005 | 1219 | 1144 | 75 | 93.8474 | 75 |
| naive_30min | NL-006 | 1240 | 1187 | 53 | 95.7258 | 53 |
| naive_30min | NL-007 | 1239 | 1185 | 54 | 95.6416 | 54 |
| naive_30min | NL-008 | 1239 | 1183 | 56 | 95.4802 | 56 |
| naive_30min | NL-009 | 1245 | 1198 | 47 | 96.2249 | 47 |
| naive_30min | NL-010 | 1229 | 1165 | 64 | 94.7925 | 64 |

### Seasonal-naive 24-hour baseline

| model | road_id | validation_rows | valid_predictions | missing_predictions | coverage_pct | historical_reference_row_absent |
| --- | --- | --- | --- | --- | --- | --- |
| seasonal_naive_24h | NL-001 | 1236 | 1177 | 59 | 95.2265 | 59 |
| seasonal_naive_24h | NL-002 | 1232 | 1165 | 67 | 94.5617 | 67 |
| seasonal_naive_24h | NL-003 | 1226 | 1160 | 66 | 94.6166 | 66 |
| seasonal_naive_24h | NL-004 | 1231 | 1168 | 63 | 94.8822 | 63 |
| seasonal_naive_24h | NL-005 | 1219 | 1145 | 74 | 93.9295 | 74 |
| seasonal_naive_24h | NL-006 | 1240 | 1189 | 51 | 95.8871 | 51 |
| seasonal_naive_24h | NL-007 | 1239 | 1185 | 54 | 95.6416 | 54 |
| seasonal_naive_24h | NL-008 | 1239 | 1185 | 54 | 95.6416 | 54 |
| seasonal_naive_24h | NL-009 | 1245 | 1199 | 46 | 96.3052 | 46 |
| seasonal_naive_24h | NL-010 | 1229 | 1163 | 66 | 94.6298 | 66 |

## 5. Findings

The naive 30-minute baseline produced 29,258 valid predictions out of 30,788 validation rows, giving 95.03% prediction coverage.

The seasonal-naive 24-hour baseline produced 29,248 valid predictions out of 30,788 validation rows, giving approximately 95.00% prediction coverage.

The investigation classified all missing predictions as historical_reference_row_absent. This means the required road-and-reference-timestamp combination was not found in the source rows considered by the investigation.

## 6. Interpretation and limitation

Missing predictions indicate that a baseline could not retrieve the required historical traffic value. They should not be treated as zero traffic or as forecasting errors.

The investigation describes reference availability in the supplied source datasets. The classification should be interpreted in the context of the eligible rows used by the baseline evaluation.
