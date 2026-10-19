# FlowCast Day 12: OOP Baseline Coverage Analysis

## 1. Objective

Apply object-oriented Python design to compare baseline prediction coverage and identify roads requiring further investigation.

## 2. Configuration

- Coverage threshold: 95.00%
- Baselines evaluated: 2
- Road/model records examined: 50
- Low-coverage road/model records: 24

## 3. Baseline comparison

| model | validation_rows | valid_predictions | missing_predictions | prediction_coverage_pct | missing_cases_classified | best_coverage_pct | coverage_gap_pp | is_best_coverage | coverage_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_30min | 30788 | 29258 | 1530 | 95.0305 | 1530 | 95.0305 | 0.0 | True | MEETS_THRESHOLD |
| seasonal_naive_24h | 30788 | 29248 | 1540 | 94.9981 | 1540 | 95.0305 | 0.0324 | False | BELOW_THRESHOLD |

## 4. Low-coverage road analysis

Road/model records are flagged when their coverage is strictly below the configured threshold.

| model | road_id | validation_rows | valid_predictions | missing_predictions | coverage_pct | historical_reference_row_absent | coverage_threshold_pct | coverage_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_30min | NL-015 | 1211 | 1131 | 80 | 93.3939 | 80 | 95.0 | LOW_COVERAGE |
| seasonal_naive_24h | NL-015 | 1211 | 1131 | 80 | 93.3939 | 80 | 95.0 | LOW_COVERAGE |
| naive_30min | NL-005 | 1219 | 1144 | 75 | 93.8474 | 75 | 95.0 | LOW_COVERAGE |
| seasonal_naive_24h | NL-005 | 1219 | 1145 | 74 | 93.9295 | 74 | 95.0 | LOW_COVERAGE |
| seasonal_naive_24h | NL-020 | 1221 | 1147 | 74 | 93.9394 | 74 | 95.0 | LOW_COVERAGE |
| naive_30min | NL-016 | 1223 | 1153 | 70 | 94.2764 | 70 | 95.0 | LOW_COVERAGE |
| naive_30min | NL-013 | 1224 | 1154 | 70 | 94.281 | 70 | 95.0 | LOW_COVERAGE |
| seasonal_naive_24h | NL-016 | 1223 | 1154 | 69 | 94.3581 | 69 | 95.0 | LOW_COVERAGE |
| naive_30min | NL-020 | 1221 | 1153 | 68 | 94.4308 | 68 | 95.0 | LOW_COVERAGE |
| naive_30min | NL-024 | 1223 | 1155 | 68 | 94.4399 | 68 | 95.0 | LOW_COVERAGE |
| naive_30min | NL-014 | 1225 | 1157 | 68 | 94.449 | 68 | 95.0 | LOW_COVERAGE |
| naive_30min | NL-003 | 1226 | 1159 | 67 | 94.5351 | 67 | 95.0 | LOW_COVERAGE |
| seasonal_naive_24h | NL-002 | 1232 | 1165 | 67 | 94.5617 | 67 | 95.0 | LOW_COVERAGE |
| seasonal_naive_24h | NL-024 | 1223 | 1157 | 66 | 94.6034 | 66 | 95.0 | LOW_COVERAGE |
| seasonal_naive_24h | NL-003 | 1226 | 1160 | 66 | 94.6166 | 66 | 95.0 | LOW_COVERAGE |
| seasonal_naive_24h | NL-010 | 1229 | 1163 | 66 | 94.6298 | 66 | 95.0 | LOW_COVERAGE |
| seasonal_naive_24h | NL-013 | 1224 | 1159 | 65 | 94.6895 | 65 | 95.0 | LOW_COVERAGE |
| seasonal_naive_24h | NL-014 | 1225 | 1160 | 65 | 94.6939 | 65 | 95.0 | LOW_COVERAGE |
| naive_30min | NL-010 | 1229 | 1165 | 64 | 94.7925 | 64 | 95.0 | LOW_COVERAGE |
| seasonal_naive_24h | NL-019 | 1231 | 1167 | 64 | 94.801 | 64 | 95.0 | LOW_COVERAGE |

## 5. Findings

Highest-coverage baseline(s): naive_30min

24 road/model records fall below the 95.00% coverage threshold.

## 6. Interpretation

Prediction coverage measures how many validation rows received predictions. It does not measure forecast accuracy. A model with higher coverage may still have larger prediction errors.

Low coverage identifies records for additional investigation. It does not, by itself, establish the cause of missing historical references.

## 7. Output files

- `baseline_coverage_comparison.csv`
- `road_coverage_flags.csv`
- `day12_oop_analysis_report.md`
