# FlowCast Day 13: Forecasting Strategy Evaluation

## Objective

Implement forecasting strategies through an object-oriented interface and compare their prediction coverage and forecasting errors.

## Evaluation results

| model | validation_rows | valid_predictions | missing_predictions | prediction_coverage_pct | mae | rmse |
| --- | --- | --- | --- | --- | --- | --- |
| naive_30min | 30788 | 29258 | 1530 | 95.0305 | 68.8243 | 100.0342 |
| seasonal_naive_24h | 30788 | 29248 | 1540 | 94.9981 | 94.6994 | 141.1094 |

## Interpretation

Prediction coverage is the percentage of validation rows for which a prediction is available.

MAE measures average absolute forecasting error. RMSE penalizes larger errors more strongly.

Compare MAE and RMSE on a consistent evaluation population. Coverage differences can affect which observations contribute to the error metrics.

These are baseline models. Their results provide reference points for more advanced forecasting models in later FlowCast development.
