# FlowCast Day 10 — Baseline Evaluation

## Objective

Establish reference forecasting performance before training
machine-learning models.

## Dataset

- Training rows: 109,460
- Validation rows: 30,788
- Test rows: 31,866
- Target: `traffic_volume`

## Validation results

                 model  evaluated_rows       mae       rmse
       Naive 30-minute           29258 68.824287 100.034189
Seasonal naive 24-hour           29248 94.699364 141.109444

## Evaluation protocol

The evaluation uses rolling one-step-ahead predictions.
At each timestamp, the model uses historical target values
from the same road and the required earlier timestamp.

The actual observation is added to history after its
prediction is made. This assumes the latest observed traffic
is available before the next prediction.

## Important limitations

- These are simple historical baselines, not trained ML models.
- Results depend on the availability of the required historical
  observation.
- The validation period is used for baseline comparison.
- The test period has not been used for model selection.
- Feature-generation logic must be audited separately before
  declaring the full feature pipeline leakage-free.
- The evaluation assumes rolling one-step-ahead forecasting,
  not a full-day forecast made in advance.

## Next step

Inspect validation metrics, compare errors by road, verify the
feature-generation implementation, and establish a documented
baseline before fitting a learned model.
