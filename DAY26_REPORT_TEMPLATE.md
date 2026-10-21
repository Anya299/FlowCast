# FlowCast Day 26 — Reliability Review

## Run metadata
- Date:
- Git commit:
- Test split:
- Model artifact:
- Analysis command:

## Overall metrics
- MAE:
- RMSE:
- R²:
- MAPE (non-zero targets only):
- Mean signed error:

## Worst-performing roads
Record the top five roads by RMSE and their sample counts.

## Time-of-day findings
Compare morning peak, evening peak, and off-peak metrics.

## Residual findings
- Overprediction vs underprediction:
- P90/P95/P99 absolute error:
- Largest error examples reviewed:

## Feature availability / leakage review
- `public_holiday` source and availability time:
- `event_flag` source and availability time:
- `roadwork_flag` source and availability time:
- Lag/rolling features confirmed shifted before aggregation?:
- Forecast horizon and target timestamp clarified?:

## Actions
Separate confirmed defects from hypotheses. Use validation—not test—data for future model selection.

## Limitations
Explain data coverage, small groups, MAPE behavior at zero targets, and retrospective-vs-live availability.
