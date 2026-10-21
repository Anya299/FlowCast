# FlowCast Day 25 — Model Pipeline Report

## Model
- Estimator: `HistGradientBoostingRegressor` inside a scikit-learn `Pipeline`.
- Target: `traffic_volume`.
- Features: 28 approved Day 8 features.
- Missing features: median imputation fitted inside the pipeline, on training data only.
- Split: existing chronological Day 9 train / validation / test CSVs; no random shuffle.

## Validation results
| Model | MAE | RMSE | R² | MAPE (%; nonzero targets only) |
|---|---:|---:|---:|---:|
| Learned model | 39.1843 | 57.6351 | 0.9565 | 9.666822717477661 |
| Previous-period persistence | 76.7935 | 118.4094 | 0.8164 | 18.888315624363514 |
| Training-mean baseline | 225.1868 | 276.4846 | -0.0011 | 85.85211440294384 |

## Held-out test results
| Model | MAE | RMSE | R² | MAPE (%; nonzero targets only) |
|---|---:|---:|---:|---:|
| Final model (fit on train + validation) | 39.7615 | 58.1351 | 0.9591 | 9.405461989662479 |
| Previous-period persistence | 80.1343 | 121.9411 | 0.8201 | 19.093861537800798 |
| Training-mean baseline | 233.0474 | 287.7410 | -0.0014 | 82.57255162369087 |

## Leakage and reproducibility controls
- Only the explicit feature allowlist is passed to the estimator.
- `traffic_volume`, `vehicle_count`, `volume_capacity_ratio`, contemporaneous speed/occupancy/travel-time/accident fields, and `congestion_level` are excluded from model inputs.
- Imputation is fit as part of the pipeline, not on the full dataset.
- Model artifact is reloaded and checked for finite predictions.
- Test metrics are reported after the fixed configuration has been chosen.

## Interpretation cautions
- Review the test metrics against the baselines; a learned model is not automatically useful just because it trained successfully.
- Verify event/holiday/roadwork flags are available at the exact prediction cutoff in a real deployment.
- This report does not claim the PRD's target metrics have been met unless the measured values demonstrate that.
