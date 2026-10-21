# FlowCast Day 25 — Model Pipeline

Day 25 trains and evaluates a first reproducible regression model for `traffic_volume`.

## Run
See the PowerShell commands supplied with this package.

## Engineering controls
- Uses the chronological Day 9 train/validation/test splits.
- Passes only the approved feature allowlist to the estimator.
- Fits median imputation only within the training pipeline.
- Compares with previous-period persistence and training-mean baselines.
- Keeps test data out of model selection.
- Saves a reloadable artifact, metadata, evaluation JSON, and a report.

## Before production
Confirm that event, holiday, and roadwork indicators are known at the prediction cutoff. Review errors by road and time period, add automated CI, and integrate model loading into the existing `src/day19` API only after testing compatibility.
