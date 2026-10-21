# FlowCast Day 25 package

This package adds a baseline model-training and evaluation stage to FlowCast.

## Expected existing inputs
- `data/processed/day9/flowcast_train.csv`
- `data/processed/day9/flowcast_validation.csv`
- `data/processed/day9/flowcast_test.csv`

Each CSV must contain `traffic_volume` and all 28 approved Day 8 feature columns listed in `src/day25/model_pipeline.py`.

## Outputs
- `models/day25/traffic_model.joblib`
- `models/day25/evaluation.json`
- `models/day25/metadata.json`
- `reports/day25/DAY25_MODEL_PIPELINE_REPORT.md`

Model artifacts and generated data are machine-specific outputs. Review `.gitignore` before deciding whether to version any binary model artifact. Do not commit private data or secrets.

## Important
A successful training run is not proof that the model meets product requirements. Inspect whether it beats the simple baselines, analyze errors by road/time, and validate prediction-time availability of every feature.
