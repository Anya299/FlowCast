"""Command-line entry point for FlowCast Day 25 model training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Supports running as: python src/day25/train_model.py from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.day25.model_pipeline import train_and_evaluate  # noqa: E402


def main() -> int:
    try:
        result = train_and_evaluate(PROJECT_ROOT)
    except Exception as exc:
        print(f"[DAY 25 FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("=" * 72)
    print("FLOWCAST — DAY 25 MODEL TRAINING AND EVALUATION")
    print("=" * 72)
    print(f"Target: {result['target']}")
    print(f"Model: {result['model']}")
    print(f"Features: {result['feature_count']}")
    print(f"Artifact: {result['model_artifact']}")
    print("\nValidation RMSE")
    print(f"  Model:       {result['validation_metrics']['model']['rmse']:.4f}")
    print(f"  Persistence: {result['validation_metrics']['persistence_lag_30m_baseline']['rmse']:.4f}")
    print(f"  Mean:        {result['validation_metrics']['training_mean_baseline']['rmse']:.4f}")
    print("\nHeld-out test RMSE")
    print(f"  Model:       {result['test_metrics']['model']['rmse']:.4f}")
    print(f"  Persistence: {result['test_metrics']['persistence_lag_30m_baseline']['rmse']:.4f}")
    print(f"  Mean:        {result['test_metrics']['training_mean_baseline']['rmse']:.4f}")
    print("\nOutputs:")
    print("  models/day25/traffic_model.joblib")
    print("  models/day25/evaluation.json")
    print("  models/day25/metadata.json")
    print("  reports/day25/DAY25_MODEL_PIPELINE_REPORT.md")
    print("\nFull evaluation JSON:")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
