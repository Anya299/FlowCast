"""Leakage-conscious model pipeline for FlowCast Day 25."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET = "traffic_volume"

# Day 8 approved feature set. Do not add current-period outcomes or sensor
# measurements that would not be known at the actual prediction cutoff.
FEATURES = [
    "traffic_lag_30m",
    "traffic_lag_1h",
    "traffic_lag_90m",
    "traffic_lag_24h",
    "traffic_lag_7d",
    "traffic_rolling_mean_3h",
    "traffic_rolling_mean_6h",
    "traffic_rolling_mean_24h",
    "traffic_rolling_std_24h",
    "hour_of_day",
    "minute_of_hour",
    "day_of_week_number",
    "day_of_month",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "road_id_feature",
    "latitude",
    "longitude",
    "road_capacity",
    "public_holiday",
    "event_flag",
    "roadwork_flag",
]

FORBIDDEN_FEATURES = {
    TARGET,
    "vehicle_count",
    "volume_capacity_ratio",
    "avg_speed",
    "occupancy",
    "travel_time",
    "accident_count",
    "congestion_level",
}


def build_pipeline() -> Pipeline:
    """Build a reproducible numeric model pipeline.

    The imputer is fitted only on the training partition because it lives
    inside the sklearn Pipeline. Missing feature values are not imputed before
    the chronological split.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.08,
                    max_iter=250,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def validate_feature_contract(frame: pd.DataFrame, split_name: str) -> None:
    """Fail fast if the expected feature contract is not satisfied."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{split_name} split is missing required model features: {missing}"
        )

    leaked = sorted(set(FEATURES) & FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(f"Forbidden target/leakage features configured: {leaked}")


def prepare_xy(frame: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Select only approved features and valid non-negative targets."""
    if TARGET not in frame.columns:
        raise ValueError(f"{split_name} split does not contain target '{TARGET}'")

    validate_feature_contract(frame, split_name)

    working = frame.copy()
    y = pd.to_numeric(working[TARGET], errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan)
    valid_target = y.notna() & (y >= 0)

    X = working.loc[valid_target, FEATURES].copy()
    y = y.loc[valid_target].astype(float)

    for column in FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.empty:
        raise ValueError(f"{split_name} split has no rows with a valid target")
    if len(X) != len(y):
        raise RuntimeError(f"{split_name} feature/target row counts do not match")

    return X, y


def regression_metrics(y_true: pd.Series | np.ndarray,
                       y_pred: np.ndarray) -> dict[str, Any]:
    """Return standard regression metrics and a guarded MAPE."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have matching shapes")
    if not np.isfinite(predicted).all():
        raise ValueError("Predictions contain NaN or infinite values")

    rmse = float(math.sqrt(mean_squared_error(actual, predicted)))
    nonzero = np.abs(actual) > 1e-8
    mape = (
        float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any()
        else None
    )
    return {
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": rmse,
        "r2": float(r2_score(actual, predicted)),
        "mape_percent_nonzero_targets_only": mape,
        "zero_target_rows_excluded_from_mape": int((~nonzero).sum()),
    }


def persistence_baseline(X: pd.DataFrame) -> np.ndarray:
    """Predict next-period traffic using the previous-period traffic lag."""
    lag = pd.to_numeric(X["traffic_lag_30m"], errors="coerce").to_numpy(dtype=float)
    # For rows with no prior-period lag, use the median lag from the supplied
    # split. This is a baseline only; the learned model uses train-fitted imputation.
    finite = lag[np.isfinite(lag)]
    fallback = float(np.median(finite)) if finite.size else 0.0
    return np.where(np.isfinite(lag), lag, fallback)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def train_and_evaluate(project_root: Path) -> dict[str, Any]:
    """Train on train, compare on validation, refit train+validation, test once."""
    split_dir = project_root / "data" / "processed" / "day9"
    train_path = split_dir / "flowcast_train.csv"
    validation_path = split_dir / "flowcast_validation.csv"
    test_path = split_dir / "flowcast_test.csv"

    for path in (train_path, validation_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required chronological split not found: {path}\n"
                "Run/repair the Day 9 split-generation step before Day 25."
            )

    train_frame = pd.read_csv(train_path)
    validation_frame = pd.read_csv(validation_path)
    test_frame = pd.read_csv(test_path)

    X_train, y_train = prepare_xy(train_frame, "train")
    X_val, y_val = prepare_xy(validation_frame, "validation")
    X_test, y_test = prepare_xy(test_frame, "test")

    model = build_pipeline()
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    validation_model_metrics = regression_metrics(y_val, val_pred)
    validation_persistence_metrics = regression_metrics(
        y_val, persistence_baseline(X_val)
    )
    train_mean = float(y_train.mean())
    validation_mean_baseline = regression_metrics(
        y_val, np.full(len(y_val), train_mean, dtype=float)
    )

    # Refit the fixed model configuration on train + validation after validation
    # evaluation. The test partition is not used for feature/model selection.
    X_trainval = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    final_model = build_pipeline()
    final_model.fit(X_trainval, y_trainval)

    test_pred = final_model.predict(X_test)
    test_model_metrics = regression_metrics(y_test, test_pred)
    test_persistence_metrics = regression_metrics(y_test, persistence_baseline(X_test))
    test_mean_baseline = regression_metrics(
        y_test, np.full(len(y_test), train_mean, dtype=float)
    )

    model_dir = project_root / "models" / "day25"
    report_dir = project_root / "reports" / "day25"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "traffic_model.joblib"
    joblib.dump(final_model, model_path)

    # Verify that the persisted artifact reloads and produces finite predictions.
    reloaded_model = joblib.load(model_path)
    reload_pred = reloaded_model.predict(X_test.head(min(25, len(X_test))))
    if not np.isfinite(reload_pred).all():
        raise RuntimeError("Reloaded model generated non-finite predictions")

    summary: dict[str, Any] = {
        "project": "FlowCast",
        "day": 25,
        "target": TARGET,
        "model": "HistGradientBoostingRegressor",
        "model_artifact": str(model_path.relative_to(project_root)),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "forbidden_features_checked": sorted(FORBIDDEN_FEATURES),
        "split_source": "data/processed/day9",
        "source_rows": {
            "train": int(len(train_frame)),
            "validation": int(len(validation_frame)),
            "test": int(len(test_frame)),
        },
        "valid_target_rows": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "validation_metrics": {
            "model": validation_model_metrics,
            "persistence_lag_30m_baseline": validation_persistence_metrics,
            "training_mean_baseline": validation_mean_baseline,
        },
        "test_metrics": {
            "model": test_model_metrics,
            "persistence_lag_30m_baseline": test_persistence_metrics,
            "training_mean_baseline": test_mean_baseline,
        },
        "reload_prediction_check": {
            "passed": True,
            "rows_checked": int(min(25, len(X_test))),
        },
        "caveats": [
            "Metrics describe this dataset and split; they are not a guarantee of future performance.",
            "Confirm public_holiday, event_flag, and roadwork_flag are known at prediction time.",
            "The persistence baseline fills missing lag values with the median lag within each evaluated split; interpret those fallback rows cautiously.",
            "MAPE excludes zero-target rows and should not be the sole selection metric.",
        ],
    }

    write_json(model_dir / "evaluation.json", summary)
    write_json(model_dir / "metadata.json", {
        "model": summary["model"],
        "target": TARGET,
        "features": FEATURES,
        "training_rows_including_validation": int(len(y_trainval)),
        "random_state": 42,
        "training_code": "src/day25/train_model.py",
    })

    report = f"""# FlowCast Day 25 — Model Pipeline Report

## Model
- Estimator: `HistGradientBoostingRegressor` inside a scikit-learn `Pipeline`.
- Target: `{TARGET}`.
- Features: {len(FEATURES)} approved Day 8 features.
- Missing features: median imputation fitted inside the pipeline, on training data only.
- Split: existing chronological Day 9 train / validation / test CSVs; no random shuffle.

## Validation results
| Model | MAE | RMSE | R² | MAPE (%; nonzero targets only) |
|---|---:|---:|---:|---:|
| Learned model | {validation_model_metrics['mae']:.4f} | {validation_model_metrics['rmse']:.4f} | {validation_model_metrics['r2']:.4f} | {validation_model_metrics['mape_percent_nonzero_targets_only'] if validation_model_metrics['mape_percent_nonzero_targets_only'] is not None else 'N/A'} |
| Previous-period persistence | {validation_persistence_metrics['mae']:.4f} | {validation_persistence_metrics['rmse']:.4f} | {validation_persistence_metrics['r2']:.4f} | {validation_persistence_metrics['mape_percent_nonzero_targets_only'] if validation_persistence_metrics['mape_percent_nonzero_targets_only'] is not None else 'N/A'} |
| Training-mean baseline | {validation_mean_baseline['mae']:.4f} | {validation_mean_baseline['rmse']:.4f} | {validation_mean_baseline['r2']:.4f} | {validation_mean_baseline['mape_percent_nonzero_targets_only'] if validation_mean_baseline['mape_percent_nonzero_targets_only'] is not None else 'N/A'} |

## Held-out test results
| Model | MAE | RMSE | R² | MAPE (%; nonzero targets only) |
|---|---:|---:|---:|---:|
| Final model (fit on train + validation) | {test_model_metrics['mae']:.4f} | {test_model_metrics['rmse']:.4f} | {test_model_metrics['r2']:.4f} | {test_model_metrics['mape_percent_nonzero_targets_only'] if test_model_metrics['mape_percent_nonzero_targets_only'] is not None else 'N/A'} |
| Previous-period persistence | {test_persistence_metrics['mae']:.4f} | {test_persistence_metrics['rmse']:.4f} | {test_persistence_metrics['r2']:.4f} | {test_persistence_metrics['mape_percent_nonzero_targets_only'] if test_persistence_metrics['mape_percent_nonzero_targets_only'] is not None else 'N/A'} |
| Training-mean baseline | {test_mean_baseline['mae']:.4f} | {test_mean_baseline['rmse']:.4f} | {test_mean_baseline['r2']:.4f} | {test_mean_baseline['mape_percent_nonzero_targets_only'] if test_mean_baseline['mape_percent_nonzero_targets_only'] is not None else 'N/A'} |

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
"""
    (report_dir / "DAY25_MODEL_PIPELINE_REPORT.md").write_text(report, encoding="utf-8")
    return summary
