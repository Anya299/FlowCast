"""Day 26: reliability analysis for the trained FlowCast traffic-volume model.

Run from the repository root:
    python src/day26/run_analysis.py

The analysis evaluates the already-trained Day 25 artifact on the untouched
chronological test partition. It does not fit a new model or select features.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# The feature contract is imported from the Day 25 pipeline so the analysis
# uses exactly the same ordered feature list.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.day25.model_pipeline import FEATURES, TARGET, prepare_xy  # noqa: E402


TIMESTAMP_CANDIDATES = (
    "timestamp", "datetime", "date_time", "observation_time",
    "recorded_at", "date"
)
ROAD_CANDIDATES = (
    "road_id", "road_segment_id", "road_name", "road", "road_id_feature"
)
MANUAL_REVIEW_FEATURES = {
    "public_holiday": (
        "Confirm the holiday calendar was published/known at the forecast cutoff."
    ),
    "event_flag": (
        "Confirm event schedules were known at the forecast cutoff and do not "
        "encode events inferred from the observed target period."
    ),
    "roadwork_flag": (
        "Confirm roadwork information was known at the forecast cutoff."
    ),
}
LAG_FEATURES = [
    "traffic_lag_30m", "traffic_lag_1h", "traffic_lag_90m",
    "traffic_lag_24h", "traffic_lag_7d",
    "traffic_rolling_mean_3h", "traffic_rolling_mean_6h",
    "traffic_rolling_mean_24h", "traffic_rolling_std_24h",
]


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def regression_metrics(actual: pd.Series | np.ndarray,
                        predicted: pd.Series | np.ndarray) -> dict[str, Any]:
    """Calculate error metrics and directional bias for a group."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.shape != p.shape:
        raise ValueError("Actual and predicted arrays must have matching shapes.")
    if y.size == 0:
        return {"rows": 0}
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics received non-finite actuals or predictions.")
    error = p - y  # positive means overprediction
    nonzero = np.abs(y) > 1e-8
    mape = (
        float(np.mean(np.abs(error[nonzero] / y[nonzero])) * 100)
        if nonzero.any() else None
    )
    return {
        "rows": int(y.size),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if y.size >= 2 else None,
        "mape_percent_nonzero_targets_only": mape,
        "mean_error_predicted_minus_actual": float(np.mean(error)),
        "mean_absolute_percentage_error_denominator_rows_excluded": int((~nonzero).sum()),
    }


def detect_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Find a likely identifier column, case-insensitively."""
    lookup = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def add_diagnostics(source: pd.DataFrame, actual: pd.Series,
                    predicted: np.ndarray) -> pd.DataFrame:
    """Return row-aligned prediction/residual diagnostics."""
    diagnostics = source.loc[actual.index].copy()
    diagnostics["actual_traffic_volume"] = np.asarray(actual, dtype=float)
    diagnostics["predicted_traffic_volume"] = np.asarray(predicted, dtype=float)
    diagnostics["error_predicted_minus_actual"] = (
        diagnostics["predicted_traffic_volume"] - diagnostics["actual_traffic_volume"]
    )
    diagnostics["absolute_error"] = diagnostics["error_predicted_minus_actual"].abs()
    denominator = diagnostics["actual_traffic_volume"].abs().replace(0, np.nan)
    diagnostics["absolute_percentage_error"] = (
        diagnostics["absolute_error"] / denominator * 100
    )
    diagnostics["prediction_direction"] = np.select(
        [
            diagnostics["error_predicted_minus_actual"] > 0,
            diagnostics["error_predicted_minus_actual"] < 0,
        ],
        ["overprediction", "underprediction"],
        default="exact",
    )
    return diagnostics


def grouped_metrics(frame: pd.DataFrame, group_column: str,
                    minimum_rows: int = 30) -> pd.DataFrame:
    """Metrics by group; very small groups are omitted from ranking tables."""
    if group_column not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_column, dropna=False, sort=True):
        if len(group) < minimum_rows:
            continue
        metrics = regression_metrics(
            group["actual_traffic_volume"], group["predicted_traffic_volume"]
        )
        rows.append({group_column: str(key), **metrics})
    if not rows:
        return pd.DataFrame(columns=[group_column, "rows", "mae", "rmse"])
    return pd.DataFrame(rows).sort_values(
        ["rmse", "mae"], ascending=[False, False]
    ).reset_index(drop=True)


def build_feature_availability_audit(feature_columns: list[str]) -> dict[str, Any]:
    """Create an explicit audit checklist; code cannot prove operational availability."""
    included = [name for name in MANUAL_REVIEW_FEATURES if name in feature_columns]
    return {
        "status": "REQUIRES_MANUAL_VERIFICATION",
        "message": (
            "Presence in a dataset does not prove a feature was available at the "
            "prediction cutoff. Verify each source's publication/ingestion time."
        ),
        "manual_review": [
            {"feature": name, "question": MANUAL_REVIEW_FEATURES[name]}
            for name in included
        ],
        "lag_and_rolling_features": {
            "features": [name for name in LAG_FEATURES if name in feature_columns],
            "required_rule": (
                "Every lag/rolling window must use observations strictly before "
                "the forecast timestamp; rolling windows must be shifted before "
                "aggregation. Confirm this in the Day 8 feature-generation code."
            ),
            "status": "REQUIRES_SOURCE_CODE_AND_TIMESTAMP_AUDIT",
        },
        "current_period_sensor_outcomes_excluded_from_model_features": True,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _save_plots(diagnostics: pd.DataFrame, report_dir: Path) -> list[str]:
    """Save basic diagnostic plots; skip gracefully if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    created: list[str] = []
    # Residual distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(diagnostics["error_predicted_minus_actual"], bins=60)
    ax.axvline(0, linestyle="--")
    ax.set_title("FlowCast Day 26 — Prediction Error Distribution")
    ax.set_xlabel("Predicted traffic volume − actual traffic volume")
    ax.set_ylabel("Number of observations")
    fig.tight_layout()
    path = report_dir / "residual_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(str(path.name))

    # Actual vs predicted sample, deterministically capped for readability.
    plot_data = diagnostics[["actual_traffic_volume", "predicted_traffic_volume"]]
    if len(plot_data) > 8000:
        plot_data = plot_data.sample(n=8000, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        plot_data["actual_traffic_volume"],
        plot_data["predicted_traffic_volume"],
        alpha=0.25,
        s=8,
    )
    max_value = float(max(
        plot_data["actual_traffic_volume"].max(),
        plot_data["predicted_traffic_volume"].max(),
    ))
    min_value = float(min(
        plot_data["actual_traffic_volume"].min(),
        plot_data["predicted_traffic_volume"].min(),
    ))
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    ax.set_title("FlowCast Day 26 — Actual vs Predicted")
    ax.set_xlabel("Actual traffic volume")
    ax.set_ylabel("Predicted traffic volume")
    fig.tight_layout()
    path = report_dir / "actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(str(path.name))

    # RMSE by hour if available
    if "hour_of_day" in diagnostics.columns:
        by_hour = grouped_metrics(diagnostics, "hour_of_day", minimum_rows=1)
        if not by_hour.empty:
            by_hour = by_hour.sort_values("hour_of_day", key=lambda s: s.astype(float))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(by_hour["hour_of_day"].astype(str), by_hour["rmse"])
            ax.set_title("FlowCast Day 26 — RMSE by Hour")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("RMSE")
            fig.tight_layout()
            path = report_dir / "rmse_by_hour.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            created.append(str(path.name))
    return created


def run_analysis(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    split_dir = root / "data" / "processed" / "day9"
    test_path = split_dir / "flowcast_test.csv"
    model_path = root / "models" / "day25" / "traffic_model.joblib"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. Run Day 25 training first."
        )

    # low_memory=False avoids mixed-type chunk inference warnings in metadata columns.
    raw = pd.read_csv(test_path, low_memory=False)
    X, y = prepare_xy(raw, "test")
    model = joblib.load(model_path)
    predicted = np.asarray(model.predict(X), dtype=float)
    if predicted.shape[0] != len(y) or not np.isfinite(predicted).all():
        raise RuntimeError("Model returned invalid predictions for the test split.")

    diagnostics = add_diagnostics(raw, y, predicted)
    timestamp_column = detect_column(list(diagnostics.columns), TIMESTAMP_CANDIDATES)
    road_column = detect_column(list(diagnostics.columns), ROAD_CANDIDATES)

    if timestamp_column:
        diagnostics[timestamp_column] = pd.to_datetime(
            diagnostics[timestamp_column], errors="coerce"
        )
        diagnostics["analysis_date"] = diagnostics[timestamp_column].dt.date.astype(str)
        diagnostics["analysis_weekday"] = diagnostics[timestamp_column].dt.day_name()
        diagnostics["analysis_hour"] = diagnostics[timestamp_column].dt.hour
    elif "hour_of_day" in diagnostics.columns:
        diagnostics["analysis_hour"] = pd.to_numeric(
            diagnostics["hour_of_day"], errors="coerce"
        )
    if "hour_of_day" in diagnostics.columns:
        hour = pd.to_numeric(diagnostics["hour_of_day"], errors="coerce")
        diagnostics["period_of_day"] = np.select(
            [hour.between(7, 10), hour.between(16, 19)],
            ["morning_peak", "evening_peak"],
            default="off_peak_or_other",
        )

    report_dir = root / "reports" / "day26"
    report_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = report_dir / "test_predictions_with_residuals.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    tables: dict[str, pd.DataFrame] = {}
    if road_column:
        tables["error_by_road"] = grouped_metrics(diagnostics, road_column)
    if "hour_of_day" in diagnostics.columns:
        tables["error_by_hour"] = grouped_metrics(diagnostics, "hour_of_day", 1)
    if "day_of_week_number" in diagnostics.columns:
        tables["error_by_weekday"] = grouped_metrics(
            diagnostics, "day_of_week_number", 1
        )
    if "period_of_day" in diagnostics.columns:
        tables["error_by_period"] = grouped_metrics(
            diagnostics, "period_of_day", 1
        )
    if timestamp_column and "analysis_weekday" in diagnostics.columns:
        tables["error_by_calendar_weekday"] = grouped_metrics(
            diagnostics, "analysis_weekday", 1
        )

    table_files: dict[str, str] = {}
    for name, table in tables.items():
        path = report_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        table_files[name] = str(path.relative_to(root))

    overall = regression_metrics(
        diagnostics["actual_traffic_volume"],
        diagnostics["predicted_traffic_volume"],
    )
    residual = diagnostics["error_predicted_minus_actual"]
    direction_counts = diagnostics["prediction_direction"].value_counts().to_dict()
    top_error_rows = diagnostics.nlargest(20, "absolute_error")
    top_errors_path = report_dir / "top_20_absolute_errors.csv"
    top_error_rows.to_csv(top_errors_path, index=False)

    feature_audit = build_feature_availability_audit(FEATURES)
    feature_audit_path = report_dir / "feature_availability_audit.json"
    _write_json(feature_audit_path, feature_audit)

    plots = _save_plots(diagnostics, report_dir)
    summary: dict[str, Any] = {
        "project": "FlowCast",
        "day": 26,
        "analysis": "Day 25 model reliability and error analysis",
        "evaluated_partition": "data/processed/day9/flowcast_test.csv",
        "model_artifact": str(model_path.relative_to(root)),
        "target": TARGET,
        "features": FEATURES,
        "rows_evaluated": int(len(diagnostics)),
        "overall_metrics_recomputed": overall,
        "residual_summary": {
            "mean_error_predicted_minus_actual": _safe_float(residual.mean()),
            "median_error_predicted_minus_actual": _safe_float(residual.median()),
            "error_std": _safe_float(residual.std(ddof=1)),
            "overprediction_rows": int(direction_counts.get("overprediction", 0)),
            "underprediction_rows": int(direction_counts.get("underprediction", 0)),
            "exact_prediction_rows": int(direction_counts.get("exact", 0)),
            "p90_absolute_error": _safe_float(diagnostics["absolute_error"].quantile(0.90)),
            "p95_absolute_error": _safe_float(diagnostics["absolute_error"].quantile(0.95)),
            "p99_absolute_error": _safe_float(diagnostics["absolute_error"].quantile(0.99)),
        },
        "group_tables": table_files,
        "diagnostics_csv": str(diagnostics_path.relative_to(root)),
        "top_20_errors_csv": str(top_errors_path.relative_to(root)),
        "feature_availability_audit": str(feature_audit_path.relative_to(root)),
        "plots": plots,
        "interpretation_guardrails": [
            "This analysis re-evaluates the fixed Day 25 artifact; it does not retrain or tune it.",
            "Error by group is descriptive and does not establish causality.",
            "MAPE excludes zero-target rows; interpret alongside MAE, RMSE, and bias.",
            "The test split has already been evaluated in Day 25; do not use it to tune features or hyperparameters.",
            "Feature availability and time-series leakage require source-code and operational-cutoff verification.",
        ],
    }
    summary_path = report_dir / "day26_reliability_summary.json"
    _write_json(summary_path, summary)
    _write_markdown_report(report_dir / "DAY26_RELIABILITY_REPORT.md", summary, tables, feature_audit)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"\nDay 26 outputs written to: {report_dir}")
    return summary


def _write_markdown_report(path: Path, summary: dict[str, Any],
                           tables: dict[str, pd.DataFrame],
                           feature_audit: dict[str, Any]) -> None:
    metrics = summary["overall_metrics_recomputed"]
    residual = summary["residual_summary"]
    lines = [
        "# FlowCast Day 26 — Model Reliability and Error Analysis",
        "",
        "## Scope",
        "",
        "This report evaluates the fixed Day 25 model on the chronological test partition. "
        "No model fitting or hyperparameter tuning is performed.",
        "",
        "## Overall test metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {metrics['rows']} |",
        f"| MAE | {metrics['mae']:.4f} |",
        f"| RMSE | {metrics['rmse']:.4f} |",
        f"| R² | {metrics['r2']:.5f} |",
        f"| MAPE (non-zero targets only) | {metrics['mape_percent_nonzero_targets_only']:.4f}% |",
        f"| Mean error (prediction − actual) | {residual['mean_error_predicted_minus_actual']:.4f} |",
        "",
        "## Residual diagnostics",
        "",
        f"- Overprediction rows: {residual['overprediction_rows']}",
        f"- Underprediction rows: {residual['underprediction_rows']}",
        f"- 90th percentile absolute error: {residual['p90_absolute_error']:.4f}",
        f"- 95th percentile absolute error: {residual['p95_absolute_error']:.4f}",
        f"- 99th percentile absolute error: {residual['p99_absolute_error']:.4f}",
        "",
        "## Group-level error tables",
        "",
    ]
    if not tables:
        lines.append("No group tables were produced; inspect the test split's identifier columns.")
    for name, table in tables.items():
        lines.extend(["", f"### {name.replace('_', ' ').title()}", ""])
        if table.empty:
            lines.append("No groups met the minimum sample threshold.")
        else:
            preview = table.head(10).copy()
            columns = list(preview.columns)
            def fmt_cell(value: Any) -> str:
                if isinstance(value, (float, np.floating)):
                    return f"{value:.3f}"
                return str(value)
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
            for _, row in preview.iterrows():
                lines.append("| " + " | ".join(fmt_cell(row[column]) for column in columns) + " |")
    lines.extend([
        "",
        "## Feature availability and leakage checklist",
        "",
        f"Status: **{feature_audit['status']}**",
        "",
        feature_audit["message"],
        "",
        "### Manual checks",
        "",
    ])
    for item in feature_audit["manual_review"]:
        lines.append(f"- **{item['feature']}** — {item['question']}")
    lines.extend([
        "",
        f"- **Lag/rolling features:** {feature_audit['lag_and_rolling_features']['required_rule']}",
        "",
        "## Interpretation and limitations",
        "",
        "- Group-level metrics are descriptive, not causal.",
        "- MAPE excludes zero-target rows and should not be used alone.",
        "- Do not tune against the test set; use validation data for model changes.",
        "- A good retrospective score does not prove inputs were available at forecast time.",
        "- The report is an analysis artifact, not a production monitoring guarantee.",
        "",
        "## Generated artifacts",
        "",
        "- `day26_reliability_summary.json`",
        "- `test_predictions_with_residuals.csv`",
        "- `top_20_absolute_errors.csv`",
        "- `feature_availability_audit.json`",
        "- Group-level CSV files and diagnostic plots (if matplotlib is installed)",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_analysis()
