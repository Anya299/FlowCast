from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "day10"

NAIVE_FILE = REPORT_DIR / "naive_validation_predictions.csv"
SEASONAL_FILE = REPORT_DIR / "seasonal_validation_predictions.csv"


def calculate_metrics(actual, predicted):
    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    return mae, rmse


def main():
    print("FLOWCAST DAY 10 — FAIR BASELINE COMPARISON")

    naive = pd.read_csv(NAIVE_FILE)
    seasonal = pd.read_csv(SEASONAL_FILE)

    keys = ["road_id", "timestamp"]

    naive = naive.rename(
        columns={
            "actual": "actual_naive",
            "predicted": "predicted_naive",
        }
    )

    seasonal = seasonal.rename(
        columns={
            "actual": "actual_seasonal",
            "predicted": "predicted_seasonal",
        }
    )

    comparison = naive[
        keys + ["actual_naive", "predicted_naive"]
    ].merge(
        seasonal[
            keys + ["actual_seasonal", "predicted_seasonal"]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    actuals_match = np.isclose(
        comparison["actual_naive"],
        comparison["actual_seasonal"],
        equal_nan=True,
    )

    if not actuals_match.all():
        raise ValueError("The actual target values do not match between files.")

    valid = (
        comparison["actual_naive"].notna()
        & comparison["predicted_naive"].notna()
        & comparison["predicted_seasonal"].notna()
        & np.isfinite(comparison["actual_naive"])
        & np.isfinite(comparison["predicted_naive"])
        & np.isfinite(comparison["predicted_seasonal"])
    )

    common = comparison.loc[valid]

    total_rows = len(comparison)
    common_rows = len(common)

    coverage = (
        common_rows / total_rows * 100
        if total_rows
        else 0
    )

    print(f"\nTotal matched validation rows: {total_rows:,}")
    print(f"Rows predicted by BOTH models: {common_rows:,}")
    print(f"Common-row coverage: {coverage:.2f}%")

    if common_rows == 0:
        raise ValueError("No common valid prediction rows were found.")

    actual = common["actual_naive"].to_numpy(dtype=float)

    naive_predictions = common["predicted_naive"].to_numpy(dtype=float)
    seasonal_predictions = common["predicted_seasonal"].to_numpy(dtype=float)

    naive_mae, naive_rmse = calculate_metrics(
        actual, naive_predictions
    )

    seasonal_mae, seasonal_rmse = calculate_metrics(
        actual, seasonal_predictions
    )

    results = pd.DataFrame(
        [
            {
                "model": "Naive 30-minute",
                "rows_evaluated": common_rows,
                "MAE": naive_mae,
                "RMSE": naive_rmse,
            },
            {
                "model": "Seasonal naive 24-hour",
                "rows_evaluated": common_rows,
                "MAE": seasonal_mae,
                "RMSE": seasonal_rmse,
            },
        ]
    )

    print("\nFAIR COMPARISON — SAME VALIDATION ROWS")
    print(results.to_string(index=False))

    if naive_mae < seasonal_mae:
        print("\nWinner by MAE: Naive 30-minute")
    elif seasonal_mae < naive_mae:
        print("\nWinner by MAE: Seasonal naive 24-hour")
    else:
        print("\nBoth models have equal MAE.")

    if naive_rmse < seasonal_rmse:
        print("Winner by RMSE: Naive 30-minute")
    elif seasonal_rmse < naive_rmse:
        print("Winner by RMSE: Seasonal naive 24-hour")
    else:
        print("Both models have equal RMSE.")

    output_file = REPORT_DIR / "common_rows_baseline_comparison.csv"
    results.to_csv(output_file, index=False)

    print(f"\nComparison saved to: {output_file}")


if __name__ == "__main__":
    main()