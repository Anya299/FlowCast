from pathlib import Path

import numpy as np
import pandas as pd


# Project root: FlowCast/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports" / "day10"

FILES = {
    "Naive 30-minute": REPORT_DIR / "naive_validation_predictions.csv",
    "Seasonal naive 24-hour": REPORT_DIR / "seasonal_validation_predictions.csv",
}


def verify_predictions(model_name: str, file_path: Path) -> None:
    print("\n" + "=" * 65)
    print(f"MODEL: {model_name}")
    print("=" * 65)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    required_columns = {"actual", "predicted", "road_id", "timestamp"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(f"ERROR: Missing columns: {sorted(missing_columns)}")
        return

    # Only evaluate rows with finite actual and predicted values.
    valid = (
        df["actual"].notna()
        & df["predicted"].notna()
        & np.isfinite(df["actual"])
        & np.isfinite(df["predicted"])
    )

    actual = df.loc[valid, "actual"].to_numpy(dtype=float)
    predicted = df.loc[valid, "predicted"].to_numpy(dtype=float)

    total_rows = len(df)
    valid_rows = len(actual)
    missing_predictions = int(df["predicted"].isna().sum())
    coverage = valid_rows / total_rows * 100 if total_rows else 0

    print(f"Total validation rows: {total_rows:,}")
    print(f"Valid prediction pairs: {valid_rows:,}")
    print(f"Missing predictions: {missing_predictions:,}")
    print(f"Prediction coverage: {coverage:.2f}%")

    if valid_rows == 0:
        print("ERROR: No valid prediction pairs.")
        return

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    print(f"Recalculated MAE: {mae:.6f}")
    print(f"Recalculated RMSE: {rmse:.6f}")

    print("\nFirst five predictions:")
    print(
        df[
            ["road_id", "timestamp", "actual", "predicted"]
        ].head().to_string(index=False)
    )


def main() -> None:
    print("FLOWCAST DAY 10 — RESULTS VERIFICATION")

    for model_name, file_path in FILES.items():
        verify_predictions(model_name, file_path)

    print("\nVerification completed.")


if __name__ == "__main__":
    main()

