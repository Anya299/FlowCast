"""
FLOWCAST DAY 13
Object-Oriented Forecasting Strategies

Concepts:
- Abstract base classes
- Inheritance
- Polymorphism
- Strategy pattern
- Data validation
- Forecast evaluation
- Automated reporting
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "day9"
    / "flowcast_train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "day9"
    / "flowcast_validation.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "day13"

ROAD_COLUMN = "road_id"
TIME_COLUMN = "timestamp"
TARGET_COLUMN = "traffic_volume"


# ============================================================
# 2. ABSTRACT BASE CLASS
# ============================================================

class ForecastStrategy(ABC):
    """
    Common interface for FlowCast forecasting strategies.

    Every concrete forecasting strategy must implement
    the predict() method.
    """

    def __init__(
        self,
        name: str,
        reference_minutes: int,
    ) -> None:

        if not name.strip():
            raise ValueError("Strategy name cannot be empty.")

        if reference_minutes <= 0:
            raise ValueError(
                "Reference interval must be positive."
            )

        self.name = name
        self.reference_minutes = reference_minutes

    @abstractmethod
    def predict(
        self,
        history: pd.DataFrame,
        validation: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions for validation rows.

        Concrete strategies must implement this method.
        """
        raise NotImplementedError


# ============================================================
# 3. NAIVE FORECASTING STRATEGY
# ============================================================

class NaiveForecastStrategy(ForecastStrategy):
    """
    Predict traffic using the most recent observation
    exactly 30 minutes before the forecast timestamp.
    """

    def __init__(self) -> None:
        super().__init__(
            name="naive_30min",
            reference_minutes=30,
        )

    def predict(
        self,
        history: pd.DataFrame,
        validation: pd.DataFrame,
    ) -> pd.DataFrame:

        return self._generate_predictions(
            history,
            validation,
        )

    def _generate_predictions(
        self,
        history: pd.DataFrame,
        validation: pd.DataFrame,
    ) -> pd.DataFrame:

        return generate_reference_predictions(
            history=history,
            validation=validation,
            strategy_name=self.name,
            reference_minutes=self.reference_minutes,
        )


# ============================================================
# 4. SEASONAL-NAIVE FORECASTING STRATEGY
# ============================================================

class SeasonalNaiveForecastStrategy(ForecastStrategy):
    """
    Predict traffic using the observation exactly 24 hours
    before the forecast timestamp.
    """

    def __init__(self) -> None:
        super().__init__(
            name="seasonal_naive_24h",
            reference_minutes=24 * 60,
        )

    def predict(
        self,
        history: pd.DataFrame,
        validation: pd.DataFrame,
    ) -> pd.DataFrame:

        return generate_reference_predictions(
            history=history,
            validation=validation,
            strategy_name=self.name,
            reference_minutes=self.reference_minutes,
        )


# ============================================================
# 5. DATA VALIDATION
# ============================================================

def validate_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Validate and standardize the required traffic columns.
    """

    required_columns = {
        ROAD_COLUMN,
        TIME_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    result = dataframe.copy()

    if result[ROAD_COLUMN].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing road identifiers."
        )

    result[TIME_COLUMN] = pd.to_datetime(
        result[TIME_COLUMN],
        errors="raise",
    )

    result[TARGET_COLUMN] = pd.to_numeric(
        result[TARGET_COLUMN],
        errors="raise",
    )

    duplicate_mask = result.duplicated(
        subset=[ROAD_COLUMN, TIME_COLUMN],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            f"{dataset_name} contains duplicate road/timestamp keys."
        )

    return result


# ============================================================
# 6. COMMON REFERENCE-BASED PREDICTION ENGINE
# ============================================================

def generate_reference_predictions(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    strategy_name: str,
    reference_minutes: int,
) -> pd.DataFrame:
    """
    Generate predictions using a specified historical interval.

    Only earlier observations are used as references.

    The history supplied to this function must contain only
    observations available before the relevant forecast time.
    """

    if reference_minutes <= 0:
        raise ValueError(
            "Reference interval must be positive."
        )

    history = validate_dataset(
        history,
        "Historical dataset",
    )

    validation = validate_dataset(
        validation,
        "Validation dataset",
    )

    # Build a road/timestamp lookup table.
    lookup = history[
        [
            ROAD_COLUMN,
            TIME_COLUMN,
            TARGET_COLUMN,
        ]
    ].copy()

    lookup = lookup.rename(
        columns={
            TIME_COLUMN: "reference_timestamp",
            TARGET_COLUMN: "predicted",
        }
    )

    # Calculate the exact historical timestamp required.
    forecast_rows = validation[
        [
            ROAD_COLUMN,
            TIME_COLUMN,
            TARGET_COLUMN,
        ]
    ].copy()

    forecast_rows = forecast_rows.rename(
        columns={
            TARGET_COLUMN: "actual",
        }
    )

    forecast_rows["reference_timestamp"] = (
        forecast_rows[TIME_COLUMN]
        - pd.Timedelta(minutes=reference_minutes)
    )

    # Retrieve the observation from the same road and
    # the exact reference timestamp.
    predictions = forecast_rows.merge(
        lookup,
        on=[
            ROAD_COLUMN,
            "reference_timestamp",
        ],
        how="left",
        validate="many_to_one",
    )

    # Record which strategy produced each prediction.
    predictions["model"] = strategy_name

    # Preserve the expected output column order.
    predictions = predictions[
        [
            ROAD_COLUMN,
            TIME_COLUMN,
            "actual",
            "predicted",
            "reference_timestamp",
            "model",
        ]
    ]

    return predictions


# ============================================================
# 7. SHARED FORECAST EVALUATOR
# ============================================================

class ForecastEvaluator:
    """
    Evaluate predictions produced by any ForecastStrategy.

    This class works with the common output interface
    instead of depending on a specific forecasting class.
    """

    REQUIRED_COLUMNS = {
        "road_id",
        "timestamp",
        "actual",
        "predicted",
        "reference_timestamp",
        "model",
    }

    def evaluate(
        self,
        predictions: pd.DataFrame,
    ) -> dict[str, object]:
        """
        Calculate prediction coverage, MAE and RMSE.
        """

        missing_columns = (
            self.REQUIRED_COLUMNS
            - set(predictions.columns)
        )

        if missing_columns:
            raise ValueError(
                "Prediction data is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if predictions.empty:
            raise ValueError(
                "Cannot evaluate an empty prediction dataset."
            )

        result = predictions.copy()

        result["actual"] = pd.to_numeric(
            result["actual"],
            errors="raise",
        )

        result["predicted"] = pd.to_numeric(
            result["predicted"],
            errors="raise",
        )

        total_rows = len(result)

        valid_mask = (
            result["actual"].notna()
            & result["predicted"].notna()
        )

        valid = result.loc[valid_mask].copy()

        valid_rows = len(valid)

        missing_predictions = total_rows - valid_rows

        coverage = (
            valid_rows / total_rows * 100
            if total_rows
            else 0.0
        )

        if valid_rows:
            errors = (
                valid["actual"] - valid["predicted"]
            )

            mae = errors.abs().mean()

            rmse = (
                (errors ** 2).mean()
            ) ** 0.5

        else:
            mae = float("nan")
            rmse = float("nan")

        model_names = result["model"].dropna().unique()

        if len(model_names) != 1:
            raise ValueError(
                "Evaluate one forecasting strategy at a time."
            )

        return {
            "model": str(model_names[0]),
            "validation_rows": total_rows,
            "valid_predictions": valid_rows,
            "missing_predictions": missing_predictions,
            "prediction_coverage_pct": round(
                coverage,
                4,
            ),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
        }


# ============================================================
# 8. MARKDOWN REPORT GENERATOR
# ============================================================

def create_markdown_report(
    summary: pd.DataFrame,
) -> str:
    """
    Generate a Markdown report from evaluation results.
    """

    if summary.empty:
        raise ValueError(
            "Cannot create a report from an empty summary."
        )

    columns = list(summary.columns)

    lines = [
        "# FlowCast Day 13: Forecasting Strategy Evaluation",
        "",
        "## Objective",
        "",
        (
            "Implement forecasting strategies through an "
            "object-oriented interface and compare their "
            "prediction coverage and forecasting errors."
        ),
        "",
        "## Evaluation results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(
            ["---"] * len(columns)
        ) + " |",
    ]

    for row in summary.itertuples(
        index=False,
        name=None,
    ):
        values = [
            "" if pd.isna(value) else str(value)
            for value in row
        ]

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        (
            "Prediction coverage is the percentage of validation "
            "rows for which a prediction is available."
        ),
        "",
        (
            "MAE measures average absolute forecasting error. "
            "RMSE penalizes larger errors more strongly."
        ),
        "",
        (
            "Compare MAE and RMSE on a consistent evaluation "
            "population. Coverage differences can affect which "
            "observations contribute to the error metrics."
        ),
        "",
        (
            "These are baseline models. Their results provide "
            "reference points for more advanced forecasting "
            "models in later FlowCast development."
        ),
        "",
    ])

    return "\n".join(lines)


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main() -> None:
    """
    Execute the Day 13 forecasting strategy pipeline.
    """

    print("=" * 65)
    print("FLOWCAST DAY 13")
    print("OBJECT-ORIENTED FORECASTING STRATEGIES")
    print("=" * 65)

    # --------------------------------------------------------
    # Load the existing Day 9 datasets.
    # --------------------------------------------------------

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_PATH}"
        )

    print("\nLoading datasets...")

    train = validate_dataset(
        pd.read_csv(TRAIN_PATH),
        "Training dataset",
    )

    validation = validate_dataset(
        pd.read_csv(VALIDATION_PATH),
        "Validation dataset",
    )

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))

    # --------------------------------------------------------
    # ROLLING HISTORICAL EVALUATION
    #
    # Validation observations can serve as historical
    # references for later validation timestamps.
    #
    # Each strategy requests an exact earlier timestamp.
    # The reference timestamp is always before the forecast
    # timestamp, so future observations are not used.
    # --------------------------------------------------------

    history = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    history = history.sort_values(
        [ROAD_COLUMN, TIME_COLUMN]
    ).reset_index(drop=True)

    print(
        "Historical reference rows:",
        len(history),
    )

    # --------------------------------------------------------
    # Create strategies.
    # --------------------------------------------------------

    strategies: list[ForecastStrategy] = [
        NaiveForecastStrategy(),
        SeasonalNaiveForecastStrategy(),
    ]

    evaluator = ForecastEvaluator()

    prediction_frames = []
    evaluation_results = []

    # --------------------------------------------------------
    # POLYMORPHISM:
    # The same predict() call works for each strategy.
    # --------------------------------------------------------

    for strategy in strategies:

        print("\nEvaluating strategy:", strategy.name)

        predictions = strategy.predict(
            history,
            validation,
        )

        metrics = evaluator.evaluate(
            predictions,
        )

        prediction_frames.append(
            predictions
        )

        evaluation_results.append(
            metrics
        )

        print(
            "Prediction coverage:",
            f"{metrics['prediction_coverage_pct']:.4f}%",
        )

        print(
            "MAE:",
            metrics["mae"],
        )

        print(
            "RMSE:",
            metrics["rmse"],
        )

    # --------------------------------------------------------
    # Combine evaluation results.
    # --------------------------------------------------------

    summary = pd.DataFrame(
        evaluation_results
    )

    predictions_all = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Save reports.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR / "forecast_strategy_comparison.csv"
    )

    predictions_path = (
        OUTPUT_DIR / "forecast_strategy_predictions.csv"
    )

    report_path = (
        OUTPUT_DIR / "day13_forecasting_strategy_report.md"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    predictions_all.to_csv(
        predictions_path,
        index=False,
    )

    report_path.write_text(
        create_markdown_report(summary),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Display final results.
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("DAY 13 COMPLETED")
    print("=" * 65)

    print("\nFORECAST STRATEGY COMPARISON")

    print(
        summary.to_string(
            index=False,
            col_space=12,
        )
    )

    print("\nREPORTS SAVED")

    print(summary_path)
    print(predictions_path)
    print(report_path)


if __name__ == "__main__":
    main()