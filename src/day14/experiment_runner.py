"""
FlowCast Day 14
Reproducible forecasting experiment runner.
"""

import json
import logging
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ---------------------------------------------------------
# IMPORT EXISTING DAY 13 COMPONENTS
# ---------------------------------------------------------

from src.day13_forecasting_strategies import (
    validate_dataset,
    NaiveForecastStrategy,
    SeasonalNaiveForecastStrategy,
    ForecastEvaluator,
    ROAD_COLUMN,
    TIME_COLUMN,
)


from src.day14.experiment_config import (
    ExperimentConfig,
    create_experiment_id,
)


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

def configure_logging(
    config: ExperimentConfig,
) -> logging.Logger:
    """
    Configure console and file logging.

    Each experiment gets its own log file.
    """

    config.create_directories()

    experiment_id = create_experiment_id()

    logger = logging.getLogger(
        f"flowcast.day14.{experiment_id}"
    )

    logger.setLevel(
        getattr(logging, config.log_level)
    )

    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console_handler = logging.StreamHandler()

    console_handler.setLevel(
        getattr(logging, config.log_level)
    )

    console_handler.setFormatter(
        formatter
    )

    file_handler = logging.FileHandler(
        config.log_dir
        / f"{experiment_id}.log",
        encoding="utf-8",
    )

    file_handler.setLevel(
        getattr(logging, config.log_level)
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    return logger


# ---------------------------------------------------------
# STRATEGY REGISTRY
# ---------------------------------------------------------

def build_strategies(
    strategy_names: tuple[str, ...],
) -> list[Any]:
    """
    Construct supported forecasting strategies.
    """

    registry = {
        "naive_30min": NaiveForecastStrategy,
        "seasonal_naive_24h": (
            SeasonalNaiveForecastStrategy
        ),
    }

    strategies = []

    for name in strategy_names:
        if name not in registry:
            raise ValueError(
                f"Unsupported strategy: {name}"
            )

        strategies.append(
            registry[name]()
        )

    return strategies


# ---------------------------------------------------------
# DATASET METADATA
# ---------------------------------------------------------

def dataset_metadata(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Summarize a dataset without copying its full contents.
    """

    return {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "unique_roads": int(
            dataframe[ROAD_COLUMN].nunique()
        ),
        "start_timestamp": str(
            dataframe[TIME_COLUMN].min()
        ),
        "end_timestamp": str(
            dataframe[TIME_COLUMN].max()
        ),
    }


# ---------------------------------------------------------
# EXPERIMENT RUNNER
# ---------------------------------------------------------

class ExperimentRunner:
    """
    Coordinates data loading, forecasting, evaluation,
    logging, and experiment artifact creation.
    """

    def __init__(
        self,
        config: ExperimentConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ExperimentConfig()
        )

        self.config.validate()

        self.logger = configure_logging(
            self.config
        )

    def run(self) -> dict[str, Any]:
        """
        Execute one rolling forecasting experiment.

        Actual validation observations can act as historical
        references for later validation timestamps. This
        represents rolling one-step evaluation, not a
        multi-step forecast made entirely in advance.
        """

        experiment_id = create_experiment_id()

        started_at = datetime.now(
            timezone.utc
        )

        self.logger.info(
            "Experiment started: %s",
            experiment_id,
        )

        self.logger.info(
            "Experiment name: %s",
            self.config.experiment_name,
        )

        try:
            result = self._execute(
                experiment_id=experiment_id,
                started_at=started_at,
            )

            self.logger.info(
                "Experiment completed successfully: %s",
                experiment_id,
            )

            return result

        except Exception:
            self.logger.exception(
                "Experiment failed: %s",
                experiment_id,
            )

            raise

    def _execute(
        self,
        experiment_id: str,
        started_at: datetime,
    ) -> dict[str, Any]:

        # ---------------------------------------------
        # 1. CHECK INPUT FILES
        # ---------------------------------------------

        if not self.config.train_path.exists():
            raise FileNotFoundError(
                "Training dataset not found: "
                f"{self.config.train_path}"
            )

        if not self.config.validation_path.exists():
            raise FileNotFoundError(
                "Validation dataset not found: "
                f"{self.config.validation_path}"
            )

        self.logger.info(
            "Loading training dataset."
        )

        train = validate_dataset(
            pd.read_csv(
                self.config.train_path,
                low_memory=False,
            ),
            "Training dataset",
        )

        self.logger.info(
            "Loading validation dataset."
        )

        validation = validate_dataset(
            pd.read_csv(
                self.config.validation_path,
                low_memory=False,
            ),
            "Validation dataset",
        )

        self.logger.info(
            "Training rows: %d",
            len(train),
        )

        self.logger.info(
            "Validation rows: %d",
            len(validation),
        )

        # ---------------------------------------------
        # 2. PREPARE ROLLING HISTORY
        # ---------------------------------------------

        history = pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )

        history = history.sort_values(
            [
                ROAD_COLUMN,
                TIME_COLUMN,
            ]
        ).reset_index(
            drop=True
        )

        self.logger.info(
            "Historical reference rows: %d",
            len(history),
        )

        # ---------------------------------------------
        # 3. CREATE FORECASTING STRATEGIES
        # ---------------------------------------------

        strategies = build_strategies(
            self.config.strategies
        )

        evaluator = ForecastEvaluator()

        evaluation_results = []

        prediction_frames = []

        # ---------------------------------------------
        # 4. RUN EACH STRATEGY
        # ---------------------------------------------

        for strategy in strategies:

            self.logger.info(
                "Evaluating strategy: %s",
                strategy.name,
            )

            predictions = strategy.predict(
                history,
                validation,
            )

            metrics = evaluator.evaluate(
                predictions
            )

            evaluation_results.append(
                metrics
            )

            prediction_frames.append(
                predictions
            )

            self.logger.info(
                "Strategy %s: coverage=%.4f%%",
                strategy.name,
                metrics[
                    "prediction_coverage_pct"
                ],
            )

            self.logger.info(
                "Strategy %s: MAE=%s, RMSE=%s",
                strategy.name,
                metrics["mae"],
                metrics["rmse"],
            )

        # ---------------------------------------------
        # 5. BUILD RESULTS
        # ---------------------------------------------

        comparison = pd.DataFrame(
            evaluation_results
        )

        predictions_all = pd.concat(
            prediction_frames,
            ignore_index=True,
        )

        finished_at = datetime.now(
            timezone.utc
        )

        duration_seconds = (
            finished_at - started_at
        ).total_seconds()

        # ---------------------------------------------
        # 6. SAVE ARTIFACTS
        # ---------------------------------------------

        comparison_path = (
            self.config.output_dir
            / f"{experiment_id}_comparison.csv"
        )

        predictions_path = (
            self.config.output_dir
            / f"{experiment_id}_predictions.csv"
        )

        metadata_path = (
            self.config.output_dir
            / f"{experiment_id}_metadata.json"
        )

        report_path = (
            self.config.output_dir
            / f"{experiment_id}_report.md"
        )

        comparison.to_csv(
            comparison_path,
            index=False,
        )

        predictions_all.to_csv(
            predictions_path,
            index=False,
        )

        metadata = {
            "experiment_id": experiment_id,
            "experiment_name": (
                self.config.experiment_name
            ),
            "status": "success",
            "started_at_utc": (
                started_at.isoformat()
            ),
            "finished_at_utc": (
                finished_at.isoformat()
            ),
            "duration_seconds": duration_seconds,
            "strategies": list(
                self.config.strategies
            ),
            "evaluation_mode": (
                "rolling_one_step_evaluation"
            ),
            "training_dataset": {
                "path": str(
                    self.config.train_path
                ),
                **dataset_metadata(train),
            },
            "validation_dataset": {
                "path": str(
                    self.config.validation_path
                ),
                **dataset_metadata(validation),
            },
            "historical_reference_rows": int(
                len(history)
            ),
            "artifacts": {
                "comparison_csv": str(
                    comparison_path
                ),
                "predictions_csv": str(
                    predictions_path
                ),
                "metadata_json": str(
                    metadata_path
                ),
                "report_markdown": str(
                    report_path
                ),
            },
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=4,
            ),
            encoding="utf-8",
        )

        report = self._build_report(
            metadata,
            comparison,
        )

        report_path.write_text(
            report,
            encoding="utf-8",
        )

        self.logger.info(
            "Comparison saved: %s",
            comparison_path,
        )

        self.logger.info(
            "Predictions saved: %s",
            predictions_path,
        )

        self.logger.info(
            "Metadata saved: %s",
            metadata_path,
        )

        self.logger.info(
            "Report saved: %s",
            report_path,
        )

        return {
            "metadata": metadata,
            "comparison": comparison,
            "comparison_path": comparison_path,
            "predictions_path": predictions_path,
            "metadata_path": metadata_path,
            "report_path": report_path,
        }

    @staticmethod
    def _build_report(
        metadata: dict[str, Any],
        comparison: pd.DataFrame,
    ) -> str:
        """
        Generate a Markdown report from experiment results.
        """

        comparison_table = comparison.to_markdown(
            index=False
        )

        return f"""# FlowCast Day 14
## Forecasting Experiment Report

### Experiment information

- **Experiment ID:** {metadata['experiment_id']}
- **Experiment name:** {metadata['experiment_name']}
- **Status:** {metadata['status']}
- **Evaluation mode:** {metadata['evaluation_mode']}
- **Started (UTC):** {metadata['started_at_utc']}
- **Finished (UTC):** {metadata['finished_at_utc']}
- **Duration:** {metadata['duration_seconds']:.3f} seconds

### Dataset summary

| Property | Training | Validation |
|---|---:|---:|
| Rows | {metadata['training_dataset']['rows']} | {metadata['validation_dataset']['rows']} |
| Columns | {metadata['training_dataset']['columns']} | {metadata['validation_dataset']['columns']} |
| Unique roads | {metadata['training_dataset']['unique_roads']} | {metadata['validation_dataset']['unique_roads']} |
| Start | {metadata['training_dataset']['start_timestamp']} | {metadata['validation_dataset']['start_timestamp']} |
| End | {metadata['training_dataset']['end_timestamp']} | {metadata['validation_dataset']['end_timestamp']} |

### Forecasting strategy comparison

{comparison_table}

### Interpretation

Prediction coverage measures the percentage of validation rows for which
a prediction is available. MAE measures mean absolute error, while RMSE
penalizes larger errors more strongly.

Metrics should be compared using a consistent evaluation population.
Coverage differences may affect the observations contributing to each
strategy's error metrics.

This experiment uses rolling one-step evaluation. Actual observations
from earlier validation timestamps may serve as historical references
for later timestamps. These results should not be interpreted as a
multi-step forecast produced entirely before the validation period.

### Reproducibility information

The experiment configuration, input dataset paths, execution timestamps,
strategy names, and output artifact paths are recorded in the metadata JSON.

### Generated artifacts

- Comparison CSV: `{metadata['artifacts']['comparison_csv']}`
- Predictions CSV: `{metadata['artifacts']['predictions_csv']}`
- Metadata JSON: `{metadata['artifacts']['metadata_json']}`
- Markdown report: `{metadata['artifacts']['report_markdown']}`
"""

# ---------------------------------------------------------
# COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":

    config = ExperimentConfig()

    runner = ExperimentRunner(config)

    results = runner.run()

    print("\n" + "=" * 65)
    print("FLOWCAST DAY 14 — EXPERIMENT COMPLETED")
    print("=" * 65)

    print("\nExperiment ID:")
    print(
        results["metadata"]["experiment_id"]
    )

    print("\nForecasting comparison:")
    print(
        results["comparison"].to_string(
            index=False
        )
    )

    print("\nGenerated files:")

    print(
        "Comparison:",
        results["comparison_path"],
    )

    print(
        "Predictions:",
        results["predictions_path"],
    )

    print(
        "Metadata:",
        results["metadata_path"],
    )

    print(
        "Report:",
        results["report_path"],
    )