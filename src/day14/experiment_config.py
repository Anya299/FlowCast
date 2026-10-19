"""
FlowCast Day 14
Experiment configuration and validation.
"""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Immutable configuration for one forecasting experiment.
    """

    experiment_name: str = "flowcast_baseline_evaluation"

    train_path: Path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "day9"
        / "flowcast_train.csv"
    )

    validation_path: Path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "day9"
        / "flowcast_validation.csv"
    )

    output_dir: Path = (
        PROJECT_ROOT
        / "reports"
        / "day14"
    )

    log_dir: Path = PROJECT_ROOT / "logs"

    log_level: str = "INFO"

    strategies: tuple[str, ...] = (
        "naive_30min",
        "seasonal_naive_24h",
    )

    def validate(self) -> None:
        """
        Validate experiment settings before execution.
        """

        if not self.experiment_name.strip():
            raise ValueError(
                "Experiment name cannot be empty."
            )

        if not re.fullmatch(
            r"[A-Za-z0-9_-]+",
            self.experiment_name,
        ):
            raise ValueError(
                "Experiment name may contain only "
                "letters, numbers, underscores, and hyphens."
            )

        if not isinstance(self.train_path, Path):
            raise TypeError(
                "train_path must be a pathlib.Path."
            )

        if not isinstance(self.validation_path, Path):
            raise TypeError(
                "validation_path must be a pathlib.Path."
            )

        if not isinstance(self.output_dir, Path):
            raise TypeError(
                "output_dir must be a pathlib.Path."
            )

        if not isinstance(self.log_dir, Path):
            raise TypeError(
                "log_dir must be a pathlib.Path."
            )

        if not self.strategies:
            raise ValueError(
                "At least one forecasting strategy is required."
            )

        supported_strategies = {
            "naive_30min",
            "seasonal_naive_24h",
        }

        if len(set(self.strategies)) != len(
            self.strategies
        ):
            raise ValueError(
                "Duplicate forecasting strategies are not allowed."
            )

        unsupported = (
            set(self.strategies)
            - supported_strategies
        )

        if unsupported:
            raise ValueError(
                f"Unsupported strategies: {sorted(unsupported)}"
            )

        if self.log_level not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        }:
            raise ValueError(
                "log_level must be DEBUG, INFO, "
                "WARNING, or ERROR."
            )

    def create_directories(self) -> None:
        """
        Create output and log directories.
        """

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


def create_experiment_id() -> str:
    """
    Create a timestamp-based UTC experiment identifier
    with a random UUID suffix to avoid rapid-call collisions.
    """

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    unique_suffix = uuid4().hex[:8]

    return f"flowcast_{timestamp}_{unique_suffix}"