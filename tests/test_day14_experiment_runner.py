"""
Tests for FlowCast Day 14 experiment management.
"""

import logging

from pathlib import Path

import pytest

from src.day14.experiment_config import (
    ExperimentConfig,
    create_experiment_id,
)

from src.day14.experiment_runner import (
    ExperimentRunner,
    build_strategies,
    configure_logging,
)


# ---------------------------------------------------------
# CONFIGURATION TESTS
# ---------------------------------------------------------

def test_default_configuration_is_valid():
    config = ExperimentConfig()

    config.validate()

    assert (
        config.experiment_name
        == "flowcast_baseline_evaluation"
    )


def test_empty_experiment_name_is_rejected():
    config = ExperimentConfig(
        experiment_name="   "
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_experiment_name_is_rejected():
    config = ExperimentConfig(
        experiment_name="invalid experiment!"
    )

    with pytest.raises(ValueError):
        config.validate()


def test_empty_strategy_list_is_rejected():
    config = ExperimentConfig(
        strategies=()
    )

    with pytest.raises(ValueError):
        config.validate()


def test_duplicate_strategies_are_rejected():
    config = ExperimentConfig(
        strategies=(
            "naive_30min",
            "naive_30min",
        )
    )

    with pytest.raises(ValueError):
        config.validate()


def test_unknown_strategy_is_rejected():
    config = ExperimentConfig(
        strategies=("unknown_strategy",)
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_log_level_is_rejected():
    config = ExperimentConfig(
        log_level="INVALID"
    )

    with pytest.raises(ValueError):
        config.validate()


def test_directories_are_created(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
    )

    config.create_directories()

    assert (
        tmp_path / "results"
    ).is_dir()

    assert (
        tmp_path / "logs"
    ).is_dir()


# ---------------------------------------------------------
# EXPERIMENT ID TESTS
# ---------------------------------------------------------

def test_experiment_id_has_expected_prefix():
    experiment_id = create_experiment_id()

    assert experiment_id.startswith(
        "flowcast_"
    )


def test_experiment_ids_are_different():
    first = create_experiment_id()
    second = create_experiment_id()

    assert first != second


# ---------------------------------------------------------
# STRATEGY REGISTRY TESTS
# ---------------------------------------------------------

def test_builds_both_supported_strategies():
    strategies = build_strategies(
        (
            "naive_30min",
            "seasonal_naive_24h",
        )
    )

    assert len(strategies) == 2

    assert {
        strategy.name
        for strategy in strategies
    } == {
        "naive_30min",
        "seasonal_naive_24h",
    }


def test_builds_single_strategy():
    strategies = build_strategies(
        ("naive_30min",)
    )

    assert len(strategies) == 1

    assert (
        strategies[0].name
        == "naive_30min"
    )


def test_unknown_strategy_fails():
    with pytest.raises(ValueError):
        build_strategies(
            ("not_a_real_strategy",)
        )


# ---------------------------------------------------------
# LOGGING TESTS
# ---------------------------------------------------------

def test_logging_creates_log_file(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
    )

    logger = configure_logging(config)

    try:
        logger.info(
            "Day 14 logging test."
        )

        for handler in logger.handlers:
            handler.flush()

        log_files = list(
            (tmp_path / "logs").glob("*.log")
        )

        assert len(log_files) == 1

        content = log_files[0].read_text(
            encoding="utf-8"
        )

        assert "Day 14 logging test." in content

    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        logging.Logger.manager.loggerDict.pop(
            logger.name,
            None,
        )


# ---------------------------------------------------------
# RUNNER INITIALIZATION TESTS
# ---------------------------------------------------------

def test_runner_accepts_valid_configuration(
    tmp_path,
):
    config = ExperimentConfig(
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
    )

    runner = ExperimentRunner(config)

    assert runner.config is config

    for handler in runner.logger.handlers[:]:
        handler.close()
        runner.logger.removeHandler(handler)


def test_runner_rejects_invalid_configuration(
    tmp_path,
):
    config = ExperimentConfig(
        experiment_name="invalid name",
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
    )

    with pytest.raises(ValueError):
        ExperimentRunner(config)


def test_runner_rejects_missing_training_dataset(
    tmp_path,
):
    config = ExperimentConfig(
        train_path=tmp_path / "missing_train.csv",
        validation_path=tmp_path / "missing_validation.csv",
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
    )

    runner = ExperimentRunner(config)

    try:
        with pytest.raises(FileNotFoundError):
            runner.run()

    finally:
        for handler in runner.logger.handlers[:]:
            handler.close()
            runner.logger.removeHandler(handler)