# FlowCast — Day 14 Technical Documentation

## Title

Experiment Management and Reproducible Forecast Evaluation

## 1. Objective

The objective of Day 14 was to introduce a reusable experiment-management
layer around the forecasting strategies implemented in Day 13.

The system centralizes configuration, validates experiment settings,
records execution logs, evaluates forecasting strategies, and saves
structured experiment artifacts.

## 2. Architecture

The experiment workflow contains the following components:

1. Experiment configuration
2. Configuration validation
3. Dataset loading and validation
4. Forecasting strategy construction
5. Prediction generation
6. Metric evaluation
7. Artifact generation
8. Logging and execution tracking

## 3. Configuration Management

An immutable configuration object stores the experiment name, dataset
paths, output directories, logging level, and forecasting strategies.

Configuration validation prevents unsupported strategies, duplicate
strategy names, invalid logging levels, and invalid experiment names.

## 4. Forecasting Strategies

The experiment runner reuses the strategies implemented in Day 13:

- Naive forecast using the observation 30 minutes earlier.
- Seasonal naive forecast using the observation 24 hours earlier.

The strategies are constructed through a registry, allowing supported
strategies to be selected through configuration.

## 5. Evaluation Method

The experiment uses rolling one-step evaluation.

Training and validation observations are combined into a historical
reference table. Each prediction strategy requests a historical value
at an earlier timestamp.

Actual observations from earlier validation timestamps can therefore
serve as references for later validation timestamps.

This evaluation setup is different from generating a complete
multi-step forecast before the validation period begins.

## 6. Logging

The runner records experiment lifecycle events, dataset loading,
strategy evaluation, output paths, and execution failures.

Each run receives a timestamp-based experiment identifier and a
separate log file.

## 7. Generated Artifacts

Each experiment produces:

- A strategy comparison CSV.
- A prediction-level CSV.
- A JSON metadata file.
- A Markdown experiment report.
- A separate execution log.

These artifacts make it easier to inspect and compare experiment runs.

## 8. Automated Testing

Automated tests cover:

- Configuration validation.
- Experiment identifier generation.
- Strategy registry behavior.
- Logging output.
- Runner initialization.
- Missing dataset handling.

The existing Day 11, Day 12, Day 13, and data-contract tests are also
run to check for regressions.

## 9. Limitations

The timestamp-based experiment identifier is not a cryptographic
identifier and does not guarantee collision-free execution.

The runner records dataset paths and metadata, but does not calculate
cryptographic hashes of the input datasets.

The experiment does not yet provide model versioning, parameter search,
or a centralized experiment database.

## 10. Future Improvements

Potential future improvements include:

- Dataset content hashes.
- Explicit experiment configuration files.
- Model parameter tracking.
- Experiment comparison dashboards.
- Automated experiment validation in CI.
- Forecasting model selection based on defined evaluation criteria.

## 11. Conclusion

Day 14 introduces a reusable experiment-management workflow around
FlowCast's existing forecasting strategies.

The implementation separates configuration, execution, evaluation,
logging, and reporting, making future forecasting experiments easier
to reproduce, test, and inspect.