# FlowCast — Day 9 Experiment Setup Report

## Objective

Prepare chronological datasets for supervised traffic-volume prediction.

## Prediction target

- Target: `traffic_volume`
- Baseline features: 28
- Random shuffling: not used
- Model training: not performed

## Chronological boundaries

| Split | Period | Rows |
|---|---|---:|
| Train | Jan 1 – Apr 6, 2025 | 109,460 |
| Validation | Apr 7 – May 3, 2025 | 30,788 |
| Test | May 4 – May 31, 2025 | 31,866 |

## Target quality

- Negative targets invalidated: 0
- Missing target rows in source: 4,587

## Missing feature policy

Missing lag and rolling features are preserved. No forward filling or future-based imputation is used.

Rows with missing target values are excluded from supervised-learning splits.

## Interpretation

This stage creates experiment-ready partitions but does not train or evaluate a model. Model selection must use the training and validation periods. The test period should remain untouched until final evaluation.

Feature availability at the real prediction cutoff must be verified before production use.
