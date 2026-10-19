# FlowCast Day 5 — Timeline Audit Report

## 1. Objective

Reconstruct the expected 30-minute road-sensor timeline and identify missing observations, duplicate road-time keys, timestamp alignment problems, and coverage gaps.

## 2. Dataset overview

- Source rows: 176,701
- Source columns: 29
- Unique roads found: 25
- First valid timestamp: 2025-01-01 00:00:00
- Last valid timestamp: 2025-05-31 23:30:00
- Expected interval: 30 minutes
- Expected road-time combinations: 181,200

## 3. Timeline completeness

- Observed expected intervals: 176,701
- Missing intervals: 4,499
- Overall timeline coverage: 97.517%

Coverage is calculated from the expected road-time grid. It is not the same as the percentage of non-missing traffic-volume values.

## 4. Data quality findings

- Invalid timestamps: 0
- Missing road identifiers: 0
- Off-grid timestamp rows: 0
- Road-time keys with duplicates: 0
- Consecutive missing-gap events: 4,386

## 5. Five roads with the lowest coverage

| Road ID | Expected | Observed | Missing | Coverage |
|---|---:|---:|---:|---:|
| NL-015 | 7248 | 7045 | 203 | 97.20% |
| NL-019 | 7248 | 7045 | 203 | 97.20% |
| NL-022 | 7248 | 7052 | 196 | 97.30% |
| NL-006 | 7248 | 7053 | 195 | 97.31% |
| NL-016 | 7248 | 7053 | 195 | 97.31% |

## 6. Five dates with the lowest coverage

| Date | Expected | Observed | Missing | Coverage |
|---|---:|---:|---:|---:|
| 2025-05-29 | 1200 | 1157 | 43 | 96.42% |
| 2025-01-24 | 1200 | 1159 | 41 | 96.58% |
| 2025-05-09 | 1200 | 1159 | 41 | 96.58% |
| 2025-03-29 | 1200 | 1159 | 41 | 96.58% |
| 2025-05-12 | 1200 | 1159 | 41 | 96.58% |

## 7. Engineering decisions

1. The integrated source dataset is not modified.
2. Missing intervals are reported, not imputed.
3. Duplicate observations are identified, not deleted.
4. Off-grid timestamps are flagged, not automatically rounded.
5. Missing intervals are not automatically interpreted as sensor failures.
6. All roads are compared against a common time grid. This assumption must be checked against sensor deployment and operational records.

## 8. Output artifacts

- `missing_timeline_slots.csv`
- `duplicate_road_time_keys.csv`
- `off_grid_timestamps.csv`
- `timestamp_quality_issues.csv`
- `road_coverage.csv`
- `daily_coverage.csv`
- `gap_events.csv`
- `timeline_audit_summary.json`
- `day5_timeline_audit_report.md`

## 9. Interpretation and limitations

A missing road-time interval indicates that no valid timestamped row was observed for that road and interval. It does not independently prove sensor failure. The common timeline assumes all identified roads were expected to report across the full observed time range.

A row absent from the timeline can indicate missing data, a sensor reporting problem, a deployment boundary, or an incomplete extraction. These possibilities require further investigation.

Timeline coverage measures whether a road-time key exists. It does not establish that the associated traffic measurements are valid or complete.

## 10. Reproducibility

Run from the FlowCast project root:

```powershell
python src/day5_timeline_audit.py
```
