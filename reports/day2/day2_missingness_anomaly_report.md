# FlowCast — Day 2 Engineering Report

## Missingness and Anomaly Investigation

**Project:** FlowCast — Traffic Forecasting and AI Engineering
**Day:** 2
**Investigation:** Missingness and suspicious measurement analysis
**Input dataset:** `data/interim/flowcast_integrated.csv`
**Output directory:** `reports/day2/`

---

## 1. Objective

The objective of Day 2 was to investigate missing measurements and suspicious values in the integrated FlowCast dataset.

The investigation examined missingness by variable, road, date, and hour. It also applied screening rules to selected numerical measurements to identify records requiring further investigation.

The goal was to establish evidence for a data-quality policy before feature engineering and forecasting.

## 2. Dataset Overview

The integrated dataset contained:

* 176,701 records.
* 29 original columns.
* Traffic, road, weather, calendar, and event-related fields.

The investigation generated additional analytical fields and reports without intentionally deleting the source observations.

## 3. Missingness Analysis

### 3.1 Principal findings

| Variable         | Missing records | Missing percentage |
| ---------------- | --------------: | -----------------: |
| holiday_name     |         169,690 |           96.0323% |
| event_name       |         169,663 |           96.0170% |
| congestion_level |          26,624 |           15.0673% |
| traffic_volume   |           4,587 |            2.5959% |
| occupancy        |           4,344 |            2.4584% |
| avg_speed        |           4,343 |            2.4578% |
| temperature      |           2,708 |            1.5325% |
| visibility       |           1,806 |            1.0221% |

### 3.2 Interpretation

The descriptive fields `holiday_name` and `event_name` have high missingness. Their missingness requires interpretation in the context of their intended meanings.

An absent event name may be appropriate when no named event is associated with a record. Similarly, a missing holiday name may be expected on ordinary days. These assumptions must be verified against the source data and data-generation process.

The `congestion_level` field has approximately 15.07% missingness. Its definition and relationship to other traffic variables require investigation before it is used as a model target or feature.

Traffic volume, average speed, and occupancy each have approximately 2.5% missingness. These fields require careful treatment because they may be important for forecasting.

Temperature and visibility have lower missingness percentages, but their impact should still be assessed.

### 3.3 Missingness limitations

Missingness percentages alone do not reveal why measurements are absent.

The current investigation does not establish whether the missingness is completely random, related to observed variables, or related to unobserved conditions.

Further investigation is required before choosing an imputation strategy or excluding records.

## 4. Anomaly Investigation

### 4.1 Screening results

| Variable       | Screening rule          | Flagged records |
| -------------- | ----------------------- | --------------: |
| traffic_volume | Value below zero        |               0 |
| vehicle_count  | Value below zero        |               0 |
| avg_speed      | Below zero or above 200 |             236 |
| occupancy      | Below zero or above 100 |             229 |
| travel_time    | Value below zero        |               0 |
| rainfall       | Value below zero        |               0 |
| visibility     | Value below zero        |               0 |

The investigation produced 465 anomaly flags across the screening rules.

These flags must not be interpreted as 465 distinct observations because an individual record can violate multiple rules.

### 4.2 Interpretation

The screening rules identified speed and occupancy measurements that require further examination.

The selected thresholds are preliminary screening assumptions. They are not a substitute for verifying the units, measurement definitions, source-system constraints, or domain-specific limits.

The absence of flags for other variables means only that the specified screening rules did not identify values violating those rules. It does not prove that every remaining measurement is correct.

### 4.3 Road-level investigation

Flagged records occurred across multiple road identifiers.

The generated `anomalies_by_road.csv` file provides counts by road and anomaly variable.

Counts alone do not establish which road has the worst data quality. A rate-based comparison, using the number of valid opportunities or total records per road as the denominator, is needed.

## 5. Temporal Investigation

The investigation generated date-level and hour-level missingness summaries.

These outputs can be used to examine whether missing measurements are concentrated in particular periods.

No conclusion about a specific temporal pattern should be made until the complete summaries and visualisations have been inspected.

## 6. Data-Quality Decisions

The following decisions guide subsequent development:

1. Preserve the original integrated dataset.
2. Do not automatically delete anomaly-flagged records.
3. Distinguish absent descriptive metadata from missing numerical measurements.
4. Investigate the meaning and missingness of `congestion_level`.
5. Verify units and domain constraints before finalising anomaly thresholds.
6. Evaluate missingness by road and time before selecting imputation methods.
7. Document any future exclusions, transformations, or imputations.
8. Keep the data-cleaning process reproducible.

## 7. Generated Artifacts

The investigation generates the following outputs where applicable:

* `missingness_by_variable.csv`
* `missingness_by_road.csv`
* `missingness_by_date.csv`
* `missingness_by_hour.csv`
* `anomaly_summary.csv`
* `flagged_anomaly_records.csv`
* `anomalies_by_road.csv`
* `day2_summary.json`
* `figures/`

## 8. Limitations

The investigation uses preliminary screening rules.

It does not yet establish the underlying causes of missingness, prove that flagged values are erroneous, or determine the optimal treatment of each field.

The findings describe the supplied dataset and should not be generalised beyond its coverage and documented provenance.

## 9. Next Steps

Day 3 will establish an explicit data-quality policy.

The next stage will define field-level rules, distinguish invalid measurements from missing targets and missing features, document plausible extremes, and implement executable validation checks.

## 10. Completion Criteria

Day 2 is complete when:

* The investigation script runs successfully.
* Generated reports are inspected.
* The visualisations are reviewed.
* Findings and limitations are documented.
* The report and script are saved in the project.
* The work is committed to version control.
