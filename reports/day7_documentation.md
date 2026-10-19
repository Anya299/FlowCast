\# FlowCast — Day 7 Engineering Documentation



\## 1. Objective



The objective of Day 7 was to implement a reproducible data-cleaning pipeline for the FlowCast traffic prediction system. The pipeline applies explicit validation rules, standardizes selected categorical values, identifies duplicate road-time records, and records data-quality issues for further investigation.



\## 2. Input Dataset



The pipeline reads the integrated dataset produced by the earlier FlowCast data-engineering stages.



\* Input: `data/interim/flowcast\_integrated.csv`

\* Data grain: road segment and 30-minute time window

\* Expected road segments: 25

\* Coverage: January 1 through May 31, 2025



The source dataset is preserved without direct modification.



\## 3. Data Quality Controls



The pipeline implements the following controls:



\* Timestamp parsing and invalid timestamp detection.

\* Duplicate detection using the road identifier and timestamp.

\* Numeric conversion for selected measurement columns.

\* Validation of traffic volume, speed, occupancy, and road capacity.

\* Detection of negative counts and physically invalid negative measurements.

\* Standardization of known congestion and weather labels.

\* Derivation of missing congestion labels from the volume-to-capacity ratio when the required inputs are valid.



\## 4. Congestion Derivation



The volume-to-capacity ratio is calculated as:



V/C = traffic volume / (road capacity / 2)



The denominator uses half of the hourly road capacity because the traffic records represent 30-minute windows.



The configured congestion categories are:



\* V/C below 0.50: Free-flow

\* V/C from 0.50 to below 0.80: Moderate

\* V/C from 0.80 to below 1.00: Heavy

\* V/C of 1.00 or higher: Severe



These categories follow the project's data dictionary.



\## 5. Auditability



The pipeline generates a cleaned dataset, a machine-readable quality summary, a Markdown report, a duplicate-record file, and a file containing records that require measurement review.



Rows with missing target values are retained. The pipeline does not silently replace missing traffic targets with zero or automatically impute missing sensor measurements.



\## 6. Output Artifacts



\* `data/processed/flowcast\_cleaned.csv`

\* `reports/day7/cleaning\_summary.json`

\* `reports/day7/cleaning\_report.md`

\* `reports/day7/rows\_requiring\_review.csv`

\* `reports/day7/duplicate\_records.csv`



\## 7. Engineering Considerations



Data cleaning decisions can affect model performance and reliability. Therefore, validation rules are explicit, problematic records remain traceable, and the original integrated dataset is preserved.



The pipeline's output must be inspected before model development. Missing values, remaining invalid timestamps, target availability, and the number of derived congestion labels must be checked.



This stage does not establish that every sensor reading is correct. It creates a controlled dataset and an auditable record of issues identified by the implemented rules.



\## 8. Next Steps



The next stage is to design model-ready features, including time encodings, historical traffic lags, rolling statistics, weather indicators, and parsed vehicle-type distributions.



Feature engineering and subsequent model evaluation must respect the chronological structure of the data to prevent information leakage.



