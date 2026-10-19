\# FlowCast — Day 4 Data Contract Report



\## 1. Objective



Implement a reusable data validation layer for the integrated

FlowCast traffic dataset.



\## 2. Input



`data/interim/flowcast\_integrated.csv`



\## 3. Implementation



Implemented a reusable validation module that checks:



\- Expected dataset schema

\- Required road and timestamp identifiers

\- Date and time parsing

\- Numeric measurement types

\- Negative measurement values

\- Speed and occupancy screening ranges

\- Binary indicator values

\- Hour ranges

\- Duplicate road-time keys



\## 4. Outputs



\- `data/processed/flowcast\_validated.csv`

\- `data/processed/quarantined\_records.csv`

\- `reports/day4/validation\_summary.json`

\- `reports/day4/validation\_issues.csv`

\- `reports/day4/missingness\_report.csv`

\- `reports/day4/duplicate\_records.csv`



\## 5. Data preservation



The validated output preserves the original number of input rows.

Records with blocking errors are identified in a separate quarantine

output. Suspicious records are flagged for review rather than

automatically deleted.



\## 6. Automated tests



The test suite covers:



\- Preservation of valid rows

\- Quarantine of negative traffic counts

\- Review of suspicious speed values

\- Duplicate road-time detection

\- Reporting of missing optional measurements



Record the actual test results after running pytest.



\## 7. Evidence



Record the actual input row count, validation findings, quarantined

row count, duplicate count, and test results here.



Do not insert invented metrics.



\## 8. Limitations



The current module validates the integrated dataset. It does not

replace the full raw-to-processed ingestion and cleaning pipeline.



\## 9. Next steps



\- Strengthen the raw-to-processed pipeline.

\- Validate source joins and temporal alignment.

\- Implement documented congestion-label derivation.

\- Add temporal completeness and leakage checks.

\- Build model-ready features after validation.

