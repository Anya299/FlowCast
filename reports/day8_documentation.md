\# FlowCast — Day 8 Engineering Documentation



\## 1. Objective



Day 8 implements a reproducible feature-engineering pipeline for traffic-volume forecasting.



The pipeline transforms cleaned traffic observations into a temporal dataset containing historical traffic measurements, rolling statistics, calendar features, and selected road-context features.



\## 2. Input



\* `data/processed/flowcast\_cleaned.csv`



The pipeline validates the required columns, parses timestamps, and checks the uniqueness of each road-time key before generating features.



\## 3. Historical Lag Features



The following historical features are generated:



\* Traffic volume 30 minutes earlier.

\* Traffic volume one hour earlier.

\* Traffic volume 90 minutes earlier.

\* Traffic volume 24 hours earlier.

\* Traffic volume seven days earlier.



Historical measurements are matched using the same road identifier and exact timestamp.



This prevents missing intervals from silently shifting a lag to the wrong time.



\## 4. Rolling Statistics



The pipeline generates:



\* Three-hour rolling mean.

\* Six-hour rolling mean.

\* Twenty-four-hour rolling mean.

\* Twenty-four-hour rolling standard deviation.



The rolling calculations exclude the current timestamp.



Missing historical observations are not automatically replaced with zero or future values.



\## 5. Temporal Features



The pipeline generates:



\* Hour of day.

\* Minute of hour.

\* Day of week.

\* Day of month.

\* Month.

\* Weekend indicator.

\* Sine and cosine encodings for hour, weekday, and month.



Cyclical encodings allow periodic relationships to be represented without treating the beginning and end of a cycle as unrelated values.



\## 6. Leakage Control



The prediction target is `traffic\_volume`.



The current target is excluded from the baseline model feature list. Contemporaneous vehicle counts, speed, occupancy, travel time, accident counts, congestion labels, and observed weather are not included in the baseline feature set.



Observed weather is excluded because its availability at the future prediction time has not yet been established.



Historical lag features and rolling statistics use earlier observations only.



These controls reduce leakage risk, but the eventual training pipeline must still use chronological validation and verify that every feature would be available at prediction time.



\## 7. Output Artifacts



\* `data/processed/flowcast\_features.csv`

\* `data/processed/flowcast\_model\_features.csv`

\* `reports/day8/feature\_catalog.csv`

\* `reports/day8/feature\_engineering\_summary.json`

\* `reports/day8/feature\_engineering\_report.md`

\* `reports/day8/leakage\_control\_report.md`



\## 8. Validation



The pipeline checks that:



\* Feature engineering preserves the input row count.

\* Road-time keys remain unique.

\* The target is not included in the model feature list.

\* The feature catalog and quality reports are generated.



These checks establish structural consistency, not predictive accuracy.



\## 9. Limitations



This stage does not train or evaluate a model.



Rows may have missing lag features when the necessary historical observations are unavailable. The handling of these rows must be decided according to the intended forecasting setup.



The feature dataset also requires a final review of prediction-time availability before it is used for model training.



\## 10. Next Stage



The next stage will focus on preparing the modelling dataset and defining a chronological training, validation, and test strategy. Model evaluation must compare against simple forecasting baselines before more complex machine-learning models are introduced.



