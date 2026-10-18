# FlowCast — Traffic Quality-Check Report

## Dataset
- Input rows: 176,701
- Output rows: 176,701
- Rows removed: 0

## Rules applied
- Converted selected measurement columns to numeric values.
- Replaced negative traffic-volume values with missing values.
- Added a flag for speed values above 200.
- Added a flag for occupancy values above 100.
- Preserved all rows.
- Preserved suspicious speed and occupancy values for investigation.
- Did not impute missing values.

## Counts recorded before correction
- Negative traffic-volume values: 239
- Speed values above 200: 236
- Occupancy values above 100: 229

## Missing values after quality checks
- traffic_volume: 4,587\n- avg_speed: 4,343\n- occupancy: 4,344\n- travel_time: 0\n- accident_count: 0\n
## Limitations
This is an initial quality-check stage, not a final modelling dataset.
Outlier flags require further investigation. Missing-value treatment,
feature engineering, temporal alignment, and train/validation/test
splitting remain outstanding.
