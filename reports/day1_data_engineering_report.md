# FlowCast — Day 1 Data Engineering Report

## Dataset integration
- Integrated rows: 176,701
- Unique roads: 25
- Unique dates: 151
- Duplicate road/date/time keys: 0
- Missing required columns: 0

## Important missing values
- traffic_volume: 4,587\n- avg_speed: 4,343\n- occupancy: 4,344\n- temperature: 2,708\n- rainfall: 0\n- visibility: 1,806\n- weather_condition: 0\n
## Completed work
- Loaded and structurally validated the three raw datasets.
- Investigated and removed exact duplicate traffic records.
- Audited traffic timestamps, missing observations, and numeric anomalies.
- Created a traffic quality-checked copy.
- Audited weather and calendar records.
- Standardized weather labels.
- Joined traffic to weather by station, date, and hour.
- Joined traffic to calendar by date.
- Preserved traffic rows during integration.

## Assumptions and limitations
- Each hourly weather reading represents both half-hour traffic windows
  within the same hour.
- Missing measurements have not been generally imputed.
- Suspicious speed and occupancy measurements remain flagged for review.
- Missing traffic time slots have been audited but not yet reindexed.
- No forecasting features or ML models have been built yet.
- Model evaluation must use chronological splits and leakage-safe features.
