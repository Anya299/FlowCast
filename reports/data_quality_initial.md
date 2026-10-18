# FlowCast — Initial Data Quality Report

## Traffic dataset

- File: `traffic_sensor_log.csv`
- Rows: 178,468
- Columns: 17
- Duplicate complete rows: 1,767
- Missing cells: 40,035
- Missing required columns: None

### Missing values by column

- `traffic_volume`: 4,387
- `avg_speed`: 4,382
- `occupancy`: 4,383
- `congestion_level`: 26,883

## Weather dataset

- File: `weather_observations.csv`
- Rows: 10,872
- Columns: 7
- Duplicate complete rows: 0
- Missing cells: 278
- Missing required columns: None

### Missing values by column

- `temperature`: 167
- `visibility`: 111

## Calendar dataset

- File: `calendar_events.csv`
- Rows: 151
- Columns: 6
- Duplicate complete rows: 0
- Missing cells: 290
- Missing required columns: None

### Missing values by column

- `holiday_name`: 145
- `event_name`: 145

## Initial consistency checks

- Unique road IDs: 25
- Unique traffic dates: 151
- Unique weather stations: 3
- Calendar date rows: 151
- Duplicate traffic keys (road_id + date + time): 1,767

### Important limitations

- Missing values, duplicates, and outliers have not yet been repaired.
- Date parsing and weather/calendar joins are not yet performed.
- This report describes raw data; it does not certify modelling readiness.
