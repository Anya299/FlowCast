# FlowCast — Day 8 Leakage Control Report

## Prediction target

`traffic_volume`

## Included baseline features

- `traffic_lag_30m`
- `traffic_lag_1h`
- `traffic_lag_90m`
- `traffic_lag_24h`
- `traffic_lag_7d`
- `traffic_rolling_mean_3h`
- `traffic_rolling_mean_6h`
- `traffic_rolling_mean_24h`
- `traffic_rolling_std_24h`
- `hour_of_day`
- `minute_of_hour`
- `day_of_week_number`
- `day_of_month`
- `month`
- `is_weekend`
- `hour_sin`
- `hour_cos`
- `day_of_week_sin`
- `day_of_week_cos`
- `month_sin`
- `month_cos`
- `road_id_feature`
- `latitude`
- `longitude`
- `road_capacity`
- `public_holiday`
- `event_flag`
- `roadwork_flag`

## Excluded fields

- `traffic_volume`
- `vehicle_count`
- `avg_speed`
- `occupancy`
- `travel_time`
- `accident_count`
- `congestion_level`
- `temperature`
- `rainfall`
- `visibility`
- `timestamp`
- `date`
- `time`

## Leakage safeguards

- Historical lag features are matched using the same road and an exact earlier timestamp.
- Missing timestamps are not silently replaced by the previous available row.
- Rolling statistics exclude the current timestamp.
- The current traffic target is never included as a feature.
- Contemporaneous speed, occupancy, travel time, and vehicle counts are excluded from the baseline.
- Observed weather is excluded from the baseline until future-weather availability is established.
- Missing lag values are preserved instead of being filled with future information.

## Remaining considerations

A feature can still leak information if it is unavailable at the actual prediction cutoff or was computed from future records. The training and validation pipeline must use chronological splits.

The event and roadwork indicators must be checked against their real operational availability before deployment.

The seven-day lag is useful only when sufficient historical observations exist for that road and timestamp.
