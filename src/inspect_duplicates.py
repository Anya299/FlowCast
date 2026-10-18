from pathlib import Path
import pandas as pd

path = Path("data/raw/traffic_sensor_log.csv")
df = pd.read_csv(path, low_memory=False)

keys = ["road_id", "date", "time"]
duplicates = df[df.duplicated(subset=keys, keep=False)].copy()
duplicates = duplicates.sort_values(keys)

print("Total traffic rows:", len(df))
print("Rows involved in duplicate keys:", len(duplicates))
print("Duplicate key groups:", duplicates.groupby(keys).ngroups)

print("\nFirst 15 duplicate records:")
print(duplicates.head(15).to_string(index=False))

print("\nConflicting duplicate groups:")
conflicts = (
    duplicates.groupby(keys)
    .nunique(dropna=False)
    .gt(1)
    .any(axis=1)
)
print("Groups with conflicting values:", int(conflicts.sum()))
print("Groups with identical values:", int((~conflicts).sum()))

out = Path("reports/duplicate_traffic_investigation.csv")
duplicates.to_csv(out, index=False)
print(f"\nInvestigation saved to: {out}")
