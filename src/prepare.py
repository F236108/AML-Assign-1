"""
Task 1 – Load and prepare the data (Plant 1 only).

Steps
-----
1.2  Convert DATE_TIME to proper datetime (formats differ between files).
1.3  Sum AC_POWER and DC_POWER over all inverters → plant-level power.
1.4  Merge plant-level power with sensor file on the timestamp.
1.5  Resample to hourly means → ~816 rows.
1.6  Report statistics.

Saves  data/plant1_hourly.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from load_data import load_raw


def prepare():
    raw = load_raw()

    gen1 = raw["gen1"].copy()
    sensor1 = raw["sensor1"].copy()

    # ------------------------------------------------------------------ 1.2
    # Inspect raw DATE_TIME formats
    print("=== Step 1.2: Convert DATE_TIME ===")
    print(f"  gen1 DATE_TIME sample:    '{gen1['DATE_TIME'].iloc[0]}'")
    print(f"  sensor1 DATE_TIME sample: '{sensor1['DATE_TIME'].iloc[0]}'")

    # gen1 format: '15-05-2020 00:00'  → dd-mm-yyyy HH:MM
    gen1["DATE_TIME"] = pd.to_datetime(gen1["DATE_TIME"],
                                       format="%d-%m-%Y %H:%M")

    # sensor1 format: '2020-05-15 00:00:00' → yyyy-mm-dd HH:MM:SS
    sensor1["DATE_TIME"] = pd.to_datetime(sensor1["DATE_TIME"],
                                          format="%Y-%m-%d %H:%M:%S")

    print(f"  gen1   first={gen1['DATE_TIME'].min()}, last={gen1['DATE_TIME'].max()}")
    print(f"  sensor1 first={sensor1['DATE_TIME'].min()}, last={sensor1['DATE_TIME'].max()}")

    # ------------------------------------------------------------------ 1.3
    # Sum AC_POWER and DC_POWER over all inverters at each timestamp
    print("\n=== Step 1.3: Plant-level power ===")
    plant_power = (gen1.groupby("DATE_TIME")[["AC_POWER", "DC_POWER"]]
                       .sum()
                       .reset_index())
    print(f"  Plant-level power rows: {len(plant_power)}")

    # ------------------------------------------------------------------ 1.4
    # Merge plant-level power with sensor file on timestamp
    print("\n=== Step 1.4: Merge power + sensor ===")

    # Sensor data: take mean across sensors (there's typically one)
    sensor_agg = (sensor1.groupby("DATE_TIME")[["AMBIENT_TEMPERATURE",
                                                 "MODULE_TEMPERATURE",
                                                 "IRRADIATION"]]
                         .mean()
                         .reset_index())

    # Check how many timestamps exist in only one file
    power_ts = set(plant_power["DATE_TIME"])
    sensor_ts = set(sensor_agg["DATE_TIME"])
    only_power = power_ts - sensor_ts
    only_sensor = sensor_ts - power_ts
    print(f"  Timestamps in power only : {len(only_power)}")
    print(f"  Timestamps in sensor only: {len(only_sensor)}")
    print(f"  Timestamps in both       : {len(power_ts & sensor_ts)}")

    merged = pd.merge(plant_power, sensor_agg, on="DATE_TIME", how="inner")
    print(f"  Merged rows (inner join) : {len(merged)}")

    # ------------------------------------------------------------------ 1.5
    # Resample to hourly means
    print("\n=== Step 1.5: Resample to hourly means ===")
    merged = merged.set_index("DATE_TIME")
    hourly = merged.resample("h").mean()

    # Drop rows where all values are NaN (hours with no data at all)
    hourly = hourly.dropna(how="all")

    # Rename columns to match required output
    hourly = hourly.rename(columns={
        "AC_POWER": "ac_power",
        "DC_POWER": "dc_power",
        "AMBIENT_TEMPERATURE": "ambient_temp",
        "MODULE_TEMPERATURE": "module_temp",
        "IRRADIATION": "irradiation",
    })
    hourly.index.name = "datetime"

    print(f"  Hourly rows: {len(hourly)}")

    # ------------------------------------------------------------------ 1.6
    # Report statistics
    print("\n=== Step 1.6: Report ===")
    missing = hourly.isnull().sum()
    rows_with_missing = hourly.isnull().any(axis=1).sum()
    print(f"  Number of hourly rows       : {len(hourly)}")
    print(f"  Rows with missing values    : {rows_with_missing}")
    print(f"  Missing values per column:\n{missing.to_string()}")

    # Handle missing values: forward-fill then back-fill for small gaps
    if rows_with_missing > 0:
        hourly = hourly.fillna(method="ffill").fillna(method="bfill")
        print(f"  → Handled via forward-fill then back-fill")
        remaining_missing = hourly.isnull().sum().sum()
        print(f"  Remaining missing after fill: {remaining_missing}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "plant1_hourly.csv")
    hourly.to_csv(out_path)
    print(f"\n  Saved to {out_path}")
    print(f"  Columns: {list(hourly.columns)}")
    print(f"  Shape  : {hourly.shape}")

    # Also print Table 1 data
    print("\n=== Table 1 — Data preparation ===")
    print(f"  Raw generation rows (Plant 1)   : {len(raw['gen1'])}")
    print(f"  Raw sensor rows (Plant 1)       : {len(raw['sensor1'])}")
    print(f"  Timestamps in only one file     : {len(only_power) + len(only_sensor)}")
    print(f"  Hourly rows after resampling    : {len(hourly)}")
    print(f"  Hourly rows with missing values : {rows_with_missing}")

    return hourly


if __name__ == "__main__":
    hourly = prepare()
    print("\nDone! Preview:")
    print(hourly.head(10).to_string())
