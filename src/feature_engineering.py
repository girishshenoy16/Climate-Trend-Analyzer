# src/feature_engineering.py

import pandas as pd
import numpy as np

from config import PROCESSED_PATH


# ======================================================
# START LOG
# ======================================================

print("=" * 60)
print("FEATURE ENGINEERING STARTED (LEAKAGE FREE V3)")
print("=" * 60)


# ======================================================
# FILE PATHS
# ======================================================

INPUT_FILE = PROCESSED_PATH + "clean_climate_data.csv"
OUTPUT_FILE = PROCESSED_PATH + "featured_climate_data.csv"


# ======================================================
# LOAD CLEAN DATA
# ======================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# ======================================================
# INITIAL CHECK
# ======================================================

print("Shape:", df.shape)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values Per Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# ======================================================
# DATE FEATURES
# ======================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

df["weekday"] = df["date"].dt.weekday

df["weekofyear"] = (
    df["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["quarter"] = df["date"].dt.quarter

df["is_month_start"] = (
    df["date"]
    .dt.is_month_start
    .astype(int)
)

df["is_month_end"] = (
    df["date"]
    .dt.is_month_end
    .astype(int)
)

df["is_weekend"] = df["weekday"].isin(
    [5, 6]
).astype(int)

print("\n✅ Date Features Created")


# ======================================================
# LAG FEATURES (SAFE)
# ======================================================

df["temp_lag_1"] = df["meantemp"].shift(1)
df["temp_lag_7"] = df["meantemp"].shift(7)
df["temp_lag_30"] = df["meantemp"].shift(30)

df["humidity_lag_1"] = df["humidity"].shift(1)
df["wind_lag_1"] = df["wind_speed"].shift(1)
df["pressure_lag_1"] = df["meanpressure"].shift(1)

print("✅ Lag Features Created")


# ======================================================
# ROLLING FEATURES (SHIFTED = SAFE)
# ======================================================

df["temp_roll_mean_7"] = (
    df["meantemp"]
    .shift(1)
    .rolling(7)
    .mean()
)

df["temp_roll_mean_30"] = (
    df["meantemp"]
    .shift(1)
    .rolling(30)
    .mean()
)

df["temp_roll_std_7"] = (
    df["meantemp"]
    .shift(1)
    .rolling(7)
    .std()
)

df["humidity_roll_mean_7"] = (
    df["humidity"]
    .shift(1)
    .rolling(7)
    .mean()
)

df["wind_roll_mean_7"] = (
    df["wind_speed"]
    .shift(1)
    .rolling(7)
    .mean()
)

df["pressure_roll_mean_7"] = (
    df["meanpressure"]
    .shift(1)
    .rolling(7)
    .mean()
)

print("✅ Rolling Features Created")


# ======================================================
# CYCLICAL FEATURES
# ======================================================

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

df["weekday_sin"] = np.sin(
    2 * np.pi * df["weekday"] / 7
)

df["weekday_cos"] = np.cos(
    2 * np.pi * df["weekday"] / 7
)

print("✅ Cyclical Features Created")


# ======================================================
# SAFE WEATHER INTERACTION FEATURES
# (NO meantemp used)
# ======================================================

df["humidity_x_wind"] = (
    df["humidity"] * df["wind_speed"]
)

df["humidity_x_pressure"] = (
    df["humidity"] * df["meanpressure"]
)

df["wind_x_pressure"] = (
    df["wind_speed"] * df["meanpressure"]
)

print("✅ Leakage-Free Interaction Features Created")


# ======================================================
# HANDLE NaN FROM SHIFTS / ROLLING
# ======================================================

feature_cols = [
    "temp_lag_1",
    "temp_lag_7",
    "temp_lag_30",
    "humidity_lag_1",
    "wind_lag_1",
    "pressure_lag_1",
    "temp_roll_mean_7",
    "temp_roll_mean_30",
    "temp_roll_std_7",
    "humidity_roll_mean_7",
    "wind_roll_mean_7",
    "pressure_roll_mean_7"
]

for col in feature_cols:
    df[col] = df[col].fillna(0)

print("✅ Lag/Rolling Missing Values Handled")


# ======================================================
# FINAL SAFETY CHECK
# ======================================================

bool_cols = df.select_dtypes(
    include="bool"
).columns

for col in bool_cols:
    df[col] = df[col].astype(int)

object_cols = df.select_dtypes(
    include="object"
).columns

if len(object_cols) > 0:
    print("\nWARNING: Object Columns Still Present")
    print(object_cols.tolist())
else:
    print("\n✅ No Object Columns Remaining")


# ======================================================
# FINAL REPORT
# ======================================================

print("\n" + "=" * 60)
print("FINAL FEATURE DATASET CHECK")
print("=" * 60)

print("Shape:", df.shape)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values Per Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# ======================================================
# SAVE FILE
# ======================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n✅ Feature Engineering Complete")
print("Saved:", OUTPUT_FILE)