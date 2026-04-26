import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    PROCESSED_PATH,
    OUTPUT_PATH
)


# ======================================================
# START LOG
# ======================================================

print("=" * 75)
print("CLIMATE TREND ANALYZER - ANOMALY DETECTION")
print("=" * 75)


# ======================================================
# FILE PATHS
# ======================================================

DATA_FILE = PROCESSED_PATH + "clean_climate_data.csv"

ANOMALY_FILE = OUTPUT_PATH + "climate_anomalies.csv"
SUMMARY_FILE = OUTPUT_PATH + "anomaly_summary.json"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print("Dataset Shape:", df.shape)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ======================================================
# BASIC PROFILE
# ======================================================

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("Duplicate Rows:", df.duplicated().sum())


# ======================================================
# Z-SCORE FUNCTION
# ======================================================

def z_score(series):

    mean_val = series.mean()
    std_val = series.std()

    if std_val == 0:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (series - mean_val) / std_val


# ======================================================
# TEMPERATURE ANOMALIES
# ======================================================

df["temp_zscore"] = z_score(
    df["meantemp"]
)

df["temp_anomaly"] = np.where(
    abs(df["temp_zscore"]) >= 2.5,
    1,
    0
)


# ======================================================
# HUMIDITY ANOMALIES
# ======================================================

df["humidity_zscore"] = z_score(
    df["humidity"]
)

df["humidity_anomaly"] = np.where(
    abs(df["humidity_zscore"]) >= 2.5,
    1,
    0
)


# ======================================================
# PRESSURE ANOMALIES
# ======================================================

df["pressure_zscore"] = z_score(
    df["meanpressure"]
)

df["pressure_anomaly"] = np.where(
    abs(df["pressure_zscore"]) >= 2.5,
    1,
    0
)


# ======================================================
# HEATWAVE / COLDWAVE
# ======================================================

temp_95 = df["meantemp"].quantile(0.95)
temp_05 = df["meantemp"].quantile(0.05)

df["heatwave"] = np.where(
    df["meantemp"] >= temp_95,
    1,
    0
)

df["coldwave"] = np.where(
    df["meantemp"] <= temp_05,
    1,
    0
)


# ======================================================
# FINAL ANOMALY FLAG
# ======================================================

df["is_anomaly"] = np.where(
    (
        (df["temp_anomaly"] == 1) |
        (df["humidity_anomaly"] == 1) |
        (df["pressure_anomaly"] == 1) |
        (df["heatwave"] == 1) |
        (df["coldwave"] == 1)
    ),
    1,
    0
)


# ======================================================
# EXTRACT ANOMALIES
# ======================================================

anomaly_df = df[
    df["is_anomaly"] == 1
].copy()

print("\nTotal Anomaly Days:", len(anomaly_df))


# ======================================================
# SAVE CSV
# ======================================================

save_cols = [
    "date",
    "meantemp",
    "humidity",
    "wind_speed",
    "meanpressure",
    "temp_anomaly",
    "humidity_anomaly",
    "pressure_anomaly",
    "heatwave",
    "coldwave"
]

anomaly_df[save_cols].to_csv(
    ANOMALY_FILE,
    index=False
)

print("Saved: climate_anomalies.csv")


# ======================================================
# PLOT 1 - TEMP SERIES WITH ANOMALIES
# ======================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["date"],
    df["meantemp"],
    label="Temperature",
    linewidth=1.8
)

plt.scatter(
    anomaly_df["date"],
    anomaly_df["meantemp"],
    s=30,
    alpha=0.8,
    label="Anomaly"
)

plt.title("Temperature Trend with Anomalies")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH + "temperature_anomalies.png"
)

print("Saved: temperature_anomalies.png")


# ======================================================
# PLOT 2 - HUMIDITY
# ======================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["date"],
    df["humidity"],
    linewidth=1.5
)

plt.title("Humidity Trend")
plt.xlabel("Date")
plt.ylabel("Humidity")
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH + "humidity_trend.png"
)

print("Saved: humidity_trend.png")


# ======================================================
# PLOT 3 - PRESSURE
# ======================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["date"],
    df["meanpressure"],
    linewidth=1.5
)

plt.title("Pressure Trend")
plt.xlabel("Date")
plt.ylabel("Mean Pressure")
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH + "pressure_trend.png"
)

print("Saved: pressure_trend.png")


# ======================================================
# SUMMARY JSON
# ======================================================

summary = {
    "Total_Days": int(len(df)),
    "Anomaly_Days": int(len(anomaly_df)),
    "Heatwave_Days": int(df["heatwave"].sum()),
    "Coldwave_Days": int(df["coldwave"].sum()),
    "Temperature_Anomalies": int(df["temp_anomaly"].sum()),
    "Humidity_Anomalies": int(df["humidity_anomaly"].sum()),
    "Pressure_Anomalies": int(df["pressure_anomaly"].sum())
}

with open(SUMMARY_FILE, "w") as f:
    json.dump(
        summary,
        f,
        indent=4
    )

print("Saved: anomaly_summary.json")


# ======================================================
# PREVIEW
# ======================================================

print("\nTop 10 Anomaly Dates:")
print(
    anomaly_df[
        ["date", "meantemp", "heatwave", "coldwave"]
    ].head(10)
)


# ======================================================
# COMPLETE
# ======================================================

print("=" * 75)
print("ANOMALY DETECTION COMPLETE")
print("=" * 75)