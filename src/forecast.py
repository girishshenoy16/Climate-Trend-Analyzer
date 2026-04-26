import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    PROCESSED_PATH,
    MODEL_PATH,
    OUTPUT_PATH
)


# ======================================================
# START LOG
# ======================================================

print("=" * 75)
print("CLIMATE TREND ANALYZER - FUTURE TEMPERATURE FORECAST")
print("=" * 75)


# ======================================================
# FILE PATHS
# ======================================================

DATA_FILE = PROCESSED_PATH + "featured_climate_data.csv"
METRICS_FILE = OUTPUT_PATH + "metrics.json"

FORECAST_FILE = OUTPUT_PATH + "future_forecast.csv"
SUMMARY_FILE = OUTPUT_PATH + "forecast_summary.json"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ======================================================
# LOAD BEST MODEL
# ======================================================

with open(METRICS_FILE, "r") as f:
    meta = json.load(f)

model_file = meta["Model_File"]

print("Loading Model:", model_file)

model = joblib.load(MODEL_PATH + model_file)


# ======================================================
# LOAD FEATURE DATA
# ======================================================

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print("Dataset Shape:", df.shape)


# ======================================================
# SETTINGS
# ======================================================

FORECAST_DAYS = 30

history_temp = df["meantemp"].tolist()

all_predictions = []


# ======================================================
# FUTURE LOOP
# ======================================================

last_row = df.iloc[-1].copy()
last_date = pd.to_datetime(last_row["date"])

print("Last Available Date:", last_date.date())
print("Forecasting Next", FORECAST_DAYS, "Days...")


for i in range(1, FORECAST_DAYS + 1):

    future_date = last_date + pd.Timedelta(days=i)

    row = {}

    # --------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------

    row["humidity"] = df["humidity"].tail(7).mean()
    row["wind_speed"] = df["wind_speed"].tail(7).mean()
    row["meanpressure"] = df["meanpressure"].tail(7).mean()

    row["year"] = future_date.year
    row["month"] = future_date.month
    row["day"] = future_date.day
    row["weekday"] = future_date.weekday()

    row["weekofyear"] = int(
        future_date.isocalendar()[1]
    )

    row["quarter"] = future_date.quarter

    row["is_month_start"] = int(
        future_date.is_month_start
    )

    row["is_month_end"] = int(
        future_date.is_month_end
    )

    row["is_weekend"] = int(
        future_date.weekday() >= 5
    )

    # --------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------

    row["temp_lag_1"] = history_temp[-1]
    row["temp_lag_7"] = np.mean(history_temp[-7:])
    row["temp_lag_30"] = np.mean(history_temp[-30:])

    row["humidity_lag_1"] = row["humidity"]
    row["wind_lag_1"] = row["wind_speed"]
    row["pressure_lag_1"] = row["meanpressure"]

    # --------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------

    row["temp_roll_mean_7"] = np.mean(
        history_temp[-7:]
    )

    row["temp_roll_mean_30"] = np.mean(
        history_temp[-30:]
    )

    row["temp_roll_std_7"] = np.std(
        history_temp[-7:]
    )

    row["humidity_roll_mean_7"] = row["humidity"]
    row["wind_roll_mean_7"] = row["wind_speed"]
    row["pressure_roll_mean_7"] = row["meanpressure"]

    # --------------------------------------------------
    # CYCLICAL FEATURES
    # --------------------------------------------------

    row["month_sin"] = np.sin(
        2 * np.pi * row["month"] / 12
    )

    row["month_cos"] = np.cos(
        2 * np.pi * row["month"] / 12
    )

    row["weekday_sin"] = np.sin(
        2 * np.pi * row["weekday"] / 7
    )

    row["weekday_cos"] = np.cos(
        2 * np.pi * row["weekday"] / 7
    )

    # --------------------------------------------------
    # INTERACTION FEATURES
    # --------------------------------------------------

    row["humidity_x_wind"] = (
        row["humidity"] *
        row["wind_speed"]
    )

    row["humidity_x_pressure"] = (
        row["humidity"] *
        row["meanpressure"]
    )

    row["wind_x_pressure"] = (
        row["wind_speed"] *
        row["meanpressure"]
    )

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    X_future = pd.DataFrame([row])

    X_future = X_future.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    pred_temp = model.predict(X_future)[0]

    pred_temp = round(float(pred_temp), 2)

    # Save history for recursive forecast
    history_temp.append(pred_temp)

    # Save output
    all_predictions.append({
        "date": future_date.strftime("%Y-%m-%d"),
        "predicted_meantemp": pred_temp
    })


# ======================================================
# SAVE FORECAST CSV
# ======================================================

forecast_df = pd.DataFrame(all_predictions)

forecast_df.to_csv(
    FORECAST_FILE,
    index=False
)

print("\nSaved:", FORECAST_FILE)


# ======================================================
# PLOT FORECAST
# ======================================================

plt.figure(figsize=(12, 6))

plt.plot(
    forecast_df["date"],
    forecast_df["predicted_meantemp"],
    marker="o",
    linewidth=2
)

plt.title("30 Day Temperature Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Mean Temp")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH + "future_forecast.png"
)

print("Saved: future_forecast.png")


# ======================================================
# SAVE SUMMARY JSON
# ======================================================

summary = {
    "Forecast_Days": FORECAST_DAYS,
    "Average_Predicted_Temp": float(
        forecast_df["predicted_meantemp"].mean()
    ),
    "Max_Predicted_Temp": float(
        forecast_df["predicted_meantemp"].max()
    ),
    "Min_Predicted_Temp": float(
        forecast_df["predicted_meantemp"].min()
    )
}

with open(SUMMARY_FILE, "w") as f:
    json.dump(
        summary,
        f,
        indent=4
    )

print("Saved: forecast_summary.json")


# ======================================================
# PREVIEW
# ======================================================

print("\nForecast Preview:")
print(forecast_df.head(10))


# ======================================================
# COMPLETE
# ======================================================

print("=" * 75)
print("FORECAST COMPLETE")
print("=" * 75)