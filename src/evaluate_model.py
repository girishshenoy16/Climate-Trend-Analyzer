import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import (
    PROCESSED_PATH,
    MODEL_PATH,
    OUTPUT_PATH
)


# ======================================================
# START LOG
# ======================================================

print("=" * 75)
print("CLIMATE TREND ANALYZER - MODEL EVALUATION")
print("=" * 75)


# ======================================================
# FILE PATHS
# ======================================================

DATA_FILE = PROCESSED_PATH + "featured_climate_data.csv"
METRICS_FILE = OUTPUT_PATH + "metrics.json"
SUMMARY_FILE = OUTPUT_PATH + "evaluation_summary.json"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ======================================================
# LOAD BEST MODEL INFO
# ======================================================

with open(METRICS_FILE, "r") as f:
    meta = json.load(f)

model_file = meta["Model_File"]

print("Loading Model:", model_file)

model = joblib.load(MODEL_PATH + model_file)


# ======================================================
# LOAD DATASET
# ======================================================

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print("\nFeature Dataset")
print("=" * 60)

print("Shape:", df.shape)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())


# ======================================================
# PREPARE FEATURES / TARGET
# ======================================================

TARGET = "meantemp"

X = df.drop(
    columns=[
        "meantemp",
        "date"
    ]
)

y = df[TARGET]


# ======================================================
# SAME TIME SPLIT AS TRAINING
# ======================================================

split_date = df["date"].quantile(0.80)

X_valid = X[df["date"] > split_date]
y_valid = y[df["date"] > split_date]
date_valid = df["date"][df["date"] > split_date]

print("\nValidation Shape:", X_valid.shape)


# ======================================================
# PREDICT
# ======================================================

pred = model.predict(X_valid)


# ======================================================
# SAFE MAPE
# ======================================================

def mape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    return np.mean(
        np.abs(
            (y_true[mask] - y_pred[mask]) /
            y_true[mask]
        )
    ) * 100


# ======================================================
# METRICS
# ======================================================

mae = mean_absolute_error(y_valid, pred)

rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        pred
    )
)

mape_val = mape(
    y_valid,
    pred
)

r2 = r2_score(
    y_valid,
    pred
)

print("\nFINAL VALIDATION METRICS")
print("-" * 45)

print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("MAPE:", round(mape_val, 3))
print("R2  :", round(r2, 4))


# ======================================================
# ACTUAL VS PREDICTED PLOT
# ======================================================

sample_n = min(200, len(y_valid))

plt.figure(figsize=(12, 6))

plt.plot(
    date_valid.iloc[:sample_n],
    y_valid.values[:sample_n],
    label="Actual",
    linewidth=2
)

plt.plot(
    date_valid.iloc[:sample_n],
    pred[:sample_n],
    label="Predicted",
    linewidth=2
)

plt.title("Actual vs Predicted Temperature")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH + "actual_vs_predicted_temp.png"
)

print("Saved: actual_vs_predicted_temp.png")


# ======================================================
# RESIDUAL PLOT
# ======================================================

residuals = y_valid.values - pred

plt.figure(figsize=(10, 6))

plt.scatter(
    pred,
    residuals,
    alpha=0.4
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Residual Plot")
plt.xlabel("Predicted Temperature")
plt.ylabel("Residual Error")
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH + "residual_plot.png"
)

print("Saved: residual_plot.png")


# ======================================================
# ERROR DISTRIBUTION
# ======================================================

plt.figure(figsize=(10, 6))

plt.hist(
    residuals,
    bins=40
)

plt.title("Residual Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH + "error_distribution.png"
)

print("Saved: error_distribution.png")


# ======================================================
# FEATURE IMPORTANCE
# ======================================================

if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False
    ).head(20)

    plt.figure(figsize=(10, 7))

    plt.barh(
        importance["feature"],
        importance["importance"]
    )

    plt.gca().invert_yaxis()

    plt.title("Top 20 Feature Importance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH + "feature_importance.png"
    )

    print("Saved: feature_importance.png")


# ======================================================
# SAVE SUMMARY JSON
# ======================================================

summary = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "MAPE": float(mape_val),
    "R2": float(r2)
}

with open(SUMMARY_FILE, "w") as f:
    json.dump(
        summary,
        f,
        indent=4
    )

print("Saved: evaluation_summary.json")


# ======================================================
# COMPLETE
# ======================================================

print("=" * 75)
print("EVALUATION COMPLETE")
print("=" * 75)