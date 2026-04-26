# src/train_model.py

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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

print("=" * 70)
print("CLIMATE TREND ANALYZER - FINAL MODEL TRAINING")
print("=" * 70)


# ======================================================
# FILE PATHS
# ======================================================

INPUT_FILE = PROCESSED_PATH + "featured_climate_data.csv"

LEADERBOARD_FILE = OUTPUT_PATH + "leaderboard.csv"
METRICS_FILE = OUTPUT_PATH + "metrics.json"


# ======================================================
# CREATE FOLDERS
# ======================================================

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# ======================================================
# FAST MODE
# ======================================================

FAST_MODE = False

if FAST_MODE:
    print("\nFAST MODE ENABLED")
    df = df.tail(500).reset_index(drop=True)


# ======================================================
# DATA CHECK
# ======================================================

print("\nFeature Dataset")
print("=" * 60)

print("Shape:", df.shape)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())


# ======================================================
# PREPARE X / y
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
# TIME BASED SPLIT
# ======================================================

split_date = df["date"].quantile(0.80)

X_train = X[df["date"] <= split_date]
X_valid = X[df["date"] > split_date]

y_train = y[df["date"] <= split_date]
y_valid = y[df["date"] > split_date]

print("\nTrain Shape:", X_train.shape)
print("Validation Shape:", X_valid.shape)


# ======================================================
# MODELS
# ======================================================

models = {

    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}


# ======================================================
# MODEL FILE MAP
# ======================================================

model_file_map = {
    "LinearRegression": "lr_model.pkl",
    "RandomForest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}


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
# TRAIN + COMPARE
# ======================================================

results = []

best_rmse = float("inf")
best_model = None
best_name = None

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_valid)

    mae = mean_absolute_error(
        y_valid,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            pred
        )
    )

    r2 = r2_score(
        y_valid,
        pred
    )

    mape_val = mape(
        y_valid,
        pred
    )

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE": round(mape_val, 3),
        "R2": round(r2, 4)
    })

    print(f"{name} RMSE: {rmse:.3f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_name = name


# ======================================================
# LEADERBOARD
# ======================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

results_df.index = results_df.index + 1

print("\nMODEL LEADERBOARD")
print(results_df)

results_df.to_csv(
    LEADERBOARD_FILE,
    index_label="Rank"
)

print("\nSaved:", LEADERBOARD_FILE)


# ======================================================
# SAVE BEST MODEL
# ======================================================

best_file = model_file_map[best_name]

joblib.dump(
    best_model,
    MODEL_PATH + best_file
)

print(f"\n✅ Best Model : {best_name}")
print(f"✅ Saved As   : {best_file}")


# ======================================================
# SAVE METRICS JSON
# ======================================================

best_row = results_df.iloc[0]

metrics = {
    "Best_Model": best_name,
    "Model_File": best_file,
    "MAE": float(best_row["MAE"]),
    "RMSE": float(best_row["RMSE"]),
    "MAPE": float(best_row["MAPE"]),
    "R2": float(best_row["R2"])
}

with open(METRICS_FILE, "w") as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print("Saved:", METRICS_FILE)


# ======================================================
# FEATURE IMPORTANCE
# ======================================================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(20)

    importance.to_csv(
        OUTPUT_PATH + "feature_importance.csv",
        index=False
    )

    print("Saved: feature_importance.csv")


# ======================================================
# COMPLETE
# ======================================================

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)