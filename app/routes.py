import os
import json
import pandas as pd

from flask import render_template, send_from_directory


def register_routes(app):

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    OUTPUT_PATH = os.path.join(BASE_DIR, "outputs")

    # ======================================
    # Serve output files (charts/csv/json)
    # ======================================
    @app.route("/outputs/<path:filename>")
    def output_files(filename):
        return send_from_directory(
            OUTPUT_PATH,
            filename
        )

    # ======================================
    # HOME
    # ======================================
    @app.route("/")
    def home():

        metrics = {
            "Best_Model": "LinearRegression",
            "RMSE": 0,
            "R2": 0,
            "MAPE": 0
        }

        file_path = os.path.join(
            OUTPUT_PATH,
            "metrics.json"
        )

        if os.path.exists(file_path):

            with open(file_path, "r") as f:
                data = json.load(f)

            metrics["Best_Model"] = data.get(
                "Best_Model",
                data.get("Model", "LinearRegression")
            )

            metrics["RMSE"] = round(
                float(data.get("RMSE", 0)), 2
            )

            metrics["R2"] = round(
                float(data.get("R2", 0)), 2
            )

            metrics["MAPE"] = round(
                float(data.get("MAPE", 0)), 2
            )

        return render_template(
            "index.html",
            metrics=metrics
        )

    # ======================================
    # DASHBOARD
    # ======================================
    @app.route("/dashboard")
    def dashboard():

        file_path = os.path.join(
            OUTPUT_PATH,
            "leaderboard.csv"
        )

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            leaderboard = df.to_dict(
                orient="records"
            )
        else:
            leaderboard = []

        return render_template(
            "dashboard.html",
            leaderboard=leaderboard
        )

    # ======================================
    # FORECAST
    # ======================================
    @app.route("/forecast")
    def forecast():

        file_path = os.path.join(
            OUTPUT_PATH,
            "future_forecast.csv"
        )

        if os.path.exists(file_path):

            df = pd.read_csv(file_path)

            df["predicted_meantemp"] = (
                df["predicted_meantemp"]
                .astype(float)
                .round(2)
            )

            avg_temp = round(
                df["predicted_meantemp"].mean(),
                2
            )

            forecast_data = df.to_dict(
                orient="records"
            )

        else:
            forecast_data = []
            avg_temp = 0

        return render_template(
            "forecast.html",
            forecast=forecast_data,
            avg_temp=avg_temp
        )

    # ======================================
    # ANOMALY
    # ======================================
    @app.route("/anomaly")
    def anomaly():

        file_path = os.path.join(
            OUTPUT_PATH,
            "climate_anomalies.csv"
        )

        if os.path.exists(file_path):

            df = pd.read_csv(file_path)

            keep_cols = [
                "date",
                "meantemp",
                "humidity",
                "meanpressure"
            ]

            df = df[keep_cols].copy()

            df["meantemp"] = df["meantemp"].round(2)
            df["humidity"] = df["humidity"].round(2)
            df["meanpressure"] = df["meanpressure"].round(2)

            df["severity"] = (
                abs(df["meantemp"]) +
                abs(df["humidity"]) / 10 +
                abs(df["meanpressure"]) / 100
            ).round(2)

            df = df.sort_values(
                by="severity",
                ascending=False
            ).head(100)

            anomaly_data = df.to_dict(
                orient="records"
            )

        else:
            anomaly_data = []

        return render_template(
            "anomaly.html",
            anomalies=anomaly_data
        )