import os
import json
import pandas as pd
import joblib


def test_model_file_exists():

    path = "models/lr_model.pkl"

    assert os.path.exists(path)


def test_metrics_json_exists():

    path = "outputs/metrics.json"

    assert os.path.exists(path)


def test_metrics_keys_present():

    with open(
        "outputs/metrics.json",
        "r"
    ) as f:

        data = json.load(f)

    keys = [
        "Best_Model",
        "RMSE",
        "R2",
        "MAPE",
        "Model_File"
    ]

    for key in keys:
        assert key in data


def test_model_loads_successfully():

    model = joblib.load(
        "models/lr_model.pkl"
    )

    assert model is not None


def test_rmse_reasonable():

    with open(
        "outputs/metrics.json",
        "r"
    ) as f:

        data = json.load(f)

    assert float(data["RMSE"]) < 5




def test_model_prediction_shape():

    model = joblib.load(
        "models/lr_model.pkl"
    )

    df = pd.read_csv(
        "data/processed/featured_climate_data.csv"
    )

    X = df.drop(
        columns=["date", "meantemp"]
    ).head(5)

    preds = model.predict(X)

    assert len(preds) == 5


def test_model_quality_threshold():

    with open(
        "outputs/metrics.json",
        "r"
    ) as f:

        data = json.load(f)

    assert float(data["R2"]) > 0.90
    assert float(data["RMSE"]) < 3