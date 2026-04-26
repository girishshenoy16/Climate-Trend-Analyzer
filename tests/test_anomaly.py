import os
import pandas as pd


def test_anomaly_file_exists():

    path = "outputs/climate_anomalies.csv"

    assert os.path.exists(path)


def test_anomaly_not_empty():

    df = pd.read_csv(
        "outputs/climate_anomalies.csv"
    )

    assert len(df) > 0

def test_anomaly_columns():

    df = pd.read_csv(
        "outputs/climate_anomalies.csv"
    )

    required = [
        "date",
        "meantemp",
        "humidity",
        "meanpressure"
    ]

    for col in required:
        assert col in df.columns


def test_anomaly_rows_reasonable():

    df = pd.read_csv(
        "outputs/climate_anomalies.csv"
    )

    assert len(df) > 0
    assert df["date"].nunique() == len(df)


def test_temperature_values_valid():

    df = pd.read_csv(
        "outputs/climate_anomalies.csv"
    )

    assert df["meantemp"].between(
        -10, 60
    ).all()