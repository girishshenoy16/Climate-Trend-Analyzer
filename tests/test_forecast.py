import os
import pandas as pd


def test_forecast_file_exists():

    path = "outputs/future_forecast.csv"

    assert os.path.exists(path)


def test_forecast_not_empty():

    df = pd.read_csv(
        "outputs/future_forecast.csv"
    )

    assert len(df) > 0


def test_forecast_columns():

    df = pd.read_csv(
        "outputs/future_forecast.csv"
    )

    required = [
        "date",
        "predicted_meantemp"
    ]

    for col in required:
        assert col in df.columns

    assert pd.to_datetime(df.date).is_monotonic_increasing


def test_forecast_30_days():

    df = pd.read_csv(
        "outputs/future_forecast.csv"
    )

    assert len(df) == 30


def test_forecast_values_reasonable():

    df = pd.read_csv(
        "outputs/future_forecast.csv"
    )

    assert df["predicted_meantemp"].between(
        -10, 60
    ).all()


def test_forecast_dates_continuous():

    df = pd.read_csv(
        "outputs/future_forecast.csv"
    )

    dates = pd.to_datetime(df["date"])

    diff = dates.diff().dropna().dt.days

    assert (diff == 1).all()