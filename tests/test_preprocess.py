import os
import pandas as pd


def test_clean_file_exists():

    path = "data/processed/clean_climate_data.csv"

    assert os.path.exists(path)


def test_clean_file_not_empty():

    df = pd.read_csv(
        "data/processed/clean_climate_data.csv"
    )

    assert len(df) > 0


def test_required_columns_present():

    df = pd.read_csv(
        "data/processed/clean_climate_data.csv"
    )

    required_cols = [
        "date",
        "meantemp",
        "humidity",
        "wind_speed",
        "meanpressure"
    ]

    for col in required_cols:
        assert col in df.columns

    assert df["date"].is_monotonic_increasing


def test_no_missing_values():

    df = pd.read_csv(
        "data/processed/clean_climate_data.csv"
    )

    assert df.isnull().sum().sum() == 0


def test_no_duplicates():

    df = pd.read_csv(
        "data/processed/clean_climate_data.csv"
    )

    assert df.duplicated().sum() == 0