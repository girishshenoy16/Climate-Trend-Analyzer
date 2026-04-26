import os
import pandas as pd


def test_feature_file_exists():

    path = "data/processed/featured_climate_data.csv"

    assert os.path.exists(path)


def test_feature_file_not_empty():

    df = pd.read_csv(
        "data/processed/featured_climate_data.csv"
    )

    assert len(df) > 0


def test_engineered_columns_exist():

    df = pd.read_csv(
        "data/processed/featured_climate_data.csv"
    )

    cols = [
        "temp_lag_1",
        "temp_lag_7",
        "temp_roll_mean_7",
        "month_sin",
        "weekday_cos"
    ]

    for col in cols:
        assert col in df.columns


def test_no_missing_values():

    df = pd.read_csv(
        "data/processed/featured_climate_data.csv"
    )

    assert df.isnull().sum().sum() == 0


def test_row_count_positive():

    df = pd.read_csv(
        "data/processed/featured_climate_data.csv"
    )

    assert df.shape[0] > 0


import pandas as pd


def test_no_target_leakage():

    df = pd.read_csv(
        "data/processed/featured_climate_data.csv"
    )

    forbidden = [
        "future_temp",
        "next_day_temp",
        "target_shift_minus1"
    ]

    for col in forbidden:
        assert col not in df.columns