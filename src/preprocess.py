import os
import numpy as np
import pandas as pd

from config import DATA_PATH, PROCESSED_PATH


# ======================================================
# FILE PATHS
# ======================================================

TRAIN_FILE = DATA_PATH + "DailyDelhiClimateTrain.csv"
TEST_FILE = DATA_PATH + "DailyDelhiClimateTest.csv"
OUTPUT_FILE = PROCESSED_PATH + "clean_climate_data.csv"


# ======================================================
# PREPROCESSOR CLASS
# ======================================================

class ClimatePreprocessor:
    """
    Climate Trend Analyzer Pro

    Handles:
    - Load train dataset
    - Load test dataset
    - Dataset profiling
    - Merge datasets
    - Missing values
    - Duplicate rows
    - Duplicate dates
    - Outlier treatment
    - Save clean dataset
    """

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    def read_csv(self, path):
        """
        Safe CSV loader
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        df = pd.read_csv(path)

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        if "date" in df.columns:
            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

        return df

    # --------------------------------------------------
    # PROFILE DATA
    # --------------------------------------------------

    def profile_data(
        self,
        df,
        name="Dataset"
    ):

        print("\n" + "=" * 70)
        print(name.upper())
        print("=" * 70)

        print("Shape:", df.shape)
        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])

        print("\nColumn Names:")
        print(df.columns.tolist())

        print("\nMissing Values Per Column:")
        print(df.isnull().sum())

        print(
            "\nTotal Missing Values:",
            df.isnull().sum().sum()
        )

        print("\nNaN Values Per Column:")
        print(df.isna().sum())

        print(
            "\nTotal NaN Values:",
            df.isna().sum().sum()
        )

        print(
            "\nDuplicate Rows:",
            df.duplicated().sum()
        )

        if "date" in df.columns:

            print(
                "Duplicate Dates:",
                df["date"].duplicated().sum()
            )

            print(
                "\nDate Range:",
                df["date"].min(),
                "to",
                df["date"].max()
            )

        print("=" * 70)

    # --------------------------------------------------
    # CLEANING
    # --------------------------------------------------

    def remove_duplicate_rows(self, df):

        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        print(
            f"\nRemoved duplicate rows: {before - after}"
        )

        return df

    def remove_duplicate_dates(self, df):

        if "date" not in df.columns:
            return df

        before = len(df)

        df = df.drop_duplicates(
            subset="date",
            keep="first"
        )

        after = len(df)

        print(
            f"Removed duplicate dates: {before - after}"
        )

        return df

    def fill_missing_values(self, df):

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_cols:
            df[col] = df[col].fillna(
                df[col].median()
            )

        print("Missing numeric values handled.")

        return df

    def treat_outliers(self, df):

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_cols:

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            df[col] = np.where(
                df[col] < lower,
                lower,
                df[col]
            )

            df[col] = np.where(
                df[col] > upper,
                upper,
                df[col]
            )

        print("Outliers treated using IQR.")

        return df

    def sort_by_date(self, df):

        if "date" in df.columns:
            df = df.sort_values(
                by="date"
            ).reset_index(drop=True)

        return df

    # --------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------

    def save_file(self, df):

        os.makedirs(
            PROCESSED_PATH,
            exist_ok=True
        )

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            f"\nSaved cleaned file: {OUTPUT_FILE}"
        )

    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------

    def run_pipeline(self):

        # -------------------------------
        # LOAD DATA
        # -------------------------------

        train = self.read_csv(TRAIN_FILE)
        test = self.read_csv(TEST_FILE)

        self.profile_data(
            train,
            "Train Dataset"
        )

        self.profile_data(
            test,
            "Test Dataset"
        )

        # -------------------------------
        # MERGE
        # -------------------------------

        full = pd.concat(
            [train, test],
            ignore_index=True
        )

        self.profile_data(
            full,
            "Merged Dataset Before Cleaning"
        )

        # -------------------------------
        # CLEANING
        # -------------------------------

        full = self.remove_duplicate_rows(full)

        full = self.remove_duplicate_dates(full)

        full = self.fill_missing_values(full)

        full = self.treat_outliers(full)

        full = self.sort_by_date(full)

        # -------------------------------
        # FINAL PROFILE
        # -------------------------------

        self.profile_data(
            full,
            "Merged Dataset After Cleaning"
        )

        # -------------------------------
        # SAVE
        # -------------------------------

        self.save_file(full)

        print("\n✅ Preprocessing Complete")

        return full


# ======================================================
# MAIN FUNCTION FOR src/main.py
# ======================================================

def run():
    """
    Use inside main.py
    """
    processor = ClimatePreprocessor()

    clean_df = processor.run_pipeline()

    return clean_df


# ======================================================
# LOCAL RUN
# ======================================================

if __name__ == "__main__":
    run()