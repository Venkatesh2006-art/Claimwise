import os
import pandas as pd


# --------------------------------------------------
# CLAIMWISE DATA PATH
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "train.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data(path: str = DATA_PATH) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at:\n{path}"
        )

    df = pd.read_csv(path)

    return df


# --------------------------------------------------
# DATA SUMMARY
# --------------------------------------------------

def get_data_summary() -> dict:

    df = load_data()

    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],

        "columns": list(df.columns),

        "dtypes": {
            col: str(df[col].dtype)
            for col in df.columns
        },

        "missing_counts": {
            col: int(df[col].isnull().sum())
            for col in df.columns
        },

        "preview": df.head(10).to_dict("records"),
    }

    return summary


# --------------------------------------------------
# TEST THE DATASET
# --------------------------------------------------

if __name__ == "__main__":

    summary = get_data_summary()

    print("==========================================")
    print("       CLAIMWISE DATA SUMMARY")
    print("==========================================")

    print("Rows:", summary["n_rows"])
    print("Columns:", summary["n_cols"])

    print("\nColumns:")
    for column in summary["columns"]:
        print(column)

    print("\nFirst 10 rows:")

    for row in summary["preview"]:
        print(row)