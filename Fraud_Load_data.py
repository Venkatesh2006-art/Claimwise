import os
import pandas as pd


PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FRAUD_DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "Fraud",
    "fraud_oracle.csv"
)


def load_fraud_data(
    path: str = FRAUD_DATA_PATH
) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fraud dataset not found at:\n{path}"
        )

    return pd.read_csv(path)


def get_fraud_data_summary() -> dict:

    df = load_fraud_data()

    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),

        "missing_counts": {
            column: int(df[column].isnull().sum())
            for column in df.columns
        },

        "fraud_distribution": (
            df["FraudFound_P"]
            .value_counts()
            .to_dict()
            if "FraudFound_P" in df.columns
            else {}
        ),

        "preview": df.head(10).to_dict(
            "records"
        )
    }

    return summary


if __name__ == "__main__":

    summary = get_fraud_data_summary()

    print("=" * 60)
    print("CLAIMWISE - FRAUD DATA SUMMARY")
    print("=" * 60)

    print("\nRows:", summary["n_rows"])
    print("Columns:", summary["n_cols"])

    print("\nColumns:")
    for column in summary["columns"]:
        print("-", column)

    print("\nMissing Values:")

    total_missing = 0

    for column, count in summary[
        "missing_counts"
    ].items():

        if count > 0:
            print(
                f"{column}: {count}"
            )
            total_missing += count

    if total_missing == 0:
        print("No missing values found.")

    print("\nFraud Distribution:")

    for label, count in summary[
        "fraud_distribution"
    ].items():

        if label == 0:
            name = "Not Fraud"
        elif label == 1:
            name = "Fraud"
        else:
            name = "Unknown"

        print(
            f"{name} ({label}): {count}"
        )

    print("\nFirst 5 Rows:")

    for row in summary["preview"][:5]:
        print(row)

    print("\n" + "=" * 60)
    print("FRAUD DATA LOADED SUCCESSFULLY")
    print("=" * 60)