import os
import pandas as pd
import matplotlib.pyplot as plt

from Fraud_Load_data import load_fraud_data


PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CHART_DIR = os.path.join(
    PROJECT_DIR,
    "static",
    "charts"
)

os.makedirs(
    CHART_DIR,
    exist_ok=True
)


def run_fraud_eda():

    df = load_fraud_data()

    print("=" * 60)
    print("CLAIMWISE - FRAUD EDA")
    print("=" * 60)

    # --------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------

    print("\nDataset Shape:")
    print(df.shape)

    print("\nNumber of Rows:", df.shape[0])
    print("Number of Columns:", df.shape[1])

    # --------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------

    print("\nFraud Distribution:")

    fraud_counts = df["FraudFound_P"].value_counts()

    print(
        "Not Fraud (0):",
        fraud_counts.get(0, 0)
    )

    print(
        "Fraud (1):",
        fraud_counts.get(1, 0)
    )

    fraud_percentage = (
        df["FraudFound_P"].mean() * 100
    )

    print(
        f"Fraud Percentage: "
        f"{fraud_percentage:.2f}%"
    )

    # --------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------

    total_missing = int(
        df.isnull().sum().sum()
    )

    print("\nTotal Missing Values:")
    print(total_missing)

    # --------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------

    duplicate_count = int(
        df.duplicated().sum()
    )

    print("\nDuplicate Rows:")
    print(duplicate_count)

    # --------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------

    categorical_columns = list(
        df.select_dtypes(
            include=["object"]
        ).columns
    )

    numerical_columns = list(
        df.select_dtypes(
            include=["int64", "float64"]
        ).columns
    )

    print("\nCategorical Features:")
    print(len(categorical_columns))

    print("\nNumerical Features:")
    print(len(numerical_columns))

    print("\nCategorical Columns:")
    for column in categorical_columns:
        print("-", column)

    print("\nNumerical Columns:")
    for column in numerical_columns:
        print("-", column)

    # --------------------------------------------------
    # AGE STATISTICS
    # --------------------------------------------------

    if "Age" in df.columns:

        print("\nAge Statistics:")

        print(
            "Minimum Age:",
            df["Age"].min()
        )

        print(
            "Maximum Age:",
            df["Age"].max()
        )

        print(
            "Average Age:",
            round(
                df["Age"].mean(),
                2
            )
        )

    # --------------------------------------------------
    # CHART 1 - FRAUD DISTRIBUTION
    # --------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    fraud_counts.sort_index().plot(
        kind="bar"
    )

    plt.title(
        "Fraud vs Non-Fraud Claims"
    )

    plt.xlabel(
        "Fraud Status"
    )

    plt.ylabel(
        "Number of Claims"
    )

    plt.xticks(
        [0, 1],
        ["Not Fraud", "Fraud"],
        rotation=0
    )

    plt.tight_layout()

    fraud_chart = os.path.join(
        CHART_DIR,
        "fraud_distribution.png"
    )

    plt.savefig(
        fraud_chart,
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # CHART 2 - AGE DISTRIBUTION
    # --------------------------------------------------

    if "Age" in df.columns:

        plt.figure(
            figsize=(8, 5)
        )

        df["Age"].plot(
            kind="hist",
            bins=20
        )

        plt.title(
            "Age Distribution"
        )

        plt.xlabel(
            "Age"
        )

        plt.ylabel(
            "Number of Claims"
        )

        plt.tight_layout()

        age_chart = os.path.join(
            CHART_DIR,
            "fraud_age_distribution.png"
        )

        plt.savefig(
            age_chart,
            dpi=150
        )

        plt.close()

    # --------------------------------------------------
    # CHART 3 - ACCIDENT AREA
    # --------------------------------------------------

    if "AccidentArea" in df.columns:

        area_fraud = pd.crosstab(
            df["AccidentArea"],
            df["FraudFound_P"]
        )

        area_fraud.plot(
            kind="bar",
            figsize=(8, 5)
        )

        plt.title(
            "Fraud by Accident Area"
        )

        plt.xlabel(
            "Accident Area"
        )

        plt.ylabel(
            "Number of Claims"
        )

        plt.xticks(
            rotation=0
        )

        plt.legend(
            ["Not Fraud", "Fraud"]
        )

        plt.tight_layout()

        area_chart = os.path.join(
            CHART_DIR,
            "fraud_by_accident_area.png"
        )

        plt.savefig(
            area_chart,
            dpi=150
        )

        plt.close()

    # --------------------------------------------------
    # CHART 4 - VEHICLE CATEGORY
    # --------------------------------------------------

    if "VehicleCategory" in df.columns:

        vehicle_fraud = pd.crosstab(
            df["VehicleCategory"],
            df["FraudFound_P"]
        )

        vehicle_fraud.plot(
            kind="bar",
            figsize=(8, 5)
        )

        plt.title(
            "Fraud by Vehicle Category"
        )

        plt.xlabel(
            "Vehicle Category"
        )

        plt.ylabel(
            "Number of Claims"
        )

        plt.xticks(
            rotation=0
        )

        plt.legend(
            ["Not Fraud", "Fraud"]
        )

        plt.tight_layout()

        vehicle_chart = os.path.join(
            CHART_DIR,
            "fraud_by_vehicle_category.png"
        )

        plt.savefig(
            vehicle_chart,
            dpi=150
        )

        plt.close()

    # --------------------------------------------------
    # CHART 5 - BASE POLICY
    # --------------------------------------------------

    if "BasePolicy" in df.columns:

        policy_fraud = pd.crosstab(
            df["BasePolicy"],
            df["FraudFound_P"]
        )

        policy_fraud.plot(
            kind="bar",
            figsize=(8, 5)
        )

        plt.title(
            "Fraud by Base Policy"
        )

        plt.xlabel(
            "Base Policy"
        )

        plt.ylabel(
            "Number of Claims"
        )

        plt.xticks(
            rotation=0
        )

        plt.legend(
            ["Not Fraud", "Fraud"]
        )

        plt.tight_layout()

        policy_chart = os.path.join(
            CHART_DIR,
            "fraud_by_base_policy.png"
        )

        plt.savefig(
            policy_chart,
            dpi=150
        )

        plt.close()

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    results = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "categorical_features": len(
            categorical_columns
        ),
        "numerical_features": len(
            numerical_columns
        ),
        "missing_values": total_missing,
        "duplicate_rows": duplicate_count,
        "not_fraud": int(
            fraud_counts.get(0, 0)
        ),
        "fraud": int(
            fraud_counts.get(1, 0)
        ),
        "fraud_percentage": round(
            fraud_percentage,
            2
        )
    }

    print("\n" + "=" * 60)
    print("FRAUD EDA COMPLETED")
    print("=" * 60)

    print("\nCharts saved in:")
    print(CHART_DIR)

    return results


if __name__ == "__main__":

    run_fraud_eda()