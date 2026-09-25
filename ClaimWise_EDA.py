import os
import pandas as pd
import matplotlib.pyplot as plt

from Load_data import load_data


# ============================================================
# CLAIMWISE - EXPLORATORY DATA ANALYSIS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(PROJECT_DIR, "static", "charts")

os.makedirs(CHART_DIR, exist_ok=True)


def run_eda():

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. IDENTIFY COLUMNS
    # --------------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Remove ID from numerical features
    if "id" in numerical_columns:
        numerical_columns.remove("id")

    # --------------------------------------------------------
    # 3. BASIC INFORMATION
    # --------------------------------------------------------

    n_rows = df.shape[0]
    n_cols = df.shape[1]

    # --------------------------------------------------------
    # 4. MISSING VALUES
    # --------------------------------------------------------

    missing_values = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    missing_values = missing_values[
        missing_values > 0
    ]

    # --------------------------------------------------------
    # 5. LOSS STATISTICS
    # --------------------------------------------------------

    loss_stats = {
        "mean": round(df["loss"].mean(), 2),
        "median": round(df["loss"].median(), 2),
        "std": round(df["loss"].std(), 2),
        "minimum": round(df["loss"].min(), 2),
        "maximum": round(df["loss"].max(), 2),
        "q1": round(df["loss"].quantile(0.25), 2),
        "q3": round(df["loss"].quantile(0.75), 2),
    }

    # --------------------------------------------------------
    # 6. LOSS DISTRIBUTION
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["loss"],
        bins=50
    )

    plt.title("Claim Loss Distribution")
    plt.xlabel("Loss")
    plt.ylabel("Number of Claims")

    plt.tight_layout()

    loss_chart = os.path.join(
        CHART_DIR,
        "loss_distribution.png"
    )

    plt.savefig(loss_chart)
    plt.close()

    # --------------------------------------------------------
    # 7. CONTINUOUS FEATURE DISTRIBUTION
    # --------------------------------------------------------

    continuous_columns = [
        col for col in numerical_columns
        if col != "loss"
    ]

    # Create a combined distribution chart
    plt.figure(figsize=(12, 7))

    for column in continuous_columns:
        plt.hist(
            df[column],
            bins=30,
            alpha=0.25,
            label=column
        )

    plt.title("Continuous Feature Distributions")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(
        fontsize=7,
        ncol=2
    )

    plt.tight_layout()

    continuous_chart = os.path.join(
        CHART_DIR,
        "continuous_features.png"
    )

    plt.savefig(continuous_chart)
    plt.close()

    # --------------------------------------------------------
    # 8. RETURN RESULTS
    # --------------------------------------------------------

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,

        "categorical_count": len(
            categorical_columns
        ),

        "numerical_count": len(
            numerical_columns
        ),

        "categorical_columns":
            categorical_columns,

        "numerical_columns":
            numerical_columns,

        "missing_values":
            missing_values.to_dict(),

        "loss_stats":
            loss_stats,

        "loss_chart":
            "/static/charts/loss_distribution.png",

        "continuous_chart":
            "/static/charts/continuous_features.png",
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    results = run_eda()

    print("\n==========================================")
    print("        CLAIMWISE EDA")
    print("==========================================")

    print("\nDataset:")
    print("Rows:", results["n_rows"])
    print("Columns:", results["n_cols"])

    print("\nCategorical Features:")
    print(results["categorical_count"])

    print("\nNumerical Features:")
    print(results["numerical_count"])

    print("\nLoss Statistics:")

    for key, value in results["loss_stats"].items():
        print(f"{key}: {value}")

    print("\nMissing Values:")

    if results["missing_values"]:
        for column, count in results["missing_values"].items():
            print(f"{column}: {count}")
    else:
        print("No missing values found.")

    print("\nEDA completed successfully.")