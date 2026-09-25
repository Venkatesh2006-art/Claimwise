import os
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# RESULT FILE PATHS
# ============================================================

LOGISTIC_RESULTS_PATH = os.path.join(
    PROJECT_DIR,
    "fraud_logistic_results.csv"
)

RANDOM_FOREST_RESULTS_PATH = os.path.join(
    PROJECT_DIR,
    "fraud_random_forest_results.csv"
)

COMPARISON_PATH = os.path.join(
    PROJECT_DIR,
    "fraud_model_comparison_results.csv"
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_model_comparison():

    print("=" * 60)
    print("CLAIMWISE - FRAUD MODEL COMPARISON")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(
        LOGISTIC_RESULTS_PATH
    ):
        raise FileNotFoundError(
            "fraud_logistic_results.csv "
            "was not found.\n"
            "Please run Fraud_Logistic_Regression.py first."
        )

    if not os.path.exists(
        RANDOM_FOREST_RESULTS_PATH
    ):
        raise FileNotFoundError(
            "fraud_random_forest_results.csv "
            "was not found.\n"
            "Please run Fraud_Random_Forest.py first."
        )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    logistic_df = pd.read_csv(
        LOGISTIC_RESULTS_PATH
    )

    random_forest_df = pd.read_csv(
        RANDOM_FOREST_RESULTS_PATH
    )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    comparison_df = pd.concat(
        [
            logistic_df,
            random_forest_df
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Display comparison
    # --------------------------------------------------------

    print("\nModel Comparison:")
    print("-" * 60)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False
    )

    print(
        "\nComparison saved to:"
    )

    print(
        COMPARISON_PATH
    )

    # --------------------------------------------------------
    # Detailed comparison
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DETAILED MODEL METRICS")
    print("=" * 60)

    for _, row in comparison_df.iterrows():

        print(
            f"\nModel: {row['Model']}"
        )

        print(
            f"Accuracy : {row['Accuracy']:.4f}"
        )

        print(
            f"Precision: {row['Precision']:.4f}"
        )

        print(
            f"Recall   : {row['Recall']:.4f}"
        )

        print(
            f"F1 Score : {row['F1_Score']:.4f}"
        )

        print(
            f"ROC-AUC  : {row['ROC_AUC']:.4f}"
        )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "FRAUD MODEL COMPARISON COMPLETED"
    )

    print(
        "=" * 60
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_model_comparison()