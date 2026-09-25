import os
import pandas as pd

from sklearn.model_selection import train_test_split


PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FRAUD_DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "Fraud",
    "fraud_oracle.csv"
)

PROCESSED_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed"
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


def load_fraud_data():

    if not os.path.exists(FRAUD_DATA_PATH):
        raise FileNotFoundError(
            f"Fraud dataset not found at:\n"
            f"{FRAUD_DATA_PATH}"
        )

    return pd.read_csv(
        FRAUD_DATA_PATH
    )


def run_fraud_preprocessing():

    print("=" * 60)
    print("CLAIMWISE - FRAUD PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    df = load_fraud_data()

    print("\nOriginal Dataset Shape:")
    print(df.shape)

    # --------------------------------------------------
    # CHECK TARGET
    # --------------------------------------------------

    target_column = "FraudFound_P"

    if target_column not in df.columns:
        raise ValueError(
            "FraudFound_P target column was not found."
        )

    print(
        "\nTarget Column:",
        target_column
    )

    # --------------------------------------------------
    # CHECK MISSING VALUES
    # --------------------------------------------------

    missing_values = int(
        df.isnull().sum().sum()
    )

    print(
        "\nTotal Missing Values:",
        missing_values
    )

    # --------------------------------------------------
    # CHECK DUPLICATES
    # --------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    print(
        "Duplicate Rows:",
        duplicate_rows
    )

    # --------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------

    print("\nTarget Distribution:")

    target_counts = (
        df[target_column]
        .value_counts()
        .sort_index()
    )

    for value, count in target_counts.items():

        if value == 0:
            label = "Not Fraud"
        elif value == 1:
            label = "Fraud"
        else:
            label = "Unknown"

        print(
            f"{label} ({value}): {count}"
        )

    # --------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # --------------------------------------------------

    X = df.drop(
        columns=[target_column]
    ).copy()

    y = df[target_column].copy()

    print(
        "\nFeature Shape Before Cleaning:",
        X.shape
    )

    print(
        "Target Shape:",
        y.shape
    )

    # --------------------------------------------------
    # REMOVE POLICY NUMBER
    # --------------------------------------------------
    #
    # PolicyNumber is an identifier rather than
    # a meaningful predictive feature.
    #
    # We keep the original dataset unchanged.
    # It is removed only from the model features.
    # --------------------------------------------------

    if "PolicyNumber" in X.columns:

        X = X.drop(
            columns=["PolicyNumber"]
        )

        print(
            "\nPolicyNumber removed "
            "from model features."
        )

    # --------------------------------------------------
    # IDENTIFY DATA TYPES
    # --------------------------------------------------

    categorical_columns = list(
        X.select_dtypes(
            include=["object", "str"]
        ).columns
    )

    numerical_columns = list(
        X.select_dtypes(
            include=["number"]
        ).columns
    )

    print(
        "\nCategorical Features:",
        len(categorical_columns)
    )

    print(
        "Numerical Features:",
        len(numerical_columns)
    )

    print("\nCategorical Columns:")

    for column in categorical_columns:
        print("-", column)

    print("\nNumerical Columns:")

    for column in numerical_columns:
        print("-", column)

    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------
    #
    # stratify=y keeps the fraud/non-fraud ratio
    # approximately the same in both sets.
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print(
        "\nTraining Feature Shape:",
        X_train.shape
    )

    print(
        "Testing Feature Shape:",
        X_test.shape
    )

    print(
        "Training Target Shape:",
        y_train.shape
    )

    print(
        "Testing Target Shape:",
        y_test.shape
    )

    # --------------------------------------------------
    # CHECK TRAINING FRAUD DISTRIBUTION
    # --------------------------------------------------

    print(
        "\nTraining Target Distribution:"
    )

    train_counts = (
        y_train
        .value_counts()
        .sort_index()
    )

    for value, count in train_counts.items():

        if value == 0:
            label = "Not Fraud"
        else:
            label = "Fraud"

        percentage = (
            count / len(y_train)
        ) * 100

        print(
            f"{label}: {count} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------
    # CHECK TEST FRAUD DISTRIBUTION
    # --------------------------------------------------

    print(
        "\nTesting Target Distribution:"
    )

    test_counts = (
        y_test
        .value_counts()
        .sort_index()
    )

    for value, count in test_counts.items():

        if value == 0:
            label = "Not Fraud"
        else:
            label = "Fraud"

        percentage = (
            count / len(y_test)
        ) * 100

        print(
            f"{label}: {count} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------
    # SAVE SPLIT DATA
    # --------------------------------------------------

    X_train_path = os.path.join(
        PROCESSED_DIR,
        "fraud_X_train.csv"
    )

    X_test_path = os.path.join(
        PROCESSED_DIR,
        "fraud_X_test.csv"
    )

    y_train_path = os.path.join(
        PROCESSED_DIR,
        "fraud_y_train.csv"
    )

    y_test_path = os.path.join(
        PROCESSED_DIR,
        "fraud_y_test.csv"
    )

    X_train.to_csv(
        X_train_path,
        index=False
    )

    X_test.to_csv(
        X_test_path,
        index=False
    )

    y_train.to_csv(
        y_train_path,
        index=False
    )

    y_test.to_csv(
        y_test_path,
        index=False
    )

    # --------------------------------------------------
    # SAVE FEATURE INFORMATION
    # --------------------------------------------------

    feature_information = pd.DataFrame({
        "feature": X.columns,
        "data_type": [
            str(X[column].dtype)
            for column in X.columns
        ],
        "feature_type": [
            (
                "categorical"
                if column in categorical_columns
                else "numerical"
            )
            for column in X.columns
        ]
    })

    feature_info_path = os.path.join(
        PROCESSED_DIR,
        "fraud_feature_information.csv"
    )

    feature_information.to_csv(
        feature_info_path,
        index=False
    )

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("\nSaved Files:")

    print(
        "-",
        X_train_path
    )

    print(
        "-",
        X_test_path
    )

    print(
        "-",
        y_train_path
    )

    print(
        "-",
        y_test_path
    )

    print(
        "-",
        feature_info_path
    )

    print("\n" + "=" * 60)
    print("FRAUD PREPROCESSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    run_fraud_preprocessing()