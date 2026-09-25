import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROCESSED_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed"
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


# ============================================================
# FILE PATHS
# ============================================================

X_TRAIN_PATH = os.path.join(
    PROCESSED_DIR,
    "fraud_X_train.csv"
)

X_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "fraud_X_test.csv"
)

Y_TRAIN_PATH = os.path.join(
    PROCESSED_DIR,
    "fraud_y_train.csv"
)

Y_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "fraud_y_test.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(X_TRAIN_PATH):
        raise FileNotFoundError(
            f"Training data not found:\n{X_TRAIN_PATH}"
        )

    if not os.path.exists(X_TEST_PATH):
        raise FileNotFoundError(
            f"Testing data not found:\n{X_TEST_PATH}"
        )

    if not os.path.exists(Y_TRAIN_PATH):
        raise FileNotFoundError(
            f"Training target not found:\n{Y_TRAIN_PATH}"
        )

    if not os.path.exists(Y_TEST_PATH):
        raise FileNotFoundError(
            f"Testing target not found:\n{Y_TEST_PATH}"
        )

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")
    y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

    return X_train, X_test, y_train, y_test


# ============================================================
# MAIN
# ============================================================

def run_random_forest():

    print("=" * 60)
    print("CLAIMWISE - FRAUD RANDOM FOREST")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    print("\nTraining Shape:")
    print(X_train.shape)

    print("\nTesting Shape:")
    print(X_test.shape)

    # --------------------------------------------------------
    # Identify feature types
    # --------------------------------------------------------

    categorical_columns = list(
        X_train.select_dtypes(
            include=["object", "str"]
        ).columns
    )

    numerical_columns = list(
        X_train.select_dtypes(
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

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Combine preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        ]
    )

    # --------------------------------------------------------
    # Random Forest model
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Complete pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    pipeline.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RANDOM FOREST RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Not Fraud",
                "Fraud"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    confusion_matrix_path = os.path.join(
        CHART_DIR,
        "fraud_random_forest_confusion_matrix.png"
    )

    try:

        import matplotlib.pyplot as plt

        plt.figure(
            figsize=(6, 5)
        )

        plt.imshow(
            cm
        )

        plt.title(
            "Fraud Random Forest - Confusion Matrix"
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        plt.xticks(
            [0, 1],
            ["Not Fraud", "Fraud"]
        )

        plt.yticks(
            [0, 1],
            ["Not Fraud", "Fraud"]
        )

        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        plt.tight_layout()

        plt.savefig(
            confusion_matrix_path,
            dpi=150
        )

        plt.close()

        print(
            "\nConfusion matrix saved to:"
        )

        print(
            confusion_matrix_path
        )

    except Exception as e:

        print(
            "\nCould not save confusion matrix:"
        )

        print(e)

    # --------------------------------------------------------
    # Save model results
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "Model": [
                "Random Forest"
            ],
            "Accuracy": [
                accuracy
            ],
            "Precision": [
                precision
            ],
            "Recall": [
                recall
            ],
            "F1_Score": [
                f1
            ],
            "ROC_AUC": [
                roc_auc
            ]
        }
    )

    results_path = os.path.join(
        PROJECT_DIR,
        "fraud_random_forest_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        results_path
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FRAUD RANDOM FOREST COMPLETED"
    )

    print(
        "=" * 60
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_random_forest()