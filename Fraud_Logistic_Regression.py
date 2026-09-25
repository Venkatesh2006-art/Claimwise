import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


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


def run_logistic_regression():

    print("=" * 60)
    print("CLAIMWISE - FRAUD LOGISTIC REGRESSION")
    print("=" * 60)

    # --------------------------------------------------
    # LOAD PREPROCESSED DATA
    # --------------------------------------------------

    X_train = pd.read_csv(
        os.path.join(
            PROCESSED_DIR,
            "fraud_X_train.csv"
        )
    )

    X_test = pd.read_csv(
        os.path.join(
            PROCESSED_DIR,
            "fraud_X_test.csv"
        )
    )

    y_train = pd.read_csv(
        os.path.join(
            PROCESSED_DIR,
            "fraud_y_train.csv"
        )
    ).squeeze()

    y_test = pd.read_csv(
        os.path.join(
            PROCESSED_DIR,
            "fraud_y_test.csv"
        )
    ).squeeze()

    print("\nTraining Shape:")
    print(X_train.shape)

    print("\nTesting Shape:")
    print(X_test.shape)

    # --------------------------------------------------
    # IDENTIFY FEATURES
    # --------------------------------------------------

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

    # --------------------------------------------------
    # CATEGORICAL PIPELINE
    # --------------------------------------------------

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

    # --------------------------------------------------
    # NUMERICAL PIPELINE
    # --------------------------------------------------

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

    # --------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------

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

    # --------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    )

    # --------------------------------------------------
    # COMPLETE PIPELINE
    # --------------------------------------------------

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

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    pipeline.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

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

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION RESULTS")
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

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    print(
        "\nClassification Report:"
    )

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

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    # --------------------------------------------------
    # SAVE CONFUSION MATRIX
    # --------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Fraud",
            "Fraud"
        ]
    )

    display.plot()

    plt.title(
        "Fraud Detection - Logistic Regression"
    )

    plt.tight_layout()

    confusion_path = os.path.join(
        CHART_DIR,
        "fraud_logistic_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    results = pd.DataFrame({
        "Model": [
            "Logistic Regression"
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
    })

    results_path = os.path.join(
        PROJECT_DIR,
        "fraud_logistic_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    print(
        "\nConfusion matrix saved to:"
    )

    print(
        confusion_path
    )

    print(
        "\nResults saved to:"
    )

    print(
        results_path
    )

    print("\n" + "=" * 60)
    print(
        "FRAUD LOGISTIC REGRESSION COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":

    run_logistic_regression()