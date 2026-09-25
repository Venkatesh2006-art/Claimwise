import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CLAIMWISE - FRAUD PREDICTION
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROCESSED_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed"
)

X_TRAIN_PATH = os.path.join(
    PROCESSED_DIR,
    "fraud_X_train.csv"
)

Y_TRAIN_PATH = os.path.join(
    PROCESSED_DIR,
    "fraud_y_train.csv"
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    if not os.path.exists(X_TRAIN_PATH):
        raise FileNotFoundError(
            f"Training features not found:\n"
            f"{X_TRAIN_PATH}"
        )

    if not os.path.exists(Y_TRAIN_PATH):
        raise FileNotFoundError(
            f"Training target not found:\n"
            f"{Y_TRAIN_PATH}"
        )

    X_train = pd.read_csv(
        X_TRAIN_PATH
    )

    y_train = pd.read_csv(
        Y_TRAIN_PATH
    ).squeeze("columns")

    return X_train, y_train


# ============================================================
# BUILD FRAUD MODEL
# ============================================================

def build_model(X_train):

    categorical_columns = list(
        X_train.select_dtypes(
            include=["object", "string"]
        ).columns
    )

    numerical_columns = list(
        X_train.select_dtypes(
            include=["number"]
        ).columns
    )

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

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

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

    return pipeline


# ============================================================
# TRAIN MODEL
# ============================================================

def train_fraud_model():

    X_train, y_train = load_training_data()

    model = build_model(
        X_train
    )

    print(
        "Training Fraud Random Forest..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Fraud model training completed."
    )

    return model, X_train


# ============================================================
# PREDICT CLAIM
# ============================================================

def predict_claim(
    model,
    claim_data,
    threshold=0.50
):

    claim_df = pd.DataFrame(
        [claim_data]
    )

    probabilities = model.predict_proba(
        claim_df
    )[0]

    classes = model.named_steps[
        "model"
    ].classes_

    fraud_index = list(
        classes
    ).index(1)

    probability = float(
        probabilities[fraud_index]
    )

    prediction = (
        1
        if probability >= threshold
        else 0
    )

    return prediction, probability


# ============================================================
# CREATE EXAMPLE CLAIM
# ============================================================

def create_example_claim(X_train):

    categorical_columns = list(
        X_train.select_dtypes(
            include=["object", "string"]
        ).columns
    )

    numerical_columns = list(
        X_train.select_dtypes(
            include=["number"]
        ).columns
    )

    example_claim = {}

    for column in categorical_columns:

        mode_values = X_train[
            column
        ].mode()

        if not mode_values.empty:

            example_claim[column] = (
                mode_values.iloc[0]
            )

        else:

            example_claim[column] = ""

    for column in numerical_columns:

        example_claim[column] = float(
            X_train[column].median()
        )

    return example_claim


# ============================================================
# STANDALONE TEST
# ============================================================

def run_prediction():

    print("=" * 60)
    print("CLAIMWISE - FRAUD PREDICTION")
    print("=" * 60)

    X_train, y_train = load_training_data()

    print(
        "\nTraining Shape:"
    )

    print(
        X_train.shape
    )

    print(
        "\nCategorical Features:",
        len(
            X_train.select_dtypes(
                include=["object", "string"]
            ).columns
        )
    )

    print(
        "Numerical Features:",
        len(
            X_train.select_dtypes(
                include=["number"]
            ).columns
        )
    )

    model = build_model(
        X_train
    )

    print(
        "\nTraining Random Forest..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    print(
        "\nCreating example claim..."
    )

    example_claim = create_example_claim(
        X_train
    )

    print(
        "Example claim created."
    )

    print(
        "\nGenerating prediction..."
    )

    prediction, probability = predict_claim(
        model,
        example_claim
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FRAUD PREDICTION RESULT"
    )

    print(
        "=" * 60
    )

    print(
        "\nFraud Probability:"
    )

    print(
        f"{probability * 100:.2f}%"
    )

    print(
        "\nPrediction:"
    )

    if prediction == 1:

        print(
            "FRAUD"
        )

    else:

        print(
            "NOT FRAUD"
        )

    print(
        "\nDecision Threshold:"
    )

    print(
        "50%"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FRAUD PREDICTION COMPLETED"
    )

    print(
        "=" * 60
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_prediction()