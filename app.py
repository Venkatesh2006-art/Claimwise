import os
import traceback

import pandas as pd

from flask import (
    Flask,
    render_template,
    request
)

from Load_data import (
    get_data_summary
)

from ClaimWise_EDA import (
    run_eda
)

from Fraud_Prediction import (
    load_training_data,
    build_model,
    predict_claim
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# FRAUD MODEL
# ============================================================

fraud_model = None
fraud_X_train = None


def initialize_fraud_model():

    global fraud_model
    global fraud_X_train

    try:

        print(
            "\n" + "=" * 60
        )

        print(
            "CLAIMWISE - LOADING FRAUD MODEL"
        )

        print(
            "=" * 60
        )

        fraud_X_train, fraud_y_train = (
            load_training_data()
        )

        fraud_model = build_model(
            fraud_X_train
        )

        print(
            "\nTraining fraud model..."
        )

        fraud_model.fit(
            fraud_X_train,
            fraud_y_train
        )

        print(
            "Fraud model loaded successfully."
        )

        print(
            "=" * 60
        )

    except Exception as e:

        fraud_model = None
        fraud_X_train = None

        print(
            "\nFraud model could not be loaded:"
        )

        print(e)

        traceback.print_exc()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="home"
    )


# ============================================================
# DATA LOADING
# ============================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:

        summary = get_data_summary()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        traceback.print_exc()

        error = (
            f"Unexpected error: {e}"
        )

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda_page():

    error = None
    results = None

    try:

        results = run_eda()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        traceback.print_exc()

        error = str(e)

    return render_template(
        "EDA.html",
        active="eda",
        results=results,
        error=error
    )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing():

    error = None

    try:

        project_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        processed_dir = os.path.join(
            project_dir,
            "data",
            "processed"
        )

        train_clean_path = os.path.join(
            processed_dir,
            "train_clean.csv"
        )

        test_clean_path = os.path.join(
            processed_dir,
            "test_clean.csv"
        )

        feature_info_path = os.path.join(
            processed_dir,
            "feature_information.csv"
        )

        if not os.path.exists(
            train_clean_path
        ):

            error = (
                "train_clean.csv was not found. "
                "Please run Preprocessing.py first."
            )

        elif not os.path.exists(
            test_clean_path
        ):

            error = (
                "test_clean.csv was not found. "
                "Please run Preprocessing.py first."
            )

        elif not os.path.exists(
            feature_info_path
        ):

            error = (
                "feature_information.csv was not found. "
                "Please run Preprocessing.py first."
            )

    except Exception as e:

        traceback.print_exc()

        error = (
            f"Error loading preprocessing information: {e}"
        )

    return render_template(
        "Preprocessing.html",
        active="preprocessing",
        error=error
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

@app.route("/model-comparison")
def model_comparison():

    return render_template(
        "model_comparison.html",
        active="model-comparison"
    )


# ============================================================
# CLAIM PREDICTION
# ============================================================

@app.route(
    "/claim-prediction",
    methods=["GET", "POST"]
)
def claim_prediction():

    prediction = None
    claim_id = None
    error = None

    if request.method == "POST":

        claim_id = request.form.get(
            "claim_id",
            ""
        ).strip()

        if not claim_id:

            error = (
                "Please enter a Claim ID."
            )

        else:

            try:

                project_dir = os.path.dirname(
                    os.path.abspath(__file__)
                )

                prediction_file = os.path.join(
                    project_dir,
                    "predictions.csv"
                )

                if not os.path.exists(
                    prediction_file
                ):

                    error = (
                        "predictions.csv was not found. "
                        "Please run Claim_Prediction.py first."
                    )

                else:

                    predictions_df = pd.read_csv(
                        prediction_file
                    )

                    predictions_df["id"] = (
                        predictions_df["id"]
                        .astype(str)
                        .str.strip()
                    )

                    matching_row = (
                        predictions_df[
                            predictions_df["id"]
                            == claim_id
                        ]
                    )

                    if matching_row.empty:

                        error = (
                            f"Claim ID {claim_id} "
                            "was not found in the predictions."
                        )

                    else:

                        prediction = float(
                            matching_row.iloc[0][
                                "loss"
                            ]
                        )

            except Exception as e:

                traceback.print_exc()

                error = (
                    f"Error while finding prediction: {e}"
                )

    return render_template(
        "claim_prediction.html",
        active="claim-prediction",
        prediction=prediction,
        claim_id=claim_id,
        error=error
    )


# ============================================================
# FRAUD DETECTION
# ============================================================

@app.route(
    "/fraud-detection",
    methods=["GET", "POST"]
)
def fraud_detection():

    prediction = None
    probability = None
    error = None

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # Make sure model exists
            # ------------------------------------------------

            if fraud_model is None:

                raise RuntimeError(
                    "Fraud model is not available. "
                    "Please restart the Flask application."
                )

            # ------------------------------------------------
            # Collect claim data
            # ------------------------------------------------

            claim_data = {}

            for column in fraud_X_train.columns:

                value = request.form.get(
                    column,
                    ""
                ).strip()

                # --------------------------------------------
                # Numerical columns
                # --------------------------------------------

                if pd.api.types.is_numeric_dtype(
                    fraud_X_train[column]
                ):

                    if value == "":

                        claim_data[column] = None

                    else:

                        claim_data[column] = float(
                            value
                        )

                # --------------------------------------------
                # Categorical columns
                # --------------------------------------------

                else:

                    claim_data[column] = value

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            prediction, probability = predict_claim(
                fraud_model,
                claim_data,
                threshold=0.50
            )

        except Exception as e:

            traceback.print_exc()

            error = (
                f"Error while predicting fraud: {e}"
            )

    return render_template(
        "fraud_detection.html",
        active="fraud-detection",
        prediction=prediction,
        probability=probability,
        error=error
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    initialize_fraud_model()

    app.run(
        debug=True
    )