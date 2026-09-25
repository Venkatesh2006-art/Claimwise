import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from Load_data import load_data


print("==========================================")
print("       CLAIMWISE - CLAIM PREDICTION")
print("==========================================")


# -------------------------------------------------
# 1. LOAD TRAINING DATA
# -------------------------------------------------

train_df = load_data()

print("\nTraining dataset loaded.")
print("Training rows:", train_df.shape[0])
print("Training columns:", train_df.shape[1])


# -------------------------------------------------
# 2. LOAD TEST DATA
# -------------------------------------------------

project_dir = os.path.dirname(os.path.abspath(__file__))

test_path = os.path.join(
    project_dir,
    "data",
    "test.csv"
)

if not os.path.exists(test_path):
    raise FileNotFoundError(
        f"Test dataset not found at:\n{test_path}"
    )

test_df = pd.read_csv(test_path)

print("\nTest dataset loaded.")
print("Test rows:", test_df.shape[0])
print("Test columns:", test_df.shape[1])


# -------------------------------------------------
# 3. PREPARE TRAINING DATA
# -------------------------------------------------

X_train = train_df.drop(
    columns=["loss", "id"]
)

y_train = np.log(
    train_df["loss"]
)


# -------------------------------------------------
# 4. PREPARE TEST DATA
# -------------------------------------------------

test_ids = test_df["id"]

X_test = test_df.drop(
    columns=["id"]
)


# -------------------------------------------------
# 5. IDENTIFY FEATURES
# -------------------------------------------------

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))


# -------------------------------------------------
# 6. PREPROCESSING
# -------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ]
)


# -------------------------------------------------
# 7. RANDOM FOREST MODEL
# -------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# -------------------------------------------------
# 8. TRAIN MODEL
# -------------------------------------------------

print("\nTraining Random Forest model...")
print("Please wait...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# -------------------------------------------------
# 9. PREDICT TEST CLAIM COST
# -------------------------------------------------

print("\nPredicting claim costs...")

log_predictions = model.predict(
    X_test
)


# Convert log prediction back to original loss
predicted_loss = np.exp(
    log_predictions
)


# -------------------------------------------------
# 10. CREATE PREDICTION FILE
# -------------------------------------------------

predictions = pd.DataFrame({
    "id": test_ids,
    "loss": predicted_loss
})


output_path = os.path.join(
    project_dir,
    "predictions.csv"
)

predictions.to_csv(
    output_path,
    index=False
)


# -------------------------------------------------
# 11. DISPLAY RESULTS
# -------------------------------------------------

print("\n==========================================")
print("        CLAIM PREDICTION RESULTS")
print("==========================================")

print("Predictions created:", len(predictions))

print("\nFirst 10 predictions:")
print(predictions.head(10).to_string(index=False))

print("\nPrediction file saved at:")
print(output_path)

print("\n==========================================")
print("      CLAIM PREDICTION COMPLETED")
print("==========================================")