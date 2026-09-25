import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Load_data import load_data


# ============================================================
# CLAIMWISE - LINEAR REGRESSION
# ============================================================

print("==========================================")
print("       CLAIMWISE - LINEAR REGRESSION")
print("==========================================")


# ============================================================
# 1. LOAD DATA
# ============================================================

df = load_data()

print("\nDataset loaded successfully.")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["loss", "id"]
)

y = np.log(df["loss"])


# ============================================================
# 3. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nCategorical features:",
      len(categorical_features))

print("Numerical features:",
      len(numerical_features))


# ============================================================
# 4. PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
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


# ============================================================
# 5. CREATE MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:",
      X_train.shape[0])

print("Testing samples:",
      X_test.shape[0])


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining Linear Regression model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 8. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 9. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 10. RESULTS
# ============================================================

print("\n==========================================")
print("       LINEAR REGRESSION RESULTS")
print("==========================================")

print(
    "MAE:",
    round(mae, 4)
)

print(
    "MSE:",
    round(mse, 4)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "R² Score:",
    round(r2, 4)
)

print("\nLinear Regression completed successfully.")