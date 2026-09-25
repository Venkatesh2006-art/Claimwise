import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from Load_data import load_data


print("==========================================")
print("       CLAIMWISE - FINAL MODEL COMPARISON")
print("==========================================")


# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------

df = load_data()

print("\nDataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# -------------------------------------------------
# 2. FEATURES AND TARGET
# -------------------------------------------------

X = df.drop(columns=["loss", "id"])
y = np.log(df["loss"])


# -------------------------------------------------
# 3. IDENTIFY FEATURES
# -------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))


# -------------------------------------------------
# 4. PREPROCESSING
# -------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
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
# 5. TRAIN / TEST SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# -------------------------------------------------
# 6. LINEAR REGRESSION
# -------------------------------------------------

print("\nTraining Linear Regression...")

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_pred)

print("Linear Regression completed.")


# -------------------------------------------------
# 7. RIDGE REGRESSION
# -------------------------------------------------

print("\nTraining Ridge Regression...")

ridge_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ]
)

ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, ridge_pred)

print("Ridge Regression completed.")


# -------------------------------------------------
# 8. RANDOM FOREST
# -------------------------------------------------

print("\nTraining Random Forest...")
print("Please wait...")

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
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

random_forest_model.fit(X_train, y_train)

rf_pred = random_forest_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest completed.")


# -------------------------------------------------
# 9. CREATE COMPARISON TABLE
# -------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Random Forest"
    ],

    "MAE": [
        linear_mae,
        ridge_mae,
        rf_mae
    ],

    "MSE": [
        linear_mse,
        ridge_mse,
        rf_mse
    ],

    "RMSE": [
        linear_rmse,
        ridge_rmse,
        rf_rmse
    ],

    "R2 Score": [
        linear_r2,
        ridge_r2,
        rf_r2
    ]
})


# -------------------------------------------------
# 10. DISPLAY RESULTS
# -------------------------------------------------

print("\n==========================================")
print("          FINAL MODEL COMPARISON")
print("==========================================")

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# -------------------------------------------------
# 11. SAVE RESULTS
# -------------------------------------------------

results.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\nResults saved to:")
print("model_comparison_results.csv")

print("\n==========================================")
print("       MODEL COMPARISON COMPLETED")
print("==========================================")