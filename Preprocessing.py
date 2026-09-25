import os
import pandas as pd

from Load_data import load_data


# =========================================================
# CLAIMWISE - DATA PREPROCESSING
# =========================================================

print("==========================================")
print("       CLAIMWISE - DATA PREPROCESSING")
print("==========================================")


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

PROCESSED_DIR = os.path.join(
    DATA_DIR,
    "processed"
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# =========================================================
# LOAD TRAINING DATA
# =========================================================

print("\nLoading training dataset...")

train_path = os.path.join(
    DATA_DIR,
    "train.csv"
)

if not os.path.exists(train_path):

    raise FileNotFoundError(
        f"Training dataset not found:\n{train_path}"
    )


train_df = pd.read_csv(train_path)


print("Training dataset loaded.")
print("Rows:", train_df.shape[0])
print("Columns:", train_df.shape[1])


# =========================================================
# LOAD TEST DATA
# =========================================================

print("\nLoading test dataset...")

test_path = os.path.join(
    DATA_DIR,
    "test.csv"
)

if not os.path.exists(test_path):

    raise FileNotFoundError(
        f"Test dataset not found:\n{test_path}"
    )


test_df = pd.read_csv(test_path)


print("Test dataset loaded.")
print("Rows:", test_df.shape[0])
print("Columns:", test_df.shape[1])


# =========================================================
# ORIGINAL DATASET INFORMATION
# =========================================================

print("\n==========================================")
print("DATASET INFORMATION")
print("==========================================")


print("\nTraining shape:")
print(train_df.shape)


print("\nTest shape:")
print(test_df.shape)


print("\nTarget column:")

if "loss" in train_df.columns:

    print("loss")

else:

    print("Target column not found!")


# =========================================================
# IDENTIFY ID
# =========================================================

if "id" in train_df.columns:

    print("\nID column found:")
    print("id")

else:

    print("\nID column not found.")


# =========================================================
# IDENTIFY CATEGORICAL FEATURES
# =========================================================

categorical_features = train_df.select_dtypes(
    include=["object", "str"]
).columns.tolist()


# Remove target if accidentally detected
if "loss" in categorical_features:

    categorical_features.remove("loss")


# Remove ID if accidentally detected
if "id" in categorical_features:

    categorical_features.remove("id")


# =========================================================
# IDENTIFY NUMERICAL FEATURES
# =========================================================

numerical_features = train_df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# Remove ID
if "id" in numerical_features:

    numerical_features.remove("id")


# Remove target
if "loss" in numerical_features:

    numerical_features.remove("loss")


print("\n==========================================")
print("FEATURE INFORMATION")
print("==========================================")


print(
    "\nCategorical features:",
    len(categorical_features)
)


print(
    "Numerical features:",
    len(numerical_features)
)


print("\nCategorical columns:")

for column in categorical_features:

    print(column)


print("\nNumerical columns:")

for column in numerical_features:

    print(column)


# =========================================================
# MISSING VALUE ANALYSIS
# =========================================================

print("\n==========================================")
print("MISSING VALUE ANALYSIS")
print("==========================================")


train_missing = train_df.isnull().sum()

train_missing = train_missing[
    train_missing > 0
]


if train_missing.empty:

    print("\nNo missing values found in training data.")

else:

    print("\nMissing values in training data:")

    print(train_missing)


# =========================================================
# DUPLICATE ANALYSIS
# =========================================================

print("\n==========================================")
print("DUPLICATE ANALYSIS")
print("==========================================")


train_duplicates = train_df.duplicated().sum()

test_duplicates = test_df.duplicated().sum()


print(
    "\nDuplicate rows in training data:",
    train_duplicates
)


print(
    "Duplicate rows in test data:",
    test_duplicates
)


# =========================================================
# TARGET ANALYSIS
# =========================================================

print("\n==========================================")
print("TARGET ANALYSIS")
print("==========================================")


if "loss" in train_df.columns:

    print(
        "\nLoss mean:",
        round(train_df["loss"].mean(), 2)
    )

    print(
        "Loss median:",
        round(train_df["loss"].median(), 2)
    )

    print(
        "Loss minimum:",
        round(train_df["loss"].min(), 2)
    )

    print(
        "Loss maximum:",
        round(train_df["loss"].max(), 2)
    )


# =========================================================
# REMOVE ID FROM MODEL DATA
# =========================================================

print("\n==========================================")
print("FEATURE PREPARATION")
print("==========================================")


X_train = train_df.drop(
    columns=["loss", "id"],
    errors="ignore"
)


y_train = train_df["loss"]


X_test = test_df.drop(
    columns=["id"],
    errors="ignore"
)


print(
    "\nTraining features after removing ID and target:",
    X_train.shape
)


print(
    "Test features after removing ID:",
    X_test.shape
)


# =========================================================
# HANDLE MISSING VALUES FOR BASIC CLEAN DATA
# =========================================================

print("\n==========================================")
print("MISSING VALUE HANDLING")
print("==========================================")


# Numerical columns:
# fill missing values using training median

for column in numerical_features:

    if column in X_train.columns:

        median_value = X_train[column].median()

        X_train[column] = X_train[column].fillna(
            median_value
        )

        if column in X_test.columns:

            X_test[column] = X_test[column].fillna(
                median_value
            )


# Categorical columns:
# fill missing values using training mode

for column in categorical_features:

    if column in X_train.columns:

        mode_values = X_train[column].mode()

        if len(mode_values) > 0:

            mode_value = mode_values.iloc[0]

        else:

            mode_value = "Unknown"


        X_train[column] = X_train[column].fillna(
            mode_value
        )

        if column in X_test.columns:

            X_test[column] = X_test[column].fillna(
                mode_value
            )


print(
    "\nMissing values handled successfully."
)


# =========================================================
# VERIFY CLEAN DATA
# =========================================================

print("\n==========================================")
print("PREPROCESSING VERIFICATION")
print("==========================================")


remaining_train_missing = (
    X_train.isnull().sum().sum()
)


remaining_test_missing = (
    X_test.isnull().sum().sum()
)


print(
    "\nRemaining missing values in training:",
    remaining_train_missing
)


print(
    "Remaining missing values in test:",
    remaining_test_missing
)


# =========================================================
# SAVE CLEAN DATA
# =========================================================

print("\n==========================================")
print("SAVING PREPROCESSED DATA")
print("==========================================")


# Add ID back separately for reference
train_ids = train_df["id"]

test_ids = test_df["id"]


# Save cleaned training features
clean_train_features = X_train.copy()

clean_train_features.insert(
    0,
    "id",
    train_ids
)

clean_train_features["loss"] = y_train.values


clean_train_path = os.path.join(
    PROCESSED_DIR,
    "train_clean.csv"
)


clean_train_features.to_csv(
    clean_train_path,
    index=False
)


# Save cleaned test features
clean_test_features = X_test.copy()

clean_test_features.insert(
    0,
    "id",
    test_ids
)


clean_test_path = os.path.join(
    PROCESSED_DIR,
    "test_clean.csv"
)


clean_test_features.to_csv(
    clean_test_path,
    index=False
)


# =========================================================
# SAVE FEATURE INFORMATION
# =========================================================

feature_info = pd.DataFrame({

    "feature": (
        categorical_features
        + numerical_features
    ),

    "type": (
        ["categorical"] * len(categorical_features)
        + ["numerical"] * len(numerical_features)
    )

})


feature_info_path = os.path.join(
    PROCESSED_DIR,
    "feature_information.csv"
)


feature_info.to_csv(
    feature_info_path,
    index=False
)


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n==========================================")
print("PREPROCESSING COMPLETED")
print("==========================================")


print("\nFiles created:")

print(
    "\n1. train_clean.csv"
)

print(
    "2. test_clean.csv"
)

print(
    "3. feature_information.csv"
)


print("\nLocation:")

print(PROCESSED_DIR)


print("\nFinal training shape:")

print(clean_train_features.shape)


print("\nFinal test shape:")

print(clean_test_features.shape)


print("\nClaimWise preprocessing completed successfully.")