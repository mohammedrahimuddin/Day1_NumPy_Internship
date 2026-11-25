
# ============================================================
# Week 2 - Day 5
# Topic: End-to-End Titanic Preprocessing Pipeline
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Create data folder if it does not exist
os.makedirs("data", exist_ok=True)

# STEP 1: Load Titanic Dataset
file_path = "data/titanic.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        "Titanic dataset not found! "
        "Please place titanic.csv inside the data folder."
    )

df = pd.read_csv(file_path)

print("\nSTEP 1: Dataset Loaded")
print("Dataset Shape:", df.shape)
print(df.head())

# STEP 2: Exploratory Data Analysis
print("\nSTEP 2: Dataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe(include="all"))

print("\nMissing Values")
print(df.isnull().sum())

# Visualize missing values
plt.figure(figsize=(10, 5))
df.isnull().sum().plot(kind="bar")

plt.title("Missing Values in Titanic Dataset")
plt.xlabel("Columns")
plt.ylabel("Missing Values Count")
plt.tight_layout()

plt.savefig("data/day10_missing_values.png", dpi=100)
plt.close()

# STEP 3: Separate Target and Features
y = df["Survived"]

# Remove target and unnecessary columns
drop_columns = [
    "Survived",
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
]

X = df.drop(columns=drop_columns)

# STEP 4: Identify Numeric and Categorical Columns
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "str", "bool"]
).columns.tolist()

print("\nSTEP 4: Feature Information")
print("Numeric Columns:", numeric_columns)
print("Categorical Columns:", categorical_columns)

# STEP 5: Numeric Preprocessing Pipeline
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical Preprocessing Pipeline
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

# STEP 6: Combine Both Pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ],
    sparse_threshold=0
)

# STEP 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nSTEP 7: Train-Test Split")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# STEP 8: Fit and Transform Data
# Fit only on training data to prevent data leakage
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the same fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

# Get processed feature names
feature_names = preprocessor.get_feature_names_out()

# Convert processed arrays into DataFrames
X_train_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)

# Add target column
X_train_df["Survived"] = y_train
X_test_df["Survived"] = y_test

# STEP 9: Export ML-Ready Datasets
X_train_df.to_csv(
    "data/day10_titanic_train_processed.csv",
    index=False
)

X_test_df.to_csv(
    "data/day10_titanic_test_processed.csv",
    index=False
)

# Save preprocessing summary
summary = pd.DataFrame({
    "Dataset": ["Training", "Testing"],
    "Rows": [len(X_train_df), len(X_test_df)],
    "Features": [
        X_train_df.shape[1] - 1,
        X_test_df.shape[1] - 1
    ]
})

summary.to_csv(
    "data/day10_preprocessing_summary.csv",
    index=False
)

# STEP 10: Final Verification
print("\nSTEP 10: Final Dataset Verification")

print("Processed Training Shape:", X_train_df.shape)
print("Processed Testing Shape:", X_test_df.shape)

print("\nRemaining Missing Values in Training:")
print(X_train_df.isnull().sum().sum())

print("\nRemaining Missing Values in Testing:")
print(X_test_df.isnull().sum().sum())

# STEP 11: Display Saved Files
print("\nSTEP 11: Files Saved")
print("1. data/day10_missing_values.png")
print("2. data/day10_titanic_train_processed.csv")
print("3. data/day10_titanic_test_processed.csv")
print("4. data/day10_preprocessing_summary.csv")

print("\nTitanic Preprocessing Pipeline Completed Successfully!")
