
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer, KNNImputer

# Week 2 - Day 2: Missing Data Handling

np.random.seed(42)
n = 100

# STEP 1: Create sample dataset
df = pd.DataFrame({
    "age": np.random.randint(18, 60, n).astype(float),
    "income": np.random.normal(50000, 15000, n).round(0),
    "score": np.random.uniform(0, 100, n).round(1),
    "category": np.random.choice(["A", "B", "C"], n)
})

# STEP 2: Introduce missing values
for column in ["age", "income", "score", "category"]:
    missing_indices = np.random.choice(n, 10, replace=False)
    df.loc[missing_indices, column] = np.nan

print("STEP 1: Original Dataset")
print(df.head(10))
print("\nDataset Shape:", df.shape)

# STEP 3: Check missing values
print("\nSTEP 2: Missing Values Count")
print(df.isnull().sum())

print("\nMissing Values Percentage")
print((df.isnull().mean() * 100).round(2))

# STEP 4: Simple Imputer - Mean Strategy
numeric_columns = ["age", "income", "score"]

mean_imputer = SimpleImputer(strategy="mean")
mean_data = mean_imputer.fit_transform(df[numeric_columns])

mean_df = pd.DataFrame(
    mean_data, columns=numeric_columns
)

print("\nSTEP 3: SimpleImputer - Mean")
print(mean_df.head())

# STEP 5: Simple Imputer - Median Strategy
median_imputer = SimpleImputer(strategy="median")
median_data = median_imputer.fit_transform(df[numeric_columns])

median_df = pd.DataFrame(
    median_data, columns=numeric_columns
)

print("\nSTEP 4: SimpleImputer - Median")
print(median_df.head())

# STEP 6: Categorical Missing Values - Mode
mode_imputer = SimpleImputer(strategy="most_frequent")

category_data = mode_imputer.fit_transform(df[["category"]])

category_df = pd.DataFrame(
    category_data, columns=["category"]
)

print("\nSTEP 5: Categorical Imputation - Mode")
print(category_df.head())

# STEP 7: KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)

knn_data = knn_imputer.fit_transform(df[numeric_columns])

knn_df = pd.DataFrame(
    knn_data, columns=numeric_columns
)

print("\nSTEP 6: KNN Imputer")
print(knn_df.head())

# STEP 8: Create final cleaned dataset
cleaned_df = pd.concat(
    [knn_df, category_df], axis=1
)

print("\nSTEP 7: Cleaned Dataset")
print(cleaned_df.head(10))

print("\nRemaining Missing Values:")
print(cleaned_df.isnull().sum())

# STEP 9: Save cleaned dataset
cleaned_df.to_csv(
    "data/day7_missing_data_cleaned.csv",
    index=False
)

# STEP 10: Plot missing values before and after
before = df.isnull().sum()
after = cleaned_df.isnull().sum()

plt.figure(figsize=(9, 5))

x = np.arange(len(before))
width = 0.35

plt.bar(x - width/2, before.values, width, label="Before")
plt.bar(x + width/2, after.values, width, label="After")

plt.xticks(x, before.index)
plt.xlabel("Columns")
plt.ylabel("Missing Values Count")
plt.title("Missing Values Before and After Imputation")
plt.legend()
plt.tight_layout()

plt.savefig("data/day7_missing_data_comparison.png", dpi=100)
plt.close()

print("\nSTEP 8: Files Saved")
print("CSV: data/day7_missing_data_cleaned.csv")
print("Plot: data/day7_missing_data_comparison.png")

print("\nMissing Data Handling Completed Successfully!")
