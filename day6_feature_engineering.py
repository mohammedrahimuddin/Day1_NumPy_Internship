
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)
from sklearn.feature_selection import SelectKBest, f_regression

# Week 2 - Day 1: Feature Engineering and Encoding
np.random.seed(42)

# STEP 1: Create a sample student performance dataset
n = 100

df = pd.DataFrame({
    "study_hours": np.random.uniform(1, 8, n).round(1),
    "attendance": np.random.randint(40, 101, n),
    "assignments": np.random.randint(1, 11, n),
    "department": np.random.choice(
        ["ISE", "CSE", "ECE", "ME"], n
    ),
    "performance": np.random.choice(
        ["Low", "Medium", "High"], n
    )
})

df["final_score"] = (
    df["study_hours"] * 6
    + df["attendance"] * 0.3
    + df["assignments"] * 2
    + np.random.normal(0, 5, n)
).clip(0, 100).round(1)

print("STEP 1: Original Dataset")
print(df.head())
print("\nDataset shape:", df.shape)

# STEP 2: Label Encoding
label_encoder = LabelEncoder()
df["performance_encoded"] = label_encoder.fit_transform(
    df["performance"]
)

print("\nSTEP 2: Label Encoding")
print(df[["performance", "performance_encoded"]].head())

# STEP 3: One-Hot Encoding
one_hot = pd.get_dummies(
    df["department"], prefix="department", dtype=int
)

print("\nSTEP 3: One-Hot Encoding")
print(one_hot.head())

# STEP 4: Ordinal Encoding
ordinal_encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)

df["performance_ordinal"] = ordinal_encoder.fit_transform(
    df[["performance"]]
).astype(int)

print("\nSTEP 4: Ordinal Encoding")
print(df[["performance", "performance_ordinal"]].head())

# STEP 5: Feature Scaling
numeric_features = df[
    ["study_hours", "attendance", "assignments"]
]

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

standard_scaled = standard_scaler.fit_transform(numeric_features)
minmax_scaled = minmax_scaler.fit_transform(numeric_features)
robust_scaled = robust_scaler.fit_transform(numeric_features)

print("\nSTEP 5: Feature Scaling")
print("StandardScaler - first 5 rows:")
print(pd.DataFrame(
    standard_scaled, columns=numeric_features.columns
).head().round(2))

print("\nMinMaxScaler - first 5 rows:")
print(pd.DataFrame(
    minmax_scaled, columns=numeric_features.columns
).head().round(2))

print("\nRobustScaler - first 5 rows:")
print(pd.DataFrame(
    robust_scaled, columns=numeric_features.columns
).head().round(2))

# STEP 6: Select the top 5 features
encoded_df = pd.concat(
    [numeric_features, one_hot],
    axis=1
)

selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(encoded_df, df["final_score"])

feature_scores = pd.DataFrame({
    "Feature": encoded_df.columns,
    "Score": selector.scores_
}).sort_values(by="Score", ascending=False)

print("\nSTEP 6: Top 5 Features")
print(feature_scores.head(5).to_string(index=False))

# STEP 7: Plot feature importance scores
top_features = feature_scores.head(5).sort_values(
    by="Score", ascending=True
)

plt.figure(figsize=(9, 5))
plt.barh(top_features["Feature"], top_features["Score"])
plt.xlabel("F-Regression Score")
plt.title("Top 5 Features for Predicting Final Score")
plt.tight_layout()
plt.savefig("data/day6_top_features.png", dpi=100)
plt.close()

# STEP 8: Save the processed dataset
processed_df = pd.concat(
    [df, one_hot],
    axis=1
)

processed_df.to_csv("data/day6_feature_engineered.csv", index=False)

print("\nSTEP 7: Files Saved")
print("Feature plot: data/day6_top_features.png")
print("Processed dataset: data/day6_feature_engineered.csv")
print("\nFeature engineering completed successfully!")
