
# Week 2 - Day 4
# Topic: Train/Test Split & Cross-Validation

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("data", exist_ok=True)

# STEP 1: Create sample classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=5,
    n_redundant=2,
    weights=[0.70, 0.30],
    random_state=42
)

feature_names = [f"feature_{i}" for i in range(1, 9)]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("STEP 1: Dataset Information")
print("Dataset Shape:", df.shape)
print(df.head())

# STEP 2: Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# STEP 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nSTEP 3: Train-Test Split")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 4: Apply different scaling methods
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

scaling_results = []

for name, scaler in scalers.items():

    # Fit only on training data to avoid data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaling_results.append({
        "Scaler": name,
        "Train Mean": np.mean(X_train_scaled),
        "Train Std": np.std(X_train_scaled),
        "Test Mean": np.mean(X_test_scaled)
    })

    print(f"\n{name}")
    print("Scaled training shape:", X_train_scaled.shape)
    print("Scaled testing shape:", X_test_scaled.shape)

scaling_df = pd.DataFrame(scaling_results)
print("\nSTEP 4: Scaling Comparison")
print(scaling_df.round(3))

# STEP 5: Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nSTEP 5: Model Evaluation")
print("Test Accuracy:", round(accuracy, 4))
print(classification_report(y_test, y_pred))

# STEP 6: K-Fold Cross Validation
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

kfold_scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("\nSTEP 6: K-Fold Cross Validation")
print("Fold Scores:", kfold_scores)
print("Average Accuracy:", round(kfold_scores.mean(), 4))

# STEP 7: Stratified K-Fold Cross Validation
stratified_kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

stratified_scores = cross_val_score(
    model,
    X,
    y,
    cv=stratified_kfold,
    scoring="accuracy"
)

print("\nSTEP 7: Stratified K-Fold")
print("Fold Scores:", stratified_scores)
print("Average Accuracy:", round(stratified_scores.mean(), 4))

# STEP 8: Save cross-validation results
cv_results = pd.DataFrame({
    "Fold": range(1, 6),
    "KFold Accuracy": kfold_scores,
    "Stratified KFold Accuracy": stratified_scores
})

cv_results.to_csv(
    "data/day9_cross_validation_results.csv",
    index=False
)

scaling_df.to_csv(
    "data/day9_scaling_comparison.csv",
    index=False
)

# STEP 9: Visualize cross-validation scores
plt.figure(figsize=(9, 5))

plt.plot(
    range(1, 6),
    kfold_scores,
    marker="o",
    label="K-Fold"
)

plt.plot(
    range(1, 6),
    stratified_scores,
    marker="s",
    label="Stratified K-Fold"
)

plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.title("Cross-Validation Accuracy Comparison")
plt.xticks(range(1, 6))
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "data/day9_cross_validation.png",
    dpi=100
)
plt.close()

print("\nSTEP 9: Files Saved")
print("1. data/day9_cross_validation_results.csv")
print("2. data/day9_scaling_comparison.csv")
print("3. data/day9_cross_validation.png")

print("\nTrain/Test Split & Cross-Validation Completed Successfully!")
