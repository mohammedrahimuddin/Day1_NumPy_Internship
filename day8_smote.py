
# Week 2 - Day 3
# Topic: Handling Imbalanced Data using SMOTE

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from imblearn.over_sampling import SMOTE

# Create output folder
os.makedirs("data", exist_ok=True)

# STEP 1: Create an imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_clusters_per_class=1,
    weights=[0.90, 0.10],
    flip_y=0,
    random_state=42
)

print("STEP 1: Original Dataset")
print("Dataset Shape:", X.shape)

# STEP 2: Check class distribution
unique, counts = np.unique(y, return_counts=True)

print("\nSTEP 2: Original Class Distribution")
for label, count in zip(unique, counts):
    print(f"Class {label}: {count} records")

# STEP 3: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nSTEP 3: Train-Test Split")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# STEP 4: Apply SMOTE only to training data
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

print("\nSTEP 4: After Applying SMOTE")
unique_after, counts_after = np.unique(
    y_train_resampled, return_counts=True
)

for label, count in zip(unique_after, counts_after):
    print(f"Class {label}: {count} records")

# STEP 5: Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].bar(
    unique,
    counts,
    tick_label=["Class 0", "Class 1"]
)
axes[0].set_title("Before SMOTE")
axes[0].set_ylabel("Number of Records")

axes[1].bar(
    unique_after,
    counts_after,
    tick_label=["Class 0", "Class 1"]
)
axes[1].set_title("After SMOTE")
axes[1].set_ylabel("Number of Records")

plt.tight_layout()
plt.savefig("data/day8_class_distribution.png", dpi=100)
plt.close()

# STEP 6: Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_resampled, y_train_resampled)

# STEP 7: Make predictions
y_pred = model.predict(X_test)

print("\nSTEP 7: Model Evaluation")
print(classification_report(y_test, y_pred))

# STEP 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

display.plot(cmap="Blues")
plt.title("Confusion Matrix - SMOTE")
plt.tight_layout()
plt.savefig("data/day8_confusion_matrix.png", dpi=100)
plt.close()

# STEP 9: Save balanced training dataset
feature_columns = [
    f"feature_{i}" for i in range(X_train_resampled.shape[1])
]

balanced_df = pd.DataFrame(
    X_train_resampled,
    columns=feature_columns
)

balanced_df["target"] = y_train_resampled

balanced_df.to_csv(
    "data/day8_smote_balanced_train.csv",
    index=False
)

print("\nSTEP 9: Files Saved")
print("1. data/day8_class_distribution.png")
print("2. data/day8_confusion_matrix.png")
print("3. data/day8_smote_balanced_train.csv")

print("\nSMOTE Implementation Completed Successfully!")
