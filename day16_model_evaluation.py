
# Week 4 - Day 1
# Model Evaluation Metrics: Precision, Recall, F1, AUC

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report
)

# Create output folder
os.makedirs("data", exist_ok=True)

# 1. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Target Classes:", data.target_names)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 5. Evaluation metrics
results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_prob)
}

print("\nMODEL EVALUATION RESULTS")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Save metrics
pd.DataFrame(
    list(results.items()),
    columns=["Metric", "Score"]
).to_csv("data/day16_evaluation_metrics.csv", index=False)

# 6. Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=data.target_names,
    cmap="Blues",
    ax=ax
)
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("data/day16_confusion_matrix.png")
plt.close()

# 7. ROC Curve
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.set_title("ROC Curve")
plt.tight_layout()
plt.savefig("data/day16_roc_curve.png")
plt.close()

# 8. Precision-Recall Curve
fig, ax = plt.subplots(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.set_title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("data/day16_precision_recall_curve.png")
plt.close()

# 9. Stratified K-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model, X, y,
    cv=cv,
    scoring="accuracy"
)

print("\nSTRATIFIED 5-FOLD CROSS VALIDATION")
print("Fold Scores:", cv_scores)
print("Mean Accuracy:", round(cv_scores.mean(), 4))

pd.DataFrame({
    "Fold": range(1, 6),
    "Accuracy": cv_scores
}).to_csv("data/day16_cross_validation.csv", index=False)

# 10. Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, marker="o", label="Training Score")
plt.plot(train_sizes, test_mean, marker="o", label="Validation Score")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/day16_learning_curve.png")
plt.close()

print("\nAll Day 1 tasks completed successfully!")
print("All output files saved inside the data folder.")
