
# Week 3 - Day 2
# Topic: Logistic Regression & Classification

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_blobs, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)
from sklearn.preprocessing import label_binarize

os.makedirs("data", exist_ok=True)

# ------------------------------------------------
# 1. Binary Classification Dataset
# ------------------------------------------------

X, y = make_classification(
    n_samples=1000,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# 2. Train Logistic Regression
# ------------------------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("BINARY CLASSIFICATION RESULTS")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# ------------------------------------------------
# 3. Evaluation Metrics
# ------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nAccuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save metrics
metrics_df = pd.DataFrame([{
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC AUC": roc_auc
}])

metrics_df.to_csv("data/day12_binary_metrics.csv", index=False)

# ------------------------------------------------
# 4. Confusion Matrix
# ------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("data/day12_confusion_matrix.png")
plt.close()

# ------------------------------------------------
# 5. ROC Curve
# ------------------------------------------------

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("Logistic Regression ROC Curve")
plt.tight_layout()
plt.savefig("data/day12_roc_curve.png")
plt.close()

# ------------------------------------------------
# 6. Decision Boundary Visualization
# ------------------------------------------------

X_blobs, y_blobs = make_blobs(
    n_samples=400,
    centers=2,
    n_features=2,
    random_state=42,
    cluster_std=2.0
)

boundary_model = LogisticRegression()
boundary_model.fit(X_blobs, y_blobs)

x_min, x_max = X_blobs[:, 0].min() - 1, X_blobs[:, 0].max() + 1
y_min, y_max = X_blobs[:, 1].min() - 1, X_blobs[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

Z = boundary_model.predict(
    np.c_[xx.ravel(), yy.ravel()]
).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(
    X_blobs[:, 0],
    X_blobs[:, 1],
    c=y_blobs,
    cmap="coolwarm",
    edgecolors="black"
)
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig("data/day12_decision_boundary.png")
plt.close()

# ------------------------------------------------
# 7. Multiclass Classification: OvR vs Softmax
# ------------------------------------------------

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris,
    y_iris,
    test_size=0.2,
    random_state=42,
    stratify=y_iris
)

# One-vs-Rest
ovr_model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000)
)
ovr_model.fit(X_train_i, y_train_i)
ovr_pred = ovr_model.predict(X_test_i)

# Softmax / Multinomial Logistic Regression
softmax_model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)
softmax_model.fit(X_train_i, y_train_i)
softmax_pred = softmax_model.predict(X_test_i)

ovr_accuracy = accuracy_score(y_test_i, ovr_pred)
softmax_accuracy = accuracy_score(y_test_i, softmax_pred)

comparison = pd.DataFrame({
    "Method": ["One-vs-Rest", "Softmax"],
    "Accuracy": [ovr_accuracy, softmax_accuracy]
})

comparison.to_csv("data/day12_ovr_softmax_comparison.csv", index=False)

print("\nMULTICLASS COMPARISON")
print(comparison)

print("\nDay 2 completed successfully!")

print("Generated files:")
print("- data/day12_binary_metrics.csv")
print("- data/day12_confusion_matrix.png")
print("- data/day12_roc_curve.png")
print("- data/day12_decision_boundary.png")
print("- data/day12_ovr_softmax_comparison.csv")
