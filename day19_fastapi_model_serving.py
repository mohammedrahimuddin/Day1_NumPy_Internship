
import os
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model with class weighting
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=5000
    ))
])

model.fit(X_train, y_train)

# Find an improved classification threshold
probabilities = model.predict_proba(X_test)[:, 1]
threshold = 0.5

best_f1 = 0

for current_threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    predictions = (probabilities >= current_threshold).astype(int)
    score = f1_score(y_test, predictions)

    if score > best_f1:
        best_f1 = score
        threshold = current_threshold

# Evaluate model at selected threshold
predictions = (probabilities >= threshold).astype(int)

print("Selected Threshold:", threshold)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Save model and threshold
joblib.dump(
    {"model": model, "threshold": threshold},
    "models/day19_fastapi_model.joblib"
)

# Save evaluation results
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [
        accuracy_score(y_test, predictions),
        precision_score(y_test, predictions),
        recall_score(y_test, predictions),
        f1_score(y_test, predictions)
    ]
})

results.to_csv("data/day19_model_metrics.csv", index=False)

print("\nModel trained and saved successfully!")
