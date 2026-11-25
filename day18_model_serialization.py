
import os
import pickle
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

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

# Build model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

# Train model
model.fit(X_train, y_train)

# Save model using Joblib
joblib.dump(model, "models/day18_model.joblib")

# Save model using Pickle
with open("models/day18_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully using Joblib and Pickle!")

# Load model using Joblib
loaded_joblib_model = joblib.load("models/day18_model.joblib")

# Load model using Pickle
with open("models/day18_model.pkl", "rb") as file:
    loaded_pickle_model = pickle.load(file)

# Predictions using loaded models
joblib_predictions = loaded_joblib_model.predict(X_test)
pickle_predictions = loaded_pickle_model.predict(X_test)

# Prediction probabilities
y_probability = loaded_joblib_model.predict_proba(X_test)[:, 1]

# Verify predictions
print("\nJoblib and Pickle Prediction Match:",
      (joblib_predictions == pickle_predictions).all())

# Evaluation metrics
metrics = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    "Score": [
        accuracy_score(y_test, joblib_predictions),
        precision_score(y_test, joblib_predictions),
        recall_score(y_test, joblib_predictions),
        f1_score(y_test, joblib_predictions),
        roc_auc_score(y_test, y_probability)
    ]
}

metrics_df = pd.DataFrame(metrics)

print("\nModel Evaluation Results:")
print(metrics_df)

metrics_df.to_csv("data/day18_model_metrics.csv", index=False)

# Confusion Matrix
cm = confusion_matrix(y_test, joblib_predictions)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.target_names
).plot(cmap="Blues")

plt.title("Day 18 - Confusion Matrix")
plt.tight_layout()
plt.savefig("data/day18_confusion_matrix.png")
plt.close()

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_probability)
plt.title("Day 18 - ROC Curve")
plt.tight_layout()
plt.savefig("data/day18_roc_curve.png")
plt.close()

# Precision Recall Curve
PrecisionRecallDisplay.from_predictions(y_test, y_probability)
plt.title("Day 18 - Precision Recall Curve")
plt.tight_layout()
plt.savefig("data/day18_precision_recall_curve.png")
plt.close()

print("\nAll outputs generated successfully!")
print("Saved models in models/ folder.")
print("Saved evaluation results and graphs in data/ folder.")
