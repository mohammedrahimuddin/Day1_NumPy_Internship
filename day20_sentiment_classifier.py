
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)

os.makedirs("data", exist_ok=True)

# Small educational sentiment dataset
reviews = [
    "This product is amazing and wonderful",
    "I absolutely love this product",
    "Excellent quality and great service",
    "Very happy with my purchase",
    "The experience was fantastic",
    "I really enjoyed using this",
    "Highly recommended product",
    "Amazing performance and quality",
    "Very satisfied with the service",
    "Best purchase I have made",
    "The product works perfectly",
    "I am impressed with the quality",
    "Fantastic value for money",
    "Really useful and easy to use",
    "Wonderful experience overall",
    "Great product and fast delivery",
    "I am extremely pleased",
    "The quality is outstanding",
    "This is worth buying",
    "Very good experience",
    "This product is terrible",
    "I hate this product",
    "Very poor quality",
    "I am disappointed with this",
    "The experience was horrible",
    "Not worth the money",
    "Completely waste of money",
    "Very bad customer service",
    "The product stopped working",
    "I regret buying this",
    "Poor performance and quality",
    "I am unhappy with the purchase",
    "The product is useless",
    "Very frustrating experience",
    "Worst purchase ever",
    "The service was disappointing",
    "I would not recommend this",
    "Low quality and expensive",
    "Very dissatisfied with the product",
    "Bad experience overall"
]

# 1 = Positive, 0 = Negative
labels = [1] * 20 + [0] * 20

df = pd.DataFrame({
    "review": reviews,
    "sentiment": labels
})

df.to_csv("data/day20_sentiment_dataset.csv", index=False)

print("Dataset loaded successfully!")
print(df.head())

# Convert text into numerical features
X = df["review"]
y = df["sentiment"]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

# Train Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)
rf_probabilities = random_forest_model.predict_proba(X_test)[:, 1]

# Evaluation function
def evaluate_model(name, actual, predictions, probabilities):
    print(f"\n{name} Classification Report")
    print(classification_report(
        actual,
        predictions,
        target_names=["Negative", "Positive"],
        zero_division=0
    ))

    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, predictions),
        "Precision": precision_score(actual, predictions, zero_division=0),
        "Recall": recall_score(actual, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(actual, probabilities)
    }

# Compare models
results = []

results.append(evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_predictions,
    logistic_probabilities
))

results.append(evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions,
    rf_probabilities
))

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)

results_df.to_csv(
    "data/day20_model_comparison.csv",
    index=False
)

# Confusion Matrix - Logistic Regression
ConfusionMatrixDisplay.from_predictions(
    y_test,
    logistic_predictions,
    display_labels=["Negative", "Positive"],
    cmap="Blues"
)

plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("data/day20_logistic_confusion_matrix.png")
plt.close()

# Confusion Matrix - Random Forest
ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_predictions,
    display_labels=["Negative", "Positive"],
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("data/day20_random_forest_confusion_matrix.png")
plt.close()

# ROC Curve comparison
RocCurveDisplay.from_predictions(
    y_test,
    logistic_probabilities,
    name="Logistic Regression"
)

RocCurveDisplay.from_predictions(
    y_test,
    rf_probabilities,
    name="Random Forest"
)

plt.title("Sentiment Classifier ROC Curve Comparison")
plt.tight_layout()
plt.savefig("data/day20_roc_comparison.png")
plt.close()

# Accuracy comparison graph
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Logistic Regression vs Random Forest")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("data/day20_model_accuracy.png")
plt.close()

print("\nAll Day 5 outputs generated successfully!")
