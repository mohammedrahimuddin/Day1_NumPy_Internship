
# Week 3 - Day 4
# Topic: SVM, KNN and Ensemble Learning

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Create output folder
os.makedirs("data", exist_ok=True)

# 2. Load dataset
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

print("Dataset Shape:", X.shape)
print("\nTarget Classes:", dataset.target_names)

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create models
models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=1.0))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=50,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42
    )
}

# 5. Train and evaluate models
results = []
predictions = {}

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

    print("\n", name)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(
        y_test,
        y_pred,
        target_names=dataset.target_names
    ))

# 6. Save model comparison
results_df = pd.DataFrame(results)
results_df.to_csv("data/day14_model_comparison.csv", index=False)

print("\nModel Comparison:")
print(results_df.round(4))

# 7. Plot model accuracy
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("SVM, KNN and Ensemble Model Comparison")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("data/day14_model_accuracy.png")
plt.close()

# 8. Confusion matrix for SVM
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions["SVM"],
    display_labels=dataset.target_names,
    cmap="Blues"
)
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("data/day14_svm_confusion_matrix.png")
plt.close()

# 9. Random Forest Feature Importance
rf_model = models["Random Forest"]

importance = pd.DataFrame({
    "Feature": dataset.feature_names,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "data/day14_feature_importance.csv",
    index=False
)

# Plot top 10 important features
top_features = importance.head(10)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest - Top 10 Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("data/day14_feature_importance.png")
plt.close()

# 10. Final result
print("\nDay 4 completed successfully!")

print("\nGenerated Files:")
print("- data/day14_model_comparison.csv")
print("- data/day14_model_accuracy.png")
print("- data/day14_svm_confusion_matrix.png")
print("- data/day14_feature_importance.csv")
print("- data/day14_feature_importance.png")
