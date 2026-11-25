
# Week 3 - Day 3
# Topic: Decision Trees & Random Forests

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Decision Tree using Gini
gini_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)
gini_model.fit(X_train, y_train)

gini_pred = gini_model.predict(X_test)

# 5. Decision Tree using Entropy
entropy_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)
entropy_model.fit(X_train, y_train)

entropy_pred = entropy_model.predict(X_test)

# 6. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# 7. Compare model performance
models = {
    "Decision Tree - Gini": (gini_model, gini_pred),
    "Decision Tree - Entropy": (entropy_model, entropy_pred),
    "Random Forest": (rf_model, rf_pred)
}

results = []

for name, (model, predictions) in models.items():

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    print("\n", name)
    print("Accuracy:", round(accuracy, 4))
    print(classification_report(y_test, predictions))

# Save comparison
results_df = pd.DataFrame(results)
results_df.to_csv("data/day13_model_comparison.csv", index=False)

print("\nModel Comparison:")
print(results_df)

# 8. Compare training and testing accuracy
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)

print("\nRandom Forest Training Accuracy:", round(train_accuracy, 4))
print("Random Forest Testing Accuracy:", round(test_accuracy, 4))

# 9. Visualize Decision Tree
plt.figure(figsize=(20, 10))

plot_tree(
    gini_model,
    feature_names=dataset.feature_names,
    class_names=dataset.target_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)

plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.savefig("data/day13_decision_tree.png")
plt.close()

# 10. Random Forest Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.target_names
)

disp.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("data/day13_confusion_matrix.png")
plt.close()

# 11. Feature Importance
importance = pd.DataFrame({
    "Feature": dataset.feature_names,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "data/day13_feature_importance.csv",
    index=False
)

top_features = importance.head(10)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("data/day13_feature_importance.png")
plt.close()

# 12. Final result
print("\nDay 3 completed successfully!")

print("\nGenerated Files:")
print("- data/day13_model_comparison.csv")
print("- data/day13_decision_tree.png")
print("- data/day13_confusion_matrix.png")
print("- data/day13_feature_importance.csv")
print("- data/day13_feature_importance.png")
