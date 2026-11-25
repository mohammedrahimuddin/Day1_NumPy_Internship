
# Week 3 - Day 5
# Topic: Hyperparameter Tuning using GridSearchCV and RandomizedSearchCV

import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from scipy.stats import randint

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

# 4. Create base model
rf_model = RandomForestClassifier(random_state=42)

# 5. Define parameter grid for GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# 6. Cross-validation strategy
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 7. GridSearchCV
print("\nStarting GridSearchCV...")

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print("\nGridSearchCV Results")
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", round(grid_search.best_score_, 4))
print("Execution Time:", round(grid_time, 2), "seconds")

# 8. RandomizedSearchCV
param_distributions = {
    "n_estimators": randint(50, 201),
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 6)
}

print("\nStarting RandomizedSearchCV...")

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

print("\nRandomizedSearchCV Results")
print("Best Parameters:", random_search.best_params_)
print("Best CV Score:", round(random_search.best_score_, 4))
print("Execution Time:", round(random_time, 2), "seconds")

# 9. Evaluate best models on test data
grid_pred = grid_search.best_estimator_.predict(X_test)
random_pred = random_search.best_estimator_.predict(X_test)

grid_accuracy = accuracy_score(y_test, grid_pred)
random_accuracy = accuracy_score(y_test, random_pred)

print("\nGridSearch Test Accuracy:", round(grid_accuracy, 4))
print("RandomizedSearch Test Accuracy:", round(random_accuracy, 4))

print("\nGridSearch Classification Report:")
print(classification_report(y_test, grid_pred))

print("\nRandomizedSearch Classification Report:")
print(classification_report(y_test, random_pred))

# 10. Save comparison results
comparison = pd.DataFrame([
    {
        "Search Method": "GridSearchCV",
        "Best CV Score": grid_search.best_score_,
        "Test Accuracy": grid_accuracy,
        "Execution Time": grid_time
    },
    {
        "Search Method": "RandomizedSearchCV",
        "Best CV Score": random_search.best_score_,
        "Test Accuracy": random_accuracy,
        "Execution Time": random_time
    }
])

comparison.to_csv(
    "data/day15_search_comparison.csv",
    index=False
)

# 11. Save best parameters
best_parameters = pd.DataFrame([
    {
        "Search Method": "GridSearchCV",
        "Best Parameters": str(grid_search.best_params_)
    },
    {
        "Search Method": "RandomizedSearchCV",
        "Best Parameters": str(random_search.best_params_)
    }
])

best_parameters.to_csv(
    "data/day15_best_parameters.csv",
    index=False
)

# 12. Visualize model comparison
plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Search Method"],
    comparison["Test Accuracy"]
)

plt.xlabel("Search Method")
plt.ylabel("Test Accuracy")
plt.title("GridSearchCV vs RandomizedSearchCV")
plt.ylim(0, 1.05)

plt.tight_layout()
plt.savefig("data/day15_search_comparison.png")
plt.close()

# 13. Confusion matrix for best model
if grid_accuracy >= random_accuracy:
    best_model = grid_search.best_estimator_
    best_predictions = grid_pred
    best_name = "GridSearchCV"
else:
    best_model = random_search.best_estimator_
    best_predictions = random_pred
    best_name = "RandomizedSearchCV"

cm = confusion_matrix(y_test, best_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.target_names
)

disp.plot(cmap="Blues")
plt.title(f"Best Model Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("data/day15_confusion_matrix.png")
plt.close()

# 14. Final result
print("\nBest Model:", best_name)
print("\nDay 5 completed successfully!")

print("\nGenerated Files:")
print("- data/day15_search_comparison.csv")
print("- data/day15_best_parameters.csv")
print("- data/day15_search_comparison.png")
print("- data/day15_confusion_matrix.png")
