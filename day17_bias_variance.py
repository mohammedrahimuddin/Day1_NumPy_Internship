
# Week 4 - Day 2
# Bias-Variance Tradeoff & Regularisation

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Create output folder
os.makedirs("data", exist_ok=True)

# 1. Generate regression dataset
X, y = make_regression(
    n_samples=500,
    n_features=8,
    noise=25,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset Shape:", X.shape)
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# 2. Bias-Variance: Compare Decision Tree depths
depths = [1, 2, 3, 5, 8, 12, None]
tree_results = []

for depth in depths:
    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    tree_results.append({
        "Max Depth": str(depth),
        "Training MSE": train_mse,
        "Testing MSE": test_mse
    })

tree_df = pd.DataFrame(tree_results)
print("\nDecision Tree Depth Comparison:")
print(tree_df.to_string(index=False))

tree_df.to_csv(
    "data/day17_bias_variance_comparison.csv",
    index=False
)

# Plot bias-variance comparison
plt.figure(figsize=(9, 5))
plt.plot(
    tree_df["Max Depth"],
    tree_df["Training MSE"],
    marker="o",
    label="Training MSE"
)
plt.plot(
    tree_df["Max Depth"],
    tree_df["Testing MSE"],
    marker="s",
    label="Testing MSE"
)
plt.title("Bias-Variance Tradeoff")
plt.xlabel("Decision Tree Maximum Depth")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/day17_bias_variance_plot.png")
plt.close()

# 3. Ridge and Lasso Regularisation
ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=10000))
])

ridge_pipeline.fit(X_train, y_train)
lasso_pipeline.fit(X_train, y_train)

models = {
    "Ridge": ridge_pipeline,
    "Lasso": lasso_pipeline
}

regularisation_results = []

for name, model in models.items():
    predictions = model.predict(X_test)

    regularisation_results.append({
        "Model": name,
        "MSE": mean_squared_error(y_test, predictions),
        "R2 Score": r2_score(y_test, predictions)
    })

regularisation_df = pd.DataFrame(regularisation_results)

print("\nRidge vs Lasso Comparison:")
print(regularisation_df.to_string(index=False))

regularisation_df.to_csv(
    "data/day17_regularisation_comparison.csv",
    index=False
)

# 4. Grid Search for Ridge
ridge_search_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

grid_parameters = {
    "model__alpha": [0.01, 0.1, 1, 10, 100, 1000]
}

grid_search = GridSearchCV(
    ridge_search_pipeline,
    grid_parameters,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid_search.fit(X_train, y_train)

print("\nGrid Search Best Parameters:")
print(grid_search.best_params_)

print("Grid Search Best CV MSE:")
print(-grid_search.best_score_)

# 5. Randomized Search for Ridge
random_search_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

random_parameters = {
    "model__alpha": np.logspace(-3, 3, 30)
}

random_search = RandomizedSearchCV(
    random_search_pipeline,
    random_parameters,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42
)

random_search.fit(X_train, y_train)

print("\nRandomized Search Best Parameters:")
print(random_search.best_params_)

print("Randomized Search Best CV MSE:")
print(-random_search.best_score_)

# 6. Compare Search Results
search_results = pd.DataFrame([
    {
        "Search Method": "Grid Search",
        "Best Alpha": grid_search.best_params_["model__alpha"],
        "Best CV MSE": -grid_search.best_score_
    },
    {
        "Search Method": "Randomized Search",
        "Best Alpha": random_search.best_params_["model__alpha"],
        "Best CV MSE": -random_search.best_score_
    }
])

print("\nGrid Search vs Randomized Search:")
print(search_results.to_string(index=False))

search_results.to_csv(
    "data/day17_search_comparison.csv",
    index=False
)

# 7. Plot Search Comparison
plt.figure(figsize=(8, 5))
plt.bar(
    search_results["Search Method"],
    search_results["Best CV MSE"]
)
plt.title("Grid Search vs Randomized Search")
plt.ylabel("Best Cross-Validation MSE")
plt.tight_layout()
plt.savefig("data/day17_search_comparison.png")
plt.close()

print("\nAll Week 4 Day 2 tasks completed successfully!")
print("Check the data folder for generated CSV files and graphs.")
