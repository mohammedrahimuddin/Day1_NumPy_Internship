
# Week 3 - Day 1
# Topic: Linear Regression, Ridge and Lasso

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Create output folder
os.makedirs("data", exist_ok=True)

# 1. Load real-world regression dataset
dataset = load_diabetes()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="target")

print("Dataset Shape:", X.shape)
print("\nFeatures:")
print(X.head())

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# 3. Initialize three regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000)
}

results = []
predictions = {}

# 4. Train and evaluate each model
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    })

    print(f"\n{name}")
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("MAE:", round(mae, 2))
    print("R2 Score:", round(r2, 4))

# 5. Save model comparison table
results_df = pd.DataFrame(results)
results_df.to_csv("data/day11_model_comparison.csv", index=False)

print("\nModel Comparison:")
print(results_df.round(3))

# 6. Actual vs Predicted plot (Linear Regression)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions["Linear Regression"], alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Actual vs Predicted")
plt.tight_layout()
plt.savefig("data/day11_actual_vs_predicted.png")
plt.close()

# 7. Residual plot (Linear Regression)
residuals = y_test - predictions["Linear Regression"]

plt.figure(figsize=(8, 5))
plt.scatter(predictions["Linear Regression"], residuals, alpha=0.7)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Linear Regression: Residual Plot")
plt.tight_layout()
plt.savefig("data/day11_residual_plot.png")
plt.close()

print("\nDay 1 completed successfully!")
print("Generated files:")
print("- data/day11_model_comparison.csv")
print("- data/day11_actual_vs_predicted.png")
print("- data/day11_residual_plot.png")
