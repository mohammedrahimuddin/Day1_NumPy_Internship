
import os
import pandas as pd
from sklearn.datasets import fetch_openml

# Create data folder
os.makedirs("data", exist_ok=True)

print("Downloading UCI Adult Income dataset...")

# Download dataset
adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

df = adult.frame.copy()

# Save dataset locally
df.to_csv("data/day21_adult_income.csv", index=False)

print("\nDataset downloaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nGender distribution:")
print(df["sex"].value_counts())

print("\nIncome distribution:")
print(df["class"].value_counts())

# Gender-wise income fairness analysis

# Clean income labels
df["income_clean"] = (
    df["class"].astype(str).str.strip().str.replace(".", "", regex=False)
)

# Calculate positive income rate for each gender
gender_analysis = df.groupby("sex").agg(
    total_people=("income_clean", "count"),
    high_income=("income_clean", lambda x: (x == ">50K").sum())
)

gender_analysis["high_income_rate"] = (
    gender_analysis["high_income"] /
    gender_analysis["total_people"]
)

print("\nGender-wise Income Analysis:")
print(gender_analysis)

# Calculate Disparate Impact
male_rate = gender_analysis.loc["Male", "high_income_rate"]
female_rate = gender_analysis.loc["Female", "high_income_rate"]

disparate_impact = female_rate / male_rate

print("\nDisparate Impact Ratio:", round(disparate_impact, 4))

if disparate_impact < 0.8:
    print("Potential fairness concern detected.")
else:
    print("Disparate Impact is at or above 0.8.")

# Save fairness results
gender_analysis.to_csv("data/day21_gender_fairness.csv")

print("\nGender fairness results saved successfully!")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt

# Prepare features and target
X = df.drop(columns=["class", "income_clean"])
y = (df["income_clean"] == ">50K").astype(int)

# Convert categorical features into numerical values
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
import joblib
joblib.dump(model, "models/day21_logistic_model.joblib")

# SHAP Explainability
print("\nGenerating SHAP explanation...")

# Use a small sample to keep explanation faster
X_sample = X_train.sample(
    n=min(200, len(X_train)),
    random_state=42
)

scaled_sample = model.named_steps["scaler"].transform(X_sample)

explainer = shap.LinearExplainer(
    model.named_steps["classifier"],
    model.named_steps["scaler"].transform(
        X_train.sample(n=min(100, len(X_train)), random_state=1)
    )
)

shap_values = explainer(scaled_sample)

plt.figure()
shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()
plt.savefig("data/day21_shap_summary.png", bbox_inches="tight")
plt.close()

print("\nSHAP summary plot saved successfully!")

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from sklearn.metrics import accuracy_score

print("\nStarting AIF360 Reweighing...")

# Prepare fairness dataset
fair_df = df.copy()

# Convert target into binary labels
fair_df["income_label"] = (
    fair_df["income_clean"] == ">50K"
).astype(int)

# Keep required columns
fair_df = fair_df[["sex", "income_label"]].copy()

# Encode gender
fair_df["sex"] = fair_df["sex"].map({
    "Male": 1,
    "Female": 0
})

# Create AIF360 dataset
dataset_aif = BinaryLabelDataset(
    df=fair_df,
    label_names=["income_label"],
    protected_attribute_names=["sex"],
    favorable_label=1,
    unfavorable_label=0
)

# Define privileged and unprivileged groups
privileged_groups = [{"sex": 1}]
unprivileged_groups = [{"sex": 0}]

# Apply Reweighing
reweighing = Reweighing(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

transformed_dataset = reweighing.fit_transform(dataset_aif)

# Display sample weights
print("\nOriginal dataset records:", len(dataset_aif.features))
print("Transformed dataset records:", len(transformed_dataset.features))

print("\nSample weights:")
print(transformed_dataset.instance_weights[:10])

print("\nReweighing completed successfully!")

# Save weights
pd.DataFrame({
    "gender": fair_df["sex"],
    "income_label": fair_df["income_label"],
    "sample_weight": transformed_dataset.instance_weights
}).to_csv("data/day21_reweighing_weights.csv", index=False)

print("Reweighing weights saved successfully!")

from sklearn.base import clone

print("\nTraining model with Reweighing weights...")

# Convert AIF360 weights into a pandas Series
sample_weights = pd.Series(
    transformed_dataset.instance_weights,
    index=df.index
)

# Get training weights using training indexes
train_weights = sample_weights.loc[X_train.index]

# Create a new Logistic Regression pipeline
reweighed_model = clone(model)

# Train model using reweighing sample weights
reweighed_model.fit(
    X_train,
    y_train,
    classifier__sample_weight=train_weights
)

# Make predictions
reweighed_predictions = reweighed_model.predict(X_test)

# Calculate accuracy
reweighed_accuracy = accuracy_score(
    y_test,
    reweighed_predictions
)

# Calculate gender-wise predicted positive rates
test_gender = df.loc[X_test.index, "sex"]

female_rate = (
    reweighed_predictions[test_gender.values == "Female"] == 1
).mean()

male_rate = (
    reweighed_predictions[test_gender.values == "Male"] == 1
).mean()

reweighed_di = female_rate / male_rate if male_rate > 0 else 0

print("\nReweighed Model Results")
print("Accuracy:", round(reweighed_accuracy, 4))
print("Female predicted positive rate:", round(female_rate, 4))
print("Male predicted positive rate:", round(male_rate, 4))
print("Disparate Impact Ratio:", round(reweighed_di, 4))

# Save comparison
comparison = pd.DataFrame({
    "Model": ["Original", "Reweighed"],
    "Accuracy": [
        accuracy_score(y_test, model.predict(X_test)),
        reweighed_accuracy
    ],
    "Disparate_Impact": [
        (
            (model.predict(X_test)[test_gender.values == "Female"] == 1).mean()
            /
            (model.predict(X_test)[test_gender.values == "Male"] == 1).mean()
        ),
        reweighed_di
    ]
})

print("\nFairness Comparison:")
print(comparison)

comparison.to_csv(
    "data/day21_fairness_comparison.csv",
    index=False
)

joblib.dump(
    reweighed_model,
    "models/day21_reweighed_model.joblib"
)

print("\nReweighed model saved successfully!")

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from xml.sax.saxutils import escape

print("\nCreating Model Card PDF and India Bias Case Analysis...")

# Read actual fairness comparison results
fairness_results = pd.read_csv(
    "data/day21_fairness_comparison.csv"
)

# Prepare actual model results
original_row = fairness_results[
    fairness_results["Model"] == "Original"
].iloc[0]

reweighed_row = fairness_results[
    fairness_results["Model"] == "Reweighed"
].iloc[0]

# Create Model Card PDF
pdf_path = "data/day21_model_card.pdf"

document = SimpleDocTemplate(
    pdf_path,
    pagesize=A4
)

styles = getSampleStyleSheet()
story = []

story.append(Paragraph(
    "MODEL CARD - RESPONSIBLE AI",
    styles["Title"]
))
story.append(Spacer(1, 15))

sections = [
    ("Model Name", "Adult Income Classification - Logistic Regression"),
    ("Purpose", "Educational binary classification of income categories."),
    ("Dataset", "UCI Adult Income dataset containing 48,842 records."),
    ("Target", "Income greater than 50K or income less than or equal to 50K."),
    ("Features", "Age, education, occupation, workclass, gender and other demographic and employment attributes."),
    ("Training", "Logistic Regression with StandardScaler and class_weight='balanced'."),
    ("Explainability", "SHAP LinearExplainer used to understand feature contributions."),
    ("Bias Detection", "Gender-wise income rates and Disparate Impact analysis."),
    ("Bias Mitigation", "AIF360 Reweighing used to calculate training sample weights."),
    ("Limitations", "This is an educational model, not suitable for real hiring, lending or financial decisions.")
]

for heading, description in sections:
    story.append(Paragraph(
        f"<b>{escape(heading)}:</b> {escape(description)}",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 8))

story.append(Spacer(1, 12))
story.append(Paragraph(
    "Model Performance and Fairness Comparison",
    styles["Heading2"]
))

table_data = [
    ["Metric", "Original", "Reweighed"],
    [
        "Accuracy",
        f"{original_row['Accuracy']:.4f}",
        f"{reweighed_row['Accuracy']:.4f}"
    ],
    [
        "Disparate Impact",
        f"{original_row['Disparate_Impact']:.4f}",
        f"{reweighed_row['Disparate_Impact']:.4f}"
    ]
]

table = Table(table_data, colWidths=[150, 120, 120])

table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("PADDING", (0, 0), (-1, -1), 8),
    ("ALIGN", (1, 1), (-1, -1), "CENTER")
]))

story.append(table)
story.append(Spacer(1, 15))

story.append(Paragraph(
    "Fairness Note: A Disparate Impact ratio below 0.8 is commonly "
    "used as a screening signal, not proof of discrimination. "
    "Reweighing results must be evaluated carefully alongside "
    "accuracy, group-level errors, data quality and social context.",
    styles["BodyText"]
))

document.build(story)

# Create India AI bias case analysis
case_analysis = """
# India AI Bias Case Analysis: Facial Recognition Technology

A published 2025 research article by Intifada P. Basheer,
"Bias in the Algorithm: Issues Raised Due to Use of Facial
Recognition in India", discusses gender bias and accuracy
concerns in facial recognition technology used in India.

The article describes an audit of four facial-processing tools
tested on 32,184 images of Indian electoral candidates. The
audit found that error rates for female subjects were higher
than those for male subjects. For example, Face++ recorded
a 14.68% error rate for female subjects, while FaceX recorded
10.47%. The article also discusses the Delhi Police facial
recognition system used to trace missing children, which was
reported to correctly match less than 1% of missing children.

Root causes include unrepresentative training data, differences
in image quality, algorithm design limitations and insufficient
testing across demographic groups. Localising a dataset alone
does not guarantee fairness. Limited transparency and weak
independent evaluation can make these problems harder to detect.

Proposed solutions include testing systems on diverse Indian
demographic groups, publishing subgroup error rates, conducting
independent audits and improving dataset quality. Human review
should be mandatory before consequential decisions. Sensitive
facial data must also be protected, and systems should have
clear limits on their use.

This case demonstrates that AI fairness requires more than
high overall accuracy. Developers must evaluate who benefits,
who experiences errors and whether the system is appropriate
for its intended purpose.
 
Source:
Basheer, I. P. (2025). Bias in the Algorithm: Issues Raised
Due to Use of Facial Recognition in India.
Journal of Development Policy and Practice, 10(1), 61-79.
https://doi.org/10.1177/24551333241283992
"""

with open(
    "day21_india_bias_case.md",
    "w",
    encoding="utf-8"
) as file:
    file.write(case_analysis.strip())

print("\nModel Card PDF created:", pdf_path)
print("India bias case analysis created: day21_india_bias_case.md")
print("\nDay 6 final documents created successfully!")
