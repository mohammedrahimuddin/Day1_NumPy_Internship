import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# DAY 4 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Create reproducible student dataset
np.random.seed(42)

df = pd.DataFrame({
    "student_id": range(1, 21),
    "name": [f"Student_{i}" for i in range(1, 21)],
    "age": np.random.randint(20, 26, 20),
    "math": np.random.randint(50, 100, 20),
    "python": np.random.randint(45, 100, 20),
    "ml_score": np.random.randint(40, 100, 20),
    "attended": np.random.choice(
        [True, False], 20, p=[0.8, 0.2]
    ),
    "department": np.random.choice(
        ["ISE", "CSE", "ECE", "AIML", "DS"], 20
    )
})

# Introduce missing values for inspection
df.loc[[3, 7, 14], "ml_score"] = np.nan

# ============================================================
# 1. DATA INSPECTION
# ============================================================

print("=== DATASET OVERVIEW ===")
print(df.head())

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== DATA INFORMATION ===")
df.info()

print("\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

# Fill missing ML scores using the median
df["ml_score"] = df["ml_score"].fillna(df["ml_score"].median())

print("\n=== MISSING VALUES AFTER CLEANING ===")
print(df.isnull().sum())


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Calculate average score across subjects
df["avg_score"] = (
    df[["math", "python", "ml_score"]]
    .mean(axis=1)
    .round(1)
)

# Create grade categories
df["grade"] = pd.cut(
    df["avg_score"],
    bins=[0, 59, 74, 89, 100],
    labels=["F", "C", "B", "A"]
)

print("\n=== GRADE DISTRIBUTION ===")
print(df["grade"].value_counts().sort_index())


# ============================================================
# 4. FIVE EDA OBSERVATIONS
# ============================================================

print("\n=== FIVE EDA OBSERVATIONS ===")

print("1. The dataset contains", len(df), "students.")

print(
    "2. The average student score is",
    round(df["avg_score"].mean(), 2)
)

print(
    "3. The highest average score is",
    df["avg_score"].max()
)

print(
    "4. The lowest average score is",
    df["avg_score"].min()
)

print(
    "5. Missing ML scores were found and replaced using the median."
)


# ============================================================
# 5. DISTRIBUTIONS OF NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "age",
    "math",
    "python",
    "ml_score",
    "avg_score"
]

for column in numeric_columns:
    plt.figure(figsize=(7, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Save each distribution plot
    plt.savefig(f"data/day4_distribution_{column}.png")
    plt.close()


# ============================================================
# 6. CORRELATION HEATMAP
# ============================================================

correlation_columns = [
    "age",
    "math",
    "python",
    "ml_score",
    "avg_score"
]

correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("data/day4_correlation_heatmap.png")
plt.close()


# ============================================================
# 7. TOP CATEGORY COUNTS
# ============================================================

department_counts = df["department"].value_counts().head(10)

print("\n=== TOP CATEGORY COUNTS ===")
print(department_counts)

plt.figure(figsize=(8, 5))
department_counts.plot(kind="bar")

plt.title("Top Department Counts")
plt.xlabel("Department")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("data/day4_top_category_counts.png")
plt.close()


# ============================================================
# 8. ADDITIONAL EDA RESULTS
# ============================================================

print("\n=== TOP 5 STUDENTS ===")

print(
    df.nlargest(5, "avg_score")[
        ["name", "avg_score", "grade", "department"]
    ]
)

print("\n=== ATTENDANCE VS AVERAGE SCORE ===")

print(
    df.groupby("attended")["avg_score"]
    .agg(["mean", "count"])
    .round(2)
)


# ============================================================
# 9. 200-WORD EDA NARRATIVE
# ============================================================

print("\n=== EDA NARRATIVE ===")

narrative = """
The dataset contains information about 20 students, including age,
Mathematics score, Python score, Machine Learning score, attendance,
and department. Initial inspection using describe(), info(), and
isnull().sum() showed that the dataset contains three missing values
in the ML score column. These missing values were handled by replacing
them with the median ML score, which reduces the effect of extreme
values compared with using the mean.

The descriptive statistics show variation in student performance
across the three subjects. An average score was created to provide
one overall performance measure, and grades were assigned based on
score ranges. The distribution plots help identify how student scores
are spread across the dataset. The correlation heatmap helps identify
relationships between numerical variables. Stronger relationships
between subject scores and average score are expected because the
average score is calculated from those subjects.

The department counts show how students are distributed across
different departments. The data is relatively small, so conclusions
should not be generalized to a larger student population. More data
would be useful for reliable analysis. Additional improvements could
include checking outliers, collecting more student records, and
investigating whether attendance has a statistically significant
relationship with academic performance.
"""

print(narrative)

print("\n=== EDA ANALYSIS COMPLETED SUCCESSFULLY ===")