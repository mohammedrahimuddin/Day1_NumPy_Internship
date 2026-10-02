
import numpy as np
import pandas as pd

# Read CSV file
data = pd.read_csv("student_scores.csv")

print("===== STUDENT DATA =====")
print(data)

# Extract columns as NumPy arrays
study_hours = data["Study_Hours"].to_numpy()
exam_scores = data["Exam_Score"].to_numpy()

# Calculate Mean
print("\n===== MEAN =====")
print("Average Study Hours:", np.mean(study_hours))
print("Average Exam Score:", np.mean(exam_scores))

# Calculate Standard Deviation
print("\n===== STANDARD DEVIATION =====")
print("Study Hours:", np.std(study_hours))
print("Exam Scores:", np.std(exam_scores))

# Calculate Correlation
print("\n===== CORRELATION =====")
correlation = np.corrcoef(study_hours, exam_scores)

print(correlation)
print("Correlation Value:", correlation[0, 1])