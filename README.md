
# Day 1 - Python for ML: NumPy Fundamentals

## Internship Practical Tasks

### 1. NumPy Arrays
- Created 1D, 2D and 3D arrays.
- Verified array shapes and dimensions.

### 2. NumPy Operations
- Broadcasting
- Vectorised operations
- Matrix multiplication

### 3. Statistics Using CSV
- Read student_scores.csv using Pandas.
- Calculated Mean.
- Calculated Standard Deviation.
- Calculated Correlation.

## Technologies Used
- Python 3.14
- NumPy
- Pandas
- VS Code
- Git

## Files
- day1_numpy.py
- day1_numpy_operations.py
- day1_numpy_statistics.py
- student_scores.csv

## Day 2 – Pandas for Data Manipulation

### Dataset
Used official Government of India state/UT datasets from data.gov.in.

### Tasks Completed
- Loaded Indian state-wise dataset using Pandas
- Checked dataset shape, data types, and first 10 rows
- Performed filtering based on population density
- Used groupby() to calculate average population density
- Merged population data with area, district, and village data
- Created a pivot table for population density by category
- Exported cleaned data to CSV
- Exported cleaned data to Parquet
- Compared CSV and Parquet file sizes

### Tools and Technologies
- Python
- Pandas
- PyArrow
- VS Code
- Git and GitHub

### Output
- Merged dataset: 34 rows × 13 columns
- CSV size: 3140 bytes
- Parquet size: 12000 bytes
## Day 3 – NumPy Data Loading, Cleaning & Inspection

### Tasks Completed
- Created and explored NumPy arrays.
- Calculated shape, mean, standard deviation, minimum and maximum.
- Used Boolean masking to find values above the average.
- Performed matrix addition, dot product and transpose operations.
- Applied broadcasting for column-wise normalization.
- Implemented z-score normalization.
- Added validation tests using assertions.
- Verified that all validation tests passed successfully.

### Tools and Technologies
- Python
- NumPy
- VS Code
- Git
- GitHub

### Validation
All Day 3 NumPy validation tests passed successfully.
## Day 4 – Exploratory Data Analysis (EDA)

### Tasks Completed
- Inspected the dataset using `df.describe()`, `df.info()`, and `df.isnull().sum()`.
- Identified and handled missing ML score values using the median.
- Documented five observations from the dataset.
- Created distributions for all important numeric columns.
- Created a correlation heatmap to analyze relationships between numeric features.
- Created a top-category count chart for student departments.
- Performed feature engineering by calculating average scores and grades.
- Wrote an EDA narrative describing findings, suspicious areas, and possible improvements.

### Tools and Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- VS Code
- Git
- GitHub

### Output Evidence
EDA plots were generated and saved in the `data/` directory:
- Numeric distribution plots
- Correlation heatmap
- Top category counts

### Key Findings
The dataset contains 20 student records with some missing ML scores. Missing values were replaced using the median. Subject scores show variation across students, while the correlation analysis helps identify relationships between academic features. The dataset is small, so the findings should not be generalized to a larger population without additional data.