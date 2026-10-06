import pandas as pd

# Load Indian Government dataset
df = pd.read_csv("data/Table_2A_State_Uts.csv")

# Display basic information
print("Dataset Shape:", df.shape)

print("\nDataset Data Types:")
print(df.dtypes)

print("\nFirst 10 Rows:")
print(df.head(10))
# --------------------------------------------------
# LOAD SECOND DATASET FOR MERGE
# --------------------------------------------------

area_df = pd.read_csv("data/state_area.csv")

print("\nSecond Dataset Shape:")
print(area_df.shape)

print("\nSecond Dataset Columns:")
print(area_df.columns)

print("\nSecond Dataset First 5 Rows:")
print(area_df.head())   
# --------------------------------------------------
# MERGE OPERATION
# Merge population data with area and district data
# using State/UT as the common key
# --------------------------------------------------

merged_df = pd.merge(
    df,
    area_df,
    left_on="India/State/Union Territory",
    right_on="State/UT",
    how="inner"
)

print("\nMerged Dataset:")
print(merged_df[
    [
        "India/State/Union Territory",
        "Population 2011",
        "Population Density (per sq.km) - 2011",
        "Area [Sq. Km.] - Total",
        "Number of Districts",
        "Number of Villages"
    ]
].head(10))

print("\nMerged Dataset Shape:")
print(merged_df.shape)
# --------------------------------------------------
# PIVOT TABLE OPERATION
# Summarize average population density by category
# --------------------------------------------------

pivot_df = pd.pivot_table(
    df,
    values="Population Density (per sq.km) - 2011",
    index="Category",
    aggfunc="mean"
)

print("\nPivot Table:")
print(pivot_df)
# --------------------------------------------------
# EXPORT CLEANED DATASET
# Save merged dataset as CSV and Parquet
# --------------------------------------------------

merged_df.to_csv("data/cleaned_state_data.csv", index=False)

merged_df.to_parquet("data/cleaned_state_data.parquet", index=False)

print("\nFiles exported successfully.")

# Compare file sizes
import os

csv_size = os.path.getsize("data/cleaned_state_data.csv")
parquet_size = os.path.getsize("data/cleaned_state_data.parquet")

print(f"CSV file size: {csv_size} bytes")
print(f"Parquet file size: {parquet_size} bytes")