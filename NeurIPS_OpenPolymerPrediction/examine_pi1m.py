import pandas as pd
import os

# Change to the RawData directory
os.chdir('RawData')

# Read the PI1M file (first 10 rows)
df = pd.read_csv('PI1M.csv', nrows=10)

print("PI1M Columns:", df.columns.tolist())
print("PI1M Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check if there are any missing values
print("\nMissing values per column:")
print(df.isnull().sum()) 