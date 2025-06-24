import pandas as pd
import os

# Change to the RawData directory
os.chdir('RawData')

# Read the Excel file
df = pd.read_excel('TC_MD_20240306.xlsx')

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nSample values from first column:")
print(df.iloc[:10, 0].tolist()) 