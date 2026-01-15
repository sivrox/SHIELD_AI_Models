import pandas as pd

df_synthetic = pd.read_csv('synthetic_vitals.csv')  # Replace with actual filename


print("Synthetic dataset shape:", df_synthetic.shape)
print("\nColumn names:")
print(df_synthetic.columns.tolist())
print("\nFirst 5 rows:")
print(df_synthetic.head())
print("\nData types:")
print(df_synthetic.dtypes)
print("\nMissing values:")
print(df_synthetic.isnull().sum())
print("\nValue ranges:")
print(df_synthetic.describe())
