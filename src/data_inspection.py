
import pandas as pd
import os

DATA_PATH = "data/raw"

files = [
    "transactions.csv",
    "products.csv",
    "customers.csv",
    "discounts.csv",
    "stores.csv",
    "employees.csv"
]

for file in files:
    path = os.path.join(DATA_PATH, file)

    print("\n" + "=" * 60)
    print(f"DATASET: {file}")
    print("=" * 60)

    if not os.path.exists(path):
        print("File not found!")
        continue

    df = pd.read_csv(path)

    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nFirst 5 Rows:")
    print(df.head())
