
import pandas as pd

PRODUCTS = "data/raw/products.csv"
METRICS = "data/processed/product_level_metrics.csv"

products = pd.read_csv(
    PRODUCTS,
    usecols=["Product ID", "Category", "Sub Category", "Production Cost"]
)

metrics = pd.read_csv(METRICS)

print("\nPRODUCT COST VALIDATION")
print("Total products:", products["Product ID"].nunique())
print("Missing production costs:", products["Production Cost"].isna().sum())
print("Negative production costs:", (products["Production Cost"] < 0).sum())

print("\nPRODUCTION COST SUMMARY")
print(products["Production Cost"].describe())

print("\nPRODUCT COUNT BY CATEGORY")
print(
    products.groupby(["Category", "Sub Category"])["Product ID"]
    .nunique()
    .sort_values(ascending=False)
    .to_string()
)

print("\nPRODUCT-LEVEL METRIC CHECK")
print("Rows:", len(metrics))
print("Unique product IDs:", metrics["Product ID"].nunique())
print("Duplicate currency-product combinations:",
      metrics.duplicated(["Currency", "Product ID"]).sum())

print("\nCURRENCY COVERAGE")
print(metrics.groupby("Currency")["Product ID"].nunique())
