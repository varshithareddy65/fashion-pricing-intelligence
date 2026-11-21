
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed"

OUTPUT.mkdir(parents=True, exist_ok=True)

print("Loading transaction data...")

df = pd.read_csv(
    DATA / "transactions.csv",
    usecols=[
        "Product ID",
        "SKU",
        "Size",
        "Color",
        "Currency",
        "Transaction Type",
        "Quantity",
        "Date"
    ],
    dtype={
        "Product ID": "string",
        "SKU": "string",
        "Size": "string",
        "Color": "string",
        "Currency": "string",
        "Transaction Type": "string"
    }
)

print(f"Transaction rows: {len(df):,}")

# Basic missing-value analysis
print("\n--- Missing Values ---")
print(
    df[["Product ID", "SKU", "Size", "Color", "Currency"]]
    .isna()
    .sum()
)

# SKU and Product ID cardinality
print("\n--- Unique Counts ---")
print("Unique Product IDs:", df["Product ID"].nunique())
print("Unique SKUs:", df["SKU"].nunique())
print("Unique sizes:", df["Size"].nunique())
print("Unique colors:", df["Color"].nunique())
print("Currencies:", sorted(df["Currency"].dropna().unique()))

# Mapping: one Product ID to how many SKUs?
product_sku_counts = (
    df.groupby("Product ID")["SKU"]
    .nunique()
    .reset_index(name="Unique_SKUs")
)

# Mapping: one SKU to how many Product IDs?
sku_product_counts = (
    df.groupby("SKU")["Product ID"]
    .nunique()
    .reset_index(name="Unique_Product_IDs")
)

# Check whether a SKU maps to multiple sizes or colors
sku_variant_mapping = (
    df.groupby("SKU")
    .agg(
        Unique_Products=("Product ID", "nunique"),
        Unique_Sizes=("Size", "nunique"),
        Unique_Colors=("Color", "nunique"),
        Unique_Currencies=("Currency", "nunique"),
        Transaction_Rows=("SKU", "size")
    )
    .reset_index()
)

# Product-level SKU distribution
sku_distribution = (
    product_sku_counts["Unique_SKUs"]
    .value_counts()
    .sort_index()
    .rename_axis("Number_of_SKUs")
    .reset_index(name="Number_of_Products")
)

# Save diagnostic outputs
product_sku_counts.to_csv(
    OUTPUT / "product_sku_counts.csv",
    index=False
)

sku_product_counts.to_csv(
    OUTPUT / "sku_product_counts.csv",
    index=False
)

sku_variant_mapping.to_csv(
    OUTPUT / "sku_variant_mapping.csv",
    index=False
)

sku_distribution.to_csv(
    OUTPUT / "sku_distribution.csv",
    index=False
)

print("\n--- Product ID to SKU Mapping ---")
print("Products with one SKU:",
      (product_sku_counts["Unique_SKUs"] == 1).sum())

print("Products with multiple SKUs:",
      (product_sku_counts["Unique_SKUs"] > 1).sum())

print("\n--- SKU to Product ID Mapping ---")
print("SKUs linked to multiple products:",
      (sku_product_counts["Unique_Product_IDs"] > 1).sum())

print("\n--- SKU Variant Mapping ---")
print("SKUs with multiple sizes:",
      (sku_variant_mapping["Unique_Sizes"] > 1).sum())

print("SKUs with multiple colors:",
      (sku_variant_mapping["Unique_Colors"] > 1).sum())

print("\n--- SKU Distribution ---")
print(sku_distribution.to_string(index=False))

print("\n--- Sample SKU Mapping ---")
print(
    sku_variant_mapping
    .sort_values("Transaction_Rows", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\nAll diagnostic files saved in data/processed/")
