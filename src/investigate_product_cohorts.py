
import pandas as pd
import numpy as np
import re
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PRODUCT_FILE = BASE_DIR / "data/raw/products.csv"
COHORT_FILE = BASE_DIR / "data/processed/product_cohort_detail.csv"

OUTPUT_DIR = BASE_DIR / "data/processed"

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 65)
print("PRODUCT COHORT STRUCTURE INVESTIGATION")
print("=" * 65)

df = pd.read_csv(
    COHORT_FILE,
    dtype={"Product ID": str}
)

print("\nLoaded cohort records:", len(df))

# ============================================================
# 2. CLEAN PRODUCT IDs
# ============================================================

df["Product ID"] = df["Product ID"].astype(str).str.strip()

df["ID_Length"] = df["Product ID"].str.len()

df["ID_Prefix"] = (
    df["Product ID"]
    .str.extract(r"^([A-Za-z]+)", expand=False)
    .fillna("Numeric/Other")
)

df["ID_Numeric"] = pd.to_numeric(
    df["Product ID"].str.extract(r"(\d+)", expand=False),
    errors="coerce"
)

print("\nPRODUCT ID ANALYSIS")

id_summary = (
    df.groupby("Product_Cohort")
    .agg(
        Products=("Product ID", "nunique"),
        Average_ID_Length=("ID_Length", "mean"),
        Minimum_ID_Length=("ID_Length", "min"),
        Maximum_ID_Length=("ID_Length", "max"),
        Numeric_ID_Count=("ID_Numeric", "count"),
        Minimum_Numeric_ID=("ID_Numeric", "min"),
        Maximum_Numeric_ID=("ID_Numeric", "max")
    )
)

print(id_summary.to_string())

prefix_summary = (
    df.groupby(["Product_Cohort", "ID_Prefix"])
    .size()
    .reset_index(name="Product_Count")
)

prefix_summary.to_csv(
    OUTPUT_DIR / "cohort_id_prefix_analysis.csv",
    index=False
)

# ============================================================
# 3. PRODUCT ID RANGE AND OVERLAP
# ============================================================

print("\nPRODUCT ID RANGE")

for cohort, group in df.groupby("Product_Cohort"):

    numeric_ids = group["ID_Numeric"].dropna()

    print("\n", cohort)
    print("First IDs:", group["Product ID"].sort_values().head(10).tolist())
    print("Last IDs:", group["Product ID"].sort_values().tail(10).tolist())

    if not numeric_ids.empty:
        print("Numeric ID minimum:", numeric_ids.min())
        print("Numeric ID maximum:", numeric_ids.max())

# ============================================================
# 4. DESCRIPTION ANALYSIS
# ============================================================

description_col = "Description EN"

if description_col not in df.columns:
    raise ValueError(
        f"Missing expected description column: {description_col}"
    )

df["Description_Clean"] = (
    df[description_col]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

df["Description_Normalized"] = (
    df["Description_Clean"]
    .str.replace(r"[^a-z0-9\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["Description_Word_Count"] = (
    df["Description_Normalized"]
    .str.split()
    .str.len()
)

df["Description_Character_Count"] = (
    df["Description_Normalized"].str.len()
)

print("\nDESCRIPTION QUALITY")

description_summary = (
    df.groupby("Product_Cohort")
    .agg(
        Products=("Product ID", "nunique"),
        Missing_Descriptions=(
            "Description_Normalized",
            lambda x: (x == "").sum()
        ),
        Unique_Descriptions=(
            "Description_Normalized",
            "nunique"
        ),
        Average_Word_Count=(
            "Description_Word_Count",
            "mean"
        ),
        Average_Character_Count=(
            "Description_Character_Count",
            "mean"
        )
    )
)

print(description_summary.to_string())

# ============================================================
# 5. EXACT DESCRIPTION OVERLAP
# ============================================================

historical = df[
    df["Product_Cohort"] == "Historical - 2023 to 2024"
]

march = df[
    df["Product_Cohort"] == "First Observed - March 2025"
]

historical_descriptions = set(
    historical.loc[
        historical["Description_Normalized"] != "",
        "Description_Normalized"
    ]
)

march_descriptions = set(
    march.loc[
        march["Description_Normalized"] != "",
        "Description_Normalized"
    ]
)

common_descriptions = (
    historical_descriptions & march_descriptions
)

print("\nDESCRIPTION OVERLAP")

print("Historical unique descriptions:", len(historical_descriptions))
print("March unique descriptions:", len(march_descriptions))
print("Exact common descriptions:", len(common_descriptions))

print(
    "March descriptions found historically:",
    march["Description_Normalized"].isin(
        historical_descriptions
    ).sum()
)

# ============================================================
# 6. DUPLICATE DESCRIPTION ANALYSIS
# ============================================================

print("\nDUPLICATE DESCRIPTION ANALYSIS")

duplicate_counts = (
    df[df["Description_Normalized"] != ""]
    .groupby(
        ["Product_Cohort", "Description_Normalized"]
    )
    .size()
    .reset_index(name="Product_Count")
)

duplicate_counts = duplicate_counts[
    duplicate_counts["Product_Count"] > 1
]

print("Repeated descriptions:", len(duplicate_counts))

duplicate_counts.to_csv(
    OUTPUT_DIR / "cohort_duplicate_descriptions.csv",
    index=False
)

# ============================================================
# 7. CATALOG ATTRIBUTE COMPARISON
# ============================================================

print("\nCATALOG ATTRIBUTE COMPARISON")

for col in [
    "Category",
    "Sub Category",
    "Color",
    "Sizes"
]:

    if col not in df.columns:
        continue

    print("\nATTRIBUTE:", col)

    comparison = (
        df.groupby("Product_Cohort")[col]
        .agg(
            Unique_Values="nunique",
            Missing_Values=lambda x: x.isna().sum()
        )
    )

    print(comparison.to_string())

# ============================================================
# 8. PRODUCTION COST COMPARISON
# ============================================================

if "Production Cost" in df.columns:

    df["Production Cost"] = pd.to_numeric(
        df["Production Cost"],
        errors="coerce"
    )

    print("\nPRODUCTION COST DISTRIBUTION")

    cost_summary = (
        df.groupby("Product_Cohort")["Production Cost"]
        .agg(
            Count="count",
            Mean="mean",
            Median="median",
            Std="std",
            Minimum="min",
            Maximum="max"
        )
    )

    print(cost_summary.to_string())

    cost_summary.to_csv(
        OUTPUT_DIR / "cohort_cost_comparison.csv"
    )

# ============================================================
# 9. SUBCATEGORY BALANCE CHECK
# ============================================================

print("\nSUBCATEGORY BALANCE")

subcategory_summary = (
    df.groupby(
        ["Product_Cohort", "Category", "Sub Category"]
    )
    .agg(
        Product_Count=("Product ID", "nunique")
    )
    .reset_index()
)

print(
    subcategory_summary.to_string(index=False)
)

subcategory_summary.to_csv(
    OUTPUT_DIR / "cohort_subcategory_balance.csv",
    index=False
)

# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("INVESTIGATION COMPLETED")
print("=" * 65)

print("Historical products:", len(historical))
print("March products:", len(march))

print("Common descriptions:", len(common_descriptions))

print(
    "March descriptions matching historical descriptions:",
    march["Description_Normalized"]
    .isin(historical_descriptions)
    .sum()
)

print("\nOUTPUT FILES")

print("cohort_id_prefix_analysis.csv")
print("cohort_duplicate_descriptions.csv")
print("cohort_cost_comparison.csv")
print("cohort_subcategory_balance.csv")

print("\nINTERPRETATION CAUTION")
print(
    "Matching IDs or descriptions can indicate catalog similarities, "
    "but cannot independently establish product launch dates "
    "or dataset-generation methods."
)
