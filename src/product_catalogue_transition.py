
import warnings
warnings.filterwarnings("ignore")

import re
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

TRANSACTION_FILE = RAW / "transactions.csv"
PRODUCT_FILE = RAW / "products.csv"

MARCH_START = pd.Timestamp("2025-03-01")
CHUNK_SIZE = 500_000

PROCESSED.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD PRODUCT CATALOGUE
# ============================================================

print("\nLoading product catalogue...")

products = pd.read_csv(
    PRODUCT_FILE,
    dtype={"Product ID": "string"}
)

products = products.drop_duplicates(
    subset=["Product ID"]
).copy()

products["Product ID"] = (
    products["Product ID"].astype("string")
)

print("Catalogue products:", f"{len(products):,}")
print("Product columns:", products.columns.tolist())

# Identify available description columns.
description_columns = [
    col for col in products.columns
    if "description" in col.lower()
]

print("Description columns:", description_columns)

# ============================================================
# 2. PROCESS TRANSACTIONS IN CHUNKS
# ============================================================

print("\nReading transaction data in chunks...")

required_columns = [
    "Product ID",
    "Date",
    "Quantity",
    "Transaction Type"
]

all_parts = []
march_parts = []

reader = pd.read_csv(
    TRANSACTION_FILE,
    usecols=required_columns,
    dtype={
        "Product ID": "string",
        "Transaction Type": "string"
    },
    chunksize=CHUNK_SIZE,
    low_memory=False
)

for chunk_number, chunk in enumerate(reader, start=1):

    chunk["Date"] = pd.to_datetime(
        chunk["Date"],
        errors="coerce"
    )

    chunk["Quantity"] = pd.to_numeric(
        chunk["Quantity"],
        errors="coerce"
    )

    # Use sale transactions only.
    chunk["Transaction Type"] = (
        chunk["Transaction Type"]
        .str.strip()
        .str.lower()
    )

    chunk = chunk[
        (chunk["Transaction Type"] == "sale")
        & chunk["Product ID"].notna()
        & chunk["Date"].notna()
        & chunk["Quantity"].notna()
    ].copy()

    if chunk.empty:
        continue

    # Overall product sales history.
    grouped = (
        chunk.groupby("Product ID")
        .agg(
            First_Sale_Date=("Date", "min"),
            Last_Sale_Date=("Date", "max"),
            Sale_Lines=("Date", "size"),
            Total_Sale_Units=("Quantity", "sum")
        )
        .reset_index()
    )

    all_parts.append(grouped)

    # March 2025 sales.
    march = chunk[
        chunk["Date"] >= MARCH_START
    ]

    if not march.empty:

        march_grouped = (
            march.groupby("Product ID")
            .agg(
                March_Sale_Lines=("Date", "size"),
                March_Sale_Units=("Quantity", "sum")
            )
            .reset_index()
        )

        march_parts.append(march_grouped)

    if chunk_number % 3 == 0:
        print(
            f"Processed {chunk_number} transaction chunks..."
        )

# ============================================================
# 3. COMBINE TRANSACTION AGGREGATES
# ============================================================

print("\nCombining transaction aggregates...")

if not all_parts:
    raise ValueError("No valid sale transactions were found.")

all_sales = pd.concat(
    all_parts,
    ignore_index=True
)

sales_summary = (
    all_sales.groupby("Product ID")
    .agg(
        First_Sale_Date=("First_Sale_Date", "min"),
        Last_Sale_Date=("Last_Sale_Date", "max"),
        Sale_Lines=("Sale_Lines", "sum"),
        Total_Sale_Units=("Total_Sale_Units", "sum")
    )
    .reset_index()
)

del all_parts, all_sales

if march_parts:

    march_sales = pd.concat(
        march_parts,
        ignore_index=True
    )

    march_summary = (
        march_sales.groupby("Product ID")
        .agg(
            March_Sale_Lines=("March_Sale_Lines", "sum"),
            March_Sale_Units=("March_Sale_Units", "sum")
        )
        .reset_index()
    )

    del march_parts, march_sales

else:

    march_summary = pd.DataFrame(
        columns=[
            "Product ID",
            "March_Sale_Lines",
            "March_Sale_Units"
        ]
    )

print(
    "Products with observed sales:",
    f"{len(sales_summary):,}"
)

# ============================================================
# 4. BUILD PRODUCT LIFECYCLE DATASET
# ============================================================

print("\nBuilding product lifecycle dataset...")

lifecycle = products.merge(
    sales_summary,
    on="Product ID",
    how="left"
)

lifecycle = lifecycle.merge(
    march_summary,
    on="Product ID",
    how="left"
)

for col in [
    "Sale_Lines",
    "Total_Sale_Units",
    "March_Sale_Lines",
    "March_Sale_Units"
]:
    lifecycle[col] = lifecycle[col].fillna(0)

lifecycle["Cohort"] = np.select(
    [
        lifecycle["First_Sale_Date"].isna(),
        lifecycle["First_Sale_Date"] >= MARCH_START
    ],
    [
        "No Observed Sales",
        "March 2025 New Cohort"
    ],
    default="Historical Cohort"
)

# Historical products with and without March sales.
lifecycle["Historical_March_Status"] = np.where(
    lifecycle["Cohort"] == "Historical Cohort",
    np.where(
        lifecycle["March_Sale_Units"] > 0,
        "Sold in March 2025",
        "No March Sale Observed"
    ),
    "Not Applicable"
)

lifecycle["Weeks_Since_Last_Sale"] = (
    (
        MARCH_START - lifecycle["Last_Sale_Date"]
    ).dt.days // 7
)

# ============================================================
# 5. COHORT SUMMARY
# ============================================================

print("\n========== PRODUCT COHORT SUMMARY ==========")

cohort_summary = (
    lifecycle.groupby("Cohort")
    .agg(
        Product_Count=("Product ID", "nunique"),
        Total_Sale_Lines=("Sale_Lines", "sum"),
        Total_Historical_Units=("Total_Sale_Units", "sum"),
        March_Sale_Lines=("March_Sale_Lines", "sum"),
        March_Sale_Units=("March_Sale_Units", "sum")
    )
    .reset_index()
)

cohort_summary["Percentage_of_Catalogue"] = (
    cohort_summary["Product_Count"]
    / len(lifecycle) * 100
)

print(cohort_summary.to_string(index=False))

# ============================================================
# 6. HISTORICAL PRODUCT ACTIVITY
# ============================================================

historical = lifecycle[
    lifecycle["Cohort"] == "Historical Cohort"
].copy()

historical["March_Activity"] = np.where(
    historical["March_Sale_Units"] > 0,
    "Sales Observed",
    "No Sales Observed"
)

historical_activity = (
    historical.groupby("March_Activity")
    .agg(
        Product_Count=("Product ID", "nunique"),
        Historical_Units=("Total_Sale_Units", "sum"),
        March_Units=("March_Sale_Units", "sum"),
        Mean_Weeks_Since_Last_Sale=(
            "Weeks_Since_Last_Sale", "mean"
        )
    )
    .reset_index()
)

historical_activity["Percentage"] = (
    historical_activity["Product_Count"]
    / max(len(historical), 1) * 100
)

print("\n========== HISTORICAL PRODUCT ACTIVITY ==========")
print(historical_activity.to_string(index=False))

# ============================================================
# 7. CATEGORY AND SUBCATEGORY TRANSITION
# ============================================================

category_summary = (
    lifecycle.groupby(
        ["Cohort", "Category"]
    )
    .agg(
        Product_Count=("Product ID", "nunique"),
        March_Sale_Units=("March_Sale_Units", "sum"),
        March_Sale_Lines=("March_Sale_Lines", "sum")
    )
    .reset_index()
)

subcategory_summary = (
    lifecycle.groupby(
        ["Cohort", "Category", "Sub Category"]
    )
    .agg(
        Product_Count=("Product ID", "nunique"),
        March_Sale_Units=("March_Sale_Units", "sum"),
        March_Sale_Lines=("March_Sale_Lines", "sum")
    )
    .reset_index()
)

# ============================================================
# 8. PRODUCT DESCRIPTION NORMALIZATION
# ============================================================

def normalize_description(value):

    if pd.isna(value):
        return ""

    value = unicodedata.normalize(
        "NFKC",
        str(value)
    )

    value = value.lower().strip()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


print("\nChecking product description overlap...")

description_overlap_summary = pd.DataFrame()
description_matches = pd.DataFrame()

if description_columns:

    description_long = products[
        ["Product ID"] + description_columns
    ].melt(
        id_vars=["Product ID"],
        value_vars=description_columns,
        value_name="Description"
    )

    description_long["Normalized_Description"] = (
        description_long["Description"]
        .apply(normalize_description)
    )

    description_long = description_long[
        description_long["Normalized_Description"].str.len() >= 4
    ].drop_duplicates(
        ["Product ID", "Normalized_Description"]
    )

    description_long = description_long.merge(
        lifecycle[["Product ID", "Cohort"]],
        on="Product ID",
        how="left"
    )

    old_descriptions = description_long[
        description_long["Cohort"] == "Historical Cohort"
    ]

    new_descriptions = description_long[
        description_long["Cohort"] == "March 2025 New Cohort"
    ]

    old_lookup = (
        old_descriptions.groupby("Normalized_Description")[
            "Product ID"
        ]
        .agg(lambda x: set(x))
        .to_dict()
    )

    match_records = []

    for product_id, group in new_descriptions.groupby("Product ID"):

        matched_old_ids = set()
        shared_descriptions = set()

        for description in group["Normalized_Description"]:

            old_ids = old_lookup.get(description, set())

            if old_ids:
                matched_old_ids.update(old_ids)
                shared_descriptions.add(description)

        if matched_old_ids:

            match_records.append({
                "March_Product_ID": product_id,
                "Historical_Product_IDs": ";".join(
                    sorted(matched_old_ids)
                ),
                "Historical_Product_Match_Count": len(
                    matched_old_ids
                ),
                "Shared_Description_Count": len(
                    shared_descriptions
                ),
                "Shared_Descriptions": " | ".join(
                    sorted(shared_descriptions)
                )
            })

    description_matches = pd.DataFrame(
        match_records,
        columns=[
            "March_Product_ID",
            "Historical_Product_IDs",
            "Historical_Product_Match_Count",
            "Shared_Description_Count",
            "Shared_Descriptions"
        ]
    )

    march_product_count = (
        lifecycle["Cohort"] == "March 2025 New Cohort"
    ).sum()

    matched_product_count = (
        description_matches["March_Product_ID"].nunique()
        if not description_matches.empty else 0
    )

    old_product_match_count = (
        len(
            set(
                ";".join(
                    description_matches["Historical_Product_IDs"]
                ).split(";")
            )
        )
        if not description_matches.empty else 0
    )

    description_overlap_summary = pd.DataFrame([{
        "March_New_Product_Count": march_product_count,
        "March_Products_With_Exact_Description_Match": (
            matched_product_count
        ),
        "March_Product_Match_Percent": (
            matched_product_count / march_product_count * 100
            if march_product_count else 0
        ),
        "Distinct_Shared_Descriptions": (
            description_long[
                description_long["Normalized_Description"].isin(
                    old_lookup.keys()
                )
            ]["Normalized_Description"].nunique()
        ),
        "Historical_Products_Matched": old_product_match_count
    }])

    print("\n========== DESCRIPTION OVERLAP ==========")
    print(description_overlap_summary.to_string(index=False))

else:

    print(
        "No description columns found. "
        "Description overlap analysis skipped."
    )

# ============================================================
# 9. MARCH PRODUCTS WITH MATCHING HISTORICAL PRODUCTS
# ============================================================

if not description_matches.empty:

    march_matches = description_matches.merge(
        lifecycle,
        left_on="March_Product_ID",
        right_on="Product ID",
        how="left"
    )

else:

    march_matches = pd.DataFrame()

# ============================================================
# 10. TOP HISTORICAL PRODUCTS THAT STOPPED SELLING
# ============================================================

inactive_historical = historical[
    historical["March_Sale_Units"] == 0
].copy()

inactive_historical = inactive_historical.sort_values(
    "Total_Sale_Units",
    ascending=False
)

print("\n========== TOP HISTORICAL INACTIVE PRODUCTS ==========")

display_columns = [
    "Product ID",
    "Category",
    "Sub Category",
    "First_Sale_Date",
    "Last_Sale_Date",
    "Weeks_Since_Last_Sale",
    "Total_Sale_Units",
    "March_Sale_Units"
]

print(
    inactive_historical[
        display_columns
    ].head(15).to_string(index=False)
)

# ============================================================
# 11. TOP NEW MARCH PRODUCTS
# ============================================================

march_new = lifecycle[
    lifecycle["Cohort"] == "March 2025 New Cohort"
].copy()

march_new = march_new.sort_values(
    "March_Sale_Units",
    ascending=False
)

print("\n========== TOP NEW MARCH PRODUCTS ==========")

print(
    march_new[
        [
            "Product ID",
            "Category",
            "Sub Category",
            "First_Sale_Date",
            "March_Sale_Units",
            "March_Sale_Lines"
        ]
    ].head(15).to_string(index=False)
)

# ============================================================
# 12. SAVE OUTPUT FILES
# ============================================================

lifecycle.to_csv(
    PROCESSED / "product_cohort_lifecycle.csv",
    index=False
)

cohort_summary.to_csv(
    PROCESSED / "product_cohort_summary.csv",
    index=False
)

historical_activity.to_csv(
    PROCESSED / "historical_product_activity_summary.csv",
    index=False
)

category_summary.to_csv(
    PROCESSED / "product_cohort_category_summary.csv",
    index=False
)

subcategory_summary.to_csv(
    PROCESSED / "product_cohort_subcategory_summary.csv",
    index=False
)

inactive_historical.to_csv(
    PROCESSED / "historical_products_inactive_before_march.csv",
    index=False
)

march_new.to_csv(
    PROCESSED / "march_new_products.csv",
    index=False
)

description_overlap_summary.to_csv(
    PROCESSED / "description_overlap_summary.csv",
    index=False
)

description_matches.to_csv(
    PROCESSED / "march_historical_description_matches.csv",
    index=False
)

print("\n========== OUTPUT FILES ==========")

print("1. product_cohort_lifecycle.csv")
print("2. product_cohort_summary.csv")
print("3. historical_product_activity_summary.csv")
print("4. product_cohort_category_summary.csv")
print("5. product_cohort_subcategory_summary.csv")
print("6. historical_products_inactive_before_march.csv")
print("7. march_new_products.csv")
print("8. description_overlap_summary.csv")
print("9. march_historical_description_matches.csv")

print("\nProduct catalogue transition analysis completed.")
