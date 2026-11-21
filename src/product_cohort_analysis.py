
import pandas as pd
import numpy as np

from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PRODUCTS_FILE = BASE_DIR / "data/raw/products.csv"

DAILY_FILE = (
    BASE_DIR
    / "data/processed/product_store_daily_sales.csv"
)

OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_OUTPUT = OUTPUT_DIR / "product_cohort_detail.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "product_cohort_summary.csv"
CATEGORY_OUTPUT = OUTPUT_DIR / "cohort_category_subcategory.csv"
NUMERIC_OUTPUT = OUTPUT_DIR / "cohort_numeric_attributes.csv"
CATEGORICAL_OUTPUT = OUTPUT_DIR / "cohort_categorical_attributes.csv"

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 65)
print("PRODUCT COHORT ANALYSIS")
print("=" * 65)

print("\nLoading product catalog and daily sales...")

products = pd.read_csv(
    PRODUCTS_FILE,
    dtype={"Product ID": str}
)

daily = pd.read_csv(
    DAILY_FILE,
    dtype={"Product ID": str},
    parse_dates=["Date"]
)

print("Catalog products:", len(products))
print("Daily sales records:", len(daily))

print("\nPRODUCT CATALOG COLUMNS")
print(products.columns.tolist())

# ============================================================
# 2. VALIDATE DATA
# ============================================================

required_product_columns = ["Product ID"]

for col in required_product_columns:
    if col not in products.columns:
        raise ValueError(f"Missing product column: {col}")

if "Product ID" not in daily.columns or "Date" not in daily.columns:
    raise ValueError("Daily sales file is missing required columns.")

products["Product ID"] = products["Product ID"].astype(str).str.strip()
daily["Product ID"] = daily["Product ID"].astype(str).str.strip()

if products["Product ID"].duplicated().any():
    raise ValueError("Duplicate Product IDs in product catalog.")

# ============================================================
# 3. CALCULATE FIRST OBSERVED SALE DATE
# ============================================================

print("\nCalculating first observed sale date...")

first_seen = (
    daily.groupby("Product ID", as_index=False)
    .agg(
        First_Observed_Sale=("Date", "min"),
        Last_Observed_Sale=("Date", "max"),
        Observed_Sales_Days=("Date", "nunique")
    )
)

# ============================================================
# 4. CREATE PRODUCT COHORTS
# ============================================================

cohort_data = products.merge(
    first_seen,
    on="Product ID",
    how="left",
    validate="one_to_one"
)

def assign_cohort(date):

    if pd.isna(date):
        return "Catalog Only - No Observed Sale"

    if date <= pd.Timestamp("2024-12-31"):
        return "Historical - 2023 to 2024"

    if date <= pd.Timestamp("2025-02-28"):
        return "First Observed - Jan to Feb 2025"

    if date <= pd.Timestamp("2025-03-18"):
        return "First Observed - March 2025"

    return "First Observed - After March 2025"


cohort_data["Product_Cohort"] = (
    cohort_data["First_Observed_Sale"].apply(assign_cohort)
)

print("\nCOHORT ASSIGNMENT COMPLETED")

print(
    cohort_data["Product_Cohort"]
    .value_counts(dropna=False)
    .to_string()
)

# ============================================================
# 5. SAVE PRODUCT-LEVEL COHORT DATA
# ============================================================

cohort_data.to_csv(
    DETAIL_OUTPUT,
    index=False
)

print("\nSaved:", DETAIL_OUTPUT)

# ============================================================
# 6. COHORT SUMMARY
# ============================================================

print("\nCOHORT SUMMARY")

summary = (
    cohort_data.groupby("Product_Cohort")
    .agg(
        Product_Count=("Product ID", "nunique"),
        Products_With_Observed_Sales=(
            "First_Observed_Sale",
            "count"
        ),
        Earliest_First_Sale=("First_Observed_Sale", "min"),
        Latest_First_Sale=("First_Observed_Sale", "max")
    )
    .reset_index()
)

summary["Share_Percent"] = (
    summary["Product_Count"]
    / len(cohort_data)
    * 100
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

print(summary.to_string(index=False))

# ============================================================
# 7. CATEGORY AND SUBCATEGORY DISTRIBUTION
# ============================================================

print("\nCATEGORY AND SUBCATEGORY ANALYSIS")

category_columns = [
    col for col in ["Category", "Sub Category"]
    if col in cohort_data.columns
]

if category_columns:

    category_summary = (
        cohort_data.groupby(
            ["Product_Cohort"] + category_columns,
            dropna=False
        )
        .agg(
            Product_Count=("Product ID", "nunique")
        )
        .reset_index()
    )

    category_summary["Cohort_Share_Percent"] = (
        category_summary["Product_Count"]
        /
        category_summary.groupby("Product_Cohort")[
            "Product_Count"
        ].transform("sum")
        * 100
    )

    category_summary.to_csv(
        CATEGORY_OUTPUT,
        index=False
    )

    print(category_summary.to_string(index=False))

    print("\nSaved:", CATEGORY_OUTPUT)

else:
    print("Category or subcategory columns unavailable.")

# ============================================================
# 8. NUMERIC PRODUCT ATTRIBUTE COMPARISON
# ============================================================

print("\nNUMERIC ATTRIBUTE ANALYSIS")

excluded_numeric = {
    "Product ID",
    "First_Observed_Sale",
    "Last_Observed_Sale",
    "Observed_Sales_Days"
}

numeric_columns = [
    col
    for col in cohort_data.select_dtypes(include=np.number).columns
    if col not in excluded_numeric
]

numeric_results = []

for col in numeric_columns:

    for cohort, group in cohort_data.groupby("Product_Cohort"):

        values = pd.to_numeric(
            group[col],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        numeric_results.append(
            {
                "Product_Cohort": cohort,
                "Attribute": col,
                "Valid_Products": len(values),
                "Mean": values.mean(),
                "Median": values.median(),
                "Minimum": values.min(),
                "Maximum": values.max()
            }
        )

numeric_df = pd.DataFrame(numeric_results)

if not numeric_df.empty:

    numeric_df.to_csv(
        NUMERIC_OUTPUT,
        index=False
    )

    print(numeric_df.to_string(index=False))
    print("\nSaved:", NUMERIC_OUTPUT)

else:
    print("No numeric product attributes found.")

# ============================================================
# 9. CATEGORICAL ATTRIBUTE COMPARISON
# ============================================================

print("\nCATEGORICAL ATTRIBUTE ANALYSIS")

excluded_names = {
    "Product ID",
    "Description",
    "First_Observed_Sale",
    "Last_Observed_Sale"
}

categorical_columns = []

for col in cohort_data.select_dtypes(
    include=["object", "category"]
).columns:

    if col in excluded_names:
        continue

    unique_count = cohort_data[col].nunique(dropna=True)

    # Exclude high-cardinality text fields.
    # Keep Sizes even if it contains many combinations.
    if unique_count <= 100 or col.lower() == "sizes":
        categorical_columns.append(col)

categorical_results = []

for col in categorical_columns:

    for cohort, group in cohort_data.groupby("Product_Cohort"):

        counts = (
            group[col]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(10)
        )

        for value, count in counts.items():

            categorical_results.append(
                {
                    "Product_Cohort": cohort,
                    "Attribute": col,
                    "Attribute_Value": value,
                    "Product_Count": count
                }
            )

categorical_df = pd.DataFrame(categorical_results)

if not categorical_df.empty:

    categorical_df.to_csv(
        CATEGORICAL_OUTPUT,
        index=False
    )

    print(categorical_df.to_string(index=False))
    print("\nSaved:", CATEGORICAL_OUTPUT)

else:
    print("No suitable categorical attributes found.")

# ============================================================
# 10. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("PRODUCT COHORT ANALYSIS COMPLETED")
print("=" * 65)

print("Catalog products:", len(products))
print("Unique catalog IDs:", products["Product ID"].nunique())

print(
    "Products with observed sales:",
    cohort_data["First_Observed_Sale"].notna().sum()
)

print(
    "Products first observed in March 2025:",
    (
        cohort_data["Product_Cohort"]
        == "First Observed - March 2025"
    ).sum()
)

print("\nOUTPUT FILES")

print(DETAIL_OUTPUT)
print(SUMMARY_OUTPUT)
print(CATEGORY_OUTPUT)
print(NUMERIC_OUTPUT)
print(CATEGORICAL_OUTPUT)

print("\nIMPORTANT NOTES")
print(
    "First observed sale date is based on the available "
    "transaction history, not the actual product launch date."
)

print(
    "Production cost units and currency have not been verified."
)

print(
    "Cohort comparisons are descriptive and do not establish "
    "why products entered the transaction dataset."
)
