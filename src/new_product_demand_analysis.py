
import pandas as pd
import numpy as np

from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DAILY_FILE = (
    BASE_DIR
    / "data/processed/product_store_daily_sales.csv"
)

COHORT_FILE = (
    BASE_DIR
    / "data/processed/product_cohort_detail.csv"
)

OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCT_OUTPUT = OUTPUT_DIR / "product_demand_metrics.csv"
COHORT_OUTPUT = OUTPUT_DIR / "cohort_demand_summary.csv"
SUBCATEGORY_OUTPUT = OUTPUT_DIR / "subcategory_demand_ranking.csv"
TOP_PRODUCTS_OUTPUT = OUTPUT_DIR / "top_products_by_demand.csv"
MONTHLY_OUTPUT = OUTPUT_DIR / "cohort_monthly_demand.csv"

# Use the actual end of the available transaction period.
DATA_END_DATE = pd.Timestamp("2025-03-18")

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("NEW PRODUCT DEMAND INTELLIGENCE")
print("=" * 70)

print("\nLoading daily sales and product cohort data...")

daily = pd.read_csv(
    DAILY_FILE,
    dtype={"Product ID": str},
    parse_dates=["Date"]
)

cohorts = pd.read_csv(
    COHORT_FILE,
    dtype={"Product ID": str},
    parse_dates=["First_Observed_Sale", "Last_Observed_Sale"]
)

print("Daily sales records:", len(daily))
print("Cohort products:", len(cohorts))

# ============================================================
# 2. VALIDATE INPUT DATA
# ============================================================

required_daily = [
    "Product ID",
    "Store ID",
    "Date",
    "Currency",
    "Category",
    "Sub Category",
    "Quantity",
    "Revenue",
    "Transactions"
]

required_cohort = [
    "Product ID",
    "Product_Cohort",
    "First_Observed_Sale",
    "Category",
    "Sub Category",
    "Production Cost"
]

for col in required_daily:
    if col not in daily.columns:
        raise ValueError(f"Missing daily sales column: {col}")

for col in required_cohort:
    if col not in cohorts.columns:
        raise ValueError(f"Missing cohort column: {col}")

if cohorts["Product ID"].duplicated().any():
    raise ValueError("Duplicate Product IDs in cohort file.")

# ============================================================
# 3. CLEAN DATA
# ============================================================

daily["Product ID"] = daily["Product ID"].astype(str).str.strip()

daily["Currency"] = daily["Currency"].astype(str).str.strip()

for col in ["Quantity", "Revenue", "Transactions"]:
    daily[col] = pd.to_numeric(
        daily[col],
        errors="coerce"
    )

daily = daily.dropna(
    subset=[
        "Product ID",
        "Date",
        "Currency",
        "Quantity",
        "Revenue"
    ]
).copy()

daily = daily[
    (daily["Quantity"] > 0)
    & (daily["Revenue"] > 0)
].copy()

# Restrict analysis to the known transaction window.
daily = daily[
    daily["Date"] <= DATA_END_DATE
].copy()

# ============================================================
# 4. AGGREGATE ACROSS STORES
# ============================================================

print("\nAggregating sales across stores...")

product_daily = (
    daily.groupby(
        [
            "Product ID",
            "Date",
            "Currency",
            "Category",
            "Sub Category"
        ],
        as_index=False
    )
    .agg(
        Daily_Units=("Quantity", "sum"),
        Daily_Revenue=("Revenue", "sum"),
        Daily_Transactions=("Transactions", "sum")
    )
)

print("Product-date-currency records:", len(product_daily))

# ============================================================
# 5. PRODUCT-LEVEL DEMAND METRICS
# ============================================================

print("\nCalculating product demand metrics...")

product_metrics = (
    product_daily.groupby(
        ["Product ID", "Currency", "Category", "Sub Category"],
        as_index=False
    )
    .agg(
        Total_Units=("Daily_Units", "sum"),
        Total_Revenue=("Daily_Revenue", "sum"),
        Total_Transactions=("Daily_Transactions", "sum"),
        First_Currency_Sale=("Date", "min"),
        Last_Currency_Sale=("Date", "max"),
        Active_Sales_Days=("Date", "nunique")
    )
)

# Merge cohort and product attributes.
cohort_attributes = cohorts[
    [
        "Product ID",
        "Product_Cohort",
        "Production Cost",
        "Color",
        "Sizes",
        "Description EN",
        "First_Observed_Sale"
    ]
].copy()

product_metrics = product_metrics.merge(
    cohort_attributes,
    on="Product ID",
    how="left",
    validate="many_to_one"
)

if product_metrics["Product_Cohort"].isna().any():
    raise ValueError("Some products have no cohort assignment.")

# Calendar exposure from first observed sale in this currency
# until the end of the available transaction period.
product_metrics["Calendar_Exposure_Days"] = (
    DATA_END_DATE - product_metrics["First_Currency_Sale"]
).dt.days + 1

product_metrics["Active_Day_Rate"] = (
    product_metrics["Active_Sales_Days"]
    / product_metrics["Calendar_Exposure_Days"]
)

product_metrics["Units_Per_Calendar_Day"] = (
    product_metrics["Total_Units"]
    / product_metrics["Calendar_Exposure_Days"]
)

product_metrics["Revenue_Per_Calendar_Day"] = (
    product_metrics["Total_Revenue"]
    / product_metrics["Calendar_Exposure_Days"]
)

product_metrics["Units_Per_Active_Day"] = (
    product_metrics["Total_Units"]
    / product_metrics["Active_Sales_Days"]
)

product_metrics["Average_Selling_Price"] = (
    product_metrics["Total_Revenue"]
    / product_metrics["Total_Units"]
)

product_metrics.to_csv(
    PRODUCT_OUTPUT,
    index=False
)

print("Saved:", PRODUCT_OUTPUT)
print("Product-currency records:", len(product_metrics))

# ============================================================
# 6. COHORT DEMAND COMPARISON
# ============================================================

print("\nComparing historical and March cohorts...")

cohort_summary = (
    product_metrics.groupby(
        ["Product_Cohort", "Category", "Currency"],
        as_index=False
    )
    .agg(
        Product_Count=("Product ID", "nunique"),
        Total_Units=("Total_Units", "sum"),
        Total_Revenue=("Total_Revenue", "sum"),
        Mean_Units_Per_Day=("Units_Per_Calendar_Day", "mean"),
        Median_Units_Per_Day=("Units_Per_Calendar_Day", "median"),
        Mean_Active_Day_Rate=("Active_Day_Rate", "mean"),
        Mean_Units_Per_Active_Day=("Units_Per_Active_Day", "mean"),
        Mean_Selling_Price=("Average_Selling_Price", "mean")
    )
)

cohort_summary.to_csv(
    COHORT_OUTPUT,
    index=False
)

print(cohort_summary.to_string(index=False))
print("\nSaved:", COHORT_OUTPUT)

# ============================================================
# 7. SUBCATEGORY DEMAND RANKING
# ============================================================

print("\nRanking demand by subcategory...")

subcategory_summary = (
    product_metrics.groupby(
        [
            "Product_Cohort",
            "Category",
            "Sub Category",
            "Currency"
        ],
        as_index=False
    )
    .agg(
        Product_Count=("Product ID", "nunique"),
        Total_Units=("Total_Units", "sum"),
        Total_Revenue=("Total_Revenue", "sum"),
        Mean_Units_Per_Day=("Units_Per_Calendar_Day", "mean"),
        Median_Units_Per_Day=("Units_Per_Calendar_Day", "median"),
        Mean_Active_Day_Rate=("Active_Day_Rate", "mean"),
        Mean_Selling_Price=("Average_Selling_Price", "mean")
    )
)

subcategory_summary["Demand_Rank"] = (
    subcategory_summary.groupby(
        ["Product_Cohort", "Category", "Currency"]
    )["Mean_Units_Per_Day"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

subcategory_summary = subcategory_summary.sort_values(
    [
        "Product_Cohort",
        "Category",
        "Currency",
        "Demand_Rank"
    ]
)

subcategory_summary.to_csv(
    SUBCATEGORY_OUTPUT,
    index=False
)

print(subcategory_summary.to_string(index=False))
print("\nSaved:", SUBCATEGORY_OUTPUT)

# ============================================================
# 8. TOP PRODUCTS BY NORMALIZED DEMAND
# ============================================================

print("\nFinding top-demand products...")

product_metrics["Demand_Rank"] = (
    product_metrics.groupby(
        ["Product_Cohort", "Category", "Currency"]
    )["Units_Per_Calendar_Day"]
    .rank(method="dense", ascending=False)
    .astype(int)
)

top_products = product_metrics[
    product_metrics["Demand_Rank"] <= 10
].copy()

top_products = top_products.sort_values(
    [
        "Product_Cohort",
        "Category",
        "Currency",
        "Demand_Rank"
    ]
)

top_products.to_csv(
    TOP_PRODUCTS_OUTPUT,
    index=False
)

print(
    top_products[
        [
            "Product ID",
            "Product_Cohort",
            "Category",
            "Currency",
            "Total_Units",
            "Units_Per_Calendar_Day",
            "Average_Selling_Price",
            "Demand_Rank"
        ]
    ].to_string(index=False)
)

print("\nSaved:", TOP_PRODUCTS_OUTPUT)

# ============================================================
# 9. MONTHLY COHORT DEMAND
# ============================================================

print("\nCalculating monthly cohort demand...")

product_daily = product_daily.merge(
    cohorts[["Product ID", "Product_Cohort"]],
    on="Product ID",
    how="left",
    validate="many_to_one"
)

product_daily["Month"] = (
    product_daily["Date"].dt.to_period("M").astype(str)
)

monthly_summary = (
    product_daily.groupby(
        [
            "Product_Cohort",
            "Month",
            "Category",
            "Currency"
        ],
        as_index=False
    )
    .agg(
        Product_Count=("Product ID", "nunique"),
        Units_Sold=("Daily_Units", "sum"),
        Revenue=("Daily_Revenue", "sum"),
        Active_Product_Days=("Product ID", "size")
    )
)

monthly_summary.to_csv(
    MONTHLY_OUTPUT,
    index=False
)

print(monthly_summary.to_string(index=False))
print("\nSaved:", MONTHLY_OUTPUT)

# ============================================================
# 10. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("NEW PRODUCT DEMAND ANALYSIS COMPLETED")
print("=" * 70)

print("Product-currency records:", len(product_metrics))

print(
    "Historical cohort products:",
    product_metrics[
        product_metrics["Product_Cohort"]
        == "Historical - 2023 to 2024"
    ]["Product ID"].nunique()
)

print(
    "March cohort products:",
    product_metrics[
        product_metrics["Product_Cohort"]
        == "First Observed - March 2025"
    ]["Product ID"].nunique()
)

print("\nOUTPUT FILES")
print(PRODUCT_OUTPUT)
print(COHORT_OUTPUT)
print(SUBCATEGORY_OUTPUT)
print(TOP_PRODUCTS_OUTPUT)
print(MONTHLY_OUTPUT)

print("\nMETHODOLOGY NOTES")

print(
    "1. Currency groups are analysed independently."
)

print(
    "2. Demand rates use calendar days from first observed "
    "sale in each currency to March 18, 2025."
)

print(
    "3. Active-day demand is conditional on days with recorded sales."
)

print(
    "4. Historical and March cohorts have different observation "
    "windows and should not be compared using raw totals alone."
)

print(
    "5. Revenue is not profit."
)

print(
    "6. First observed sale does not establish actual product launch date."
)

print("\nANALYSIS FINISHED")
