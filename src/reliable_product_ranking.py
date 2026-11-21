
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "product_demand_metrics.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

MIN_EXPOSURE_DAYS = 14
MIN_ACTIVE_DAYS = 5
TOP_N = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading product demand metrics...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Run new_product_demand_analysis.py first."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Records loaded: {len(df):,}")


# --------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "Product ID",
    "Currency",
    "Category",
    "Sub Category",
    "Product_Cohort",
    "Total_Units",
    "Active_Sales_Days",
    "Calendar_Exposure_Days",
    "Units_Per_Calendar_Day",
    "Active_Day_Rate",
    "Units_Per_Active_Day",
    "Average_Selling_Price",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

numeric_columns = [
    "Total_Units",
    "Active_Sales_Days",
    "Calendar_Exposure_Days",
    "Units_Per_Calendar_Day",
    "Active_Day_Rate",
    "Units_Per_Active_Day",
    "Average_Selling_Price",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[
        "Product ID",
        "Currency",
        "Category",
        "Product_Cohort",
        "Total_Units",
        "Active_Sales_Days",
        "Calendar_Exposure_Days",
        "Units_Per_Calendar_Day",
    ]
)

# Exclude invalid or non-positive observations.
df = df[
    (df["Total_Units"] > 0)
    & (df["Active_Sales_Days"] > 0)
    & (df["Calendar_Exposure_Days"] > 0)
    & (df["Units_Per_Calendar_Day"] >= 0)
].copy()

print(f"Valid records after cleaning: {len(df):,}")


# --------------------------------------------------
# RELIABILITY FILTER
# --------------------------------------------------

df["Reliability_Status"] = "Insufficient Exposure"

reliable_mask = (
    (df["Calendar_Exposure_Days"] >= MIN_EXPOSURE_DAYS)
    & (df["Active_Sales_Days"] >= MIN_ACTIVE_DAYS)
)

df.loc[reliable_mask, "Reliability_Status"] = "Reliable"

reliable_df = df[reliable_mask].copy()
insufficient_df = df[~reliable_mask].copy()

print("\nReliability filtering")
print("---------------------")
print(f"Reliable records: {len(reliable_df):,}")
print(f"Insufficient records: {len(insufficient_df):,}")


# --------------------------------------------------
# RANK RELIABLE PRODUCTS
# --------------------------------------------------

group_columns = [
    "Product_Cohort",
    "Category",
    "Currency",
]

# Primary metric:
# Units_Per_Calendar_Day
#
# Supporting tie-breakers:
# Active_Day_Rate
# Units_Per_Active_Day
# Total_Units

sort_columns = group_columns + [
    "Units_Per_Calendar_Day",
    "Active_Day_Rate",
    "Units_Per_Active_Day",
    "Total_Units",
    "Product ID",
]

ascending_order = (
    [True] * len(group_columns)
    + [False, False, False, False, True]
)

reliable_df = reliable_df.sort_values(
    by=sort_columns,
    ascending=ascending_order,
)

# Exactly TOP_N rows per group, when enough qualify.
reliable_df["Demand_Rank"] = (
    reliable_df
    .groupby(group_columns)
    .cumcount()
    .add(1)
)

top_products = reliable_df[
    reliable_df["Demand_Rank"] <= TOP_N
].copy()


# --------------------------------------------------
# GROUP COVERAGE SUMMARY
# --------------------------------------------------

coverage = (
    df.groupby(group_columns)
    .agg(
        Total_Products=("Product ID", "nunique"),
        Reliable_Products=(
            "Reliability_Status",
            lambda x: (x == "Reliable").sum()
        ),
        Insufficient_Products=(
            "Reliability_Status",
            lambda x: (x != "Reliable").sum()
        ),
    )
    .reset_index()
)

coverage["Products_In_Top_10"] = (
    top_products.groupby(group_columns)["Product ID"]
    .nunique()
    .reindex(
        pd.MultiIndex.from_frame(coverage[group_columns])
    )
    .fillna(0)
    .values
)

coverage["Reliability_Coverage_Percent"] = (
    coverage["Reliable_Products"]
    / coverage["Total_Products"]
    * 100
)


# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------

ranking_file = OUTPUT_DIR / "reliable_product_rankings.csv"
insufficient_file = OUTPUT_DIR / "insufficient_exposure_products.csv"
coverage_file = OUTPUT_DIR / "ranking_coverage_summary.csv"

top_products.to_csv(ranking_file, index=False)
insufficient_df.to_csv(insufficient_file, index=False)
coverage.to_csv(coverage_file, index=False)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\nTop reliable products")
print("---------------------")

display_columns = [
    "Product ID",
    "Product_Cohort",
    "Category",
    "Currency",
    "Total_Units",
    "Calendar_Exposure_Days",
    "Active_Sales_Days",
    "Units_Per_Calendar_Day",
    "Active_Day_Rate",
    "Units_Per_Active_Day",
    "Demand_Rank",
]

print(
    top_products[display_columns]
    .head(20)
    .to_string(index=False)
)

print("\nRanking coverage")
print("----------------")

print(
    coverage[
        group_columns
        + [
            "Total_Products",
            "Reliable_Products",
            "Insufficient_Products",
            "Products_In_Top_10",
            "Reliability_Coverage_Percent",
        ]
    ].to_string(index=False)
)

print("\nFiles saved successfully:")
print(ranking_file)
print(insufficient_file)
print(coverage_file)

print("\nProduct ranking analysis completed.")
