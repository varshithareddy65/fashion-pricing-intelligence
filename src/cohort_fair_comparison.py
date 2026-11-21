
from pathlib import Path

import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRANSACTIONS_FILE = BASE_DIR / "data/raw/transactions.csv"
PRODUCTS_FILE = BASE_DIR / "data/raw/products.csv"
DEMAND_METRICS_FILE = (
    BASE_DIR / "data/processed/product_demand_metrics.csv"
)

OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 14


# ==================================================
# LOAD DATA
# ==================================================

print("Loading transaction, product and cohort data...")

transactions = pd.read_csv(TRANSACTIONS_FILE)
products = pd.read_csv(PRODUCTS_FILE)
demand_metrics = pd.read_csv(DEMAND_METRICS_FILE)

print(f"Transactions loaded: {len(transactions):,}")
print(f"Products loaded: {len(products):,}")
print(f"Demand metric records loaded: {len(demand_metrics):,}")


# ==================================================
# VALIDATE SCHEMAS
# ==================================================

transaction_columns = [
    "Product ID",
    "Date",
    "Quantity",
    "Currency",
    "Transaction Type",
]

product_columns = [
    "Product ID",
    "Category",
    "Sub Category",
]

cohort_columns = [
    "Product ID",
    "Product_Cohort",
]

for name, frame, required in [
    ("transactions", transactions, transaction_columns),
    ("products", products, product_columns),
    ("demand metrics", demand_metrics, cohort_columns),
]:
    missing = [col for col in required if col not in frame.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {name}: {missing}"
        )


# ==================================================
# PREPARE DATA
# ==================================================

transactions["Date"] = pd.to_datetime(
    transactions["Date"],
    errors="coerce"
)

transactions["Quantity"] = pd.to_numeric(
    transactions["Quantity"],
    errors="coerce"
)

transactions["Product ID"] = pd.to_numeric(
    transactions["Product ID"],
    errors="coerce"
)

products["Product ID"] = pd.to_numeric(
    products["Product ID"],
    errors="coerce"
)

demand_metrics["Product ID"] = pd.to_numeric(
    demand_metrics["Product ID"],
    errors="coerce"
)

transactions = transactions.dropna(
    subset=[
        "Product ID",
        "Date",
        "Quantity",
        "Currency",
        "Transaction Type",
    ]
)

products = products.dropna(
    subset=["Product ID", "Category", "Sub Category"]
)

# Product IDs must map to exactly one cohort.
cohort_map = demand_metrics[
    ["Product ID", "Product_Cohort"]
].drop_duplicates()

cohort_counts = (
    cohort_map.groupby("Product ID")["Product_Cohort"]
    .nunique()
)

if (cohort_counts > 1).any():
    raise ValueError(
        "Some Product IDs map to multiple product cohorts."
    )

cohort_map = cohort_map.drop_duplicates(
    subset=["Product ID"]
)

# Product dimension must contain one row per product.
if products["Product ID"].duplicated().any():
    raise ValueError(
        "Duplicate Product IDs found in products.csv."
    )


# ==================================================
# JOIN PRODUCT AND COHORT INFORMATION
# ==================================================

print("\nJoining product and cohort information...")

sales = transactions.merge(
    products[
        ["Product ID", "Category", "Sub Category"]
    ],
    on="Product ID",
    how="left",
    validate="many_to_one",
)

sales = sales.merge(
    cohort_map,
    on="Product ID",
    how="left",
    validate="many_to_one",
)

unmatched_products = sales["Category"].isna().sum()
unmatched_cohorts = sales["Product_Cohort"].isna().sum()

print(f"Transactions without product category: {unmatched_products:,}")
print(f"Transactions without cohort mapping: {unmatched_cohorts:,}")

if unmatched_products > 0 or unmatched_cohorts > 0:
    raise ValueError(
        "Some transactions could not be mapped to product or cohort data."
    )


# ==================================================
# FILTER SALE TRANSACTIONS
# ==================================================

sales["Transaction Type"] = (
    sales["Transaction Type"].astype(str).str.strip()
)

sales = sales[
    sales["Transaction Type"].str.casefold() == "sale"
].copy()

# Keep positive quantities for demand analysis.
sales = sales[sales["Quantity"] > 0].copy()

print(f"Sale records used: {len(sales):,}")

if sales.empty:
    raise ValueError("No valid sale transactions found.")


# ==================================================
# AGGREGATE DAILY PRODUCT DEMAND
# ==================================================

print("\nAggregating daily product demand...")

daily = (
    sales.groupby(
        [
            "Product ID",
            "Currency",
            "Date",
            "Category",
            "Sub Category",
            "Product_Cohort",
        ],
        as_index=False,
    )
    .agg(
        Daily_Units=("Quantity", "sum")
    )
)

print(f"Daily product-currency records: {len(daily):,}")


# ==================================================
# FIND FIRST OBSERVED SALE DATE
# ==================================================

print("\nFinding first observed sale dates...")

keys = [
    "Product ID",
    "Currency",
    "Category",
    "Sub Category",
    "Product_Cohort",
]

first_sales = (
    daily.groupby(keys, as_index=False)
    .agg(
        First_Observed_Sale_Date=("Date", "min")
    )
)

# Latest observed date available for each currency.
currency_end_dates = (
    daily.groupby("Currency")["Date"]
    .max()
    .to_dict()
)

first_sales["Window_End_Date"] = (
    first_sales["First_Observed_Sale_Date"]
    + pd.Timedelta(days=WINDOW_DAYS - 1)
)

first_sales["Currency_Data_End_Date"] = (
    first_sales["Currency"].map(currency_end_dates)
)

# Keep only complete 14-calendar-day observation windows.
first_sales["Complete_14_Day_Window"] = (
    first_sales["Window_End_Date"]
    <= first_sales["Currency_Data_End_Date"]
)

complete_windows = first_sales[
    first_sales["Complete_14_Day_Window"]
].copy()

incomplete_windows = first_sales[
    ~first_sales["Complete_14_Day_Window"]
].copy()

print(f"Product-currency windows: {len(first_sales):,}")
print(f"Complete windows: {len(complete_windows):,}")
print(f"Incomplete windows excluded: {len(incomplete_windows):,}")


# ==================================================
# ASSIGN DAILY SALES TO FIRST 14-DAY WINDOWS
# ==================================================

print("\nCalculating demand during the first 14 days...")

window_daily = daily.merge(
    complete_windows[
        keys + ["First_Observed_Sale_Date"]
    ],
    on=keys,
    how="inner",
    validate="many_to_one",
)

window_daily["Day_From_First_Sale"] = (
    window_daily["Date"]
    - window_daily["First_Observed_Sale_Date"]
).dt.days

window_daily = window_daily[
    (window_daily["Day_From_First_Sale"] >= 0)
    & (window_daily["Day_From_First_Sale"] < WINDOW_DAYS)
].copy()


# ==================================================
# PRODUCT-LEVEL 14-DAY METRICS
# ==================================================

product_metrics = (
    window_daily.groupby(keys, as_index=False)
    .agg(
        First_14_Day_Units=("Daily_Units", "sum"),
        Active_Sales_Days=("Date", "nunique"),
    )
)

product_metrics = product_metrics.merge(
    complete_windows[
        keys
        + [
            "First_Observed_Sale_Date",
            "Window_End_Date",
        ]
    ],
    on=keys,
    how="left",
    validate="one_to_one",
)

product_metrics["Calendar_Exposure_Days"] = WINDOW_DAYS

product_metrics["Units_Per_Calendar_Day"] = (
    product_metrics["First_14_Day_Units"] / WINDOW_DAYS
)

product_metrics["Active_Day_Rate"] = (
    product_metrics["Active_Sales_Days"] / WINDOW_DAYS
)

product_metrics["Units_Per_Active_Day"] = (
    product_metrics["First_14_Day_Units"]
    / product_metrics["Active_Sales_Days"]
)

product_metrics["No_Sale_Days"] = (
    WINDOW_DAYS - product_metrics["Active_Sales_Days"]
)

product_metrics["Cohort_Window_Status"] = "Complete"


# ==================================================
# BUILD SUBCATEGORY COMPARISON
# ==================================================

print("\nBuilding subcategory comparison...")

comparison_columns = [
    "Category",
    "Sub Category",
    "Currency",
    "Product_Cohort",
]

subcategory_summary = (
    product_metrics.groupby(
        comparison_columns,
        as_index=False,
    )
    .agg(
        Products_Compared=("Product ID", "nunique"),

        Mean_14_Day_Units=(
            "First_14_Day_Units", "mean"
        ),

        Median_14_Day_Units=(
            "First_14_Day_Units", "median"
        ),

        Mean_Units_Per_Calendar_Day=(
            "Units_Per_Calendar_Day", "mean"
        ),

        Median_Units_Per_Calendar_Day=(
            "Units_Per_Calendar_Day", "median"
        ),

        Mean_Active_Day_Rate=(
            "Active_Day_Rate", "mean"
        ),

        Mean_Units_Per_Active_Day=(
            "Units_Per_Active_Day", "mean"
        ),
    )
)


# ==================================================
# PIVOT HISTORICAL AND MARCH COHORTS
# ==================================================

print("\nComparing historical and March cohorts...")

metrics_to_compare = [
    "Products_Compared",
    "Mean_14_Day_Units",
    "Median_14_Day_Units",
    "Mean_Units_Per_Calendar_Day",
    "Median_Units_Per_Calendar_Day",
    "Mean_Active_Day_Rate",
    "Mean_Units_Per_Active_Day",
]

subcategory_comparison = (
    subcategory_summary
    .pivot(
        index=["Category", "Sub Category", "Currency"],
        columns="Product_Cohort",
        values=metrics_to_compare,
    )
)

subcategory_comparison.columns = [
    f"{metric}__{cohort}"
    for metric, cohort in subcategory_comparison.columns
]

subcategory_comparison = (
    subcategory_comparison.reset_index()
)

march_name = "First Observed - March 2025"
historical_name = "Historical - 2023 to 2024"

for metric in metrics_to_compare:
    march_col = f"{metric}__{march_name}"
    historical_col = f"{metric}__{historical_name}"

    if march_col in subcategory_comparison.columns:
        if historical_col in subcategory_comparison.columns:
            subcategory_comparison[
                f"{metric}_March_Minus_Historical"
            ] = (
                subcategory_comparison[march_col]
                - subcategory_comparison[historical_col]
            )

            if metric not in ["Products_Compared"]:
                subcategory_comparison[
                    f"{metric}_March_to_Historical_Ratio"
                ] = (
                    subcategory_comparison[march_col]
                    / subcategory_comparison[historical_col].replace(
                        0, pd.NA
                    )
                )


# ==================================================
# BUILD CATEGORY-LEVEL COMPARISON
# ==================================================

category_columns = [
    "Category",
    "Currency",
    "Product_Cohort",
]

category_summary = (
    product_metrics.groupby(
        category_columns,
        as_index=False,
    )
    .agg(
        Products_Compared=("Product ID", "nunique"),
        Mean_14_Day_Units=("First_14_Day_Units", "mean"),
        Median_14_Day_Units=("First_14_Day_Units", "median"),
        Mean_Units_Per_Calendar_Day=(
            "Units_Per_Calendar_Day", "mean"
        ),
        Mean_Active_Day_Rate=("Active_Day_Rate", "mean"),
        Mean_Units_Per_Active_Day=(
            "Units_Per_Active_Day", "mean"
        ),
    )
)

category_comparison = (
    category_summary
    .pivot(
        index=["Category", "Currency"],
        columns="Product_Cohort",
        values=[
            "Products_Compared",
            "Mean_14_Day_Units",
            "Median_14_Day_Units",
            "Mean_Units_Per_Calendar_Day",
            "Mean_Active_Day_Rate",
            "Mean_Units_Per_Active_Day",
        ],
    )
)

category_comparison.columns = [
    f"{metric}__{cohort}"
    for metric, cohort in category_comparison.columns
]

category_comparison = category_comparison.reset_index()


# ==================================================
# COVERAGE SUMMARY
# ==================================================

coverage_summary = (
    first_sales.groupby(
        ["Product_Cohort", "Category", "Currency"],
        as_index=False,
    )
    .agg(
        Total_Product_Currency_Records=(
            "Product ID", "count"
        ),
        Complete_14_Day_Windows=(
            "Complete_14_Day_Window", "sum"
        ),
    )
)

coverage_summary["Incomplete_14_Day_Windows"] = (
    coverage_summary["Total_Product_Currency_Records"]
    - coverage_summary["Complete_14_Day_Windows"]
)

coverage_summary["Complete_Window_Percent"] = (
    coverage_summary["Complete_14_Day_Windows"]
    / coverage_summary["Total_Product_Currency_Records"]
    * 100
)


# ==================================================
# RANK SUBCATEGORIES WITHIN EACH COHORT
# ==================================================

ranked_subcategories = subcategory_summary.sort_values(
    by=[
        "Product_Cohort",
        "Category",
        "Currency",
        "Mean_Units_Per_Calendar_Day",
    ],
    ascending=[True, True, True, False],
).copy()

ranked_subcategories["Subcategory_Demand_Rank"] = (
    ranked_subcategories.groupby(
        ["Product_Cohort", "Category", "Currency"]
    )
    .cumcount()
    .add(1)
)


# ==================================================
# SAVE OUTPUTS
# ==================================================

outputs = {
    "first_14_day_product_metrics.csv": product_metrics,
    "first_14_day_subcategory_summary.csv": subcategory_summary,
    "first_14_day_subcategory_comparison.csv": subcategory_comparison,
    "first_14_day_category_comparison.csv": category_comparison,
    "first_14_day_coverage_summary.csv": coverage_summary,
    "first_14_day_subcategory_rankings.csv": ranked_subcategories,
    "incomplete_14_day_windows.csv": incomplete_windows,
}

for filename, frame in outputs.items():
    output_path = OUTPUT_DIR / filename
    frame.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 65)
print("FAIR 14-DAY COHORT COMPARISON")
print("=" * 65)

print("\nCoverage summary:")
print(
    coverage_summary.to_string(index=False)
)

print("\nCategory-level comparison:")
print(
    category_comparison.head(20).to_string(index=False)
)

print("\nTop subcategories by first-14-day demand:")
print(
    ranked_subcategories.groupby(
        ["Product_Cohort", "Category", "Currency"],
        group_keys=False,
    ).head(3).to_string(index=False)
)

print("\nProduct-level records compared:", len(product_metrics))
print("Incomplete windows excluded:", len(incomplete_windows))

print("\nFair cohort comparison completed successfully.")
