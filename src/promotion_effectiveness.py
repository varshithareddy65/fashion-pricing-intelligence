
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRANSACTIONS_FILE = BASE_DIR / "data/raw/transactions.csv"
PRODUCTS_FILE = BASE_DIR / "data/raw/products.csv"
CAMPAIGNS_FILE = BASE_DIR / "data/raw/discounts.csv"

OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "promotion_effectiveness.csv"

COMPARISON_DAYS = 28

# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading transaction, product and campaign data...")

transactions = pd.read_csv(TRANSACTIONS_FILE)
products = pd.read_csv(PRODUCTS_FILE)
campaigns = pd.read_csv(CAMPAIGNS_FILE)

print("Transactions loaded:", len(transactions))
print("Products loaded:", len(products))
print("Campaign rows loaded:", len(campaigns))

# ============================================================
# 2. VALIDATE TRANSACTION AND PRODUCT COLUMNS
# ============================================================

required_transaction_columns = [
    "Invoice ID",
    "Product ID",
    "Date",
    "Currency",
    "Transaction Type",
    "Line Total",
    "Quantity"
]

required_product_columns = [
    "Product ID",
    "Category",
    "Sub Category"
]

for df, required, name in [
    (transactions, required_transaction_columns, "transactions.csv"),
    (products, required_product_columns, "products.csv")
]:
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {name}: {missing}"
        )

# ============================================================
# 3. PREPARE PRODUCT DATA
# ============================================================

products = products[
    ["Product ID", "Category", "Sub Category"]
].copy()

products["Product ID"] = (
    products["Product ID"].astype(str).str.strip()
)

if products["Product ID"].duplicated().any():
    raise ValueError(
        "Duplicate Product IDs found in products.csv."
    )

products["Category"] = products["Category"].str.strip()
products["Sub Category"] = products["Sub Category"].str.strip()

# ============================================================
# 4. CLEAN TRANSACTIONS
# ============================================================

transactions["Product ID"] = (
    transactions["Product ID"].astype(str).str.strip()
)

transactions["Date"] = pd.to_datetime(
    transactions["Date"],
    errors="coerce"
)

transactions["Line Total"] = pd.to_numeric(
    transactions["Line Total"],
    errors="coerce"
)

transactions["Quantity"] = pd.to_numeric(
    transactions["Quantity"],
    errors="coerce"
)

transactions = transactions.dropna(
    subset=[
        "Date",
        "Currency",
        "Transaction Type",
        "Line Total",
        "Quantity"
    ]
).copy()

transactions["Date"] = transactions["Date"].dt.normalize()

# Keep sale transactions only.
sales = transactions[
    transactions["Transaction Type"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "sale"
].copy()

print("\nTRANSACTION CLEANING")
print("Sale rows:", len(sales))

# ============================================================
# 5. MERGE SALES WITH PRODUCT INFORMATION
# ============================================================

print("\nMerging transactions with product information...")

sales = sales.merge(
    products,
    on="Product ID",
    how="left",
    validate="many_to_one",
    indicator=True
)

unmatched_products = (
    sales["_merge"] == "left_only"
).sum()

print("Unmatched product transactions:", unmatched_products)

sales = sales.drop(columns=["_merge"])

sales = sales.dropna(
    subset=["Category", "Sub Category"]
).copy()

print("Matched sale rows:", len(sales))
print("Categories:", sales["Category"].nunique())
print("Subcategories:", sales["Sub Category"].nunique())

# ============================================================
# 6. AGGREGATE DAILY SALES
# ============================================================

print("\nAggregating daily sales...")

daily_sales = (
    sales.groupby(
        ["Date", "Category", "Sub Category", "Currency"],
        as_index=False
    )
    .agg(
        Revenue=("Line Total", "sum"),
        Units=("Quantity", "sum"),
        Transactions=("Invoice ID", "nunique")
    )
)

data_start = daily_sales["Date"].min()
data_end = daily_sales["Date"].max()

print("Daily sales records:", len(daily_sales))
print("Transaction date range:", data_start, "to", data_end)

# ============================================================
# 7. VALIDATE AND CLEAN CAMPAIGN DATA
# ============================================================

required_campaign_columns = [
    "Start",
    "End",
    "Discont",
    "Description",
    "Category",
    "Sub Category"
]

missing = [
    col for col in required_campaign_columns
    if col not in campaigns.columns
]

if missing:
    raise ValueError(
        f"Missing columns in discounts.csv: {missing}"
    )

# Rename actual dataset columns into consistent names.
campaigns = campaigns.rename(
    columns={
        "Discont": "Discount",
        "Description": "Campaign_Name",
        "Sub Category": "Sub_Category"
    }
)

campaigns["Start"] = pd.to_datetime(
    campaigns["Start"],
    errors="coerce"
).dt.normalize()

campaigns["End"] = pd.to_datetime(
    campaigns["End"],
    errors="coerce"
).dt.normalize()

campaigns["Discount"] = pd.to_numeric(
    campaigns["Discount"],
    errors="coerce"
)

# Remove missing values before converting text columns.
campaigns = campaigns.dropna(
    subset=[
        "Start",
        "End",
        "Discount",
        "Campaign_Name",
        "Category",
        "Sub_Category"
    ]
).copy()

campaigns["Campaign_Name"] = (
    campaigns["Campaign_Name"].astype(str).str.strip()
)

campaigns["Category"] = (
    campaigns["Category"].astype(str).str.strip()
)

campaigns["Sub_Category"] = (
    campaigns["Sub_Category"].astype(str).str.strip()
)

campaigns = campaigns[
    campaigns["End"] >= campaigns["Start"]
].copy()

# Generate an ID for each campaign row.
campaigns = campaigns.reset_index(drop=True)

campaigns["Campaign_ID"] = [
    f"C{i:03d}" for i in range(1, len(campaigns) + 1)
]

all_campaigns = campaigns.copy()

print("\nCAMPAIGN DATA")
print("Valid campaign rows:", len(all_campaigns))
print("Unique campaign descriptions:", all_campaigns["Campaign_Name"].nunique())

# ============================================================
# 8. FILTER FULLY OBSERVABLE CAMPAIGNS
# ============================================================

campaigns = campaigns[
    (
        (
            campaigns["Start"]
            - pd.Timedelta(days=COMPARISON_DAYS)
        ) >= data_start
    )
    &
    (
        (
            campaigns["End"]
            + pd.Timedelta(days=COMPARISON_DAYS)
        ) <= data_end
    )
].copy()

print("\nCAMPAIGN DATE VALIDATION")
print("Transaction start:", data_start)
print("Transaction end:", data_end)
print("Original valid campaigns:", len(all_campaigns))
print("Fully observable campaigns:", len(campaigns))
print("Excluded campaigns:", len(all_campaigns) - len(campaigns))

if campaigns.empty:
    raise ValueError(
        "No campaigns have complete before/during/after windows."
    )

# ============================================================
# 9. CAMPAIGN DATE LOOKUP
# ============================================================

campaign_periods = all_campaigns[
    [
        "Campaign_ID",
        "Start",
        "End",
        "Category",
        "Sub_Category"
    ]
].copy()

# ============================================================
# 10. CAMPAIGN PERFORMANCE ANALYSIS
# ============================================================

results = []

print("\nAnalysing promotion effectiveness...")

for _, campaign in campaigns.iterrows():

    campaign_id = campaign["Campaign_ID"]
    campaign_name = campaign["Campaign_Name"]

    start_date = campaign["Start"]
    end_date = campaign["End"]

    category = campaign["Category"]
    sub_category = campaign["Sub_Category"]
    discount = campaign["Discount"]

    before_start = start_date - pd.Timedelta(
        days=COMPARISON_DAYS
    )

    before_end = start_date - pd.Timedelta(days=1)

    during_start = start_date
    during_end = end_date

    after_start = end_date + pd.Timedelta(days=1)

    after_end = end_date + pd.Timedelta(
        days=COMPARISON_DAYS
    )

    # --------------------------------------------------------
    # Identify other campaigns in the same product group.
    # --------------------------------------------------------

    other_campaigns = campaign_periods[
        (
            campaign_periods["Campaign_ID"] != campaign_id
        )
        &
        (
            campaign_periods["Category"] == category
        )
        &
        (
            campaign_periods["Sub_Category"] == sub_category
        )
    ]

    excluded_dates = set()

    for _, other in other_campaigns.iterrows():

        overlap_start = max(
            before_start,
            other["Start"]
        )

        overlap_end = min(
            after_end,
            other["End"]
        )

        if overlap_start <= overlap_end:

            dates = pd.date_range(
                overlap_start,
                overlap_end,
                freq="D"
            )

            excluded_dates.update(dates)

    # --------------------------------------------------------
    # Filter daily data for this campaign's product group.
    # --------------------------------------------------------

    category_sales = daily_sales[
        (daily_sales["Category"] == category)
        &
        (daily_sales["Sub Category"] == sub_category)
    ].copy()

    if category_sales.empty:
        continue

    # --------------------------------------------------------
    # Analyse each currency separately.
    # --------------------------------------------------------

    for currency in sorted(
        category_sales["Currency"].dropna().unique()
    ):

        currency_sales = category_sales[
            category_sales["Currency"] == currency
        ].copy()

        # BEFORE PERIOD
        before_data = currency_sales[
            (currency_sales["Date"] >= before_start)
            &
            (currency_sales["Date"] <= before_end)
        ].copy()

        # DURING PERIOD
        during_data = currency_sales[
            (currency_sales["Date"] >= during_start)
            &
            (currency_sales["Date"] <= during_end)
        ].copy()

        # AFTER PERIOD
        after_data = currency_sales[
            (currency_sales["Date"] >= after_start)
            &
            (currency_sales["Date"] <= after_end)
        ].copy()

        # Exclude other promotion dates from baseline periods.
        before_data = before_data[
            ~before_data["Date"].isin(excluded_dates)
        ]

        after_data = after_data[
            ~after_data["Date"].isin(excluded_dates)
        ]

        # ----------------------------------------------------
        # Count valid calendar days.
        # ----------------------------------------------------

        before_dates = pd.date_range(
            before_start,
            before_end,
            freq="D"
        )

        during_dates = pd.date_range(
            during_start,
            during_end,
            freq="D"
        )

        after_dates = pd.date_range(
            after_start,
            after_end,
            freq="D"
        )

        before_dates = [
            date for date in before_dates
            if date not in excluded_dates
        ]

        after_dates = [
            date for date in after_dates
            if date not in excluded_dates
        ]

        if len(before_dates) == 0 or len(after_dates) == 0:
            continue

        # ----------------------------------------------------
        # Calculate total revenue and units.
        # ----------------------------------------------------

        before_revenue = before_data["Revenue"].sum()
        during_revenue = during_data["Revenue"].sum()
        after_revenue = after_data["Revenue"].sum()

        before_units = before_data["Units"].sum()
        during_units = during_data["Units"].sum()
        after_units = after_data["Units"].sum()

        # ----------------------------------------------------
        # Calculate average daily performance.
        # ----------------------------------------------------

        avg_before_revenue = (
            before_revenue / len(before_dates)
        )

        avg_during_revenue = (
            during_revenue / len(during_dates)
        )

        avg_after_revenue = (
            after_revenue / len(after_dates)
        )

        avg_before_units = (
            before_units / len(before_dates)
        )

        avg_during_units = (
            during_units / len(during_dates)
        )

        avg_after_units = (
            after_units / len(after_dates)
        )

        # ----------------------------------------------------
        # Calculate percentage changes.
        # ----------------------------------------------------

        revenue_change = np.nan
        units_change = np.nan

        post_revenue_change = np.nan
        post_units_change = np.nan

        if avg_before_revenue != 0:

            revenue_change = (
                (avg_during_revenue - avg_before_revenue)
                / abs(avg_before_revenue)
            ) * 100

        if avg_before_units != 0:

            units_change = (
                (avg_during_units - avg_before_units)
                / abs(avg_before_units)
            ) * 100

        if avg_during_revenue != 0:

            post_revenue_change = (
                (avg_after_revenue - avg_during_revenue)
                / abs(avg_during_revenue)
            ) * 100

        if avg_during_units != 0:

            post_units_change = (
                (avg_after_units - avg_during_units)
                / abs(avg_during_units)
            ) * 100

        # ----------------------------------------------------
        # Store campaign results.
        # ----------------------------------------------------

        results.append(
            {
                "Campaign_ID": campaign_id,
                "Campaign_Name": campaign_name,
                "Category": category,
                "Sub_Category": sub_category,
                "Discount": discount,
                "Start_Date": start_date,
                "End_Date": end_date,
                "Currency": currency,

                "Before_Days": len(before_dates),
                "During_Days": len(during_dates),
                "After_Days": len(after_dates),

                "Before_Revenue": before_revenue,
                "During_Revenue": during_revenue,
                "After_Revenue": after_revenue,

                "Before_Units": before_units,
                "During_Units": during_units,
                "After_Units": after_units,

                "Avg_Daily_Revenue_Before": avg_before_revenue,
                "Avg_Daily_Revenue_During": avg_during_revenue,
                "Avg_Daily_Revenue_After": avg_after_revenue,

                "Avg_Daily_Units_Before": avg_before_units,
                "Avg_Daily_Units_During": avg_during_units,
                "Avg_Daily_Units_After": avg_after_units,

                "Revenue_Change_Percent": revenue_change,
                "Units_Change_Percent": units_change,

                "Post_Campaign_Revenue_Change_Percent": (
                    post_revenue_change
                ),

                "Post_Campaign_Units_Change_Percent": (
                    post_units_change
                )
            }
        )

# ============================================================
# 11. CREATE OUTPUT DATAFRAME
# ============================================================

performance = pd.DataFrame(results)

if performance.empty:
    raise ValueError(
        "No campaign performance records generated. "
        "Check product-category mapping and campaign data."
    )

performance = performance.sort_values(
    by=["Campaign_ID", "Currency"]
).reset_index(drop=True)

# ============================================================
# 12. SAVE RESULTS
# ============================================================

performance.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPROMOTION EFFECTIVENESS ANALYSIS COMPLETED")

print(
    "Campaign rows analysed:",
    campaigns["Campaign_ID"].nunique()
)

print(
    "Campaign-currency records:",
    len(performance)
)

print("\nOutput saved to:")
print(OUTPUT_FILE)

# ============================================================
# 13. DISPLAY TOP AND BOTTOM CAMPAIGNS
# ============================================================

display_columns = [
    "Campaign_ID",
    "Campaign_Name",
    "Category",
    "Sub_Category",
    "Discount",
    "Currency",
    "Revenue_Change_Percent",
    "Units_Change_Percent"
]

print("\nTOP 10 CAMPAIGN REVENUE CHANGES")

print(
    performance.sort_values(
        "Revenue_Change_Percent",
        ascending=False
    )[display_columns]
    .head(10)
    .to_string(index=False)
)

print("\nBOTTOM 10 CAMPAIGN REVENUE CHANGES")

print(
    performance.sort_values(
        "Revenue_Change_Percent",
        ascending=True
    )[display_columns]
    .head(10)
    .to_string(index=False)
)

print("\nMETHODOLOGY NOTE:")
print(
    "Results show before/during/after associations, not causal "
    "campaign impact. Currencies are analysed separately."
)
