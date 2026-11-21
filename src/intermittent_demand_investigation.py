
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

WEEKLY_FILE = PROCESSED / "sku_weekly_demand.csv"
ELIGIBILITY_FILE = PROCESSED / "sku_training_eligibility.csv"
SEGMENT_FILE = PROCESSED / "sku_corrected_demand_segments.csv"
PRODUCT_FILE = RAW / "products.csv"

TEST_START = pd.Timestamp("2025-01-06")
MIN_ACTIVE_WEEKS = 8

KEYS = ["SKU", "Currency"]

PROCESSED.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading SKU demand data...")

weekly = pd.read_csv(
    WEEKLY_FILE,
    parse_dates=["Week_Start"],
    dtype={
        "SKU": "string",
        "Product ID": "string",
        "Currency": "string"
    }
)

eligibility = pd.read_csv(
    ELIGIBILITY_FILE,
    parse_dates=["First_History_Week"],
    dtype={
        "SKU": "string",
        "Product ID": "string",
        "Currency": "string"
    }
)

segments = pd.read_csv(
    SEGMENT_FILE,
    dtype={
        "SKU": "string",
        "Currency": "string"
    }
)

products = pd.read_csv(
    PRODUCT_FILE,
    usecols=["Product ID", "Category", "Sub Category"],
    dtype={
        "Product ID": "string",
        "Category": "string",
        "Sub Category": "string"
    }
).drop_duplicates("Product ID")

# ============================================================
# 2. CLEAN AND VALIDATE INPUT DATA
# ============================================================

weekly["Week_Start"] = (
    pd.to_datetime(weekly["Week_Start"])
    .dt.normalize()
)

eligibility["First_History_Week"] = (
    pd.to_datetime(eligibility["First_History_Week"])
    .dt.normalize()
)

weekly = (
    weekly.groupby(
        ["SKU", "Product ID", "Currency", "Week_Start"],
        as_index=False
    )["Weekly_Units"]
    .sum()
)

eligibility = eligibility[
    [
        "SKU",
        "Product ID",
        "Currency",
        "First_History_Week",
        "Pre_Test_Active_Weeks"
    ]
].drop_duplicates(KEYS)

segments = segments[
    ["SKU", "Currency", "Demand_Segment"]
].drop_duplicates(KEYS)

print("Weekly records:", f"{len(weekly):,}")
print("Eligibility records:", f"{len(eligibility):,}")
print("Segment records:", f"{len(segments):,}")

# ============================================================
# 3. FILTER ELIGIBLE SERIES BEFORE SEGMENT JOIN
# ============================================================

print("\nFiltering historically eligible SKU series...")

eligible = eligibility[
    eligibility["Pre_Test_Active_Weeks"] >= MIN_ACTIVE_WEEKS
].copy()

eligible = eligible.merge(
    segments,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

if eligible["Demand_Segment"].isna().any():
    missing_count = eligible["Demand_Segment"].isna().sum()

    raise ValueError(
        f"{missing_count} eligible series have no demand segment."
    )

print(
    "Filtered eligible series:",
    f"{len(eligible):,}"
)

print(
    "\nIntermittent series:",
    f"{(eligible['Demand_Segment'] == 'Intermittent').sum():,}"
)

# ============================================================
# 4. SPLIT HISTORICAL AND TEST DATA
# ============================================================

history = weekly[
    weekly["Week_Start"] < TEST_START
].copy()

test = weekly[
    weekly["Week_Start"] >= TEST_START
].copy()

history = history.merge(
    eligible[KEYS],
    on=KEYS,
    how="inner"
)

test = test.merge(
    eligible[KEYS],
    on=KEYS,
    how="inner"
)

print("\nHistorical weekly records:", f"{len(history):,}")
print("Test weekly records:", f"{len(test):,}")

# ============================================================
# 5. HISTORICAL DEMAND METRICS
# ============================================================

print("\nCalculating historical demand patterns...")

historical_metrics = (
    history.groupby(KEYS)
    .agg(
        Historical_Units=("Weekly_Units", "sum"),
        Historical_Active_Weeks=("Weekly_Units", "count"),
        First_Observed_Sale_Week=("Week_Start", "min"),
        Last_Sale_Week=("Week_Start", "max")
    )
    .reset_index()
)

metrics = eligible.merge(
    historical_metrics,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

metrics["Historical_Units"] = (
    metrics["Historical_Units"].fillna(0)
)

metrics["Historical_Active_Weeks"] = (
    metrics["Historical_Active_Weeks"].fillna(0)
)

# Calendar weeks from first history week to test cutoff.
metrics["Calendar_Weeks"] = (
    (TEST_START - metrics["First_History_Week"]).dt.days // 7
).clip(lower=1)

metrics["Historical_Active_Week_Rate"] = (
    metrics["Historical_Active_Weeks"]
    / metrics["Calendar_Weeks"]
)

# Weeks since the last observed sale.
metrics["Weeks_Since_Last_Sale"] = (
    (TEST_START - metrics["Last_Sale_Week"]).dt.days // 7
)

metrics["Weeks_Since_Last_Sale"] = (
    metrics["Weeks_Since_Last_Sale"]
    .fillna(metrics["Calendar_Weeks"])
)

# ============================================================
# 6. RECENT HISTORICAL DEMAND
# ============================================================

print("\nCalculating recent 4, 8 and 12-week demand...")

for weeks in [4, 8, 12]:

    window_start = (
        TEST_START - pd.Timedelta(weeks=weeks)
    )

    recent = history[
        history["Week_Start"] >= window_start
    ]

    recent_metrics = (
        recent.groupby(KEYS)
        .agg(
            **{
                f"Units_Last_{weeks}_Weeks": (
                    "Weekly_Units", "sum"
                ),
                f"Active_Weeks_Last_{weeks}_Weeks": (
                    "Weekly_Units",
                    lambda x: (x > 0).sum()
                )
            }
        )
        .reset_index()
    )

    metrics = metrics.merge(
        recent_metrics,
        on=KEYS,
        how="left",
        validate="one_to_one"
    )

    for column in [
        f"Units_Last_{weeks}_Weeks",
        f"Active_Weeks_Last_{weeks}_Weeks"
    ]:
        metrics[column] = metrics[column].fillna(0)

# ============================================================
# 7. CLASSIFY HISTORICAL RECENCY
# ============================================================

def classify_recency(weeks):

    if pd.isna(weeks):
        return "No Historical Sale"

    if weeks <= 4:
        return "Recently Active"

    elif weeks <= 8:
        return "Inactive for 5-8 Weeks"

    elif weeks <= 16:
        return "Inactive for 9-16 Weeks"

    return "Inactive for More Than 16 Weeks"


metrics["Historical_Recency_Group"] = (
    metrics["Weeks_Since_Last_Sale"]
    .apply(classify_recency)
)

# ============================================================
# 8. TEST-PERIOD ACTIVITY
# ============================================================

print("\nCalculating test-period activity...")

test_metrics = (
    test.groupby(KEYS)
    .agg(
        Test_Units=("Weekly_Units", "sum"),
        Test_Active_Weeks=("Weekly_Units", "count"),
        First_Test_Sale_Week=("Week_Start", "min"),
        Last_Test_Sale_Week=("Week_Start", "max")
    )
    .reset_index()
)

metrics = metrics.merge(
    test_metrics,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

metrics["Test_Units"] = (
    metrics["Test_Units"].fillna(0)
)

metrics["Test_Active_Weeks"] = (
    metrics["Test_Active_Weeks"].fillna(0)
)

last_dataset_week = weekly["Week_Start"].max()

test_calendar_weeks = (
    (last_dataset_week - TEST_START).days // 7 + 1
)

metrics["Test_Calendar_Weeks"] = test_calendar_weeks

metrics["Test_Active_Week_Rate"] = (
    metrics["Test_Active_Weeks"]
    / test_calendar_weeks
)

metrics["Test_Activity_Status"] = np.where(
    metrics["Test_Units"] > 0,
    "Sales Observed",
    "No Sales Observed"
)

# ============================================================
# 9. ADD PRODUCT INFORMATION
# ============================================================

metrics = metrics.merge(
    products,
    on="Product ID",
    how="left",
    validate="many_to_one"
)

metrics["Category"] = (
    metrics["Category"].fillna("Unknown")
)

metrics["Sub Category"] = (
    metrics["Sub Category"].fillna("Unknown")
)

# ============================================================
# 10. ISOLATE INTERMITTENT PRODUCTS
# ============================================================

intermittent = metrics[
    metrics["Demand_Segment"] == "Intermittent"
].copy()

print(
    "\nIntermittent SKU-currency series:",
    f"{len(intermittent):,}"
)

# ============================================================
# 11. RECENCY SUMMARY
# ============================================================

recency_summary = (
    intermittent.groupby("Historical_Recency_Group")
    .agg(
        SKU_Series=("SKU", "count"),
        Historical_Units=("Historical_Units", "sum"),
        Mean_Active_Week_Rate=(
            "Historical_Active_Week_Rate", "mean"
        ),
        Mean_Weeks_Since_Last_Sale=(
            "Weeks_Since_Last_Sale", "mean"
        ),
        Test_Units=("Test_Units", "sum"),
        Test_Active_Series=(
            "Test_Activity_Status",
            lambda x: (x == "Sales Observed").sum()
        )
    )
    .reset_index()
)

recency_summary["Percentage_of_Intermittent"] = (
    recency_summary["SKU_Series"]
    / len(intermittent) * 100
)

# ============================================================
# 12. INTERMITTENT DEMAND BY CATEGORY AND CURRENCY
# ============================================================

category_summary = (
    intermittent.groupby(
        ["Category", "Currency"]
    )
    .agg(
        SKU_Series=("SKU", "count"),
        Historical_Units=("Historical_Units", "sum"),
        Mean_Active_Week_Rate=(
            "Historical_Active_Week_Rate", "mean"
        ),
        Test_Units=("Test_Units", "sum"),
        Test_Active_Series=(
            "Test_Activity_Status",
            lambda x: (x == "Sales Observed").sum()
        )
    )
    .reset_index()
)

category_summary["Test_Active_Series_Percent"] = (
    category_summary["Test_Active_Series"]
    / category_summary["SKU_Series"] * 100
)

# ============================================================
# 13. COMPARISON OF ALL DEMAND SEGMENTS
# ============================================================

segment_summary = (
    metrics.groupby("Demand_Segment")
    .agg(
        SKU_Series=("SKU", "count"),
        Historical_Units=("Historical_Units", "sum"),
        Mean_Active_Week_Rate=(
            "Historical_Active_Week_Rate", "mean"
        ),
        Median_Active_Week_Rate=(
            "Historical_Active_Week_Rate", "median"
        ),
        Mean_Weeks_Since_Last_Sale=(
            "Weeks_Since_Last_Sale", "mean"
        ),
        Test_Units=("Test_Units", "sum"),
        Test_Active_Series=(
            "Test_Activity_Status",
            lambda x: (x == "Sales Observed").sum()
        )
    )
    .reset_index()
)

segment_summary["Test_Active_Series_Percent"] = (
    segment_summary["Test_Active_Series"]
    / segment_summary["SKU_Series"] * 100
)

# ============================================================
# 14. RECENTLY ACTIVE INTERMITTENT PRODUCTS
# ============================================================

recently_active = intermittent[
    intermittent["Historical_Recency_Group"]
    == "Recently Active"
].copy()

recently_active = recently_active.sort_values(
    [
        "Historical_Active_Week_Rate",
        "Historical_Units"
    ],
    ascending=False
)

# ============================================================
# 15. HISTORICALLY ACTIVE BUT NOW STALE
# ============================================================

stale_products = intermittent[
    intermittent["Historical_Recency_Group"].isin(
        [
            "Inactive for 9-16 Weeks",
            "Inactive for More Than 16 Weeks"
        ]
    )
].copy()

stale_products = stale_products.sort_values(
    "Historical_Units",
    ascending=False
)

# ============================================================
# 16. SAVE OUTPUT FILES
# ============================================================

metrics.to_csv(
    PROCESSED / "sku_all_demand_activity_metrics.csv",
    index=False
)

intermittent.to_csv(
    PROCESSED / "intermittent_sku_detailed_analysis.csv",
    index=False
)

recency_summary.to_csv(
    PROCESSED / "intermittent_demand_recency_summary.csv",
    index=False
)

category_summary.to_csv(
    PROCESSED / "intermittent_demand_category_summary.csv",
    index=False
)

segment_summary.to_csv(
    PROCESSED / "demand_segment_activity_comparison.csv",
    index=False
)

recently_active.to_csv(
    PROCESSED / "intermittent_recently_active_skus.csv",
    index=False
)

stale_products.to_csv(
    PROCESSED / "intermittent_stale_skus.csv",
    index=False
)

# ============================================================
# 17. PRINT RESULTS
# ============================================================

print("\n========== INTERMITTENT DEMAND INVESTIGATION ==========")

print(
    "Intermittent series:",
    f"{len(intermittent):,}"
)

print(
    "Test calendar weeks:",
    test_calendar_weeks
)

print("\n========== HISTORICAL RECENCY DISTRIBUTION ==========")

print(
    recency_summary.to_string(index=False)
)

print(
    "\n========== INTERMITTENT DEMAND BY CATEGORY/CURRENCY =========="
)

print(
    category_summary.to_string(index=False)
)

print("\n========== ALL DEMAND SEGMENTS ==========")

print(
    segment_summary.to_string(index=False)
)

print(
    "\n========== RECENTLY ACTIVE INTERMITTENT SERIES =========="
)

print(
    recently_active[
        [
            "SKU",
            "Product ID",
            "Category",
            "Sub Category",
            "Currency",
            "Historical_Units",
            "Historical_Active_Weeks",
            "Calendar_Weeks",
            "Historical_Active_Week_Rate",
            "Weeks_Since_Last_Sale",
            "Test_Units"
        ]
    ].head(15).to_string(index=False)
)

print(
    "\n========== STALE INTERMITTENT SERIES =========="
)

print(
    stale_products[
        [
            "SKU",
            "Product ID",
            "Category",
            "Sub Category",
            "Currency",
            "Historical_Units",
            "Historical_Active_Weeks",
            "Weeks_Since_Last_Sale",
            "Test_Units"
        ]
    ].head(15).to_string(index=False)
)

print("\nSaved seven output files in data/processed/")
print("Intermittent demand investigation completed.")
