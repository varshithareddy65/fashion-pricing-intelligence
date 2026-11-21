
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
        "SKU",
        "Product ID",
        "Date",
        "Quantity",
        "Transaction Type",
        "Currency"
    ],
    dtype={
        "SKU": "string",
        "Product ID": "string",
        "Transaction Type": "string",
        "Currency": "string"
    }
)

print(f"Total transaction rows: {len(df):,}")

# Keep completed sales only; returns are not part of the sales-demand target.
df = df[df["Transaction Type"].eq("Sale")].copy()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "SKU", "Product ID", "Currency"])

# Monday-start weekly aggregation.
df["Week_Start"] = (
    df["Date"]
    .dt.to_period("W-SUN")
    .dt.start_time
)

print(f"Sale rows: {len(df):,}")
print("Aggregating SKU-level weekly demand...")

weekly = (
    df.groupby(
        ["SKU", "Product ID", "Currency", "Week_Start"],
        observed=True,
        as_index=False
    )
    .agg(
        Weekly_Units=("Quantity", "sum"),
        Transaction_Rows=("Quantity", "size")
    )
)

weekly = weekly.sort_values(
    ["SKU", "Currency", "Week_Start"]
).reset_index(drop=True)

# Overall date range
first_week = weekly["Week_Start"].min()
last_week = weekly["Week_Start"].max()

# Calculate weekly history coverage for every SKU-currency pair.
coverage = (
    weekly.groupby(["SKU", "Product ID", "Currency"], as_index=False)
    .agg(
        First_Observed_Week=("Week_Start", "min"),
        Last_Observed_Week=("Week_Start", "max"),
        Active_Weeks=("Week_Start", "nunique"),
        Total_Units=("Weekly_Units", "sum"),
        Average_Units_Per_Active_Week=("Weekly_Units", "mean")
    )
)

coverage["Calendar_Weeks"] = (
    (
        coverage["Last_Observed_Week"]
        - coverage["First_Observed_Week"]
    ).dt.days // 7
) + 1

coverage["Active_Week_Rate"] = (
    coverage["Active_Weeks"] / coverage["Calendar_Weeks"]
)

# Determine which series existed before the 2025 test period.
coverage["Available_Before_2025"] = (
    coverage["First_Observed_Week"]
    < pd.Timestamp("2025-01-01")
)

coverage["Has_8_Active_Weeks"] = (
    coverage["Active_Weeks"] >= 8
)

coverage["Has_16_Active_Weeks"] = (
    coverage["Active_Weeks"] >= 16
)

# Save outputs.
weekly.to_csv(
    OUTPUT / "sku_weekly_demand.csv",
    index=False
)

coverage.to_csv(
    OUTPUT / "sku_weekly_coverage.csv",
    index=False
)

# Print diagnostics.
print("\n========== WEEKLY DATA SUMMARY ==========")
print("First week:", first_week.date())
print("Last week:", last_week.date())
print("Weekly SKU-currency records:", f"{len(weekly):,}")
print("Unique SKUs:", f"{weekly['SKU'].nunique():,}")
print("Unique Product IDs:", f"{weekly['Product ID'].nunique():,}")
print("Unique SKU-currency series:", f"{len(coverage):,}")

print("\n========== HISTORY COVERAGE ==========")
print(
    "Series with at least 8 active weeks:",
    f"{coverage['Has_8_Active_Weeks'].sum():,}"
)
print(
    "Series with at least 16 active weeks:",
    f"{coverage['Has_16_Active_Weeks'].sum():,}"
)
print(
    "Series first observed before 2025:",
    f"{coverage['Available_Before_2025'].sum():,}"
)

eligible = coverage[
    coverage["Available_Before_2025"]
    & coverage["Has_8_Active_Weeks"]
]

print(
    "Potential historical training series (pre-2025, 8+ active weeks):",
    f"{len(eligible):,}"
)

print("\n========== COVERAGE BY CURRENCY ==========")

currency_summary = (
    coverage.groupby("Currency")
    .agg(
        Total_Series=("SKU", "size"),
        Series_8_Weeks=("Has_8_Active_Weeks", "sum"),
        Series_16_Weeks=("Has_16_Active_Weeks", "sum"),
        Pre_2025_Series=("Available_Before_2025", "sum")
    )
)

print(currency_summary.to_string())

print("\n========== ACTIVE WEEK DISTRIBUTION ==========")
print(
    coverage["Active_Weeks"]
    .describe(percentiles=[0.25, 0.50, 0.75, 0.90])
    .to_string()
)

print("\n========== MOST ACTIVE SKU SERIES ==========")
print(
    coverage.sort_values(
        "Active_Weeks",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

print("\nSaved:")
print("data/processed/sku_weekly_demand.csv")
print("data/processed/sku_weekly_coverage.csv")
