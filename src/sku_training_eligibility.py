
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

INPUT = PROCESSED / "sku_weekly_demand.csv"

# First forecast week. All earlier weeks are available as history.
TEST_START = pd.Timestamp("2025-01-06")

print("Loading SKU weekly demand...")
df = pd.read_csv(
    INPUT,
    parse_dates=["Week_Start"],
    dtype={
        "SKU": "string",
        "Product ID": "string",
        "Currency": "string"
    }
)

print(f"Weekly records: {len(df):,}")

# Separate historical records from the forecast evaluation period.
historical = df[df["Week_Start"] < TEST_START].copy()
test = df[df["Week_Start"] >= TEST_START].copy()

# Count active weeks using historical data ONLY.
history_coverage = (
    historical.groupby(
        ["SKU", "Product ID", "Currency"],
        as_index=False
    )
    .agg(
        First_History_Week=("Week_Start", "min"),
        Last_History_Week=("Week_Start", "max"),
        Pre_Test_Active_Weeks=("Week_Start", "nunique"),
        Historical_Units=("Weekly_Units", "sum")
    )
)

history_coverage["Eligible_8_Weeks"] = (
    history_coverage["Pre_Test_Active_Weeks"] >= 8
)

history_coverage["Eligible_16_Weeks"] = (
    history_coverage["Pre_Test_Active_Weeks"] >= 16
)

# Count test-period activity separately for reporting.
test_coverage = (
    test.groupby(["SKU", "Currency"], as_index=False)
    .agg(
        Test_Active_Weeks=("Week_Start", "nunique"),
        Test_Units=("Weekly_Units", "sum")
    )
)

coverage = history_coverage.merge(
    test_coverage,
    on=["SKU", "Currency"],
    how="left"
)

coverage["Test_Active_Weeks"] = (
    coverage["Test_Active_Weeks"].fillna(0).astype(int)
)

coverage["Test_Units"] = coverage["Test_Units"].fillna(0)

# Save the corrected eligibility table.
coverage.to_csv(
    PROCESSED / "sku_training_eligibility.csv",
    index=False
)

# Overall summary
print("\n========== PRE-TEST ELIGIBILITY ==========")
print("Historical records:", f"{len(historical):,}")
print("Test records:", f"{len(test):,}")
print("Historical SKU-currency series:", f"{len(history_coverage):,}")

print(
    "Series with 8+ historical active weeks:",
    f"{history_coverage['Eligible_8_Weeks'].sum():,}"
)

print(
    "Series with 16+ historical active weeks:",
    f"{history_coverage['Eligible_16_Weeks'].sum():,}"
)

# Currency-level summary
summary = (
    coverage.groupby("Currency")
    .agg(
        Total_Historical_Series=("SKU", "size"),
        Eligible_8_Weeks=("Eligible_8_Weeks", "sum"),
        Eligible_16_Weeks=("Eligible_16_Weeks", "sum"),
        Test_Active_Series=("Test_Active_Weeks",
                            lambda x: (x > 0).sum()),
        Test_Active_Weeks=("Test_Active_Weeks", "sum")
    )
)

print("\n========== CURRENCY-LEVEL ELIGIBILITY ==========")
print(summary.to_string())

# History distribution
print("\n========== PRE-TEST ACTIVE WEEK DISTRIBUTION ==========")
print(
    history_coverage["Pre_Test_Active_Weeks"]
    .describe(percentiles=[0.25, 0.50, 0.75, 0.90])
    .to_string()
)

# Compare 8-week and 16-week thresholds
print("\n========== ELIGIBILITY COMPARISON ==========")
for threshold in [8, 12, 16, 20]:
    count = (
        history_coverage["Pre_Test_Active_Weeks"] >= threshold
    ).sum()

    print(f"{threshold}+ active weeks: {count:,}")

print("\nSaved:")
print("data/processed/sku_training_eligibility.csv")
