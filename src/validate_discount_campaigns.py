
import pandas as pd

DISCOUNTS_PATH = "data/raw/discounts.csv"
TRANSACTIONS_PATH = "data/raw/transactions.csv"

# 1. Inspect campaign data
discounts = pd.read_csv(DISCOUNTS_PATH)

print("\n========== DISCOUNT CAMPAIGN DATA ==========")
print("Campaign rows:", len(discounts))
print("Columns:", discounts.columns.tolist())

discounts["Start"] = pd.to_datetime(
    discounts["Start"], errors="coerce"
)
discounts["End"] = pd.to_datetime(
    discounts["End"], errors="coerce"
)

print("\nCampaign date range:")
print("Start:", discounts["Start"].min())
print("End:", discounts["End"].max())

print("\nMissing campaign dates:")
print("Start:", discounts["Start"].isna().sum())
print("End:", discounts["End"].isna().sum())

print("\nCampaign discount values:")
print(discounts["Discont"].value_counts(dropna=False).sort_index())

print("\nCampaign categories:")
print(discounts["Category"].value_counts(dropna=False))

print("\nCampaign subcategories:")
print(discounts["Sub Category"].value_counts(dropna=False))

print("\nFirst 15 campaign records:")
print(discounts.head(15).to_string(index=False))


# 2. Inspect transaction date and discount coverage
print("\n========== TRANSACTION COVERAGE ==========")

date_min = None
date_max = None
total_rows = 0
discounted_rows = 0
sale_rows = 0
discounted_sale_rows = 0
date_overlap_rows = 0
discounted_date_overlap_rows = 0

valid_campaigns = discounts.dropna(subset=["Start", "End"])

# Check whether each transaction date falls within at least
# one campaign date interval. This is date-only overlap,
# not proof of a category-specific campaign match.
campaign_intervals = list(
    zip(valid_campaigns["Start"], valid_campaigns["End"])
)

for chunk in pd.read_csv(
    TRANSACTIONS_PATH,
    usecols=["Date", "Discount", "Transaction Type"],
    chunksize=200000
):
    chunk["Date"] = pd.to_datetime(
        chunk["Date"], errors="coerce"
    )

    if chunk["Date"].notna().any():
        chunk_min = chunk["Date"].min()
        chunk_max = chunk["Date"].max()

        date_min = chunk_min if date_min is None else min(date_min, chunk_min)
        date_max = chunk_max if date_max is None else max(date_max, chunk_max)

    total_rows += len(chunk)

    discounted = chunk["Discount"].fillna(0) > 0
    sales = chunk["Transaction Type"].eq("Sale")

    discounted_rows += discounted.sum()
    sale_rows += sales.sum()
    discounted_sale_rows += (discounted & sales).sum()

    # Date-only campaign overlap
    date_match = pd.Series(False, index=chunk.index)

    for start, end in campaign_intervals:
        date_match |= chunk["Date"].between(start, end)

    date_overlap_rows += date_match.sum()
    discounted_date_overlap_rows += (
        date_match & discounted
    ).sum()

print("Transaction rows:", total_rows)
print("Transaction date range:", date_min, "to", date_max)

print("\nTransaction discount summary:")
print("Rows with a discount:", discounted_rows)
print("Sale rows:", sale_rows)
print("Discounted sale rows:", discounted_sale_rows)

print("\nDate-only campaign overlap:")
print("Rows within at least one campaign date range:", date_overlap_rows)
print("Discounted rows within campaign date ranges:",
      discounted_date_overlap_rows)

print("\nNOTE:")
print("Date overlap does not prove that a transaction")
print("belongs to a campaign's category or subcategory.")
