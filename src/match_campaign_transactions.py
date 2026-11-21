
import pandas as pd

# ============================================================
# FILE PATHS
# ============================================================

TRANSACTIONS_PATH = "data/raw/transactions.csv"
PRODUCTS_PATH = "data/raw/products.csv"
DISCOUNTS_PATH = "data/raw/discounts.csv"

OUTPUT_PATH = "data/processed/campaign_performance.csv"


# ============================================================
# 1. LOAD PRODUCT DATA
# ============================================================

print("Loading product data...")

products = pd.read_csv(
    PRODUCTS_PATH,
    usecols=[
        "Product ID",
        "Category",
        "Sub Category"
    ]
)

print("Products loaded:", len(products))


# ============================================================
# 2. LOAD AND PREPARE CAMPAIGN DATA
# ============================================================

print("\nLoading campaign data...")

campaigns = pd.read_csv(DISCOUNTS_PATH)

campaigns["Start"] = pd.to_datetime(
    campaigns["Start"],
    errors="coerce"
)

campaigns["End"] = pd.to_datetime(
    campaigns["End"],
    errors="coerce"
)

campaigns["Discont"] = pd.to_numeric(
    campaigns["Discont"],
    errors="coerce"
)

campaigns = campaigns.dropna(
    subset=[
        "Start",
        "End",
        "Discont",
        "Category",
        "Sub Category"
    ]
).copy()

campaigns["Campaign_ID"] = range(1, len(campaigns) + 1)

campaigns = campaigns.rename(
    columns={"Discont": "Campaign_Discount"}
)

print("Valid campaigns:", len(campaigns))


# ============================================================
# 3. INITIALIZE CAMPAIGN METRICS
# ============================================================

campaign_results = {}

for campaign in campaigns.itertuples(index=False):
    campaign_results[campaign.Campaign_ID] = {
        "Matched_Rows": 0,
        "Invoice_Lines": set(),
        "Invoices": set(),
        "Matched_Quantity": 0
    }


# ============================================================
# 4. STREAM TRANSACTIONS AND MATCH CAMPAIGNS
# ============================================================

columns = [
    "Invoice ID",
    "Line",
    "Product ID",
    "Date",
    "Discount",
    "Quantity",
    "Transaction Type"
]

total_discounted_sales = 0
matched_discounted_rows = 0

print("\nStarting transaction matching...\n")

for chunk in pd.read_csv(
    TRANSACTIONS_PATH,
    usecols=columns,
    chunksize=200000
):

    chunk["Date"] = pd.to_datetime(
        chunk["Date"],
        errors="coerce"
    )

    chunk["Discount"] = pd.to_numeric(
        chunk["Discount"],
        errors="coerce"
    )

    # Only sale transactions
    sales = chunk[
        chunk["Transaction Type"].eq("Sale")
    ].copy()

    # Only discounted sales
    discounted = sales[
        sales["Discount"].fillna(0) > 0
    ].copy()

    total_discounted_sales += len(discounted)

    if discounted.empty:
        continue

    # Attach product category and subcategory
    discounted = discounted.merge(
        products,
        on="Product ID",
        how="left",
        validate="many_to_one"
    )

    # Match category, subcategory and discount
    candidates = discounted.merge(
        campaigns,
        left_on=[
            "Category",
            "Sub Category",
            "Discount"
        ],
        right_on=[
            "Category",
            "Sub Category",
            "Campaign_Discount"
        ],
        how="inner"
    )

    if candidates.empty:
        continue

    # Match transaction dates with campaign dates
    matches = candidates[
    (candidates["Date"] >= candidates["Start"]) &
    (
        candidates["Date"] <
        candidates["End"] + pd.Timedelta(days=1)
    )
].copy()
    if matches.empty:
        continue

    # Avoid duplicate matching of a transaction line
    # within the same campaign
    matches = matches.drop_duplicates(
        subset=[
            "Invoice ID",
            "Line",
            "Campaign_ID"
        ]
    )

    # Count unique matched transaction lines
    matched_line_keys = matches[
        ["Invoice ID", "Line"]
    ].drop_duplicates()

    matched_discounted_rows += len(matched_line_keys)

    # Update campaign metrics
    for campaign_id, group in matches.groupby("Campaign_ID"):

        campaign_results[campaign_id]["Matched_Rows"] += len(group)

        campaign_results[campaign_id]["Invoice_Lines"].update(
            zip(
                group["Invoice ID"],
                group["Line"]
            )
        )

        campaign_results[campaign_id]["Invoices"].update(
            group["Invoice ID"].dropna().unique()
        )

        campaign_results[campaign_id]["Matched_Quantity"] += (
            group["Quantity"].fillna(0).sum()
        )

    print(
        f"Processed discounted rows: {total_discounted_sales:,} "
        f"| Matched lines so far: {matched_discounted_rows:,}"
    )


# ============================================================
# 5. CREATE CAMPAIGN-LEVEL OUTPUT
# ============================================================

output_rows = []

for campaign in campaigns.itertuples(index=False):

    campaign_id = campaign.Campaign_ID
    result = campaign_results[campaign_id]

    output_rows.append({

        "Campaign_ID": campaign_id,

        "Start": campaign.Start,

        "End": campaign.End,

        "Description": campaign.Description,

        "Category": campaign.Category,

        # Correctly retrieve Sub Category
        "Sub_Category": campaign[5],

        "Campaign_Discount": campaign.Campaign_Discount,

        "Matched_Rows": result["Matched_Rows"],

        "Unique_Invoice_Lines": len(
            result["Invoice_Lines"]
        ),

        "Unique_Invoices": len(
            result["Invoices"]
        ),

        "Matched_Quantity": result["Matched_Quantity"]
    })


output = pd.DataFrame(output_rows)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

output.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 7. PRINT MATCHING SUMMARY
# ============================================================

unmatched_discounted_rows = (
    total_discounted_sales - matched_discounted_rows
)

print("\n" + "=" * 60)
print("CAMPAIGN MATCHING SUMMARY")
print("=" * 60)

print(
    "Total discounted sale rows:",
    f"{total_discounted_sales:,}"
)

print(
    "Matched discounted sale rows:",
    f"{matched_discounted_rows:,}"
)

print(
    "Unmatched discounted sale rows:",
    f"{unmatched_discounted_rows:,}"
)

if total_discounted_sales > 0:

    coverage = (
        matched_discounted_rows / total_discounted_sales
    ) * 100

    print(
        "Matching coverage:",
        f"{coverage:.2f}%"
    )


# ============================================================
# 8. TOP 15 CAMPAIGNS
# ============================================================

print("\nTOP 15 CAMPAIGNS BY MATCHED INVOICE LINES")

top_campaigns = output.sort_values(
    by="Unique_Invoice_Lines",
    ascending=False
).head(15)

print(
    top_campaigns.to_string(index=False)
)


# ============================================================
# 9. CAMPAIGNS WITHOUT MATCHED TRANSACTIONS
# ============================================================

zero_match_count = (
    output["Unique_Invoice_Lines"] == 0
).sum()

print(
    "\nCampaigns with zero matched invoice lines:",
    zero_match_count
)


# ============================================================
# 10. COMPLETION
# ============================================================

print("\nOUTPUT FILE CREATED:")
print(OUTPUT_PATH)

print("\nCAMPAIGN MATCHING COMPLETED!")
