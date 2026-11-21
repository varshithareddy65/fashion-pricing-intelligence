
import pandas as pd
from collections import defaultdict

# ============================================================
# FILE PATHS
# ============================================================

TRANSACTIONS_PATH = "data/raw/transactions.csv"
PRODUCTS_PATH = "data/raw/products.csv"
DISCOUNTS_PATH = "data/raw/discounts.csv"

OUTPUT_PATH = "data/processed/campaign_revenue_analysis.csv"


# ============================================================
# 1. LOAD PRODUCT DATA
# ============================================================

print("Loading products...")

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
# 2. LOAD AND PREPARE CAMPAIGNS
# ============================================================

print("\nLoading campaigns...")

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

# Normalize discount values to avoid floating-point mismatches
campaigns["Discount_Key"] = (
    campaigns["Campaign_Discount"].round(4)
)

print("Valid campaigns:", len(campaigns))


# ============================================================
# 3. INITIALIZE CAMPAIGN METRICS
# ============================================================

sales_metrics = defaultdict(lambda: {
    "Matched_Rows": 0,
    "Invoice_Lines": set(),
    "Invoices": set(),
    "Units_Sold": 0,
    "Gross_Revenue": 0.0
})

return_metrics = defaultdict(lambda: {
    "Return_Rows": 0,
    "Return_Units": 0,
    "Return_Amount": 0.0
})

total_discounted_sales = 0
matched_sale_rows = 0
ambiguous_sale_rows = 0


# ============================================================
# 4. TRANSACTION COLUMNS
# ============================================================

columns = [
    "Invoice ID",
    "Line",
    "Product ID",
    "Date",
    "Discount",
    "Quantity",
    "Line Total",
    "Currency",
    "Transaction Type"
]


# ============================================================
# 5. PROCESS TRANSACTIONS IN CHUNKS
# ============================================================

print("\nStarting campaign financial analysis...\n")

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

    chunk["Quantity"] = pd.to_numeric(
        chunk["Quantity"],
        errors="coerce"
    ).fillna(0)

    chunk["Line Total"] = pd.to_numeric(
        chunk["Line Total"],
        errors="coerce"
    ).fillna(0)

    chunk["Discount_Key"] = (
        chunk["Discount"].round(4)
    )

    # Attach product categories
    chunk = chunk.merge(
        products,
        on="Product ID",
        how="left",
        validate="many_to_one"
    )


    # --------------------------------------------------------
    # 5A. MATCH DISCOUNTED SALE TRANSACTIONS
    # --------------------------------------------------------

    sales = chunk[
        (chunk["Transaction Type"] == "Sale") &
        (chunk["Discount"].fillna(0) > 0)
    ].copy()

    total_discounted_sales += len(sales)

    if not sales.empty:

        candidates = sales.merge(
            campaigns,
            left_on=[
                "Category",
                "Sub Category",
                "Discount_Key"
            ],
            right_on=[
                "Category",
                "Sub Category",
                "Discount_Key"
            ],
            how="inner"
        )

        matches = candidates[
            (candidates["Date"] >= candidates["Start"]) &
            (
                candidates["Date"] <
                candidates["End"] + pd.Timedelta(days=1)
            )
        ].copy()

        if not matches.empty:

            # Count transaction lines that match multiple campaigns
            campaign_counts = matches.groupby(
                ["Invoice ID", "Line"]
            )["Campaign_ID"].nunique()

            ambiguous_sale_rows += (
                campaign_counts > 1
            ).sum()

            matched_lines = matches[
                ["Invoice ID", "Line"]
            ].drop_duplicates()

            matched_sale_rows += len(matched_lines)

            # Aggregate by campaign and currency
            for (campaign_id, currency), group in matches.groupby(
                ["Campaign_ID", "Currency"]
            ):

                key = (campaign_id, currency)

                sales_metrics[key]["Matched_Rows"] += len(group)

                sales_metrics[key]["Units_Sold"] += (
                    group["Quantity"].sum()
                )

                sales_metrics[key]["Gross_Revenue"] += (
                    group["Line Total"].sum()
                )

                sales_metrics[key]["Invoices"].update(
                    group["Invoice ID"].dropna().unique()
                )

                sales_metrics[key]["Invoice_Lines"].update(
                    zip(
                        group["Invoice ID"],
                        group["Line"]
                    )
                )


    # --------------------------------------------------------
    # 5B. CALCULATE CONTEXTUAL RETURNS
    # --------------------------------------------------------

    # Returns are not automatically attributable to a campaign.
    # We calculate returns in the same category/subcategory
    # and date window as contextual information only.

    returns = chunk[
        chunk["Transaction Type"] == "Return"
    ].copy()

    if not returns.empty:

        return_candidates = returns.merge(
            campaigns[
                [
                    "Campaign_ID",
                    "Start",
                    "End",
                    "Category",
                    "Sub Category"
                ]
            ],
            on=[
                "Category",
                "Sub Category"
            ],
            how="inner"
        )

        return_matches = return_candidates[
            (return_candidates["Date"] >= return_candidates["Start"]) &
            (
                return_candidates["Date"] <
                return_candidates["End"] + pd.Timedelta(days=1)
            )
        ].copy()

        if not return_matches.empty:

            for (campaign_id, currency), group in return_matches.groupby(
                ["Campaign_ID", "Currency"]
            ):

                key = (campaign_id, currency)

                return_metrics[key]["Return_Rows"] += len(group)

                return_metrics[key]["Return_Units"] += (
                    group["Quantity"].abs().sum()
                )

                return_metrics[key]["Return_Amount"] += (
                    group["Line Total"].sum()
                )


    print(
        f"Processed transactions: {total_discounted_sales:,} "
        f"| Matched sale lines: {matched_sale_rows:,}"
    )


# ============================================================
# 6. BUILD CAMPAIGN FINANCIAL REPORT
# ============================================================

print("\nBuilding campaign financial report...")

output_rows = []

for campaign in campaigns.itertuples(index=False):

    campaign_id = campaign.Campaign_ID

    # Include currencies with either matched sales or
    # contextual returns for this campaign.
    currencies = {
        currency
        for cid, currency in sales_metrics
        if cid == campaign_id
    }

    currencies.update({
        currency
        for cid, currency in return_metrics
        if cid == campaign_id
    })

    for currency in sorted(currencies):

        key = (campaign_id, currency)

        sale = sales_metrics[key]
        ret = return_metrics[key]

        units_sold = sale["Units_Sold"]
        gross_revenue = sale["Gross_Revenue"]

        return_units = ret["Return_Units"]
        return_amount = ret["Return_Amount"]

        average_selling_price = (
            gross_revenue / units_sold
            if units_sold > 0 else 0
        )

        unique_invoices = len(sale["Invoices"])

        revenue_per_invoice = (
            gross_revenue / unique_invoices
            if unique_invoices > 0 else 0
        )

        # Contextual return rate only.
        # Returns are not linked to the campaign's original
        # sale invoices, so this is NOT an attributable rate.
        contextual_return_rate = (
            return_units / units_sold * 100
            if units_sold > 0 else 0
        )

        output_rows.append({

            "Campaign_ID": campaign_id,

            "Campaign_Name": campaign.Description,

            "Start": campaign.Start,

            "End": campaign.End,

            "Category": campaign.Category,

            "Sub_Category": campaign[5],

            "Campaign_Discount": campaign.Campaign_Discount,

            "Currency": currency,

            "Matched_Sale_Rows": sale["Matched_Rows"],

            "Unique_Invoice_Lines": len(
                sale["Invoice_Lines"]
            ),

            "Unique_Invoices": unique_invoices,

            "Units_Sold": units_sold,

            "Gross_Revenue": gross_revenue,

            "Average_Selling_Price": average_selling_price,

            "Revenue_Per_Invoice": revenue_per_invoice,

            "In_Period_Return_Rows": ret["Return_Rows"],

            "In_Period_Return_Units": return_units,

            "In_Period_Return_Amount": return_amount,

            "Contextual_Return_Rate_Pct": contextual_return_rate

        })


output = pd.DataFrame(output_rows)


# ============================================================
# 7. SAVE OUTPUT
# ============================================================

output.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 8. PRINT CAMPAIGN MATCHING SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("CAMPAIGN FINANCIAL ANALYSIS SUMMARY")
print("=" * 65)

print("Total discounted sale rows:", f"{total_discounted_sales:,}")

print("Matched discounted sale lines:", f"{matched_sale_rows:,}")

print(
    "Matching coverage:",
    f"{matched_sale_rows / total_discounted_sales * 100:.2f}%"
    if total_discounted_sales else "0%"
)

print(
    "Sale lines matching multiple campaigns:",
    f"{ambiguous_sale_rows:,}"
)

print("Campaigns analyzed:", len(campaigns))

print("Campaign-currency result rows:", len(output))


# ============================================================
# 9. TOP CAMPAIGNS BY GROSS REVENUE
# ============================================================

if not output.empty:

    print("\nTOP 15 CAMPAIGN-CURRENCY RESULTS BY GROSS REVENUE")

    top_campaigns = output.sort_values(
        by="Gross_Revenue",
        ascending=False
    ).head(15)

    display_columns = [
        "Campaign_ID",
        "Campaign_Name",
        "Category",
        "Sub_Category",
        "Campaign_Discount",
        "Currency",
        "Units_Sold",
        "Gross_Revenue",
        "Average_Selling_Price",
        "Unique_Invoices",
        "Revenue_Per_Invoice"
    ]

    print(
        top_campaigns[display_columns].to_string(index=False)
    )


# ============================================================
# 10. OUTPUT FILE
# ============================================================

print("\nOUTPUT FILE CREATED:")
print(OUTPUT_PATH)

print("\nCAMPAIGN FINANCIAL ANALYSIS COMPLETED!")


# ============================================================
# IMPORTANT INTERPRETATION NOTES
# ============================================================

print("\nIMPORTANT:")
print("1. Revenue is reported in the original transaction currency.")
print("2. Different currencies must not be added together.")
print("3. In-period returns are contextual, not campaign-attributed.")
print("4. Net campaign revenue and campaign profit are not calculated.")
print("5. Duplicate source transaction rows have not been removed.")
print("6. A matched campaign does not prove incremental sales.")
