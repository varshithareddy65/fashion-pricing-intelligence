
import pandas as pd
from collections import defaultdict

PATH = "data/raw/"

# Load product information
products = pd.read_csv(
    PATH + "products.csv",
    usecols=[
        "Product ID",
        "Category",
        "Sub Category"
    ]
)

# Store aggregated results
category_data = []
discount_data = []
monthly_data = []
return_data = []

# Read transactions in chunks
for chunk in pd.read_csv(
    PATH + "transactions.csv",
    chunksize=200_000,
    usecols=[
        "Product ID",
        "Date",
        "Unit Price",
        "Quantity",
        "Discount",
        "Line Total",
        "Currency",
        "Transaction Type"
    ]
):

    chunk["Date"] = pd.to_datetime(
        chunk["Date"],
        errors="coerce"
    )

    # Add product categories
    chunk = chunk.merge(
        products,
        on="Product ID",
        how="left"
    )

    # Separate sales and returns
    sales = chunk[
        chunk["Transaction Type"] == "Sale"
    ].copy()

    returns = chunk[
        chunk["Transaction Type"] == "Return"
    ].copy()

    # --------------------------------
    # 1. CATEGORY PERFORMANCE
    # --------------------------------

    grouped = sales.groupby(
        ["Currency", "Category"]
    ).agg(
        Revenue=("Line Total", "sum"),
        Units_Sold=("Quantity", "sum"),
        Transactions=("Product ID", "count")
    ).reset_index()

    category_data.append(grouped)

    # --------------------------------
    # 2. DISCOUNT PERFORMANCE
    # --------------------------------

    grouped = sales.groupby(
        ["Currency", "Discount"]
    ).agg(
        Revenue=("Line Total", "sum"),
        Units_Sold=("Quantity", "sum"),
        Transactions=("Product ID", "count"),
        Total_Quantity=("Quantity", "sum"),
        Weighted_Discount=(
            "Discount",
            lambda x: 0
        )
    ).reset_index()

    # Recalculate quantity-weighted discount
    weighted = sales.assign(
        Discount_Quantity=(
            sales["Discount"] * sales["Quantity"]
        )
    ).groupby(
        ["Currency", "Discount"]
    )["Discount_Quantity"].sum().reset_index()

    grouped = grouped.drop(
        columns=["Weighted_Discount", "Total_Quantity"]
    ).merge(
        weighted,
        on=["Currency", "Discount"]
    )

    discount_data.append(grouped)

    # --------------------------------
    # 3. MONTHLY SALES
    # --------------------------------

    sales["Month"] = sales["Date"].dt.to_period("M").astype(str)

    grouped = sales.groupby(
        ["Currency", "Month"]
    ).agg(
        Revenue=("Line Total", "sum"),
        Units_Sold=("Quantity", "sum")
    ).reset_index()

    monthly_data.append(grouped)

    # --------------------------------
    # 4. RETURNS BY CATEGORY
    # --------------------------------

    grouped = returns.groupby(
        ["Currency", "Category"]
    ).agg(
        Return_Amount=("Line Total", "sum"),
        Returned_Units=("Quantity", "sum"),
        Return_Count=("Product ID", "count")
    ).reset_index()

    return_data.append(grouped)


# Combine chunk-level results
def combine_results(data, keys):
    result = pd.concat(data, ignore_index=True)

    return result.groupby(
        keys,
        as_index=False
    ).sum(numeric_only=True)


category = combine_results(
    category_data,
    ["Currency", "Category"]
)

discount = combine_results(
    discount_data,
    ["Currency", "Discount"]
)

monthly = combine_results(
    monthly_data,
    ["Currency", "Month"]
)

returns = combine_results(
    return_data,
    ["Currency", "Category"]
)

# Calculate weighted average discount
discount["Average_Discount"] = (
    discount["Discount_Quantity"]
    / discount["Units_Sold"]
)

# Save results
category.to_csv(
    "data/processed/category_performance.csv",
    index=False
)

discount.to_csv(
    "data/processed/discount_performance.csv",
    index=False
)

monthly.to_csv(
    "data/processed/monthly_sales.csv",
    index=False
)

returns.to_csv(
    "data/processed/returns_analysis.csv",
    index=False
)

# Display results
print("\nTOP CATEGORIES BY REVENUE")

print(
    category.sort_values(
        "Revenue",
        ascending=False
    ).groupby("Currency").head(5).to_string(index=False)
)

print("\nDISCOUNT PERFORMANCE")

print(
    discount.sort_values(
        ["Currency", "Discount"]
    ).to_string(index=False)
)

print("\nMONTHLY SALES")

print(monthly.head(20).to_string(index=False))

print("\nRETURNS BY CATEGORY")

print(
    returns.sort_values(
        "Return_Count",
        ascending=False
    ).head(15).to_string(index=False)
)

print("\nEDA COMPLETED SUCCESSFULLY!")
