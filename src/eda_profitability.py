
import pandas as pd

PATH = "data/raw/"
OUTPUT = "data/processed/"

# ------------------------------------------
# 1. LOAD PRODUCT INFORMATION
# ------------------------------------------

products = pd.read_csv(
    PATH + "products.csv",
    usecols=[
        "Product ID",
        "Category",
        "Sub Category",
        "Production Cost"
    ]
)

print("Products loaded:", len(products))

# ------------------------------------------
# 2. PROCESS TRANSACTIONS IN CHUNKS
# ------------------------------------------

product_results = []

columns = [
    "Product ID",
    "Unit Price",
    "Quantity",
    "Line Total",
    "Currency",
    "Transaction Type"
]

for chunk in pd.read_csv(
    PATH + "transactions.csv",
    chunksize=200_000,
    usecols=columns
):

    # Separate sales and returns
    chunk["Sales_Units"] = chunk["Quantity"].where(
        chunk["Transaction Type"] == "Sale", 0
    )

    chunk["Return_Units"] = chunk["Quantity"].where(
        chunk["Transaction Type"] == "Return", 0
    )

    chunk["Gross_Sales"] = chunk["Line Total"].where(
        chunk["Transaction Type"] == "Sale", 0
    )

    chunk["Return_Amount"] = chunk["Line Total"].where(
        chunk["Transaction Type"] == "Return", 0
    )

    # Aggregate at product and currency level
    grouped = chunk.groupby(
        ["Currency", "Product ID"]
    ).agg(
        Sales_Units=("Sales_Units", "sum"),
        Return_Units=("Return_Units", "sum"),
        Gross_Sales=("Gross_Sales", "sum"),
        Return_Amount=("Return_Amount", "sum")
    ).reset_index()

    product_results.append(grouped)

    print("Processed rows:", len(chunk))

# ------------------------------------------
# 3. COMBINE ALL CHUNK RESULTS
# ------------------------------------------

product_metrics = pd.concat(
    product_results,
    ignore_index=True
)

product_metrics = product_metrics.groupby(
    ["Currency", "Product ID"],
    as_index=False
).agg(
    Sales_Units=("Sales_Units", "sum"),
    Return_Units=("Return_Units", "sum"),
    Gross_Sales=("Gross_Sales", "sum"),
    Return_Amount=("Return_Amount", "sum")
)

# ------------------------------------------
# 4. JOIN PRODUCT INFORMATION
# ------------------------------------------

product_metrics = product_metrics.merge(
    products,
    on="Product ID",
    how="left",
    validate="many_to_one"
)

# ------------------------------------------
# 5. PRODUCT-LEVEL METRICS
# ------------------------------------------

product_metrics["Return_Rate"] = (
    product_metrics["Return_Units"]
    / product_metrics["Sales_Units"].replace(0, pd.NA)
) * 100

product_metrics["Net_Revenue"] = (
    product_metrics["Gross_Sales"]
    + product_metrics["Return_Amount"]
)

product_metrics.to_csv(
    OUTPUT + "product_level_metrics.csv",
    index=False
)

# ------------------------------------------
# 6. SUBCATEGORY PERFORMANCE
# ------------------------------------------

subcategory = product_metrics.groupby(
    ["Currency", "Category", "Sub Category"],
    as_index=False,
    dropna=False
).agg(
    Sales_Units=("Sales_Units", "sum"),
    Return_Units=("Return_Units", "sum"),
    Gross_Sales=("Gross_Sales", "sum"),
    Return_Amount=("Return_Amount", "sum"),
    Net_Revenue=("Net_Revenue", "sum"),
    Unique_Products=("Product ID", "nunique"),
    Average_Product_Cost=("Production Cost", "mean")
)

subcategory["Return_Rate"] = (
    subcategory["Return_Units"]
    / subcategory["Sales_Units"].replace(0, pd.NA)
) * 100

# ------------------------------------------
# 7. SAVE RESULTS
# ------------------------------------------

subcategory = subcategory.sort_values(
    "Gross_Sales",
    ascending=False
)

subcategory.to_csv(
    OUTPUT + "subcategory_performance.csv",
    index=False
)

# ------------------------------------------
# 8. DISPLAY BUSINESS INSIGHTS
# ------------------------------------------

print("\nTOP 15 SUBCATEGORIES BY REVENUE")
print(
    subcategory.head(15).to_string(index=False)
)

print("\nTOP 15 SUBCATEGORIES BY RETURN RATE")
print(
    subcategory.sort_values(
        "Return_Rate",
        ascending=False
    ).head(15).to_string(index=False)
)

print("\nCURRENCY-WISE PRODUCT COST SUMMARY")
print(
    product_metrics.groupby("Currency")[
        "Production Cost"
    ].describe().round(2)
)

print("\nOUTPUT FILES CREATED:")
print("- product_level_metrics.csv")
print("- subcategory_performance.csv")

print("\nPROFITABILITY DATA PROCESSING COMPLETED!")
