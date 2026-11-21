
import pandas as pd
from collections import Counter

PATH = "data/raw/"

# --------------------------------------------------
# 1. LOAD DIMENSION TABLES
# --------------------------------------------------

products = pd.read_csv(
    PATH + "products.csv",
    usecols=["Product ID"]
)

customers = pd.read_csv(
    PATH + "customers.csv",
    usecols=["Customer ID"],
    low_memory=False
)

stores = pd.read_csv(
    PATH + "stores.csv",
    usecols=["Store ID"]
)

employees = pd.read_csv(
    PATH + "employees.csv",
    usecols=["Employee ID"]
)

# --------------------------------------------------
# 2. CREATE LOOKUP SETS
# --------------------------------------------------

product_ids = set(products["Product ID"])
customer_ids = set(customers["Customer ID"])
store_ids = set(stores["Store ID"])
employee_ids = set(employees["Employee ID"])

# --------------------------------------------------
# 3. DIMENSION TABLE VALIDATION
# --------------------------------------------------

print("\nDIMENSION TABLE VALIDATION")
print("-" * 50)

for name, df, key in [
    ("Products", products, "Product ID"),
    ("Customers", customers, "Customer ID"),
    ("Stores", stores, "Store ID"),
    ("Employees", employees, "Employee ID")
]:

    print(
        name,
        "| Rows:", len(df),
        "| Unique IDs:", df[key].nunique(),
        "| Duplicate IDs:", df[key].duplicated().sum()
    )

# --------------------------------------------------
# 4. TRANSACTION VALIDATION
# --------------------------------------------------

results = Counter()
currency_sales = Counter()
monthly_sales = Counter()
formula_mismatches = Counter()

for chunk in pd.read_csv(
    PATH + "transactions.csv",
    chunksize=200_000
):

    results["total_rows"] += len(chunk)

    # Convert dates BEFORE filtering
    chunk["Date"] = pd.to_datetime(
        chunk["Date"],
        errors="coerce"
    )

    results["invalid_dates"] += chunk["Date"].isna().sum()

    # Foreign key validation
    results["unknown_products"] += (
        ~chunk["Product ID"].isin(product_ids)
    ).sum()

    results["unknown_customers"] += (
        ~chunk["Customer ID"].isin(customer_ids)
    ).sum()

    results["unknown_stores"] += (
        ~chunk["Store ID"].isin(store_ids)
    ).sum()

    results["unknown_employees"] += (
        ~chunk["Employee ID"].isin(employee_ids)
    ).sum()

    # Separate sales and returns
    sales = chunk[
        chunk["Transaction Type"] == "Sale"
    ].copy()

    returns = chunk[
        chunk["Transaction Type"] == "Return"
    ].copy()

    results["sales_rows"] += len(sales)
    results["return_rows"] += len(returns)

    # Check sales calculation
    expected = (
        sales["Unit Price"]
        * sales["Quantity"]
        * (1 - sales["Discount"])
    )

    formula_mismatches["sales"] += (
        (sales["Line Total"] - expected).abs() >= 0.02
    ).sum()

    # Check return amounts
    results["non_negative_returns"] += (
        returns["Line Total"] >= 0
    ).sum()

    # Monthly sales aggregation
    monthly = sales.groupby(
        sales["Date"].dt.to_period("M")
    )["Line Total"].sum()

    for month, value in monthly.items():
        monthly_sales[str(month)] += value

    # Currency-wise sales
    currency = sales.groupby(
        "Currency"
    )["Line Total"].sum()

    for name, value in currency.items():
        currency_sales[name] += value

# --------------------------------------------------
# 5. DISPLAY RESULTS
# --------------------------------------------------

print("\nFOREIGN KEY AND DATA VALIDATION")
print("-" * 50)

for key, value in results.items():
    print(f"{key}: {value:,}")

print(
    "\nSALES FORMULA MISMATCHES:",
    f"{formula_mismatches['sales']:,}"
)

print("\nSALES BY CURRENCY")
print("-" * 50)

for currency, value in currency_sales.items():
    print(currency, ":", round(value, 2))

print("\nFIRST 10 MONTHLY SALES")
print("-" * 50)

for month, value in sorted(monthly_sales.items())[:10]:
    print(month, ":", round(value, 2))

print("\nVALIDATION COMPLETED!")
