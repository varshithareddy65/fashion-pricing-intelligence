
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRANSACTIONS_FILE = BASE_DIR / "data/raw/transactions.csv"
PRODUCTS_FILE = BASE_DIR / "data/raw/products.csv"

OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_OUTPUT = OUTPUT_DIR / "product_store_daily_sales.csv"

RESULT_OUTPUT = OUTPUT_DIR / "price_elasticity_validated.csv"

TRAIN_END_DATE = pd.Timestamp("2024-12-31")
TEST_START_DATE = pd.Timestamp("2025-01-01")

MIN_OBSERVATIONS = 500
MIN_PRODUCTS = 20
MIN_STORES = 5

# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading transactions and products...")

transactions = pd.read_csv(TRANSACTIONS_FILE)
products = pd.read_csv(PRODUCTS_FILE)

print("Transactions:", len(transactions))
print("Products:", len(products))

# ============================================================
# 2. VALIDATE COLUMNS
# ============================================================

required_transactions = [
    "Invoice ID",
    "Product ID",
    "Store ID",
    "Date",
    "Unit Price",
    "Quantity",
    "Discount",
    "Line Total",
    "Currency",
    "Transaction Type"
]

required_products = [
    "Product ID",
    "Category",
    "Sub Category"
]

for df, required, filename in [
    (transactions, required_transactions, "transactions.csv"),
    (products, required_products, "products.csv")
]:
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {filename}: {missing}"
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
    raise ValueError("Duplicate Product IDs found.")

products["Category"] = products["Category"].astype(str).str.strip()

products["Sub Category"] = (
    products["Sub Category"].astype(str).str.strip()
)

# ============================================================
# 4. CLEAN TRANSACTIONS
# ============================================================

transactions["Product ID"] = (
    transactions["Product ID"].astype(str).str.strip()
)

transactions["Store ID"] = (
    transactions["Store ID"].astype(str).str.strip()
)

transactions["Date"] = pd.to_datetime(
    transactions["Date"],
    errors="coerce"
).dt.normalize()

for col in ["Unit Price", "Quantity", "Discount", "Line Total"]:
    transactions[col] = pd.to_numeric(
        transactions[col],
        errors="coerce"
    )

transactions = transactions.dropna(
    subset=[
        "Product ID",
        "Date",
        "Unit Price",
        "Quantity",
        "Discount",
        "Line Total",
        "Currency",
        "Store ID",
        "Transaction Type"
    ]
).copy()

# Sales only.
sales = transactions[
    transactions["Transaction Type"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "sale"
].copy()

sales = sales[
    (sales["Quantity"] > 0)
    &
    (sales["Unit Price"] > 0)
    &
    (sales["Discount"] >= 0)
    &
    (sales["Discount"] < 1)
].copy()

print("\nVALID SALES:", len(sales))

# ============================================================
# 5. MERGE PRODUCT INFORMATION
# ============================================================

print("\nMerging product information...")

sales = sales.merge(
    products,
    on="Product ID",
    how="left",
    validate="many_to_one"
)

unmatched = sales["Category"].isna().sum()

print("Unmatched product rows:", unmatched)

sales = sales.dropna(
    subset=["Category", "Sub Category"]
).copy()

# ============================================================
# 6. AGGREGATE AT PRODUCT-STORE-DAY LEVEL
# ============================================================

print("\nAggregating product-store-day sales...")

daily = (
    sales.groupby(
        [
            "Product ID",
            "Store ID",
            "Date",
            "Currency",
            "Category",
            "Sub Category"
        ],
        as_index=False
    )
    .agg(
        Quantity=("Quantity", "sum"),
        Revenue=("Line Total", "sum"),
        Transactions=("Invoice ID", "nunique")
    )
)

# Weighted average selling price.
daily["Selling_Price"] = (
    daily["Revenue"] / daily["Quantity"]
)

daily = daily[
    (daily["Quantity"] > 0)
    &
    (daily["Selling_Price"] > 0)
].copy()

daily["Log_Price"] = np.log(daily["Selling_Price"])
daily["Log_Quantity"] = np.log(daily["Quantity"])

daily["Month"] = daily["Date"].dt.month.astype(str)
daily["DayOfWeek"] = daily["Date"].dt.dayofweek.astype(str)

daily.to_csv(DAILY_OUTPUT, index=False)

print("Product-store-day records:", len(daily))
print("Products:", daily["Product ID"].nunique())
print("Stores:", daily["Store ID"].nunique())
print("Categories:", daily["Category"].nunique())
print("Currencies:", daily["Currency"].nunique())

print("Saved:", DAILY_OUTPUT)

# ============================================================
# 7. TEMPORAL TRAIN-TEST SPLIT
# ============================================================

train = daily[
    daily["Date"] <= TRAIN_END_DATE
].copy()

test = daily[
    daily["Date"] >= TEST_START_DATE
].copy()

print("\nTEMPORAL SPLIT")

print("Training records:", len(train))
print("Testing records:", len(test))

print("Training period:",
      train["Date"].min(), "to", train["Date"].max())

print("Testing period:",
      test["Date"].min(), "to", test["Date"].max())

if train.empty or test.empty:
    raise ValueError("Training or testing data is empty.")

# ============================================================
# 8. HELPER FUNCTION FOR FEATURE CREATION
# ============================================================

def create_features(train_group, test_group):

    product_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    )

    store_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    )

    month_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    )

    weekday_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    )

    # Fit encoders only on training data.
    product_train = product_encoder.fit_transform(
        train_group[["Product ID"]].astype(str)
    )

    product_test = product_encoder.transform(
        test_group[["Product ID"]].astype(str)
    )

    store_train = store_encoder.fit_transform(
        train_group[["Store ID"]].astype(str)
    )

    store_test = store_encoder.transform(
        test_group[["Store ID"]].astype(str)
    )

    month_train = month_encoder.fit_transform(
        train_group[["Month"]].astype(str)
    )

    month_test = month_encoder.transform(
        test_group[["Month"]].astype(str)
    )

    weekday_train = weekday_encoder.fit_transform(
        train_group[["DayOfWeek"]].astype(str)
    )

    weekday_test = weekday_encoder.transform(
        test_group[["DayOfWeek"]].astype(str)
    )

    price_train = csr_matrix(
        train_group["Log_Price"].to_numpy().reshape(-1, 1)
    )

    price_test = csr_matrix(
        test_group["Log_Price"].to_numpy().reshape(-1, 1)
    )

    X_train = hstack(
        [
            price_train,
            product_train,
            store_train,
            month_train,
            weekday_train
        ],
        format="csr"
    )

    X_test = hstack(
        [
            price_test,
            product_test,
            store_test,
            month_test,
            weekday_test
        ],
        format="csr"
    )

    return X_train, X_test

# ============================================================
# 9. PRICE ELASTICITY MODELLING
# ============================================================

print("\nStarting validated elasticity models...")

results = []

group_columns = ["Category", "Currency"]

for (category, currency), train_group in train.groupby(group_columns):

    test_group = test[
        (test["Category"] == category)
        &
        (test["Currency"] == currency)
    ].copy()

    train_group = train_group.copy()

    print("\n" + "=" * 60)
    print(f"Processing: {category} | {currency}")

    if test_group.empty:
        print("Skipped: no test observations.")
        continue

    # --------------------------------------------------------
    # Require price variation within products in training.
    # --------------------------------------------------------

    variation = (
        train_group.groupby("Product ID")["Selling_Price"]
        .nunique()
    )

    eligible_products = variation[
        variation > 1
    ].index

    train_group = train_group[
        train_group["Product ID"].isin(eligible_products)
    ].copy()

    test_group = test_group[
        test_group["Product ID"].isin(eligible_products)
    ].copy()

    if len(train_group) < MIN_OBSERVATIONS:
        print("Skipped: insufficient training records.")
        continue

    if len(test_group) < 100:
        print("Skipped: insufficient testing records.")
        continue

    if train_group["Product ID"].nunique() < MIN_PRODUCTS:
        print("Skipped: insufficient products.")
        continue

    if train_group["Store ID"].nunique() < MIN_STORES:
        print("Skipped: insufficient stores.")
        continue

    # --------------------------------------------------------
    # Create model features.
    # --------------------------------------------------------

    X_train, X_test = create_features(
        train_group,
        test_group
    )

    y_train = train_group["Log_Quantity"].to_numpy()
    y_test = test_group["Log_Quantity"].to_numpy()

    # --------------------------------------------------------
    # Fit regression model.
    # --------------------------------------------------------

    model = LinearRegression()

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    elasticity = model.coef_[0]

    # --------------------------------------------------------
    # Evaluate model.
    # Metrics are measured in log-quantity space.
    # --------------------------------------------------------

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    # --------------------------------------------------------
    # Interpret elasticity.
    # --------------------------------------------------------

    if elasticity < -1:
        interpretation = "Elastic demand"

    elif elasticity < 0:
        interpretation = "Inelastic demand"

    elif elasticity > 0:
        interpretation = "Positive price-demand association"

    else:
        interpretation = "Approximately price-insensitive"

    # --------------------------------------------------------
    # Save results.
    # --------------------------------------------------------

    results.append(
        {
            "Category": category,
            "Currency": currency,

            "Price_Elasticity": elasticity,
            "Interpretation": interpretation,

            "Train_R2": train_r2,
            "Test_R2": test_r2,

            "Test_MAE_Log": test_mae,
            "Test_RMSE_Log": test_rmse,

            "Training_Observations": len(train_group),
            "Testing_Observations": len(test_group),

            "Unique_Products": train_group["Product ID"].nunique(),
            "Unique_Stores": train_group["Store ID"].nunique(),

            "Training_Start": train_group["Date"].min(),
            "Training_End": train_group["Date"].max(),

            "Testing_Start": test_group["Date"].min(),
            "Testing_End": test_group["Date"].max()
        }
    )

    print("Training observations:", len(train_group))
    print("Testing observations:", len(test_group))

    print("Price elasticity:", round(elasticity, 4))
    print("Training R2:", round(train_r2, 4))
    print("Testing R2:", round(test_r2, 4))
    print("Test MAE (log):", round(test_mae, 4))
    print("Test RMSE (log):", round(test_rmse, 4))
    print("Interpretation:", interpretation)

# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError(
        "No models completed. Check category/currency data."
    )

results_df = results_df.sort_values(
    ["Category", "Currency"]
).reset_index(drop=True)

results_df.to_csv(
    RESULT_OUTPUT,
    index=False
)

# ============================================================
# 11. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("VALIDATED PRICE ELASTICITY MODELLING COMPLETED")
print("=" * 65)

print(
    results_df.to_string(index=False)
)

print("\nOUTPUT FILES")
print("Daily data:", DAILY_OUTPUT)
print("Model results:", RESULT_OUTPUT)

print("\nMETHODOLOGY NOTES")
print(
    "1. Model uses product, store, month and weekday effects."
)

print(
    "2. Training data ends December 2024."
)

print(
    "3. Testing data covers January 2025 to March 18, 2025."
)

print(
    "4. MAE, RMSE and R2 are evaluated on unseen dates."
)

print(
    "5. Only observed positive-sales product-store-days are included."
)

print(
    "6. Estimates are observational and do not establish causal "
    "effects of changing prices."
)
