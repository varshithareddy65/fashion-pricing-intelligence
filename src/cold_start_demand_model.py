
import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from pathlib import Path
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

TRANSACTION_FILE = RAW / "transactions.csv"
PRODUCT_FILE = RAW / "products.csv"

MARCH_START = pd.Timestamp("2025-03-01")
WINDOW_DAYS = 14
CHUNK_SIZE = 500_000

PROCESSED.mkdir(parents=True, exist_ok=True)

print("\n========== COLD-START DEMAND MODEL ==========")

# ============================================================
# 2. LOAD PRODUCT ATTRIBUTES
# ============================================================

print("\nLoading product attributes...")

products = pd.read_csv(
    PRODUCT_FILE,
    usecols=[
        "Product ID",
        "Category",
        "Sub Category",
        "Description EN",
        "Color",
        "Sizes"
    ],
    dtype={"Product ID": "string"}
)

products = products.drop_duplicates(
    subset=["Product ID"]
).copy()

for col in [
    "Category",
    "Sub Category",
    "Description EN",
    "Color",
    "Sizes"
]:
    products[col] = products[col].fillna("Unknown").astype(str)

print("Products loaded:", f"{len(products):,}")

# ============================================================
# 3. FIRST PASS: FIND FIRST OBSERVED SALE DATE
# ============================================================

print("\nFirst pass: finding first sale dates...")

required_columns = [
    "Product ID",
    "Currency",
    "Date",
    "Quantity",
    "Transaction Type"
]

first_sale_parts = []
dataset_end = None

reader = pd.read_csv(
    TRANSACTION_FILE,
    usecols=required_columns,
    dtype={
        "Product ID": "string",
        "Currency": "string",
        "Transaction Type": "string"
    },
    chunksize=CHUNK_SIZE,
    low_memory=False
)

for chunk_number, chunk in enumerate(reader, start=1):

    chunk["Date"] = pd.to_datetime(
        chunk["Date"],
        errors="coerce"
    )

    chunk["Transaction Type"] = (
        chunk["Transaction Type"]
        .str.strip()
        .str.lower()
    )

    chunk = chunk[
        (chunk["Transaction Type"] == "sale")
        & chunk["Product ID"].notna()
        & chunk["Currency"].notna()
        & chunk["Date"].notna()
    ].copy()

    if chunk.empty:
        continue

    chunk["Sale_Day"] = chunk["Date"].dt.normalize()

    chunk_max = chunk["Sale_Day"].max()

    if dataset_end is None or chunk_max > dataset_end:
        dataset_end = chunk_max

    grouped = (
        chunk.groupby(
            ["Product ID", "Currency"]
        )
        .agg(
            First_Sale_Date=("Sale_Day", "min")
        )
        .reset_index()
    )

    first_sale_parts.append(grouped)

    if chunk_number % 3 == 0:
        print(
            f"First pass processed {chunk_number} chunks..."
        )

if not first_sale_parts:
    raise ValueError("No valid sale transactions found.")

first_sales = pd.concat(
    first_sale_parts,
    ignore_index=True
)

first_sales = (
    first_sales.groupby(
        ["Product ID", "Currency"],
        as_index=False
    )["First_Sale_Date"]
    .min()
)

del first_sale_parts
gc.collect()

print(
    "Product-currency series with sales:",
    f"{len(first_sales):,}"
)

print("Last observed transaction date:", dataset_end.date())

# ============================================================
# 4. SECOND PASS: COLLECT FIRST 14-DAY SALES
# ============================================================

print("\nSecond pass: calculating first 14-day demand...")

daily_parts = []

reader = pd.read_csv(
    TRANSACTION_FILE,
    usecols=required_columns,
    dtype={
        "Product ID": "string",
        "Currency": "string",
        "Transaction Type": "string"
    },
    chunksize=CHUNK_SIZE,
    low_memory=False
)

for chunk_number, chunk in enumerate(reader, start=1):

    chunk["Date"] = pd.to_datetime(
        chunk["Date"],
        errors="coerce"
    )

    chunk["Quantity"] = pd.to_numeric(
        chunk["Quantity"],
        errors="coerce"
    )

    chunk["Transaction Type"] = (
        chunk["Transaction Type"]
        .str.strip()
        .str.lower()
    )

    chunk = chunk[
        (chunk["Transaction Type"] == "sale")
        & chunk["Product ID"].notna()
        & chunk["Currency"].notna()
        & chunk["Date"].notna()
        & chunk["Quantity"].notna()
    ].copy()

    if chunk.empty:
        continue

    chunk["Sale_Day"] = chunk["Date"].dt.normalize()

    chunk = chunk.merge(
        first_sales,
        on=["Product ID", "Currency"],
        how="inner",
        validate="many_to_one"
    )

    chunk["Window_End"] = (
        chunk["First_Sale_Date"]
        + pd.Timedelta(days=WINDOW_DAYS - 1)
    )

    chunk = chunk[
        (chunk["Sale_Day"] >= chunk["First_Sale_Date"])
        & (chunk["Sale_Day"] <= chunk["Window_End"])
    ]

    if chunk.empty:
        continue

    daily = (
        chunk.groupby(
            ["Product ID", "Currency", "Sale_Day"],
            as_index=False
        )
        .agg(
            Daily_Units=("Quantity", "sum")
        )
    )

    daily_parts.append(daily)

    if chunk_number % 3 == 0:
        print(
            f"Second pass processed {chunk_number} chunks..."
        )

if not daily_parts:
    raise ValueError("No first-window sales were found.")

daily_sales = pd.concat(
    daily_parts,
    ignore_index=True
)

del daily_parts
gc.collect()

# Combine repeated product-currency-day records across chunks.
daily_sales = (
    daily_sales.groupby(
        ["Product ID", "Currency", "Sale_Day"],
        as_index=False
    )["Daily_Units"]
    .sum()
)

# Aggregate first-window metrics.
demand_metrics = (
    daily_sales.groupby(
        ["Product ID", "Currency"],
        as_index=False
    )
    .agg(
        First_14_Day_Units=("Daily_Units", "sum"),
        First_14_Day_Active_Days=("Sale_Day", "nunique")
    )
)

del daily_sales
gc.collect()

# ============================================================
# 5. BUILD PRODUCT-CURRENCY DATASET
# ============================================================

print("\nBuilding product-currency modelling dataset...")

dataset = first_sales.merge(
    demand_metrics,
    on=["Product ID", "Currency"],
    how="left",
    validate="one_to_one"
)

dataset["First_14_Day_Units"] = (
    dataset["First_14_Day_Units"].fillna(0)
)

dataset["First_14_Day_Active_Days"] = (
    dataset["First_14_Day_Active_Days"].fillna(0)
)

dataset["Window_End"] = (
    dataset["First_Sale_Date"]
    + pd.Timedelta(days=WINDOW_DAYS - 1)
)

dataset["Window_Complete"] = (
    dataset["Window_End"] <= dataset_end
)

dataset["Cohort"] = np.where(
    dataset["First_Sale_Date"] < MARCH_START,
    "Historical",
    "March 2025"
)

dataset = dataset.merge(
    products,
    on="Product ID",
    how="left",
    validate="many_to_one"
)

# ============================================================
# 6. CHECK COHORT COVERAGE
# ============================================================

coverage = (
    dataset.groupby(["Cohort", "Window_Complete"])
    .agg(
        Product_Currency_Series=("Product ID", "count"),
        Unique_Products=("Product ID", "nunique")
    )
    .reset_index()
)

print("\n========== COHORT WINDOW COVERAGE ==========")
print(coverage.to_string(index=False))

# Keep only complete 14-day windows.
complete = dataset[
    dataset["Window_Complete"]
].copy()

train = complete[
    complete["Cohort"] == "Historical"
].copy()

test = complete[
    complete["Cohort"] == "March 2025"
].copy()

print("\nTraining product-currency records:", f"{len(train):,}")
print("Testing product-currency records:", f"{len(test):,}")

if train.empty or test.empty:
    raise ValueError(
        "Training or testing cohort is empty. "
        "Check the first-sale dates and window coverage."
    )

# ============================================================
# 7. PREPARE FEATURES
# ============================================================

print("\nPreparing product attributes...")

categorical_features = [
    "Category",
    "Sub Category",
    "Color",
    "Sizes",
    "Currency"
]

text_feature = "Description EN"

for col in categorical_features:
    train[col] = train[col].fillna("Unknown").astype(str)
    test[col] = test[col].fillna("Unknown").astype(str)

train[text_feature] = (
    train[text_feature].fillna("").astype(str)
)

test[text_feature] = (
    test[text_feature].fillna("").astype(str)
)

# Do not use Product ID, dates, or production cost as model features.
# Product IDs identify products; dates identify cohorts.
# Production cost currency/meaning is not yet validated.

# ============================================================
# 8. ONE-HOT ENCODING
# ============================================================

print("\nEncoding categorical attributes...")

combined_categories = pd.concat(
    [
        train[categorical_features],
        test[categorical_features]
    ],
    ignore_index=True
)

encoded_all = pd.get_dummies(
    combined_categories,
    columns=categorical_features,
    dtype=np.float32
)

train_size = len(train)

X_train_cat = csr_matrix(
    encoded_all.iloc[:train_size].to_numpy(
        dtype=np.float32
    )
)

X_test_cat = csr_matrix(
    encoded_all.iloc[train_size:].to_numpy(
        dtype=np.float32
    )
)

categorical_names = encoded_all.columns.tolist()

del combined_categories, encoded_all
gc.collect()

# ============================================================
# 9. TF-IDF DESCRIPTION FEATURES
# ============================================================

print("\nExtracting English description features...")

vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words="english",
    sublinear_tf=True
)

X_train_text = vectorizer.fit_transform(
    train[text_feature]
)

X_test_text = vectorizer.transform(
    test[text_feature]
)

text_names = [
    f"Description_{term}"
    for term in vectorizer.get_feature_names_out()
]

print("Categorical features:", len(categorical_names))
print("Text features:", len(text_names))

# Combine sparse matrices.
X_train = hstack(
    [X_train_cat, X_train_text],
    format="csr"
)

X_test = hstack(
    [X_test_cat, X_test_text],
    format="csr"
)

feature_names = categorical_names + text_names

print("Training matrix:", X_train.shape)
print("Testing matrix:", X_test.shape)

del X_train_cat, X_test_cat
del X_train_text, X_test_text
gc.collect()

# ============================================================
# 10. TARGET
# ============================================================

y_train = train["First_14_Day_Units"].to_numpy(
    dtype=np.float32
)

y_test = test["First_14_Day_Units"].to_numpy(
    dtype=np.float32
)

if np.any(y_train < 0) or np.any(y_test < 0):
    raise ValueError("Negative sales quantities found.")

# Log transformation handles skewed demand.
y_train_log = np.log1p(y_train)

# ============================================================
# 11. TRAIN LIGHTGBM
# ============================================================

print("\nTraining LightGBM cold-start model...")

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    max_bin=63,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

model.fit(
    X_train,
    y_train_log
)

print("LightGBM training completed.")

# ============================================================
# 12. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating cold-start predictions...")

predicted_log = model.predict(X_test)

lightgbm_predictions = np.maximum(
    np.expm1(predicted_log),
    0
)

# ============================================================
# 13. BASELINE MODELS
# ============================================================

print("\nBuilding historical median baselines...")

global_median = train[
    "First_14_Day_Units"
].median()

category_baseline = (
    train.groupby("Category")["First_14_Day_Units"]
    .median()
)

subcategory_baseline = (
    train.groupby(
        ["Sub Category", "Currency"]
    )["First_14_Day_Units"]
    .median()
)

category_currency_baseline = (
    train.groupby(
        ["Category", "Currency"]
    )["First_14_Day_Units"]
    .median()
)

pred_category = (
    test["Category"]
    .map(category_baseline)
    .fillna(global_median)
    .to_numpy()
)

pred_subcategory = [
    subcategory_baseline.get(
        (subcategory, currency),
        global_median
    )
    for subcategory, currency in zip(
        test["Sub Category"],
        test["Currency"]
    )
]

pred_category_currency = [
    category_currency_baseline.get(
        (category, currency),
        global_median
    )
    for category, currency in zip(
        test["Category"],
        test["Currency"]
    )
]

# ============================================================
# 14. EVALUATION FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    total_actual = np.abs(actual).sum()

    return {
        "Records": len(actual),
        "Actual_Units": actual.sum(),
        "Predicted_Units": predicted.sum(),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "WAPE_Percent": (
            np.abs(actual - predicted).sum()
            / total_actual * 100
            if total_actual > 0 else np.nan
        ),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) > 1 and np.var(actual) > 0
            else np.nan
        )
    }


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

predictions = {
    "LightGBM_Cold_Start": lightgbm_predictions,
    "Subcategory_Currency_Median": np.asarray(
        pred_subcategory, dtype=float
    ),
    "Category_Currency_Median": np.asarray(
        pred_category_currency, dtype=float
    ),
    "Category_Median": np.asarray(
        pred_category, dtype=float
    )
}

results = []

for model_name, prediction in predictions.items():

    result = calculate_metrics(
        y_test,
        prediction
    )

    result["Model"] = model_name
    results.append(result)

comparison = pd.DataFrame(results)

comparison = comparison[
    [
        "Model",
        "Records",
        "Actual_Units",
        "Predicted_Units",
        "MAE",
        "RMSE",
        "WAPE_Percent",
        "R2"
    ]
]

comparison = comparison.sort_values(
    "WAPE_Percent"
)

print("\n========== COLD-START MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    PROCESSED / "cold_start_model_comparison.csv",
    index=False
)

# ============================================================
# 16. PREDICTION OUTPUT
# ============================================================

prediction_output = test[
    [
        "Product ID",
        "Currency",
        "First_Sale_Date",
        "Category",
        "Sub Category",
        "Color",
        "Sizes",
        "First_14_Day_Units",
        "First_14_Day_Active_Days"
    ]
].copy()

prediction_output["LightGBM_Predicted_Units"] = (
    lightgbm_predictions
)

prediction_output["Subcategory_Currency_Baseline"] = (
    pred_subcategory
)

prediction_output["Category_Currency_Baseline"] = (
    pred_category_currency
)

prediction_output["Absolute_Error"] = (
    prediction_output["LightGBM_Predicted_Units"]
    - prediction_output["First_14_Day_Units"]
).abs()

prediction_output["Signed_Error"] = (
    prediction_output["LightGBM_Predicted_Units"]
    - prediction_output["First_14_Day_Units"]
)

prediction_output.to_csv(
    PROCESSED / "cold_start_product_predictions.csv",
    index=False
)

# ============================================================
# 17. CATEGORY-CURRENCY EVALUATION
# ============================================================

category_results = []

for (category, currency), indices in (
    test.groupby(["Category", "Currency"]).groups.items()
):

    positions = test.index.get_indexer(indices)

    actual = y_test[positions]
    predicted = lightgbm_predictions[positions]

    result = calculate_metrics(
        actual,
        predicted
    )

    result["Category"] = category
    result["Currency"] = currency

    category_results.append(result)

category_evaluation = pd.DataFrame(
    category_results
)

category_evaluation.to_csv(
    PROCESSED / "cold_start_category_currency_evaluation.csv",
    index=False
)

print("\n========== CATEGORY-CURRENCY PERFORMANCE ==========")
print(category_evaluation.to_string(index=False))

# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance.to_csv(
    PROCESSED / "cold_start_feature_importance.csv",
    index=False
)

print("\n========== TOP 20 FEATURE IMPORTANCE ==========")
print(
    importance.head(20).to_string(index=False)
)

# ============================================================
# 19. HIGH-ERROR PRODUCTS
# ============================================================

prediction_output["Percentage_Error"] = np.where(
    prediction_output["First_14_Day_Units"] > 0,
    prediction_output["Absolute_Error"]
    / prediction_output["First_14_Day_Units"] * 100,
    np.nan
)

high_error = prediction_output.sort_values(
    "Absolute_Error",
    ascending=False
)

high_error.head(100).to_csv(
    PROCESSED / "cold_start_high_error_products.csv",
    index=False
)

# ============================================================
# 20. SAVE TRAINING AND TEST DATA
# ============================================================

train_output = train[
    [
        "Product ID",
        "Currency",
        "First_Sale_Date",
        "Category",
        "Sub Category",
        "Color",
        "Sizes",
        "First_14_Day_Units",
        "First_14_Day_Active_Days"
    ]
].copy()

train_output.to_csv(
    PROCESSED / "cold_start_training_dataset.csv",
    index=False
)

coverage.to_csv(
    PROCESSED / "cold_start_window_coverage.csv",
    index=False
)

# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print("Historical training records:", f"{len(train):,}")
print("March testing records:", f"{len(test):,}")

print(
    "Unique historical products:",
    f"{train['Product ID'].nunique():,}"
)

print(
    "Unique March products:",
    f"{test['Product ID'].nunique():,}"
)

print(
    "Historical median first-14-day units:",
    round(float(np.median(y_train)), 2)
)

print(
    "March median first-14-day units:",
    round(float(np.median(y_test)), 2)
)

print("\nSaved output files:")
print("1. cold_start_model_comparison.csv")
print("2. cold_start_product_predictions.csv")
print("3. cold_start_category_currency_evaluation.csv")
print("4. cold_start_feature_importance.csv")
print("5. cold_start_high_error_products.csv")
print("6. cold_start_training_dataset.csv")
print("7. cold_start_window_coverage.csv")

print("\nCold-start demand modelling completed.")
