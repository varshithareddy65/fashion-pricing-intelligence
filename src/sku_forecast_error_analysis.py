
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PREDICTION_FILE = PROCESSED / "sku_lightgbm_test_predictions.csv"
PRODUCT_FILE = RAW / "products.csv"

OUTPUT_DIR = PROCESSED

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading forecast predictions...")

df = pd.read_csv(
    PREDICTION_FILE,
    parse_dates=["Week_Start"],
    dtype={
        "SKU": "string",
        "Product ID": "string",
        "Currency": "string"
    }
)

products = pd.read_csv(
    PRODUCT_FILE,
    usecols=["Product ID", "Category", "Sub Category"],
    dtype={
        "Product ID": "string",
        "Category": "string",
        "Sub Category": "string"
    }
).drop_duplicates("Product ID")

print("Prediction records:", f"{len(df):,}")

# ============================================================
# 2. SELECT VALID EVALUATION WEEKS
# ============================================================

weekly_actual = (
    df.groupby("Week_Start")["Weekly_Units"]
    .sum()
)

valid_weeks = weekly_actual[
    weekly_actual > 0
].index

df = df[
    df["Week_Start"].isin(valid_weeks)
].copy()

print("Valid evaluation weeks:", len(valid_weeks))
print("Evaluation records:", f"{len(df):,}")

# ============================================================
# 3. ADD PRODUCT INFORMATION
# ============================================================

df = df.merge(
    products,
    on="Product ID",
    how="left",
    validate="many_to_one"
)

df["Category"] = df["Category"].fillna("Unknown")
df["Sub Category"] = df["Sub Category"].fillna("Unknown")

# ============================================================
# 4. SKU-LEVEL AGGREGATION
# ============================================================

print("\nCalculating SKU-level forecast errors...")

sku_metrics = (
    df.groupby(
        [
            "SKU",
            "Product ID",
            "Currency",
            "Category",
            "Sub Category",
            "Demand_Segment"
        ],
        dropna=False
    )
    .agg(
        Actual_Units=("Weekly_Units", "sum"),
        LightGBM_Predicted_Units=("LightGBM_Prediction", "sum"),
        HGB_Predicted_Units=("Predicted_Units", "sum"),
        Active_Actual_Weeks=(
            "Weekly_Units",
            lambda x: (x > 0).sum()
        ),
        Evaluation_Weeks=("Week_Start", "nunique")
    )
    .reset_index()
)

# ============================================================
# 5. CALCULATE ERRORS
# ============================================================

for model in ["LightGBM", "HGB"]:

    actual = sku_metrics["Actual_Units"]

    predicted = sku_metrics[
        f"{model}_Predicted_Units"
    ]

    sku_metrics[f"{model}_Signed_Error"] = (
        predicted - actual
    )

    sku_metrics[f"{model}_Absolute_Error"] = (
        (predicted - actual).abs()
    )

    sku_metrics[f"{model}_Bias_Percent"] = np.where(
        actual > 0,
        (predicted - actual) / actual * 100,
        np.nan
    )

    sku_metrics[f"{model}_WAPE_Percent"] = np.where(
        actual > 0,
        (predicted - actual).abs() / actual * 100,
        np.nan
    )

    sku_metrics[f"{model}_Error_Type"] = np.select(
        [
            predicted > actual,
            predicted < actual
        ],
        [
            "Overprediction",
            "Underprediction"
        ],
        default="Exact Prediction"
    )

# ============================================================
# 6. IDENTIFY ZERO-ACTUAL PRODUCTS
# ============================================================

zero_actual = sku_metrics[
    sku_metrics["Actual_Units"] == 0
].copy()

zero_actual = zero_actual.sort_values(
    "LightGBM_Predicted_Units",
    ascending=False
)

# ============================================================
# 7. IDENTIFY OVERPREDICTED PRODUCTS
# ============================================================

overpredicted = sku_metrics[
    (sku_metrics["Actual_Units"] > 0)
    & (sku_metrics["LightGBM_Signed_Error"] > 0)
].copy()

overpredicted = overpredicted.sort_values(
    "LightGBM_Signed_Error",
    ascending=False
)

# ============================================================
# 8. IDENTIFY UNDERPREDICTED PRODUCTS
# ============================================================

underpredicted = sku_metrics[
    (sku_metrics["Actual_Units"] > 0)
    & (sku_metrics["LightGBM_Signed_Error"] < 0)
].copy()

underpredicted = underpredicted.sort_values(
    "LightGBM_Signed_Error",
    ascending=True
)

# ============================================================
# 9. COMPARE LIGHTGBM AND HGB PER SKU
# ============================================================

sku_metrics["Better_Model"] = np.select(
    [
        sku_metrics["LightGBM_Absolute_Error"]
        < sku_metrics["HGB_Absolute_Error"],

        sku_metrics["LightGBM_Absolute_Error"]
        > sku_metrics["HGB_Absolute_Error"]
    ],
    [
        "LightGBM",
        "HistGradientBoosting"
    ],
    default="Tie"
)

model_wins = (
    sku_metrics["Better_Model"]
    .value_counts()
    .rename_axis("Model")
    .reset_index(name="SKU_Count")
)

# ============================================================
# 10. ERROR SUMMARY BY DEMAND SEGMENT
# ============================================================

segment_summary = (
    sku_metrics.groupby("Demand_Segment")
    .agg(
        SKU_Count=("SKU", "nunique"),
        Actual_Units=("Actual_Units", "sum"),
        LightGBM_Predicted_Units=(
            "LightGBM_Predicted_Units", "sum"
        ),
        HGB_Predicted_Units=(
            "HGB_Predicted_Units", "sum"
        ),
        Mean_LightGBM_Absolute_Error=(
            "LightGBM_Absolute_Error", "mean"
        ),
        Mean_HGB_Absolute_Error=(
            "HGB_Absolute_Error", "mean"
        ),
        Overpredicted_SKUs=(
            "LightGBM_Error_Type",
            lambda x: (x == "Overprediction").sum()
        ),
        Underpredicted_SKUs=(
            "LightGBM_Error_Type",
            lambda x: (x == "Underprediction").sum()
        )
    )
    .reset_index()
)

segment_summary["LightGBM_Bias_Percent"] = np.where(
    segment_summary["Actual_Units"] > 0,
    (
        segment_summary["LightGBM_Predicted_Units"]
        - segment_summary["Actual_Units"]
    ) / segment_summary["Actual_Units"] * 100,
    np.nan
)

# ============================================================
# 11. ERROR SUMMARY BY CURRENCY
# ============================================================

currency_summary = (
    sku_metrics.groupby("Currency")
    .agg(
        SKU_Count=("SKU", "nunique"),
        Actual_Units=("Actual_Units", "sum"),
        LightGBM_Predicted_Units=(
            "LightGBM_Predicted_Units", "sum"
        ),
        HGB_Predicted_Units=(
            "HGB_Predicted_Units", "sum"
        ),
        LightGBM_Absolute_Error=(
            "LightGBM_Absolute_Error", "sum"
        ),
        HGB_Absolute_Error=(
            "HGB_Absolute_Error", "sum"
        )
    )
    .reset_index()
)

currency_summary["LightGBM_WAPE_Percent"] = np.where(
    currency_summary["Actual_Units"] > 0,
    currency_summary["LightGBM_Absolute_Error"]
    / currency_summary["Actual_Units"] * 100,
    np.nan
)

currency_summary["HGB_WAPE_Percent"] = np.where(
    currency_summary["Actual_Units"] > 0,
    currency_summary["HGB_Absolute_Error"]
    / currency_summary["Actual_Units"] * 100,
    np.nan
)

# ============================================================
# 12. ERROR SUMMARY BY CATEGORY
# ============================================================

category_summary = (
    sku_metrics.groupby(
        ["Category", "Currency"]
    )
    .agg(
        SKU_Count=("SKU", "nunique"),
        Actual_Units=("Actual_Units", "sum"),
        LightGBM_Predicted_Units=(
            "LightGBM_Predicted_Units", "sum"
        ),
        LightGBM_Absolute_Error=(
            "LightGBM_Absolute_Error", "sum"
        ),
        HGB_Absolute_Error=(
            "HGB_Absolute_Error", "sum"
        )
    )
    .reset_index()
)

category_summary["LightGBM_WAPE_Percent"] = np.where(
    category_summary["Actual_Units"] > 0,
    category_summary["LightGBM_Absolute_Error"]
    / category_summary["Actual_Units"] * 100,
    np.nan
)

category_summary["HGB_WAPE_Percent"] = np.where(
    category_summary["Actual_Units"] > 0,
    category_summary["HGB_Absolute_Error"]
    / category_summary["Actual_Units"] * 100,
    np.nan
)

# ============================================================
# 13. SAVE OUTPUT FILES
# ============================================================

sku_metrics.to_csv(
    OUTPUT_DIR / "sku_level_forecast_errors.csv",
    index=False
)

overpredicted.head(100).to_csv(
    OUTPUT_DIR / "top_100_overpredicted_skus.csv",
    index=False
)

underpredicted.head(100).to_csv(
    OUTPUT_DIR / "top_100_underpredicted_skus.csv",
    index=False
)

zero_actual.head(100).to_csv(
    OUTPUT_DIR / "zero_actual_predicted_skus.csv",
    index=False
)

model_wins.to_csv(
    OUTPUT_DIR / "sku_model_wins.csv",
    index=False
)

segment_summary.to_csv(
    OUTPUT_DIR / "sku_error_by_segment.csv",
    index=False
)

currency_summary.to_csv(
    OUTPUT_DIR / "sku_error_by_currency.csv",
    index=False
)

category_summary.to_csv(
    OUTPUT_DIR / "sku_error_by_category_currency.csv",
    index=False
)

# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n========== SKU ERROR ANALYSIS ==========")

print("Total SKU-currency series:", f"{len(sku_metrics):,}")

print(
    "Overpredicted SKU series:",
    f"{len(overpredicted):,}"
)

print(
    "Underpredicted SKU series:",
    f"{len(underpredicted):,}"
)

print(
    "Zero-actual SKU series:",
    f"{len(zero_actual):,}"
)

print("\n========== TOP 15 OVERPREDICTED SKUS ==========")

print(
    overpredicted[
        [
            "SKU",
            "Product ID",
            "Category",
            "Sub Category",
            "Currency",
            "Demand_Segment",
            "Actual_Units",
            "LightGBM_Predicted_Units",
            "LightGBM_Signed_Error",
            "LightGBM_Bias_Percent"
        ]
    ].head(15).to_string(index=False)
)

print("\n========== TOP 15 UNDERPREDICTED SKUS ==========")

print(
    underpredicted[
        [
            "SKU",
            "Product ID",
            "Category",
            "Sub Category",
            "Currency",
            "Demand_Segment",
            "Actual_Units",
            "LightGBM_Predicted_Units",
            "LightGBM_Signed_Error",
            "LightGBM_Bias_Percent"
        ]
    ].head(15).to_string(index=False)
)

print("\n========== ZERO ACTUAL BUT PREDICTED ==========")

print(
    zero_actual[
        [
            "SKU",
            "Product ID",
            "Category",
            "Sub Category",
            "Currency",
            "Demand_Segment",
            "Actual_Units",
            "LightGBM_Predicted_Units"
        ]
    ].head(15).to_string(index=False)
)

print("\n========== MODEL WINS BY SKU ==========")
print(model_wins.to_string(index=False))

print("\n========== ERROR BY DEMAND SEGMENT ==========")
print(segment_summary.to_string(index=False))

print("\n========== ERROR BY CURRENCY ==========")
print(currency_summary.to_string(index=False))

print("\n========== ERROR BY CATEGORY AND CURRENCY ==========")
print(category_summary.to_string(index=False))

print("\nSaved all SKU-level error analysis files.")
print("SKU forecast error analysis completed.")
