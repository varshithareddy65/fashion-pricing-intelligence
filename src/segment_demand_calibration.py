
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

TRAIN_FILE = DATA / "cold_start_training_dataset.csv"
TEST_FILE = DATA / "cold_start_product_predictions.csv"

VALIDATION_START = pd.Timestamp("2024-09-01")
MIN_CALIBRATION_RECORDS = 100
SHRINKAGE_STRENGTH = 100

print("\n========== SEGMENT-SPECIFIC DEMAND CALIBRATION ==========")

# ============================================================
# 2. LOAD DATA
# ============================================================

train_all = pd.read_csv(TRAIN_FILE)
march = pd.read_csv(TEST_FILE)

for df in [train_all, march]:
    df["First_Sale_Date"] = pd.to_datetime(
        df["First_Sale_Date"],
        errors="coerce"
    )

required_train = [
    "First_Sale_Date",
    "Category",
    "Sub Category",
    "Currency",
    "First_14_Day_Units"
]

required_test = [
    "First_Sale_Date",
    "Category",
    "Sub Category",
    "Currency",
    "First_14_Day_Units"
]

if train_all[required_train].isna().any().any():
    raise ValueError("Missing required values in historical training data.")

if march[required_test].isna().any().any():
    raise ValueError("Missing required values in March test data.")

# ============================================================
# 3. CHRONOLOGICAL HISTORICAL SPLIT
# ============================================================

historical_fit = train_all[
    train_all["First_Sale_Date"] < VALIDATION_START
].copy()

validation = train_all[
    train_all["First_Sale_Date"] >= VALIDATION_START
].copy()

march_test = march.copy()

if historical_fit.empty or validation.empty or march_test.empty:
    raise ValueError("One of the modelling datasets is empty.")

print("\n========== CHRONOLOGICAL SPLIT ==========")
print(
    "Historical fitting records:",
    f"{len(historical_fit):,}"
)
print(
    "Historical validation records:",
    f"{len(validation):,}"
)
print(
    "March test records:",
    f"{len(march_test):,}"
)

print(
    "Fit date range:",
    historical_fit["First_Sale_Date"].min().date(),
    "to",
    historical_fit["First_Sale_Date"].max().date()
)

print(
    "Validation date range:",
    validation["First_Sale_Date"].min().date(),
    "to",
    validation["First_Sale_Date"].max().date()
)

print(
    "March test date range:",
    march_test["First_Sale_Date"].min().date(),
    "to",
    march_test["First_Sale_Date"].max().date()
)

# ============================================================
# 4. BASELINE MEDIANS FROM HISTORICAL FIT DATA ONLY
# ============================================================

global_median = float(
    historical_fit["First_14_Day_Units"].median()
)

category_currency_medians = (
    historical_fit.groupby(
        ["Category", "Currency"]
    )["First_14_Day_Units"]
    .median()
    .to_dict()
)

subcategory_currency_medians = (
    historical_fit.groupby(
        ["Sub Category", "Currency"]
    )["First_14_Day_Units"]
    .median()
    .to_dict()
)

category_medians = (
    historical_fit.groupby("Category")
    ["First_14_Day_Units"]
    .median()
    .to_dict()
)


def predict_category_currency(df):
    return np.array([
        category_currency_medians.get(
            (category, currency),
            category_medians.get(category, global_median)
        )
        for category, currency in zip(
            df["Category"],
            df["Currency"]
        )
    ], dtype=float)


def predict_subcategory_currency(df):
    return np.array([
        subcategory_currency_medians.get(
            (subcategory, currency),
            category_currency_medians.get(
                (category, currency),
                category_medians.get(category, global_median)
            )
        )
        for subcategory, category, currency in zip(
            df["Sub Category"],
            df["Category"],
            df["Currency"]
        )
    ], dtype=float)


def predict_category(df):
    return np.array([
        category_medians.get(category, global_median)
        for category in df["Category"]
    ], dtype=float)


# ============================================================
# 5. HISTORICAL VALIDATION PREDICTIONS
# ============================================================

validation = validation.copy()

validation["Category_Currency_Prediction"] = (
    predict_category_currency(validation)
)

validation["Subcategory_Currency_Prediction"] = (
    predict_subcategory_currency(validation)
)

validation["Category_Prediction"] = (
    predict_category(validation)
)

# ============================================================
# 6. METRIC FUNCTION
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
        "Bias_Percent": (
            (predicted.sum() - actual.sum())
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
# 7. SELECT BEST STRATEGY PER CATEGORY-CURRENCY
# ============================================================

print("\nSelecting strategy using historical validation...")

strategy_rows = []

for (category, currency), group in validation.groupby(
    ["Category", "Currency"]
):

    actual = group["First_14_Day_Units"].to_numpy()

    cat_pred = group[
        "Category_Currency_Prediction"
    ].to_numpy()

    subcat_pred = group[
        "Subcategory_Currency_Prediction"
    ].to_numpy()

    cat_metrics = calculate_metrics(actual, cat_pred)
    subcat_metrics = calculate_metrics(actual, subcat_pred)

    if (
        subcat_metrics["WAPE_Percent"]
        < cat_metrics["WAPE_Percent"]
    ):
        selected_strategy = "Subcategory_Currency"
        selected_prediction = subcat_pred
        selected_wape = subcat_metrics["WAPE_Percent"]
    else:
        selected_strategy = "Category_Currency"
        selected_prediction = cat_pred
        selected_wape = cat_metrics["WAPE_Percent"]

    # Calibration is based only on historical validation records.
    actual_total = actual.sum()
    predicted_total = selected_prediction.sum()

    if predicted_total > 0:
        raw_factor = actual_total / predicted_total
    else:
        raw_factor = 1.0

    # Shrink small-sample calibration factors towards 1.
    n = len(group)

    if n >= MIN_CALIBRATION_RECORDS:
        weight = n / (n + SHRINKAGE_STRENGTH)
        calibration_factor = (
            1 + weight * (raw_factor - 1)
        )
    else:
        calibration_factor = 1.0

    strategy_rows.append({
        "Category": category,
        "Currency": currency,
        "Validation_Records": n,
        "Category_Currency_WAPE": (
            cat_metrics["WAPE_Percent"]
        ),
        "Subcategory_Currency_WAPE": (
            subcat_metrics["WAPE_Percent"]
        ),
        "Selected_Strategy": selected_strategy,
        "Selected_Validation_WAPE": selected_wape,
        "Raw_Calibration_Factor": raw_factor,
        "Calibration_Factor": calibration_factor
    })

strategy_table = pd.DataFrame(strategy_rows)

print("\n========== SELECTED STRATEGIES ==========")
print(strategy_table.to_string(index=False))

strategy_table.to_csv(
    DATA / "calibration_strategy_selection.csv",
    index=False
)

# ============================================================
# 8. APPLY SELECTED STRATEGY TO MARCH TEST
# ============================================================

march_test = march_test.copy()

march_test["Category_Currency_Baseline"] = (
    predict_category_currency(march_test)
)

march_test["Subcategory_Currency_Baseline"] = (
    predict_subcategory_currency(march_test)
)

march_test["Category_Baseline"] = (
    predict_category(march_test)
)

strategy_lookup = {
    (row["Category"], row["Currency"]): row
    for _, row in strategy_table.iterrows()
}

selected_predictions = []
calibrated_predictions = []
selected_strategies = []
applied_factors = []

for _, row in march_test.iterrows():

    key = (row["Category"], row["Currency"])
    strategy = strategy_lookup.get(key)

    if strategy is None:
        selected_strategy = "Category_Currency"
        factor = 1.0
    else:
        selected_strategy = strategy["Selected_Strategy"]
        factor = strategy["Calibration_Factor"]

    if selected_strategy == "Subcategory_Currency":
        base_prediction = row[
            "Subcategory_Currency_Baseline"
        ]
    else:
        base_prediction = row[
            "Category_Currency_Baseline"
        ]

    selected_predictions.append(base_prediction)
    calibrated_predictions.append(
        max(0.0, base_prediction * factor)
    )
    selected_strategies.append(selected_strategy)
    applied_factors.append(factor)

march_test["Selected_Strategy"] = selected_strategies

march_test["Selected_Baseline_Prediction"] = (
    selected_predictions
)

march_test["Calibration_Factor"] = applied_factors

march_test["Calibrated_Prediction"] = (
    calibrated_predictions
)

# ============================================================
# 9. MODEL COMPARISON ON UNTOUCHED MARCH TEST
# ============================================================

comparison_models = {
    "Category_Currency_Median": (
        "Category_Currency_Baseline"
    ),
    "Subcategory_Currency_Median": (
        "Subcategory_Currency_Baseline"
    ),
    "Category_Median": (
        "Category_Baseline"
    ),
    "Segment_Selected_Baseline": (
        "Selected_Baseline_Prediction"
    ),
    "Segment_Selected_Calibrated": (
        "Calibrated_Prediction"
    )
}

# Existing LightGBM result is included as a reference.
# It was trained using the full historical cohort, unlike
# this chronological validation-based calibration workflow.
if "LightGBM_Predicted_Units" in march_test.columns:
    comparison_models["Existing_LightGBM_Reference"] = (
        "LightGBM_Predicted_Units"
    )

comparison_rows = []

actual = march_test["First_14_Day_Units"].to_numpy()

for model_name, prediction_column in comparison_models.items():

    if prediction_column not in march_test.columns:
        continue

    result = calculate_metrics(
        actual,
        march_test[prediction_column].to_numpy()
    )

    result["Model"] = model_name
    comparison_rows.append(result)

comparison = pd.DataFrame(comparison_rows)

comparison = comparison.sort_values(
    "WAPE_Percent"
)

print("\n========== FINAL MARCH TEST COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    DATA / "calibrated_demand_model_comparison.csv",
    index=False
)

# ============================================================
# 10. CATEGORY-CURRENCY TEST PERFORMANCE
# ============================================================

segment_results = []

for (category, currency), group in march_test.groupby(
    ["Category", "Currency"]
):

    for model_name, prediction_column in comparison_models.items():

        if prediction_column not in group.columns:
            continue

        result = calculate_metrics(
            group["First_14_Day_Units"],
            group[prediction_column]
        )

        result["Category"] = category
        result["Currency"] = currency
        result["Model"] = model_name

        segment_results.append(result)

segment_evaluation = pd.DataFrame(segment_results)

print("\n========== MARCH CATEGORY-CURRENCY PERFORMANCE ==========")
print(
    segment_evaluation.sort_values(
        ["Category", "Currency", "WAPE_Percent"]
    ).to_string(index=False)
)

segment_evaluation.to_csv(
    DATA / "calibrated_demand_segment_evaluation.csv",
    index=False
)

# ============================================================
# 11. ERROR AND BIAS ANALYSIS
# ============================================================

march_test["Calibrated_Error"] = (
    march_test["Calibrated_Prediction"]
    - march_test["First_14_Day_Units"]
)

march_test["Calibrated_Absolute_Error"] = (
    march_test["Calibrated_Error"].abs()
)

march_test["Prediction_Direction"] = np.select(
    [
        march_test["Calibrated_Error"] > 0,
        march_test["Calibrated_Error"] < 0
    ],
    [
        "Overprediction",
        "Underprediction"
    ],
    default="Exact"
)

direction_summary = (
    march_test.groupby(
        ["Category", "Currency", "Prediction_Direction"]
    )
    .size()
    .reset_index(name="Records")
)

direction_summary.to_csv(
    DATA / "calibrated_demand_error_direction.csv",
    index=False
)

# ============================================================
# 12. SAVE PREDICTIONS
# ============================================================

march_test.to_csv(
    DATA / "calibrated_cold_start_predictions.csv",
    index=False
)

march_test.sort_values(
    "Calibrated_Absolute_Error",
    ascending=False
).head(100).to_csv(
    DATA / "calibrated_demand_largest_errors.csv",
    index=False
)

# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print("Validation strategy selection completed.")
print("Calibration factors estimated from validation only.")
print("March test actuals were not used for model selection.")

print(
    "\nBest March test model:",
    comparison.iloc[0]["Model"]
)

print(
    "Best March test WAPE:",
    round(comparison.iloc[0]["WAPE_Percent"], 2),
    "%"
)

print("\nSaved output files:")
print("1. calibration_strategy_selection.csv")
print("2. calibrated_demand_model_comparison.csv")
print("3. calibrated_demand_segment_evaluation.csv")
print("4. calibrated_demand_error_direction.csv")
print("5. calibrated_cold_start_predictions.csv")
print("6. calibrated_demand_largest_errors.csv")

print("\nDemand calibration completed successfully.")
