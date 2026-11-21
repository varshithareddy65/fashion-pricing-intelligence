
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

train = pd.read_csv(DATA / "cold_start_training_dataset.csv")
test = pd.read_csv(DATA / "cold_start_product_predictions.csv")

def metrics(group, actual_col, pred_col):
    actual = group[actual_col].to_numpy(dtype=float)
    predicted = group[pred_col].to_numpy(dtype=float)

    total = np.abs(actual).sum()

    return {
        "Records": len(group),
        "Actual_Units": actual.sum(),
        "Predicted_Units": predicted.sum(),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WAPE_Percent": (
            np.abs(actual - predicted).sum() / total * 100
            if total > 0 else np.nan
        ),
        "Bias_Percent": (
            (predicted.sum() - actual.sum()) / total * 100
            if total > 0 else np.nan
        ),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) > 1 and np.var(actual) > 0
            else np.nan
        )
    }

# --------------------------------------------------
# 1. Reconstruct historical baselines
# --------------------------------------------------

global_median = train["First_14_Day_Units"].median()

category_currency = (
    train.groupby(["Category", "Currency"])["First_14_Day_Units"]
    .median()
)

subcategory_currency = (
    train.groupby(["Sub Category", "Currency"])["First_14_Day_Units"]
    .median()
)

category = (
    train.groupby("Category")["First_14_Day_Units"]
    .median()
)

test["Category_Currency_Median"] = [
    category_currency.get((c, cur), global_median)
    for c, cur in zip(test["Category"], test["Currency"])
]

test["Subcategory_Currency_Median"] = [
    subcategory_currency.get((s, cur), global_median)
    for s, cur in zip(test["Sub Category"], test["Currency"])
]

test["Category_Median"] = (
    test["Category"].map(category).fillna(global_median)
)

# --------------------------------------------------
# 2. Overall model comparison
# --------------------------------------------------

models = {
    "LightGBM": "LightGBM_Predicted_Units",
    "Category_Currency_Median": "Category_Currency_Median",
    "Subcategory_Currency_Median": "Subcategory_Currency_Median",
    "Category_Median": "Category_Median"
}

overall = []

for name, col in models.items():
    result = metrics(
        test,
        "First_14_Day_Units",
        col
    )
    result["Model"] = name
    overall.append(result)

overall_df = pd.DataFrame(overall)

print("\n========== OVERALL COMPARISON ==========")
print(overall_df.to_string(index=False))

overall_df.to_csv(
    DATA / "cold_start_diagnostic_overall.csv",
    index=False
)

# --------------------------------------------------
# 3. Category-currency comparison
# --------------------------------------------------

segment_results = []

for (cat, cur), group in test.groupby(
    ["Category", "Currency"]
):

    for name, col in models.items():
        result = metrics(
            group,
            "First_14_Day_Units",
            col
        )

        result["Category"] = cat
        result["Currency"] = cur
        result["Model"] = name

        segment_results.append(result)

segment_df = pd.DataFrame(segment_results)

print("\n========== CATEGORY-CURRENCY COMPARISON ==========")
print(
    segment_df.sort_values(
        ["Category", "Currency", "WAPE_Percent"]
    ).to_string(index=False)
)

segment_df.to_csv(
    DATA / "cold_start_diagnostic_segments.csv",
    index=False
)

# --------------------------------------------------
# 4. Historical vs March demand distribution
# --------------------------------------------------

def distribution_summary(df, cohort_name):
    result = (
        df.groupby(["Category", "Currency"])
        ["First_14_Day_Units"]
        .agg(
            Records="count",
            Mean="mean",
            Median="median",
            Std="std",
            P75=lambda x: x.quantile(0.75),
            P90=lambda x: x.quantile(0.90),
            P95=lambda x: x.quantile(0.95)
        )
        .reset_index()
    )

    result["Cohort"] = cohort_name
    return result

historical_dist = distribution_summary(
    train, "Historical"
)

march_dist = distribution_summary(
    test, "March 2025"
)

distribution = pd.concat(
    [historical_dist, march_dist],
    ignore_index=True
)

print("\n========== DEMAND DISTRIBUTION ==========")
print(distribution.to_string(index=False))

distribution.to_csv(
    DATA / "cold_start_demand_distribution.csv",
    index=False
)

# --------------------------------------------------
# 5. Error direction
# --------------------------------------------------

test["Prediction_Error"] = (
    test["LightGBM_Predicted_Units"]
    - test["First_14_Day_Units"]
)

test["Error_Type"] = np.select(
    [
        test["Prediction_Error"] > 0,
        test["Prediction_Error"] < 0
    ],
    [
        "Overprediction",
        "Underprediction"
    ],
    default="Exact"
)

error_summary = (
    test.groupby(["Category", "Currency", "Error_Type"])
    .size()
    .reset_index(name="Record_Count")
)

print("\n========== ERROR DIRECTION ==========")
print(error_summary.to_string(index=False))

error_summary.to_csv(
    DATA / "cold_start_error_direction.csv",
    index=False
)

# --------------------------------------------------
# 6. Largest prediction errors
# --------------------------------------------------

test["Absolute_Error"] = test["Prediction_Error"].abs()

test.sort_values(
    "Absolute_Error",
    ascending=False
).head(100).to_csv(
    DATA / "cold_start_largest_errors.csv",
    index=False
)

print("\nDiagnostic files saved successfully.")
