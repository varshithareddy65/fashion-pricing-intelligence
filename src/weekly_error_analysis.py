
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data/processed/demand_model_weekly_predictions.csv"
)

OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 65)
print("WEEKLY FORECAST ERROR ANALYSIS")
print("=" * 65)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File not found: {INPUT_FILE}\n"
        "Run demand_forecasting.py first."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Weekly prediction records: {len(df):,}")


# ============================================================
# 3. VALIDATE DATA
# ============================================================

required_columns = [
    "Category",
    "Sub Category",
    "Currency",
    "Date",
    "Target_7_Day_Units",
    "Model_Prediction",
    "Baseline_Prediction",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# 4. CLEAN DATA
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)

numeric_columns = [
    "Target_7_Day_Units",
    "Model_Prediction",
    "Baseline_Prediction",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

df = df.dropna(
    subset=required_columns
).copy()

if df.empty:
    raise ValueError(
        "No valid prediction records found."
    )

df = df.sort_values(
    ["Date", "Category", "Sub Category", "Currency"]
).reset_index(drop=True)


# ============================================================
# 5. CALCULATE ERRORS
# ============================================================

df["Model_Error"] = (
    df["Model_Prediction"]
    - df["Target_7_Day_Units"]
)

df["Baseline_Error"] = (
    df["Baseline_Prediction"]
    - df["Target_7_Day_Units"]
)

df["Model_Absolute_Error"] = (
    df["Model_Error"].abs()
)

df["Baseline_Absolute_Error"] = (
    df["Baseline_Error"].abs()
)

df["Model_Squared_Error"] = (
    df["Model_Error"] ** 2
)

df["Baseline_Squared_Error"] = (
    df["Baseline_Error"] ** 2
)

df["Model_Overpredicts"] = (
    df["Model_Error"] > 0
).astype(int)

df["Model_Underpredicts"] = (
    df["Model_Error"] < 0
).astype(int)

df["Actual_Is_Zero"] = (
    df["Target_7_Day_Units"] == 0
).astype(int)


# ============================================================
# 6. METRIC FUNCTION
# ============================================================

def calculate_metrics(group):

    actual = group[
        "Target_7_Day_Units"
    ].to_numpy(dtype=float)

    predicted = group[
        "Model_Prediction"
    ].to_numpy(dtype=float)

    baseline = group[
        "Baseline_Prediction"
    ].to_numpy(dtype=float)

    model_error = predicted - actual
    baseline_error = baseline - actual

    total_actual = np.sum(np.abs(actual))

    if total_actual > 0:
        model_wape = (
            np.sum(np.abs(model_error))
            / total_actual
        ) * 100

        baseline_wape = (
            np.sum(np.abs(baseline_error))
            / total_actual
        ) * 100

        model_bias = (
            np.sum(model_error)
            / total_actual
        ) * 100
    else:
        model_wape = np.nan
        baseline_wape = np.nan
        model_bias = np.nan

    model_mae = np.mean(
        np.abs(model_error)
    )

    baseline_mae = np.mean(
        np.abs(baseline_error)
    )

    model_rmse = np.sqrt(
        np.mean(model_error ** 2)
    )

    baseline_rmse = np.sqrt(
        np.mean(baseline_error ** 2)
    )

    if len(actual) > 1 and np.var(actual) > 0:

        model_r2 = (
            1
            - (
                np.sum(model_error ** 2)
                / np.sum((actual - np.mean(actual)) ** 2)
            )
        )

    else:
        model_r2 = np.nan

    improvement = np.nan

    if baseline_mae > 0:
        improvement = (
            (baseline_mae - model_mae)
            / baseline_mae
        ) * 100

    return {
        "Forecast_Records": len(group),

        "Actual_Units": np.sum(actual),
        "Predicted_Units": np.sum(predicted),

        "Mean_Actual_Units": np.mean(actual),
        "Mean_Predicted_Units": np.mean(predicted),

        "Model_MAE": model_mae,
        "Model_RMSE": model_rmse,
        "Model_WAPE_Percent": model_wape,
        "Model_R2": model_r2,

        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
        "Baseline_WAPE_Percent": baseline_wape,

        "MAE_Improvement_Percent": improvement,

        "Model_Bias_Percent": model_bias,

        "Overprediction_Rate_Percent": (
            np.mean(model_error > 0) * 100
        ),

        "Underprediction_Rate_Percent": (
            np.mean(model_error < 0) * 100
        ),

        "Zero_Demand_Percent": (
            np.mean(actual == 0) * 100
        ),
    }


# ============================================================
# 7. OVERALL WEEKLY ERROR ANALYSIS
# ============================================================

print("\nCalculating overall weekly performance...")

overall_metrics = pd.DataFrame(
    [
        calculate_metrics(df)
    ]
)

print(
    overall_metrics.to_string(index=False)
)


# ============================================================
# 8. CATEGORY-CURRENCY ANALYSIS
# ============================================================

print("\nAnalysing category and currency segments...")

segment_results = []

for (category, currency), group in df.groupby(
    ["Category", "Currency"]
):

    result = calculate_metrics(group)

    result["Category"] = category
    result["Currency"] = currency

    segment_results.append(result)

segment_metrics = pd.DataFrame(
    segment_results
)

segment_metrics = segment_metrics.sort_values(
    "Model_WAPE_Percent",
    ascending=True,
)


# ============================================================
# 9. SUBCATEGORY ANALYSIS
# ============================================================

print("\nAnalysing subcategory performance...")

subcategory_results = []

for (category, subcategory, currency), group in df.groupby(
    ["Category", "Sub Category", "Currency"]
):

    result = calculate_metrics(group)

    result["Category"] = category
    result["Sub Category"] = subcategory
    result["Currency"] = currency

    subcategory_results.append(result)

subcategory_metrics = pd.DataFrame(
    subcategory_results
)

subcategory_metrics = subcategory_metrics.sort_values(
    "Model_WAPE_Percent",
    ascending=True,
)


# ============================================================
# 10. FORECAST-ORIGIN DATE ANALYSIS
# ============================================================

print("\nAnalysing errors by forecast origin date...")

date_results = []

for date, group in df.groupby("Date"):

    result = calculate_metrics(group)
    result["Forecast_Origin_Date"] = date

    date_results.append(result)

date_metrics = pd.DataFrame(
    date_results
).sort_values(
    "Forecast_Origin_Date"
)


# ============================================================
# 11. WORST PREDICTIONS
# ============================================================

print("\nFinding largest prediction errors...")

worst_predictions = df.sort_values(
    "Model_Absolute_Error",
    ascending=False,
).head(50).copy()

worst_predictions = worst_predictions[
    [
        "Date",
        "Category",
        "Sub Category",
        "Currency",
        "Target_7_Day_Units",
        "Model_Prediction",
        "Baseline_Prediction",
        "Model_Error",
        "Model_Absolute_Error",
    ]
]


# ============================================================
# 12. SAVE OUTPUT FILES
# ============================================================

outputs = {
    "weekly_error_overall.csv": overall_metrics,

    "weekly_error_by_category_currency.csv": segment_metrics,

    "weekly_error_by_subcategory.csv": subcategory_metrics,

    "weekly_error_by_forecast_date.csv": date_metrics,

    "weekly_worst_predictions.csv": worst_predictions,
}

print("\nSaving analysis files...")

for filename, frame in outputs.items():

    path = OUTPUT_DIR / filename

    frame.to_csv(
        path,
        index=False,
    )

    print(f"Saved: {path}")


# ============================================================
# 13. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("WEEKLY FORECAST ERROR REPORT")
print("=" * 65)

print("\nOverall performance:")
print(
    overall_metrics.to_string(index=False)
)

print("\nBest category-currency segments:")
print(
    segment_metrics.head(5).to_string(index=False)
)

print("\nWeakest category-currency segments:")
print(
    segment_metrics.tail(5).to_string(index=False)
)

print("\nBest subcategories:")
print(
    subcategory_metrics.head(10).to_string(index=False)
)

print("\nForecast performance by date:")
print(
    date_metrics[
        [
            "Forecast_Origin_Date",
            "Model_MAE",
            "Model_WAPE_Percent",
            "Model_Bias_Percent",
        ]
    ].to_string(index=False)
)

print("\nWeekly error analysis completed successfully.")
