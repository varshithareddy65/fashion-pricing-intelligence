
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

INPUT = PROCESSED / "sku_forecast_test_predictions.csv"

print("Loading SKU forecast predictions...")

df = pd.read_csv(
    INPUT,
    parse_dates=["Week_Start"],
    dtype={
        "SKU": "string",
        "Product ID": "string",
        "Currency": "string"
    }
)

def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    total_actual = np.abs(actual).sum()

    return {
        "Observations": len(actual),
        "Actual_Units": actual.sum(),
        "Predicted_Units": predicted.sum(),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WAPE_Percent": (
            np.abs(actual - predicted).sum() / total_actual * 100
            if total_actual > 0 else np.nan
        ),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) > 1 and np.var(actual) > 0
            else np.nan
        )
    }

# ------------------------------------------------------------
# 1. WEEKLY ACTUAL SALES CHECK
# ------------------------------------------------------------

weekly = (
    df.groupby("Week_Start")
    .agg(
        Actual_Units=("Weekly_Units", "sum"),
        Predicted_Units=("Predicted_Units", "sum"),
        SKU_Series=("SKU", "size"),
        Active_SKUs=("Weekly_Units", lambda x: (x > 0).sum())
    )
    .reset_index()
)

print("\n========== WEEKLY ACTUAL SALES ==========")
print(weekly.to_string(index=False))

# ------------------------------------------------------------
# 2. SELECT WEEKS WITH OBSERVED COHORT SALES
# ------------------------------------------------------------

valid_weeks = weekly.loc[
    weekly["Actual_Units"] > 0,
    "Week_Start"
]

valid = df[df["Week_Start"].isin(valid_weeks)].copy()

print("\nWeeks with observed sales:", len(valid_weeks))
print("Weeks excluded because total actual sales were zero:",
      len(weekly) - len(valid_weeks))

# ------------------------------------------------------------
# 3. COMPARE MODEL AGAINST BOTH BASELINES
# ------------------------------------------------------------

results = []

for name, prediction_column in [
    ("HistGradientBoosting", "Predicted_Units"),
    ("Previous_Week_Baseline", "Previous_Week_Prediction"),
    ("Previous_4_Week_Average", "Previous_4_Week_Average")
]:
    result = metrics(
        valid["Weekly_Units"],
        valid[prediction_column]
    )
    result["Model"] = name
    results.append(result)

evaluation = pd.DataFrame(results)

evaluation.to_csv(
    PROCESSED / "sku_forecast_valid_period_evaluation.csv",
    index=False
)

print("\n========== VALID-PERIOD EVALUATION ==========")
print(evaluation.to_string(index=False))

# ------------------------------------------------------------
# 4. WEEKLY MODEL PERFORMANCE
# ------------------------------------------------------------

weekly_results = []

for week, group in valid.groupby("Week_Start", sort=True):
    result = metrics(
        group["Weekly_Units"],
        group["Predicted_Units"]
    )

    result["Week_Start"] = week
    weekly_results.append(result)

weekly_evaluation = pd.DataFrame(weekly_results)

weekly_evaluation.to_csv(
    PROCESSED / "sku_forecast_valid_weekly_evaluation.csv",
    index=False
)

print("\n========== VALID WEEKLY PERFORMANCE ==========")
print(weekly_evaluation.to_string(index=False))

# ------------------------------------------------------------
# 5. CURRENCY-LEVEL MODEL PERFORMANCE
# ------------------------------------------------------------

currency_results = []

for currency, group in valid.groupby("Currency", sort=True):
    result = metrics(
        group["Weekly_Units"],
        group["Predicted_Units"]
    )
    result["Currency"] = currency
    currency_results.append(result)

currency_evaluation = pd.DataFrame(currency_results)

currency_evaluation.to_csv(
    PROCESSED / "sku_forecast_valid_currency_evaluation.csv",
    index=False
)

print("\n========== VALID CURRENCY PERFORMANCE ==========")
print(currency_evaluation.to_string(index=False))

print("\nSaved:")
print("sku_forecast_valid_period_evaluation.csv")
print("sku_forecast_valid_weekly_evaluation.csv")
print("sku_forecast_valid_currency_evaluation.csv")
