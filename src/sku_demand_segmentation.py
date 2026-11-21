
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

PREDICTIONS_FILE = PROCESSED / "sku_forecast_test_predictions.csv"
ELIGIBILITY_FILE = PROCESSED / "sku_training_eligibility.csv"

TEST_START = pd.Timestamp("2025-01-06")

print("Loading historical eligibility and predictions...")

predictions = pd.read_csv(
    PREDICTIONS_FILE,
    parse_dates=["Week_Start"],
    dtype={
        "SKU": "string",
        "Product ID": "string",
        "Currency": "string"
    }
)

history = pd.read_csv(
    ELIGIBILITY_FILE,
    parse_dates=["First_History_Week", "Last_History_Week"],
    dtype={
        "SKU": "string",
        "Product ID": "string",
        "Currency": "string"
    }
)

# ============================================================
# 1. CORRECT HISTORICAL EXPOSURE WINDOW
# ============================================================

print("\nCorrecting historical active-week rates...")

# Number of calendar weeks from first observed sale week
# through the final week before the test period.
history["Calendar_Weeks"] = (
    (TEST_START - history["First_History_Week"]).dt.days // 7
)

history["Calendar_Weeks"] = (
    history["Calendar_Weeks"] + 0
).clip(lower=1)

history["Active_Week_Rate"] = (
    history["Pre_Test_Active_Weeks"]
    / history["Calendar_Weeks"]
).clip(upper=1)

history["Average_Units_Per_Active_Week"] = (
    history["Historical_Units"]
    / history["Pre_Test_Active_Weeks"]
)

history["Average_Units_Per_Calendar_Week"] = (
    history["Historical_Units"]
    / history["Calendar_Weeks"]
)

# Only use series eligible before the test period.
history = history[
    history["Pre_Test_Active_Weeks"] >= 8
].copy()

# ============================================================
# 2. CLASSIFY DEMAND SEGMENTS
# ============================================================

intermittent_mask = history["Active_Week_Rate"] < 0.40

history["Demand_Segment"] = "Unassigned"

history.loc[
    intermittent_mask,
    "Demand_Segment"
] = "Intermittent"

frequent = history.loc[~intermittent_mask].copy()

volume_percentile = (
    frequent["Average_Units_Per_Calendar_Week"]
    .rank(method="average", pct=True)
)

history.loc[frequent.index, "Demand_Segment"] = np.select(
    [
        volume_percentile <= 1 / 3,
        volume_percentile <= 2 / 3
    ],
    [
        "Low Demand",
        "Medium Demand"
    ],
    default="High Demand"
)

print("\n========== CORRECTED SEGMENT COUNTS ==========")

segment_summary = (
    history.groupby("Demand_Segment")
    .agg(
        Series_Count=("SKU", "size"),
        Mean_Active_Week_Rate=("Active_Week_Rate", "mean"),
        Mean_Active_Weeks=("Pre_Test_Active_Weeks", "mean"),
        Mean_Calendar_Weeks=("Calendar_Weeks", "mean"),
        Mean_Units_Per_Active_Week=(
            "Average_Units_Per_Active_Week", "mean"
        ),
        Mean_Units_Per_Calendar_Week=(
            "Average_Units_Per_Calendar_Week", "mean"
        )
    )
    .reset_index()
)

print(segment_summary.to_string(index=False))

# ============================================================
# 3. ATTACH SEGMENTS TO TEST PREDICTIONS
# ============================================================

print("\nJoining predictions with corrected segments...")

segment_columns = [
    "SKU",
    "Currency",
    "Demand_Segment",
    "Active_Week_Rate",
    "Average_Units_Per_Calendar_Week",
    "Pre_Test_Active_Weeks"
]

results = predictions.merge(
    history[segment_columns],
    on=["SKU", "Currency"],
    how="left",
    validate="many_to_one"
)

if results["Demand_Segment"].isna().any():
    raise ValueError("Some predictions have no historical segment.")

# Keep the same observed-sales test weeks used in the previous
# valid-period analysis. This avoids treating the final three
# zero-sales weeks as normal evaluation weeks.
weekly_actual = (
    results.groupby("Week_Start")["Weekly_Units"]
    .sum()
)

valid_weeks = weekly_actual[weekly_actual > 0].index

results = results[
    results["Week_Start"].isin(valid_weeks)
].copy()

print("\nTest weeks with observed cohort sales:", len(valid_weeks))
print("Evaluation prediction records:", f"{len(results):,}")

# ============================================================
# 4. METRIC FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):
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
        ),
        "Overprediction_Rate_Percent": (
            (predicted > actual).mean() * 100
        )
    }

# ============================================================
# 5. EVALUATE EACH SEGMENT
# ============================================================

model_columns = {
    "HistGradientBoosting": "Predicted_Units",
    "Previous_Week_Baseline": "Previous_Week_Prediction",
    "Previous_4_Week_Average": "Previous_4_Week_Average"
}

segment_results = []

for segment, group in results.groupby("Demand_Segment"):

    for model_name, prediction_column in model_columns.items():

        metric = calculate_metrics(
            group["Weekly_Units"],
            group[prediction_column]
        )

        metric["Demand_Segment"] = segment
        metric["Model"] = model_name

        segment_results.append(metric)

segment_evaluation = pd.DataFrame(segment_results)

segment_evaluation.to_csv(
    PROCESSED / "sku_segment_evaluation.csv",
    index=False
)

print("\n========== CORRECTED SEGMENT EVALUATION ==========")
print(segment_evaluation.to_string(index=False))

# ============================================================
# 6. CURRENCY-LEVEL SEGMENT EVALUATION
# ============================================================

currency_results = []

for (segment, currency), group in results.groupby(
    ["Demand_Segment", "Currency"]
):

    metric = calculate_metrics(
        group["Weekly_Units"],
        group["Predicted_Units"]
    )

    metric["Demand_Segment"] = segment
    metric["Currency"] = currency

    currency_results.append(metric)

currency_evaluation = pd.DataFrame(currency_results)

currency_evaluation.to_csv(
    PROCESSED / "sku_segment_currency_evaluation.csv",
    index=False
)

print("\n========== CORRECTED CURRENCY EVALUATION ==========")
print(currency_evaluation.to_string(index=False))

# ============================================================
# 7. SAVE CORRECTED SEGMENTATION
# ============================================================

history.to_csv(
    PROCESSED / "sku_corrected_demand_segments.csv",
    index=False
)

results.to_csv(
    PROCESSED / "sku_segmented_test_predictions.csv",
    index=False
)

print("\nSaved:")
print("1. sku_corrected_demand_segments.csv")
print("2. sku_segment_evaluation.csv")
print("3. sku_segment_currency_evaluation.csv")
print("4. sku_segmented_test_predictions.csv")

print("\nCorrected SKU segmentation completed.")
