
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

WEEKLY_FILE = PROCESSED / "sku_weekly_demand.csv"
ELIGIBILITY_FILE = PROCESSED / "sku_training_eligibility.csv"
PRODUCT_FILE = RAW / "products.csv"

TEST_START = pd.Timestamp("2025-01-06")
MIN_ACTIVE_WEEKS = 8

RANDOM_STATE = 42

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading weekly SKU demand...")

weekly = pd.read_csv(
    WEEKLY_FILE,
    parse_dates=["Week_Start"],
    dtype={
        "SKU": "string",
        "Product ID": "string",
        "Currency": "string"
    }
)

eligibility = pd.read_csv(
    ELIGIBILITY_FILE,
    parse_dates=["First_History_Week", "Last_History_Week"],
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

print("Weekly records:", f"{len(weekly):,}")
print("Eligibility records:", f"{len(eligibility):,}")

# ============================================================
# 2. SELECT ELIGIBLE SKU-CURRENCY SERIES
# ============================================================

eligible = eligibility[
    eligibility["Pre_Test_Active_Weeks"] >= MIN_ACTIVE_WEEKS
].copy()

eligible = eligible[
    ["SKU", "Product ID", "Currency", "First_History_Week"]
].drop_duplicates(["SKU", "Currency"])

eligible = eligible.reset_index(drop=True)

print("\nEligible SKU-currency series:", f"{len(eligible):,}")

# ============================================================
# 3. CREATE COMPLETE WEEKLY PANEL
# ============================================================
# Missing weekly sales records are filled with zero.
# This represents zero recorded sales, not confirmed stock availability.

last_week = weekly["Week_Start"].max()

lengths = (
    (last_week - eligible["First_History_Week"]).dt.days // 7 + 1
).astype(int).to_numpy()

series_indices = np.repeat(
    np.arange(len(eligible)),
    lengths
)

week_offsets = np.concatenate([
    np.arange(length, dtype=np.int16)
    for length in lengths
])

print("\nCreating complete weekly panel...")

panel = pd.DataFrame({
    "SKU": eligible["SKU"].to_numpy()[series_indices],
    "Product ID": eligible["Product ID"].to_numpy()[series_indices],
    "Currency": eligible["Currency"].to_numpy()[series_indices],
    "First_History_Week": eligible["First_History_Week"].to_numpy()[series_indices],
    "Week_Start": (
        eligible["First_History_Week"].to_numpy()[series_indices]
        + pd.to_timedelta(week_offsets * 7, unit="D")
    )
})

# Add product category information.
panel = panel.merge(
    products,
    on="Product ID",
    how="left"
)

# Keep sales belonging to eligible series only.
sales = weekly.merge(
    eligible[["SKU", "Currency"]],
    on=["SKU", "Currency"],
    how="inner"
)

# Attach observed weekly units to the complete panel.
panel = panel.merge(
    sales[
        [
            "SKU",
            "Product ID",
            "Currency",
            "Week_Start",
            "Weekly_Units"
        ]
    ],
    on=["SKU", "Product ID", "Currency", "Week_Start"],
    how="left"
)

panel["Weekly_Units"] = (
    panel["Weekly_Units"].fillna(0).astype("float32")
)

panel = panel.sort_values(
    ["SKU", "Currency", "Week_Start"]
).reset_index(drop=True)

print("Complete panel records:", f"{len(panel):,}")
print("Zero-sales weekly records:",
      f"{(panel['Weekly_Units'] == 0).sum():,}")

# ============================================================
# 4. CREATE HISTORICAL FEATURES
# ============================================================

print("\nEngineering historical features...")

keys = ["SKU", "Currency"]

grouped = panel.groupby(keys, sort=False)

# Previous weeks' sales.
panel["Lag_1"] = grouped["Weekly_Units"].shift(1)
panel["Lag_2"] = grouped["Weekly_Units"].shift(2)
panel["Lag_4"] = grouped["Weekly_Units"].shift(4)
panel["Lag_8"] = grouped["Weekly_Units"].shift(8)

# Rolling averages, shifted to exclude the current target week.
rolling_4 = (
    panel.groupby(keys)["Weekly_Units"]
    .rolling(4, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

rolling_8 = (
    panel.groupby(keys)["Weekly_Units"]
    .rolling(8, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

panel["Rolling_4_Week_Mean"] = (
    rolling_4.groupby([panel["SKU"], panel["Currency"]])
    .shift(1)
)

panel["Rolling_8_Week_Mean"] = (
    rolling_8.groupby([panel["SKU"], panel["Currency"]])
    .shift(1)
)

# Expanding historical average, excluding the current week.
expanding_mean = (
    panel.groupby(keys)["Weekly_Units"]
    .expanding(min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

panel["Historical_Mean"] = (
    expanding_mean.groupby([panel["SKU"], panel["Currency"]])
    .shift(1)
)

# Number of previously active sales weeks.
previous_units = grouped["Weekly_Units"].shift(1)

panel["Historical_Active_Weeks"] = (
    previous_units.fillna(0).gt(0).astype("int8")
    .groupby([panel["SKU"], panel["Currency"]])
    .cumsum()
)

# Age of the series in weeks.
panel["Series_Age_Weeks"] = (
    (panel["Week_Start"] - panel["First_History_Week"])
    .dt.days // 7
)

# Calendar features for the target week.
panel["Month"] = panel["Week_Start"].dt.month.astype(str)
panel["Week_Of_Year"] = (
    panel["Week_Start"].dt.isocalendar().week.astype(int)
)

# Remove rows without any previous weekly observation.
panel = panel.dropna(subset=["Lag_1"]).copy()

print("Feature records:", f"{len(panel):,}")

# ============================================================
# 5. TEMPORAL TRAIN / TEST SPLIT
# ============================================================

train = panel[panel["Week_Start"] < TEST_START].copy()
test = panel[panel["Week_Start"] >= TEST_START].copy()

print("\n========== TEMPORAL SPLIT ==========")
print("Training rows:", f"{len(train):,}")
print("Testing rows:", f"{len(test):,}")
print("Test starts:", test["Week_Start"].min().date())
print("Test ends:", test["Week_Start"].max().date())

# ============================================================
# 6. PREPARE MODEL FEATURES
# ============================================================

numeric_features = [
    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Lag_8",
    "Rolling_4_Week_Mean",
    "Rolling_8_Week_Mean",
    "Historical_Mean",
    "Historical_Active_Weeks",
    "Series_Age_Weeks",
    "Week_Of_Year"
]

categorical_features = [
    "Category",
    "Sub Category",
    "Currency",
    "Month"
]

# Replace missing product attributes with explicit categories.
for col in categorical_features:
    train[col] = train[col].fillna("Unknown")
    test[col] = test[col].fillna("Unknown")

X_train = pd.get_dummies(
    train[numeric_features + categorical_features],
    columns=categorical_features,
    dtype=np.uint8
)

X_test = pd.get_dummies(
    test[numeric_features + categorical_features],
    columns=categorical_features,
    dtype=np.uint8
)

# Align columns between train and test.
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

y_train = train["Weekly_Units"].astype("float32")
y_test = test["Weekly_Units"].astype("float32")

print("\nModel feature count:", X_train.shape[1])

# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining HistGradientBoosting model...")

model = HistGradientBoostingRegressor(
    loss="poisson",
    learning_rate=0.08,
    max_iter=100,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

print("Model training completed.")

# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

predictions = model.predict(X_test)
predictions = np.maximum(predictions, 0)

# Baseline 1: previous week's sales.
baseline_lag1 = test["Lag_1"].to_numpy()

# Baseline 2: previous four-week average.
baseline_rolling4 = (
    test["Rolling_4_Week_Mean"]
    .fillna(0)
    .to_numpy()
)

# ============================================================
# 9. EVALUATION FUNCTIONS
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    total_actual = np.sum(np.abs(actual))

    wape = (
        np.sum(np.abs(actual - predicted)) / total_actual * 100
        if total_actual > 0
        else np.nan
    )

    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_Percent": wape,
        "R2": r2,
        "Actual_Units": actual.sum(),
        "Predicted_Units": predicted.sum(),
        "Observations": len(actual)
    }


# ============================================================
# 10. OVERALL MODEL EVALUATION
# ============================================================

model_metrics = calculate_metrics(y_test, predictions)
lag1_metrics = calculate_metrics(y_test, baseline_lag1)
rolling4_metrics = calculate_metrics(y_test, baseline_rolling4)

evaluation = pd.DataFrame([
    {"Model": "HistGradientBoosting", **model_metrics},
    {"Model": "Previous_Week_Baseline", **lag1_metrics},
    {"Model": "Previous_4_Week_Average", **rolling4_metrics}
])

evaluation.to_csv(
    PROCESSED / "sku_forecast_evaluation.csv",
    index=False
)

print("\n========== OVERALL EVALUATION ==========")
print(evaluation.to_string(index=False))

# ============================================================
# 11. CURRENCY-LEVEL EVALUATION
# ============================================================

currency_results = []

for currency, group in test.groupby("Currency", sort=True):
    idx = group.index.to_numpy()
    positions = test.index.get_indexer(idx)

    result = calculate_metrics(
        group["Weekly_Units"],
        predictions[positions]
    )

    result["Currency"] = currency
    currency_results.append(result)

currency_evaluation = pd.DataFrame(currency_results)

currency_evaluation.to_csv(
    PROCESSED / "sku_forecast_currency_evaluation.csv",
    index=False
)

print("\n========== CURRENCY-LEVEL EVALUATION ==========")
print(currency_evaluation.to_string(index=False))

# ============================================================
# 12. WEEKLY EVALUATION
# ============================================================

test_results = test[
    [
        "SKU",
        "Product ID",
        "Currency",
        "Category",
        "Sub Category",
        "Week_Start",
        "Weekly_Units"
    ]
].copy()

test_results["Predicted_Units"] = predictions
test_results["Previous_Week_Prediction"] = baseline_lag1
test_results["Previous_4_Week_Average"] = baseline_rolling4

weekly_results = []

for week, group in test_results.groupby("Week_Start", sort=True):
    result = calculate_metrics(
        group["Weekly_Units"],
        group["Predicted_Units"]
    )

    result["Week_Start"] = week
    weekly_results.append(result)

weekly_evaluation = pd.DataFrame(weekly_results)

weekly_evaluation.to_csv(
    PROCESSED / "sku_forecast_weekly_evaluation.csv",
    index=False
)

# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

test_results.to_csv(
    PROCESSED / "sku_forecast_test_predictions.csv",
    index=False
)

print("\n========== WEEKLY EVALUATION ==========")
print(weekly_evaluation.to_string(index=False))

print("\nSaved output files:")
print("1. sku_forecast_evaluation.csv")
print("2. sku_forecast_currency_evaluation.csv")
print("3. sku_forecast_weekly_evaluation.csv")
print("4. sku_forecast_test_predictions.csv")

print("\nSKU-level demand forecasting completed.")
