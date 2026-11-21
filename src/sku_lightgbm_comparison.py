
import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

WEEKLY_FILE = PROCESSED / "sku_weekly_demand.csv"
ELIGIBILITY_FILE = PROCESSED / "sku_training_eligibility.csv"
HGB_FILE = PROCESSED / "sku_forecast_test_predictions.csv"
SEGMENT_FILE = PROCESSED / "sku_corrected_demand_segments.csv"
PRODUCT_FILE = RAW / "products.csv"

TEST_START = pd.Timestamp("2025-01-06")
MIN_ACTIVE_WEEKS = 8

PROCESSED.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD INPUT DATA
# ============================================================

print("\nLoading weekly SKU data...")

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
    parse_dates=["First_History_Week"],
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

old_predictions = pd.read_csv(
    HGB_FILE,
    parse_dates=["Week_Start"],
    dtype={
        "SKU": "string",
        "Product ID": "string",
        "Currency": "string"
    }
)

segments = pd.read_csv(
    SEGMENT_FILE,
    usecols=["SKU", "Currency", "Demand_Segment"],
    dtype={
        "SKU": "string",
        "Currency": "string"
    }
).drop_duplicates(["SKU", "Currency"])

print("Weekly records:", f"{len(weekly):,}")
print("Eligibility records:", f"{len(eligibility):,}")
print("Product records:", f"{len(products):,}")
print("Existing HGB predictions:", f"{len(old_predictions):,}")

# ============================================================
# 2. SELECT HISTORICALLY ELIGIBLE SERIES
# ============================================================

eligible = eligibility[
    eligibility["Pre_Test_Active_Weeks"] >= MIN_ACTIVE_WEEKS
].copy()

eligible = eligible[
    ["SKU", "Product ID", "Currency", "First_History_Week"]
].drop_duplicates(["SKU", "Currency"])

eligible = eligible.reset_index(drop=True)

print(
    "\nEligible SKU-currency series:",
    f"{len(eligible):,}"
)

# ============================================================
# 3. BUILD COMPLETE WEEKLY PANEL
# ============================================================

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
    "First_History_Week": (
        eligible["First_History_Week"].to_numpy()[series_indices]
    ),
    "Week_Start": (
        eligible["First_History_Week"].to_numpy()[series_indices]
        + pd.to_timedelta(week_offsets * 7, unit="D")
    )
})

panel = panel.merge(
    products,
    on="Product ID",
    how="left"
)

sales = weekly.merge(
    eligible[["SKU", "Currency"]],
    on=["SKU", "Currency"],
    how="inner"
)

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
    panel["Weekly_Units"]
    .fillna(0)
    .astype("float32")
)

panel = panel.sort_values(
    ["SKU", "Currency", "Week_Start"]
).reset_index(drop=True)

print("Complete panel records:", f"{len(panel):,}")
print(
    "Zero-sales records:",
    f"{(panel['Weekly_Units'] == 0).sum():,}"
)

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\nEngineering forecasting features...")

keys = ["SKU", "Currency"]
grouped = panel.groupby(keys, sort=False)

# Lag features
for lag in [1, 2, 4, 8]:
    panel[f"Lag_{lag}"] = (
        grouped["Weekly_Units"].shift(lag)
    )

# Rolling averages
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

# Historical average
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

# Historical active weeks
previous_units = grouped["Weekly_Units"].shift(1)

panel["Historical_Active_Weeks"] = (
    previous_units.fillna(0).gt(0).astype("int8")
    .groupby([panel["SKU"], panel["Currency"]])
    .cumsum()
)

# Product series age
panel["Series_Age_Weeks"] = (
    (panel["Week_Start"] - panel["First_History_Week"])
    .dt.days // 7
)

# Calendar features
panel["Month"] = panel["Week_Start"].dt.month.astype(str)

panel["Week_Of_Year"] = (
    panel["Week_Start"].dt.isocalendar().week.astype(int)
)

panel = panel.dropna(
    subset=["Lag_1"]
).copy()

print("Feature records:", f"{len(panel):,}")

# Release feature-engineering intermediates.
del grouped, rolling_4, rolling_8, expanding_mean, previous_units
gc.collect()

# ============================================================
# 5. TEMPORAL TRAIN / TEST SPLIT
# ============================================================

train = panel[
    panel["Week_Start"] < TEST_START
].copy()

test = panel[
    panel["Week_Start"] >= TEST_START
].copy()

print("\n========== DATA SPLIT ==========")
print("Training rows:", f"{len(train):,}")
print("Testing rows:", f"{len(test):,}")

# ============================================================
# 6. CREATE FEATURE MATRICES
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

for col in categorical_features:
    train[col] = train[col].fillna("Unknown")
    test[col] = test[col].fillna("Unknown")

print("\nEncoding categorical variables...")

X_train_df = pd.get_dummies(
    train[numeric_features + categorical_features],
    columns=categorical_features,
    dtype=np.uint8
)

X_test_df = pd.get_dummies(
    test[numeric_features + categorical_features],
    columns=categorical_features,
    dtype=np.uint8
)

X_test_df = X_test_df.reindex(
    columns=X_train_df.columns,
    fill_value=0
)

feature_names = X_train_df.columns.tolist()

# Convert to compact numerical arrays.
X_train = X_train_df.to_numpy(
    dtype=np.float32,
    copy=True
)

X_test = X_test_df.to_numpy(
    dtype=np.float32,
    copy=True
)

y_train = train["Weekly_Units"].to_numpy(
    dtype=np.float32
)

print("Feature count:", len(feature_names))
print("Training matrix:", X_train.shape)
print("Testing matrix:", X_test.shape)

# Preserve metadata and baseline predictions.
test_meta = test[
    [
        "SKU",
        "Product ID",
        "Currency",
        "Week_Start",
        "Weekly_Units"
    ]
].copy()

test_meta["Previous_Week_Prediction"] = (
    test["Lag_1"].to_numpy()
)

test_meta["Previous_4_Week_Average"] = (
    test["Rolling_4_Week_Mean"]
    .fillna(0)
    .to_numpy()
)

# Free pandas DataFrames, but KEEP X_train and y_train.
del panel, train, test
del X_train_df, X_test_df
del weekly, sales, eligibility, products, eligible
gc.collect()

# ============================================================
# 7. TRAIN LIGHTGBM
# ============================================================

print("\nTraining LightGBM...")

model = lgb.LGBMRegressor(
    objective="poisson",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    max_bin=63,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

model.fit(
    X_train,
    y_train
)

print("LightGBM training completed.")

# X_train and y_train are no longer required after fitting.
del X_train, y_train
gc.collect()

# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating LightGBM predictions...")

lightgbm_predictions = np.maximum(
    model.predict(X_test),
    0
)

test_meta["LightGBM_Prediction"] = (
    lightgbm_predictions
)

del X_test, lightgbm_predictions
gc.collect()

# ============================================================
# 9. JOIN EXISTING HGB PREDICTIONS
# ============================================================

print("\nJoining existing model predictions...")

test_meta = test_meta.merge(
    old_predictions[
        [
            "SKU",
            "Currency",
            "Week_Start",
            "Predicted_Units"
        ]
    ],
    on=["SKU", "Currency", "Week_Start"],
    how="left",
    validate="one_to_one"
)

if test_meta["Predicted_Units"].isna().any():
    raise ValueError(
        "Some existing HistGradientBoosting predictions are missing."
    )

test_meta = test_meta.merge(
    segments,
    on=["SKU", "Currency"],
    how="left",
    validate="many_to_one"
)

if test_meta["Demand_Segment"].isna().any():
    raise ValueError(
        "Some SKU series have no demand segment."
    )

# ============================================================
# 10. METRIC FUNCTION
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
# 11. SELECT VALID TEST WEEKS
# ============================================================

weekly_actual = (
    test_meta.groupby("Week_Start")["Weekly_Units"]
    .sum()
)

valid_weeks = weekly_actual[
    weekly_actual > 0
].index

valid = test_meta[
    test_meta["Week_Start"].isin(valid_weeks)
].copy()

print("\nValid evaluation weeks:", len(valid_weeks))
print("Evaluation records:", f"{len(valid):,}")

model_columns = {
    "LightGBM": "LightGBM_Prediction",
    "HistGradientBoosting": "Predicted_Units",
    "Previous_Week_Baseline": "Previous_Week_Prediction",
    "Previous_4_Week_Average": "Previous_4_Week_Average"
}

# ============================================================
# 12. OVERALL MODEL COMPARISON
# ============================================================

overall_results = []

for model_name, prediction_column in model_columns.items():

    result = calculate_metrics(
        valid["Weekly_Units"],
        valid[prediction_column]
    )

    result["Model"] = model_name
    overall_results.append(result)

overall = pd.DataFrame(overall_results)

overall.to_csv(
    PROCESSED / "sku_lightgbm_overall_comparison.csv",
    index=False
)

print("\n========== OVERALL MODEL COMPARISON ==========")
print(overall.to_string(index=False))

# ============================================================
# 13. DEMAND SEGMENT COMPARISON
# ============================================================

segment_results = []

for segment, group in valid.groupby(
    "Demand_Segment",
    sort=True
):

    for model_name, prediction_column in model_columns.items():

        result = calculate_metrics(
            group["Weekly_Units"],
            group[prediction_column]
        )

        result["Demand_Segment"] = segment
        result["Model"] = model_name

        segment_results.append(result)

segment_evaluation = pd.DataFrame(
    segment_results
)

segment_evaluation.to_csv(
    PROCESSED / "sku_lightgbm_segment_comparison.csv",
    index=False
)

print("\n========== SEGMENT COMPARISON ==========")
print(segment_evaluation.to_string(index=False))

# ============================================================
# 14. CURRENCY COMPARISON
# ============================================================

currency_results = []

for currency, group in valid.groupby(
    "Currency",
    sort=True
):

    for model_name, prediction_column in model_columns.items():

        result = calculate_metrics(
            group["Weekly_Units"],
            group[prediction_column]
        )

        result["Currency"] = currency
        result["Model"] = model_name

        currency_results.append(result)

currency_evaluation = pd.DataFrame(
    currency_results
)

currency_evaluation.to_csv(
    PROCESSED / "sku_lightgbm_currency_comparison.csv",
    index=False
)

print("\n========== CURRENCY COMPARISON ==========")
print(currency_evaluation.to_string(index=False))

# ============================================================
# 15. WEEKLY LIGHTGBM PERFORMANCE
# ============================================================

weekly_results = []

for week, group in valid.groupby(
    "Week_Start",
    sort=True
):

    result = calculate_metrics(
        group["Weekly_Units"],
        group["LightGBM_Prediction"]
    )

    result["Week_Start"] = week
    weekly_results.append(result)

weekly_evaluation = pd.DataFrame(
    weekly_results
)

weekly_evaluation.to_csv(
    PROCESSED / "sku_lightgbm_weekly_evaluation.csv",
    index=False
)

print("\n========== WEEKLY LIGHTGBM PERFORMANCE ==========")
print(weekly_evaluation.to_string(index=False))

# ============================================================
# 16. FEATURE IMPORTANCE
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
    PROCESSED / "sku_lightgbm_feature_importance.csv",
    index=False
)

print("\n========== TOP 20 FEATURE IMPORTANCE ==========")
print(
    importance.head(20).to_string(index=False)
)

# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

test_meta.to_csv(
    PROCESSED / "sku_lightgbm_test_predictions.csv",
    index=False
)

print("\nSaved output files:")
print("1. sku_lightgbm_overall_comparison.csv")
print("2. sku_lightgbm_segment_comparison.csv")
print("3. sku_lightgbm_currency_comparison.csv")
print("4. sku_lightgbm_weekly_evaluation.csv")
print("5. sku_lightgbm_feature_importance.csv")
print("6. sku_lightgbm_test_predictions.csv")

print("\nLightGBM comparison completed.")
