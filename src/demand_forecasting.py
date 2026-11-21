
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRANSACTIONS_FILE = BASE_DIR / "data/raw/transactions.csv"
PRODUCTS_FILE = BASE_DIR / "data/raw/products.csv"

OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_HORIZON = 7

TRAIN_CUTOFF = pd.Timestamp("2024-12-24")
TEST_START = pd.Timestamp("2025-01-01")

RANDOM_STATE = 42

GROUP_COLUMNS = [
    "Category",
    "Sub Category",
    "Currency",
]


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 65)
print("7-DAY DEMAND FORECASTING")
print("=" * 65)

print("\nLoading transaction and product data...")

transactions = pd.read_csv(TRANSACTIONS_FILE)
products = pd.read_csv(PRODUCTS_FILE)

print(f"Transactions: {len(transactions):,}")
print(f"Products: {len(products):,}")


# ============================================================
# 3. VALIDATE SCHEMAS
# ============================================================

required_transaction_columns = [
    "Product ID",
    "Date",
    "Quantity",
    "Discount",
    "Currency",
    "Transaction Type",
]

required_product_columns = [
    "Product ID",
    "Category",
    "Sub Category",
]

for name, frame, required in [
    ("transactions", transactions, required_transaction_columns),
    ("products", products, required_product_columns),
]:
    missing = [
        col for col in required if col not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {name}: {missing}"
        )

if products["Product ID"].duplicated().any():
    raise ValueError(
        "Duplicate Product IDs found in products.csv."
    )


# ============================================================
# 4. CLEAN TRANSACTIONS
# ============================================================

print("\nCleaning transactions...")

transactions["Date"] = pd.to_datetime(
    transactions["Date"],
    errors="coerce",
).dt.normalize()

transactions["Quantity"] = pd.to_numeric(
    transactions["Quantity"],
    errors="coerce",
)

transactions["Discount"] = pd.to_numeric(
    transactions["Discount"],
    errors="coerce",
)

transactions["Product ID"] = pd.to_numeric(
    transactions["Product ID"],
    errors="coerce",
)

products["Product ID"] = pd.to_numeric(
    products["Product ID"],
    errors="coerce",
)

transactions = transactions.dropna(
    subset=[
        "Product ID",
        "Date",
        "Quantity",
        "Currency",
        "Transaction Type",
    ]
).copy()

transactions["Transaction Type"] = (
    transactions["Transaction Type"]
    .astype(str)
    .str.strip()
    .str.casefold()
)

# Use sale transactions only.
sales = transactions[
    transactions["Transaction Type"] == "sale"
].copy()

# Positive quantities represent demand.
sales = sales[
    sales["Quantity"] > 0
].copy()

sales["Discount"] = (
    sales["Discount"]
    .fillna(0)
    .clip(0, 1)
)

print(f"Valid sale transactions: {len(sales):,}")

print(
    "Normalized transaction dates:",
    sales["Date"].min(),
    "to",
    sales["Date"].max(),
)


# ============================================================
# 5. JOIN PRODUCT INFORMATION
# ============================================================

print("\nJoining product categories...")

sales = sales.merge(
    products[
        ["Product ID", "Category", "Sub Category"]
    ],
    on="Product ID",
    how="left",
    validate="many_to_one",
)

unmatched = sales["Category"].isna().sum()

print(f"Unmatched product transactions: {unmatched:,}")

if unmatched > 0:
    raise ValueError(
        "Some sale transactions could not be mapped to products."
    )


# ============================================================
# 6. AGGREGATE DAILY DEMAND
# ============================================================

print("\nAggregating daily demand...")

sales["Discounted_Units"] = np.where(
    sales["Discount"] > 0,
    sales["Quantity"],
    0,
)

sales["Discount_Weighted_Value"] = (
    sales["Quantity"] * sales["Discount"]
)

daily = (
    sales.groupby(
        ["Date"] + GROUP_COLUMNS,
        as_index=False,
    )
    .agg(
        Daily_Units=("Quantity", "sum"),
        Discounted_Units=("Discounted_Units", "sum"),
        Discount_Weighted_Value=("Discount_Weighted_Value", "sum"),
    )
)

daily["Discounted_Unit_Share"] = (
    daily["Discounted_Units"]
    / daily["Daily_Units"]
)

daily["Average_Discount"] = (
    daily["Discount_Weighted_Value"]
    / daily["Daily_Units"]
)

daily = daily.drop(
    columns=[
        "Discounted_Units",
        "Discount_Weighted_Value",
    ]
)

print(f"Daily aggregated records: {len(daily):,}")

global_start = daily["Date"].min()
global_end = daily["Date"].max()

print(f"Data start: {global_start.date()}")
print(f"Data end: {global_end.date()}")


# ============================================================
# 7. CREATE COMPLETE DAILY CALENDAR PANEL
# ============================================================

print("\nCreating complete daily calendar panel...")

panel_parts = []

for group_values, group in daily.groupby(
    GROUP_COLUMNS,
    sort=False,
):

    group = (
        group.sort_values("Date")
        .set_index("Date")
    )

    dates = pd.date_range(
        start=group.index.min(),
        end=global_end,
        freq="D",
    )

    group = group.reindex(dates)
    group.index.name = "Date"

    for column, value in zip(
        GROUP_COLUMNS,
        group_values,
    ):
        group[column] = value

    # No recorded sales are represented as zero.
    # This does not establish that stock was available.
    group["Daily_Units"] = (
        group["Daily_Units"].fillna(0)
    )

    group["Discounted_Unit_Share"] = (
        group["Discounted_Unit_Share"].fillna(0)
    )

    group["Average_Discount"] = (
        group["Average_Discount"].fillna(0)
    )

    panel_parts.append(
        group.reset_index()
    )

panel = pd.concat(
    panel_parts,
    ignore_index=True,
)

panel = (
    panel.sort_values(
        GROUP_COLUMNS + ["Date"]
    )
    .reset_index(drop=True)
)

print(f"Calendar panel records: {len(panel):,}")


# ============================================================
# 8. CALENDAR FEATURES
# ============================================================

print("\nCreating calendar features...")

panel["Day_Of_Week"] = (
    panel["Date"].dt.day_of_week
)

panel["Month"] = (
    panel["Date"].dt.month
)

panel["Week_Of_Year"] = (
    panel["Date"].dt.isocalendar().week.astype(int)
)

panel["Is_Weekend"] = (
    panel["Day_Of_Week"] >= 5
).astype(int)


# ============================================================
# 9. CREATE LAG FEATURES
# ============================================================

print("\nCreating demand lag features...")

for lag in [1, 7, 14, 28]:

    panel[f"Lag_{lag}"] = (
        panel.groupby(
            GROUP_COLUMNS,
            sort=False,
        )["Daily_Units"]
        .shift(lag)
    )


# ============================================================
# 10. CREATE ROLLING FEATURES
# ============================================================

print("\nCreating rolling demand features...")

for window in [7, 14, 28]:

    panel[f"Rolling_Mean_{window}"] = (
        panel.groupby(
            GROUP_COLUMNS,
            sort=False,
        )["Daily_Units"]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    window,
                    min_periods=window,
                )
                .mean()
            )
        )
    )

panel["Rolling_Std_7"] = (
    panel.groupby(
        GROUP_COLUMNS,
        sort=False,
    )["Daily_Units"]
    .transform(
        lambda series: (
            series.shift(1)
            .rolling(
                7,
                min_periods=7,
            )
            .std()
        )
    )
)


# ============================================================
# 11. PROMOTION FEATURES
# ============================================================

print("\nCreating historical promotion features...")

for window in [7, 28]:

    panel[f"Discount_Share_Mean_{window}"] = (
        panel.groupby(
            GROUP_COLUMNS,
            sort=False,
        )["Discounted_Unit_Share"]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    window,
                    min_periods=window,
                )
                .mean()
            )
        )
    )

    panel[f"Average_Discount_Mean_{window}"] = (
        panel.groupby(
            GROUP_COLUMNS,
            sort=False,
        )["Average_Discount"]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    window,
                    min_periods=window,
                )
                .mean()
            )
        )
    )


# ============================================================
# 12. BASELINE FORECAST
# ============================================================

panel["Baseline_7_Day_Forecast"] = (
    panel["Rolling_Mean_28"]
    * FORECAST_HORIZON
)


# ============================================================
# 13. CREATE 7-DAY TARGET
# ============================================================

print("\nCreating 7-day forecast target...")

panel["Target_7_Day_Units"] = 0.0

for day in range(1, FORECAST_HORIZON + 1):

    future_units = (
        panel.groupby(
            GROUP_COLUMNS,
            sort=False,
        )["Daily_Units"]
        .shift(-day)
    )

    panel["Target_7_Day_Units"] += (
        future_units.fillna(0)
    )

# Require a complete future 7-day target window.
panel["Target_Window_End"] = (
    panel["Date"]
    + pd.Timedelta(days=FORECAST_HORIZON)
)

panel["Target_Complete"] = (
    panel["Target_Window_End"] <= global_end
)

panel = panel[
    panel["Target_Complete"]
].copy()


# ============================================================
# 14. MODEL FEATURES
# ============================================================

categorical_features = [
    "Category",
    "Sub Category",
    "Currency",
]

numeric_features = [
    "Day_Of_Week",
    "Month",
    "Week_Of_Year",
    "Is_Weekend",

    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_28",

    "Rolling_Mean_7",
    "Rolling_Mean_14",
    "Rolling_Mean_28",
    "Rolling_Std_7",

    "Discount_Share_Mean_7",
    "Discount_Share_Mean_28",
    "Average_Discount_Mean_7",
    "Average_Discount_Mean_28",
]

feature_columns = (
    categorical_features + numeric_features
)

model_data = panel.dropna(
    subset=numeric_features + ["Target_7_Day_Units"]
).copy()

print(f"Usable model records: {len(model_data):,}")


# ============================================================
# 15. TEMPORAL TRAIN / TEST SPLIT
# ============================================================

print("\nPerforming time-based train/test split...")

train_data = model_data[
    model_data["Date"] <= TRAIN_CUTOFF
].copy()

test_data = model_data[
    model_data["Date"] >= TEST_START
].copy()

if train_data.empty or test_data.empty:
    raise ValueError(
        "Training or test data is empty."
    )

# Ensure training targets do not overlap the test period.
if (
    train_data["Date"].max()
    + pd.Timedelta(days=FORECAST_HORIZON)
    > TEST_START
):
    raise ValueError(
        "Training target window overlaps the test period."
    )

print(
    f"Training origins: {train_data['Date'].min().date()} "
    f"to {train_data['Date'].max().date()}"
)

print(
    f"Testing origins: {test_data['Date'].min().date()} "
    f"to {test_data['Date'].max().date()}"
)

print(f"Training records: {len(train_data):,}")
print(f"Testing records: {len(test_data):,}")


# ============================================================
# 16. BUILD MODEL PIPELINE
# ============================================================

print("\nBuilding HistGradientBoosting model...")

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
        (
            "numeric",
            numeric_transformer,
            numeric_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.06,
                max_leaf_nodes=20,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# ============================================================
# 17. TRAIN MODEL
# ============================================================

print("\nTraining model...")

X_train = train_data[feature_columns]
y_train = train_data["Target_7_Day_Units"]

X_test = test_data[feature_columns]
y_test = test_data["Target_7_Day_Units"]

model.fit(
    X_train,
    y_train,
)

print("Model training completed.")


# ============================================================
# 18. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

model_predictions = model.predict(X_test)

model_predictions = np.maximum(
    model_predictions,
    0,
)

baseline_predictions = np.maximum(
    test_data["Baseline_7_Day_Forecast"].to_numpy(),
    0,
)

actual_values = y_test.to_numpy()


# ============================================================
# 19. EVALUATION FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    total_actual = np.sum(
        np.abs(actual)
    )

    if total_actual > 0:

        wape = (
            np.sum(
                np.abs(actual - predicted)
            )
            / total_actual
        ) * 100

    else:
        wape = np.nan

    if len(actual) > 1 and np.var(actual) > 0:

        r2 = r2_score(
            actual,
            predicted,
        )

    else:
        r2 = np.nan

    zero_actual_percent = (
        np.mean(actual == 0) * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_Percent": wape,
        "R2": r2,
        "Zero_Actual_Percent": zero_actual_percent,
    }


# ============================================================
# 20. OVERALL EVALUATION
# ============================================================

print("\nEvaluating model and baseline...")

model_metrics = calculate_metrics(
    actual_values,
    model_predictions,
)

baseline_metrics = calculate_metrics(
    actual_values,
    baseline_predictions,
)

overall_metrics = pd.DataFrame(
    [
        {
            "Model": "HistGradientBoosting",
            **model_metrics,
        },
        {
            "Model": "28-Day Average Baseline",
            **baseline_metrics,
        },
    ]
)

print("\nOverall evaluation:")
print(
    overall_metrics.to_string(index=False)
)


# ============================================================
# 21. NON-OVERLAPPING WEEKLY EVALUATION
# ============================================================

print("\nEvaluating non-overlapping weekly forecasts...")

# Use forecast origins seven days apart.
# This prevents the same target dates from appearing
# in multiple evaluation windows.

weekly_dates = pd.date_range(
    start=TEST_START,
    end=test_data["Date"].max(),
    freq="7D",
)

weekly_test = test_data[
    test_data["Date"].isin(weekly_dates)
].copy()

if weekly_test.empty:
    raise ValueError(
        "No non-overlapping weekly test observations found."
    )

weekly_indices = weekly_test.index

weekly_positions = test_data.index.get_indexer(
    weekly_indices
)

weekly_model_predictions = model_predictions[
    weekly_positions
]

weekly_baseline_predictions = baseline_predictions[
    weekly_positions
]

weekly_actual = weekly_test[
    "Target_7_Day_Units"
].to_numpy()

weekly_metrics = pd.DataFrame(
    [
        {
            "Model": "HistGradientBoosting",
            **calculate_metrics(
                weekly_actual,
                weekly_model_predictions,
            ),
        },
        {
            "Model": "28-Day Average Baseline",
            **calculate_metrics(
                weekly_actual,
                weekly_baseline_predictions,
            ),
        },
    ]
)

print("\nNon-overlapping weekly evaluation:")
print(
    weekly_metrics.to_string(index=False)
)


# ============================================================
# 22. CATEGORY-CURRENCY EVALUATION
# ============================================================

print("\nEvaluating category and currency segments...")

evaluation = test_data[
    GROUP_COLUMNS
    + [
        "Date",
        "Target_7_Day_Units",
    ]
].copy()

evaluation["Model_Prediction"] = (
    model_predictions
)

evaluation["Baseline_Prediction"] = (
    baseline_predictions
)

segment_results = []

for group_values, group in evaluation.groupby(
    ["Category", "Currency"]
):

    actual = group[
        "Target_7_Day_Units"
    ].to_numpy()

    model_result = calculate_metrics(
        actual,
        group["Model_Prediction"].to_numpy(),
    )

    baseline_result = calculate_metrics(
        actual,
        group["Baseline_Prediction"].to_numpy(),
    )

    segment_results.append(
        {
            "Category": group_values[0],
            "Currency": group_values[1],
            "Test_Records": len(group),

            "Model_MAE": model_result["MAE"],
            "Model_RMSE": model_result["RMSE"],
            "Model_WAPE_Percent": model_result["WAPE_Percent"],
            "Model_R2": model_result["R2"],

            "Baseline_MAE": baseline_result["MAE"],
            "Baseline_RMSE": baseline_result["RMSE"],
            "Baseline_WAPE_Percent": baseline_result["WAPE_Percent"],

            "Zero_Actual_Percent": model_result[
                "Zero_Actual_Percent"
            ],
        }
    )

segment_metrics = pd.DataFrame(
    segment_results
)


# ============================================================
# 23. PREDICTION DIAGNOSTICS
# ============================================================

print("\nGenerating prediction diagnostics...")

predictions_output = test_data[
    GROUP_COLUMNS
    + [
        "Date",
        "Target_7_Day_Units",
        "Baseline_7_Day_Forecast",
    ]
].copy()

predictions_output["Actual_7_Day_Units"] = (
    actual_values
)

predictions_output["Model_Predicted_7_Day_Units"] = (
    model_predictions
)

predictions_output["Baseline_Predicted_7_Day_Units"] = (
    baseline_predictions
)

predictions_output["Model_Absolute_Error"] = (
    np.abs(
        actual_values - model_predictions
    )
)

predictions_output["Baseline_Absolute_Error"] = (
    np.abs(
        actual_values - baseline_predictions
    )
)

predictions_output["Model_Error"] = (
    model_predictions - actual_values
)

predictions_output["Baseline_Error"] = (
    baseline_predictions - actual_values
)

predictions_output["Actual_Is_Zero"] = (
    actual_values == 0
).astype(int)

predictions_output["Model_Overpredicts"] = (
    model_predictions > actual_values
).astype(int)


# ============================================================
# 24. SAVE RESULTS
# ============================================================

print("\nSaving output files...")

overall_file = (
    OUTPUT_DIR / "demand_model_evaluation.csv"
)

segment_file = (
    OUTPUT_DIR / "demand_model_segment_evaluation.csv"
)

predictions_file = (
    OUTPUT_DIR / "demand_model_test_predictions.csv"
)

weekly_file = (
    OUTPUT_DIR / "demand_model_weekly_evaluation.csv"
)

weekly_predictions_file = (
    OUTPUT_DIR / "demand_model_weekly_predictions.csv"
)

overall_metrics.to_csv(
    overall_file,
    index=False,
)

segment_metrics.to_csv(
    segment_file,
    index=False,
)

predictions_output.to_csv(
    predictions_file,
    index=False,
)

weekly_metrics.to_csv(
    weekly_file,
    index=False,
)

weekly_predictions = weekly_test[
    GROUP_COLUMNS
    + [
        "Date",
        "Target_7_Day_Units",
    ]
].copy()

weekly_predictions["Model_Prediction"] = (
    weekly_model_predictions
)

weekly_predictions["Baseline_Prediction"] = (
    weekly_baseline_predictions
)

weekly_predictions.to_csv(
    weekly_predictions_file,
    index=False,
)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("DEMAND FORECASTING RESULTS")
print("=" * 65)

print("\nOverall evaluation:")
print(
    overall_metrics.to_string(index=False)
)

print("\nNon-overlapping weekly evaluation:")
print(
    weekly_metrics.to_string(index=False)
)

print("\nCategory-currency evaluation:")
print(
    segment_metrics.to_string(index=False)
)

print("\nPrediction diagnostics:")

print(
    "Zero actual demand:",
    f"{np.mean(actual_values == 0) * 100:.2f}%"
)

print(
    "Model overprediction rate:",
    f"{np.mean(model_predictions > actual_values) * 100:.2f}%"
)

print(
    "Mean actual demand:",
    f"{np.mean(actual_values):.4f}"
)

print(
    "Mean model prediction:",
    f"{np.mean(model_predictions):.4f}"
)

print(
    "Mean baseline prediction:",
    f"{np.mean(baseline_predictions):.4f}"
)

print("\nSaved files:")
print(overall_file)
print(segment_file)
print(predictions_file)
print(weekly_file)
print(weekly_predictions_file)

print("\n7-day demand forecasting completed successfully.")
