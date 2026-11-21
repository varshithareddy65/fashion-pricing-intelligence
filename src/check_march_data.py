
import pandas as pd

df = pd.read_csv(
    "data/processed/product_store_daily_sales.csv",
    parse_dates=["Date"],
    dtype={"Product ID": str}
)

train = df[df["Date"] <= "2024-12-31"]
test = df[df["Date"] >= "2025-01-01"]

train_products = set(train["Product ID"].unique())

for month in ["2025-01", "2025-02", "2025-03"]:

    month_data = test[
        test["Date"].dt.to_period("M").astype(str) == month
    ]

    month_products = set(month_data["Product ID"].unique())

    previously_seen = month_products & train_products
    unseen = month_products - train_products

    print(f"\n{month}")
    print("Total products:", len(month_products))
    print("Seen in training:", len(previously_seen))
    print("Not seen in training:", len(unseen))
    print("Transaction records:", len(month_data))

print("\nCOMPLETED")
