import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fashion Retail Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CUSTOM LIGHT THEME
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #F5F7FB !important;
    color: #111827 !important;
}
h1, h2, h3, h4 {
    color: #111827 !important;
    font-weight: 750 !important;
}
.stApp p, .stApp label, .stApp span {
    color: #1F2937;
}
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    padding: 18px !important;
    border-radius: 12px !important;
    border: 1px solid #D1D5DB !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.04);
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    font-size: 27px !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #374151 !important;
    -webkit-text-fill-color: #374151 !important;
    font-weight: 650 !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #1F2937 !important;
}
[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #D1D5DB !important;
}
[data-baseweb="select"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #FFFFFF !important;
}
[data-baseweb="menu"] * {
    color: #111827 !important;
}
input, textarea {
    color: #111827 !important;
}
[data-testid="stDataFrame"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
}
.stButton button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}
.stButton button p {
    color: #FFFFFF !important;
}
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text,
.js-plotly-plot .legendtext,
.js-plotly-plot .annotation-text {
    fill: #111827 !important;
    color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DATA FILES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

FILES = {
    "monthly_sales": "monthly_sales.csv",
    "category_performance": "category_performance.csv",
    "discount_performance": "discount_performance.csv",
    "returns_analysis": "returns_analysis.csv",
    "campaign_revenue": "campaign_revenue_analysis.csv",
    "promotion_effectiveness": "promotion_effectiveness.csv",
    "product_demand": "product_demand_metrics.csv",
    "cohort_summary": "cohort_demand_summary.csv",
    "subcategory_ranking": "subcategory_demand_ranking.csv",
    "reliable_rankings": "reliable_product_rankings.csv",
    "cold_start_predictions": "cold_start_product_predictions.csv",
    "calibrated_predictions": "calibrated_cold_start_predictions.csv",
    "model_comparison": "calibrated_demand_model_comparison.csv",
    "segment_evaluation": "calibrated_demand_segment_evaluation.csv",
    "first_14_day_comparison": "first_14_day_subcategory_comparison.csv",
    "product_lifecycle": "product_cohort_lifecycle.csv",
    "cohort_category": "product_cohort_category_summary.csv"
}


# =========================================================
# 4. LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    result = {}

    for key, filename in FILES.items():
        path = DATA_DIR / filename

        if path.exists():
            try:
                result[key] = pd.read_csv(
                    path,
                    low_memory=False
                )
            except Exception as error:
                result[key] = pd.DataFrame()
                st.warning(f"Error loading {filename}: {error}")
        else:
            result[key] = pd.DataFrame()

    return result


data = load_data()


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================

def get_df(key):
    return data.get(key, pd.DataFrame())


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def fmt(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def fmt_decimal(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def get_currency_options(df):
    if "Currency" not in df.columns:
        return []
    return sorted(
        df["Currency"].dropna().astype(str).unique()
    )


def currency_filter(df, currency):
    if currency and "Currency" in df.columns:
        return df[df["Currency"].astype(str) == currency].copy()
    return df.copy()


def revenue_column(df):
    return find_column(
        df,
        [
            "Net Revenue",
            "Net_Revenue",
            "Revenue",
            "Total Revenue",
            "Gross Revenue",
            "Gross_Revenue"
        ]
    )


def show_table(df, height=400):
    if df.empty:
        st.info("No records available.")
    else:
        st.dataframe(
            df,
            use_container_width=True,
            height=height,
            hide_index=True
        )


def missing_file(filename):
    st.info(
        f"{filename} was not found in data/processed. "
        "Run the corresponding analysis script first."
    )


def plot_style(fig, title="", height=400):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=19, color="#111827")
        ),
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial",
            size=13,
            color="#111827"
        ),
        margin=dict(l=45, r=25, t=65, b=65),
        legend=dict(font=dict(color="#111827")),
        xaxis=dict(
            tickfont=dict(color="#111827"),
            gridcolor="#E5E7EB"
        ),
        yaxis=dict(
            tickfont=dict(color="#111827"),
            gridcolor="#E5E7EB"
        )
    )
    return fig


# =========================================================
# 6. SIDEBAR
# =========================================================

st.sidebar.title("Fashion Intelligence")
st.sidebar.caption("Retail Pricing & Promotion Analytics")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Demand Intelligence",
        "Forecast Evaluation",
        "Promotion Intelligence",
        "Product Lifecycle",
        "Data Explorer"
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    "Fashion Retail Pricing & Promotion Intelligence System"
)
st.sidebar.caption("Dataset period: Jan 2023 – Mar 2025")


# =========================================================
# 7. EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    st.title("Fashion Retail Intelligence Dashboard")

    st.markdown(
        "### Pricing, demand, promotion and product lifecycle insights"
    )

    st.caption(
        "An analytical view of fashion retail transactions, "
        "campaign performance and demand modelling."
    )

    monthly = get_df("monthly_sales")
    category = get_df("category_performance")
    campaigns = get_df("campaign_revenue")
    demand = get_df("product_demand")
    comparison = get_df("model_comparison")

    st.divider()

    # Currency selector
    currency_source = monthly if not monthly.empty else category
    currencies = get_currency_options(currency_source)

    selected_currency = None

    if currencies:
        selected_currency = st.selectbox(
            "Select reporting currency",
            currencies,
            key="overview_currency"
        )

    monthly_view = currency_filter(monthly, selected_currency)
    category_view = currency_filter(category, selected_currency)
    campaign_view = currency_filter(campaigns, selected_currency)

    # KPI calculations
    rev_col = revenue_column(monthly_view)

    total_revenue = (
        pd.to_numeric(monthly_view[rev_col], errors="coerce").sum()
        if rev_col else None
    )

    product_col = find_column(
        demand,
        ["Product ID", "Product_ID"]
    )

    unique_products = (
        demand[product_col].nunique()
        if product_col and not demand.empty else None
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Monthly Records",
        fmt(len(monthly_view))
    )

    k2.metric(
        f"{selected_currency or 'Reported'} Revenue",
        fmt_decimal(total_revenue)
    )

    k3.metric(
        "Unique Products",
        fmt(unique_products)
    )

    k4.metric(
        "Campaign Analysis Rows",
        fmt(len(campaign_view))
    )

    # Business snapshot
    st.divider()
    st.subheader("Business Snapshot")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Demand coverage**")
        st.write(
            f"The demand dataset contains "
            f"**{len(demand):,} product-currency records**."
        )

    with col2:
        st.markdown("**Promotion coverage**")
        st.write(
            f"The campaign analysis contains "
            f"**{len(campaign_view):,} records** "
            f"for the selected currency."
        )

    # Monthly trend
    st.divider()
    st.subheader("Monthly Sales Trend")

    if not monthly_view.empty and rev_col:

        date_col = find_column(
            monthly_view,
            ["Month", "Year-Month", "Date", "Invoice Month"]
        )

        chart_data = monthly_view.copy()
        chart_data[rev_col] = pd.to_numeric(
            chart_data[rev_col],
            errors="coerce"
        )

        if date_col:
            parsed = pd.to_datetime(
                chart_data[date_col],
                errors="coerce"
            )

            if parsed.notna().any():
                chart_data["_date"] = parsed
                chart_data = chart_data.sort_values("_date")
                chart_data["_month"] = parsed.dt.strftime("%b %Y")
                x_col = "_month"
            else:
                chart_data = chart_data.sort_values(date_col)
                x_col = date_col

            fig = px.line(
                chart_data,
                x=x_col,
                y=rev_col,
                markers=True
            )

            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=7)
            )

            fig.update_yaxes(
                title_text=f"Revenue ({selected_currency or 'reported currency'})"
            )

            st.plotly_chart(
                plot_style(fig, "Monthly Revenue Trend", 390),
                use_container_width=True
            )
        else:
            st.info("Month column not identified.")
    else:
        missing_file("monthly_sales.csv")

    # Category performance
    st.divider()
    st.subheader("Category Performance")

    cat_col = find_column(
        category_view,
        ["Category", "Product Category"]
    )

    cat_rev = revenue_column(category_view)

    if not category_view.empty and cat_col and cat_rev:

        temp = category_view.copy()
        temp[cat_rev] = pd.to_numeric(
            temp[cat_rev],
            errors="coerce"
        )

        summary = (
            temp.groupby(cat_col, as_index=False)[cat_rev]
            .sum()
            .sort_values(cat_rev, ascending=False)
        )

        fig = px.bar(
            summary,
            x=cat_col,
            y=cat_rev,
            color=cat_col,
            text_auto=".2s"
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(
            plot_style(fig, "Revenue by Product Category", 380),
            use_container_width=True
        )

        show_table(summary)

    elif category.empty:
        missing_file("category_performance.csv")
    else:
        st.info("Category revenue columns not identified.")

    # Forecast snapshot
    st.divider()
    st.subheader("Demand Forecasting Snapshot")

    if not comparison.empty:

        model_col = find_column(
            comparison,
            ["Model", "Model_Name", "Strategy"]
        )

        wape_col = find_column(
            comparison,
            ["WAPE_Percent", "WAPE (%)", "WAPE"]
        )

        if model_col and wape_col:

            temp = comparison.copy()
            temp[wape_col] = pd.to_numeric(
                temp[wape_col],
                errors="coerce"
            )

            temp = temp.dropna(
                subset=[wape_col]
            ).sort_values(wape_col)

            if not temp.empty:
                best = temp.iloc[0]

                st.success(
                    f"Lowest recorded WAPE: **{best[model_col]}** "
                    f"({best[wape_col]:.2f}%)."
                )
        else:
            st.info("Model and WAPE columns not identified.")
    else:
        missing_file("calibrated_demand_model_comparison.csv")

    st.caption(
        "Revenue is displayed separately by currency. "
        "CNY, EUR, GBP and USD are never combined."
    )


# =========================================================
# 8. DEMAND INTELLIGENCE
# =========================================================

elif page == "Demand Intelligence":

    st.title("Demand Intelligence")
    st.markdown(
        "### Product demand patterns and subcategory rankings"
    )

    demand = get_df("product_demand")
    ranking = get_df("subcategory_ranking")
    reliable = get_df("reliable_rankings")
    cohort = get_df("cohort_summary")

    if not demand.empty:

        units_col = find_column(
            demand,
            [
                "Total_Units",
                "Total Units",
                "Total_Quantity",
                "Total Quantity",
                "Units",
                "Quantity"
            ]
        )

        product_col = find_column(
            demand,
            ["Product ID", "Product_ID"]
        )

        category_col = find_column(
            demand,
            ["Category"]
        )

        currency_col = find_column(
            demand,
            ["Currency"]
        )

        total_units = (
            pd.to_numeric(demand[units_col], errors="coerce").sum()
            if units_col else None
        )

        a, b, c = st.columns(3)

        a.metric("Product-Currency Records", fmt(len(demand)))

        b.metric(
            "Unique Products",
            fmt(demand[product_col].nunique())
            if product_col else "N/A"
        )

        c.metric("Total Units", fmt(total_units))

        st.divider()

        chart_data = demand.copy()

        if currency_col:
            currencies = get_currency_options(chart_data)

            if currencies:
                selected = st.selectbox(
                    "Select currency",
                    currencies,
                    key="demand_currency"
                )

                chart_data = chart_data[
                    chart_data[currency_col].astype(str) == selected
                ]

        if category_col and units_col:

            chart_data[units_col] = pd.to_numeric(
                chart_data[units_col],
                errors="coerce"
            )

            summary = (
                chart_data.groupby(category_col, as_index=False)[units_col]
                .sum()
                .sort_values(units_col, ascending=False)
            )

            fig = px.bar(
                summary,
                x=category_col,
                y=units_col,
                color=category_col,
                text_auto=".2s"
            )

            st.plotly_chart(
                plot_style(fig, "Units Sold by Category", 380),
                use_container_width=True
            )

        st.subheader("Product Demand Metrics")
        show_table(demand.head(500))

    else:
        missing_file("product_demand_metrics.csv")

    st.divider()
    st.subheader("Subcategory Demand Ranking")

    if not ranking.empty:
        show_table(ranking)
    else:
        missing_file("subcategory_demand_ranking.csv")

    st.divider()
    st.subheader("Reliable Product Rankings")

    if not reliable.empty:
        show_table(reliable.head(500))
    else:
        missing_file("reliable_product_rankings.csv")

    st.divider()
    st.subheader("Cohort Demand Summary")

    if not cohort.empty:
        show_table(cohort)
    else:
        missing_file("cohort_demand_summary.csv")


# =========================================================
# 9. FORECAST EVALUATION
# =========================================================

elif page == "Forecast Evaluation":

    st.title("Demand Forecast Evaluation")
    st.markdown(
        "### Comparing demand prediction approaches"
    )

    comparison = get_df("model_comparison")
    predictions = get_df("cold_start_predictions")
    segment = get_df("segment_evaluation")
    calibrated = get_df("calibrated_predictions")

    st.subheader("Model Comparison")

    if not comparison.empty:

        model_col = find_column(
            comparison,
            ["Model", "Model_Name", "Strategy"]
        )

        wape_col = find_column(
            comparison,
            ["WAPE_Percent", "WAPE (%)", "WAPE"]
        )

        temp = comparison.copy()

        if wape_col:
            temp[wape_col] = pd.to_numeric(
                temp[wape_col],
                errors="coerce"
            )

            temp = temp.sort_values(
                wape_col,
                ascending=True
            )

        show_table(temp)

        if model_col and wape_col:

            chart = temp.dropna(
                subset=[model_col, wape_col]
            )

            fig = px.bar(
                chart,
                x=wape_col,
                y=model_col,
                orientation="h",
                text=wape_col,
                color=wape_col,
                color_continuous_scale="Blues"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            fig.update_layout(
                coloraxis_showscale=False
            )

            st.plotly_chart(
                plot_style(fig, "Model Comparison - WAPE", 430),
                use_container_width=True
            )

    else:
        missing_file("calibrated_demand_model_comparison.csv")

    st.divider()
    st.subheader("Segment-Level Evaluation")

    if not segment.empty:

        st.caption(
            "Performance varies by category and currency. "
            "Review individual segments before deployment."
        )

        show_table(segment)

        wape = find_column(
            segment,
            ["WAPE_Percent", "WAPE (%)", "WAPE"]
        )

        cat = find_column(segment, ["Category"])
        curr = find_column(segment, ["Currency"])

        if wape and cat and curr:

            chart_data = segment.copy()
            chart_data[wape] = pd.to_numeric(
                chart_data[wape],
                errors="coerce"
            )

            chart_data["Segment"] = (
                chart_data[cat].astype(str)
                + " - "
                + chart_data[curr].astype(str)
            )

            fig = px.bar(
                chart_data,
                x="Segment",
                y=wape,
                color=cat
            )

            fig.update_layout(
                xaxis_tickangle=-45
            )

            st.plotly_chart(
                plot_style(fig, "WAPE by Category and Currency", 450),
                use_container_width=True
            )

    else:
        missing_file("calibrated_demand_segment_evaluation.csv")

    st.divider()
    st.subheader("Cold-Start Product Predictions")

    if not predictions.empty:

        st.caption(
            "The baseline uses subcategory and currency median demand. "
            "The first observed sale date defines the first-14-day target "
            "window. These are not pre-launch predictions."
        )

        show_table(predictions.head(500))

        actual_col = find_column(
            predictions,
            [
                "Actual_Units",
                "First_14_Day_Units",
                "Actual Units"
            ]
        )

        predicted_col = find_column(
            predictions,
            [
                "Subcategory_Currency_Baseline",
                "Predicted_Units",
                "Prediction"
            ]
        )

        if actual_col and predicted_col:

            chart_data = predictions.copy()

            chart_data[actual_col] = pd.to_numeric(
                chart_data[actual_col],
                errors="coerce"
            )

            chart_data[predicted_col] = pd.to_numeric(
                chart_data[predicted_col],
                errors="coerce"
            )

            chart_data = chart_data.dropna(
                subset=[actual_col, predicted_col]
            ).head(100)

            fig = px.scatter(
                chart_data,
                x=actual_col,
                y=predicted_col,
                opacity=0.6
            )

            st.plotly_chart(
                plot_style(
                    fig,
                    "Actual vs Predicted First-14-Day Units",
                    400
                ),
                use_container_width=True
            )

    else:
        missing_file("cold_start_product_predictions.csv")

    st.divider()
    st.subheader("Calibration Diagnostics")

    if not calibrated.empty:

        st.warning(
            "The calibrated strategy showed substantial overprediction "
            "in the March evaluation. It is retained for diagnostics, "
            "not as the recommended model."
        )

        show_table(calibrated.head(300))

    else:
        st.info("No calibrated prediction file available.")


# =========================================================
# 10. PROMOTION INTELLIGENCE
# =========================================================

elif page == "Promotion Intelligence":

    st.title("Promotion Intelligence")
    st.markdown(
        "### Campaign performance and discount analysis"
    )

    campaigns = get_df("campaign_revenue")
    promotion = get_df("promotion_effectiveness")
    discounts = get_df("discount_performance")

    st.subheader("Campaign Revenue Analysis")

    if not campaigns.empty:

        chart_data = campaigns.copy()
        currencies = get_currency_options(chart_data)

        if currencies:

            selected_currency = st.selectbox(
                "Select campaign currency",
                currencies,
                key="campaign_currency"
            )

            chart_data = chart_data[
                chart_data["Currency"].astype(str)
                == selected_currency
            ]

        rev_col = find_column(
            chart_data,
            [
                "Gross Revenue",
                "Gross_Revenue",
                "Revenue"
            ]
        )

        campaign_col = find_column(
            chart_data,
            [
                "Campaign Name",
                "Campaign",
                "Description",
                "Campaign ID"
            ]
        )

        if rev_col and campaign_col:

            chart_data[rev_col] = pd.to_numeric(
                chart_data[rev_col],
                errors="coerce"
            )

            top = (
                chart_data.groupby(
                    campaign_col,
                    as_index=False
                )[rev_col]
                .sum()
                .sort_values(rev_col, ascending=False)
                .head(10)
            )

            fig = px.bar(
                top,
                x=campaign_col,
                y=rev_col,
                color=campaign_col,
                text_auto=".2s"
            )

            st.plotly_chart(
                plot_style(
                    fig,
                    "Top Campaigns by Matched Gross Revenue",
                    450
                ),
                use_container_width=True
            )

        show_table(chart_data)

    else:
        missing_file("campaign_revenue_analysis.csv")

    st.divider()
    st.subheader("Promotion Effectiveness")

    if not promotion.empty:

        st.caption(
            "Before-and-after revenue changes are observational "
            "comparisons and should not be interpreted as causal "
            "promotion impact."
        )

        show_table(promotion)

    else:
        missing_file("promotion_effectiveness.csv")

    st.divider()
    st.subheader("Discount Performance")

    if not discounts.empty:
        show_table(discounts)
    else:
        missing_file("discount_performance.csv")


# =========================================================
# 11. PRODUCT LIFECYCLE
# =========================================================

elif page == "Product Lifecycle":

    st.title("Product Lifecycle Intelligence")
    st.markdown(
        "### Historical and March 2025 product cohort comparison"
    )

    lifecycle = get_df("product_lifecycle")
    cohort_category = get_df("cohort_category")
    first14 = get_df("first_14_day_comparison")

    st.subheader("Product Cohort Lifecycle")

    if not lifecycle.empty:
        show_table(lifecycle)
    else:
        missing_file("product_cohort_lifecycle.csv")

    st.divider()
    st.subheader("Cohort Category Summary")

    if not cohort_category.empty:
        show_table(cohort_category)
    else:
        missing_file("product_cohort_category_summary.csv")

    st.divider()
    st.subheader("First 14-Day Demand Comparison")

    if not first14.empty:

        st.caption(
            "Compares complete first-14-calendar-day observed-sales "
            "windows between historical and March cohorts."
        )

        show_table(first14)

        category_col = find_column(
            first14,
            ["Category", "Product Category"]
        )

        ratio_col = find_column(
            first14,
            [
                "March_to_Historical_Ratio",
                "March/Historical Ratio",
                "Demand_Ratio",
                "Units_Ratio"
            ]
        )

        if category_col and ratio_col:

            chart_data = first14.copy()

            chart_data[ratio_col] = pd.to_numeric(
                chart_data[ratio_col],
                errors="coerce"
            )

            fig = px.bar(
                chart_data,
                x=category_col,
                y=ratio_col,
                color=category_col,
                text_auto=".2f"
            )

            fig.add_hline(
                y=1,
                line_dash="dash",
                annotation_text="Equal demand"
            )

            st.plotly_chart(
                plot_style(
                    fig,
                    "March-to-Historical First-14-Day Demand Ratio",
                    400
                ),
                use_container_width=True
            )

    else:
        missing_file("first_14_day_subcategory_comparison.csv")

    st.warning(
        "The dataset shows a sharp product cohort transition in March 2025. "
        "This may reflect a catalogue refresh or dataset boundary; "
        "it does not independently establish product launch or discontinuation."
    )


# =========================================================
# 12. DATA EXPLORER
# =========================================================

elif page == "Data Explorer":

    st.title("Data Explorer")

    st.markdown(
        "### Explore, filter and download analytical datasets"
    )

    available = {
        key: df.copy()
        for key, df in data.items()
        if not df.empty
    }

    if not available:

        st.warning("No processed datasets found.")

    else:

        selected_dataset = st.selectbox(
            "Choose a dataset",
            list(available.keys())
        )

        df = available[selected_dataset].copy()

        st.subheader(
            selected_dataset.replace("_", " ").title()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Total Columns", f"{len(df.columns):,}")
        c3.metric(
            "Missing Values",
            f"{df.isna().sum().sum():,}"
        )

        st.divider()
        st.subheader("Search and Filter")

        search = st.text_input(
            "Search across dataset",
            placeholder="Enter a product, category, currency..."
        )

        filtered = df.copy()

        # Search across all columns
        if search:

            mask = filtered.astype(str).apply(
                lambda col: col.str.contains(
                    search,
                    case=False,
                    na=False,
                    regex=False
                )
            ).any(axis=1)

            filtered = filtered[mask]

        # Dynamic filters
        filter_columns = [
            col for col in [
                "Currency",
                "Category",
                "Sub Category",
                "Product_Cohort"
            ]
            if col in filtered.columns
        ]

        if filter_columns:

            st.markdown("**Available Filters**")

            filter_cols = st.columns(
                min(len(filter_columns), 3)
            )

            for i, col in enumerate(filter_columns):

                with filter_cols[i % len(filter_cols)]:

                    options = sorted(
                        filtered[col].dropna().unique().tolist(),
                        key=str
                    )

                    selected = st.multiselect(
                        col.replace("_", " "),
                        options,
                        key=f"filter_{selected_dataset}_{col}"
                    )

                    if selected:

                        filtered = filtered[
                            filtered[col].isin(selected)
                        ]

        st.divider()
        st.subheader("Choose Columns")

        selected_columns = st.multiselect(
            "Columns to display",
            df.columns.tolist(),
            default=df.columns.tolist()
        )

        if not selected_columns:

            st.info("Select at least one column.")

        else:

            display_df = filtered[selected_columns].copy()

            sort_col = st.selectbox(
                "Sort by",
                ["None"] + selected_columns
            )

            if sort_col != "None":

                try:
                    display_df = display_df.sort_values(
                        sort_col,
                        ascending=True,
                        na_position="last"
                    )

                except (TypeError, ValueError):
                    st.warning(
                        f"Could not sort by {sort_col}."
                    )

            st.metric(
                "Filtered Records",
                f"{len(display_df):,}"
            )

            st.subheader("Filtered Dataset Preview")

            st.dataframe(
                display_df.head(1000),
                use_container_width=True,
                height=500,
                hide_index=True
            )

            # Download filtered data
            csv = display_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Filtered CSV",
                data=csv,
                file_name=f"{selected_dataset}_filtered.csv",
                mime="text/csv"
            )

        st.divider()
        st.subheader("Column Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isna().sum().values,
            "Unique Values": df.nunique(dropna=True).values
        })

        st.dataframe(
            info,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "Fashion Retail Pricing & Promotion Intelligence System | "
    "Historical retail transaction analysis"
)