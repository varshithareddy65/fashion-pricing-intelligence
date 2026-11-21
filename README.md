# Fashion Retail Pricing & Promotion Intelligence System

An end-to-end retail analytics and machine learning project designed to analyze fashion product demand, evaluate promotional campaigns, study product lifecycle patterns, and support demand planning through data-driven insights.

## Overview

Fashion retailers need to make decisions about product demand, pricing, promotions, and inventory planning. This project uses real-world fashion retail transaction data to build an analytical system that explores historical sales patterns, evaluates promotional performance, compares product demand cohorts, and experiments with demand forecasting models.

The project combines data engineering, exploratory data analysis, statistical modelling, machine learning, and interactive dashboard development.

## Key Features

### 1. Retail Sales Analytics
- Monthly sales and revenue analysis.
- Category and subcategory performance.
- Currency-wise revenue analysis.
- Product-level demand analysis.
- Return transaction analysis.

### 2. Pricing and Promotion Intelligence
- Discount distribution analysis.
- Campaign-to-transaction matching.
- Campaign revenue analysis.
- Before-and-after promotion performance comparisons.
- Promotion effectiveness diagnostics.

### 3. Product Demand Intelligence
- Product-level demand aggregation.
- Historical and newer product cohort comparison.
- First 14-day demand analysis.
- Reliable product ranking using minimum exposure and active-sales-day requirements.
- Subcategory-level demand comparisons.

### 4. Demand Forecasting
- Daily category and subcategory demand modelling.
- Weekly SKU-level demand forecasting experiments.
- HistGradientBoosting and LightGBM comparison.
- Forecast error analysis.
- Cold-start demand prediction experiments.
- Segment-level model evaluation.

### 5. Interactive Dashboard

Built using Streamlit and Plotly.

The dashboard contains six sections:

| Section | Purpose |
|---|---|
| Executive Overview | Business KPIs and retail performance |
| Demand Intelligence | Product and subcategory demand patterns |
| Forecast Evaluation | Model comparison and prediction analysis |
| Promotion Intelligence | Campaign and discount performance |
| Product Lifecycle | Historical and newer product cohort comparison |
| Data Explorer | Search, filter, inspect and download datasets |

## Dataset

The project uses a fashion retail transaction dataset containing:

| Dataset | Records |
|---|---:|
| Transactions | 6,416,827 |
| Products | 17,940 |
| Customers | 1,643,306 |
| Discounts | 181 |
| Stores | 35 |
| Employees | 404 |

The transaction data covers January 2023 to March 2025.

**Important data considerations**
- Transactions contain USD, CNY, EUR and GBP. Monetary metrics are evaluated separately by currency.
- Returns are represented by negative transaction values.
- Invoice totals are repeated across invoice lines and are not summed as line-level revenue.
- Inventory availability data is not available, so zero sales are not interpreted as stockouts.
- Product cost currency and per-unit semantics have not been independently validated; profitability claims are therefore excluded.

## Technology Stack

**Programming:** Python

**Data Analysis:** Pandas, NumPy

**Visualization:** Plotly, Streamlit

**Machine Learning:** Scikit-learn, LightGBM, HistGradientBoosting

**NLP:** TF-IDF

**Development:** VS Code, Git, GitHub

## Project Structure

```text
fashion-pricing-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data_inspection.py
│   ├── relationship_checks.py
│   ├── eda_pricing.py
│   ├── eda_profitability.py
│   ├── validate_product_cost.py
│   ├── validate_discount_campaigns.py
│   ├── match_campaign_transactions.py
│   ├── campaign_revenue_analysis.py
│   ├── promotion_effectiveness.py
│   ├── price_elasticity.py
│   ├── new_product_demand_analysis.py
│   ├── reliable_product_ranking.py
│   ├── cohort_fair_comparison.py
│   ├── demand_forecasting.py
│   ├── sku_demand_forecasting.py
│   ├── cold_start_demand_model.py
│   └── segment_demand_calibration.py
│
├── dashboard/
│   └── app.py
│
├── models/
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Analytical and Machine Learning Methodology

### 1. Data Validation and Preparation

- Checked primary key uniqueness and foreign key relationships.
- Validated transaction dates and sales calculations.
- Separated sales and return transactions.
- Examined product, discount, store and customer relationships.
- Preserved currency-level distinctions.

### 2. Product Demand Analysis

Aggregated sales at product and currency levels to understand demand patterns.

Additional analysis included:
- Product cohort comparisons.
- Minimum observation exposure filters.
- Active sales day analysis.
- First 14-day demand comparisons.

### 3. Demand Forecasting

Multiple approaches were evaluated, including:

- Historical median baselines.
- Category and currency-based baselines.
- Subcategory and currency-based baselines.
- HistGradientBoosting.
- LightGBM.

The SKU-level forecasting experiments also included weekly demand aggregation, intermittent demand analysis, and model benchmarking.

### 4. Cold-Start Demand Prediction

A separate experiment estimated first-14-day product demand using product attributes, category information, currency and TF-IDF representations of English product descriptions.

The experiment compared statistical baselines against a LightGBM model.

## Key Experimental Results

### Cold-Start Model Comparison

March 2025 evaluation on 2,984 products with complete first-14-day observation windows:

| Model | WAPE | R² |
|---|---:|---:|
| Subcategory + Currency Median | **30.79%** | **0.571** |
| Category + Currency Median | 32.56% | 0.492 |
| LightGBM | 38.71% | 0.262 |
| Category Median | 46.56% | — |
| Calibrated Strategy | 96.11% | — |

The subcategory-and-currency median baseline achieved the lowest WAPE in this evaluation.

This experiment demonstrates that a carefully selected statistical baseline can outperform a more complex machine learning model on a particular cold-start evaluation.

**Evaluation limitation:** The first observed sale date defines the beginning of the target window. Therefore, this