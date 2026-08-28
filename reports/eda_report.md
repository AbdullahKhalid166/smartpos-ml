# Exploratory Data Analysis Report

## Dataset Overview

The project uses the Online Retail transaction file at `data/raw/rawretaildata.csv`. It contains **1,048,575 rows** and these columns:

`Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, and `Country`.

The data is transaction-level retail history. An invoice can contain multiple products, so transactions must be aggregated before forecasting or other time-dependent modeling.

## Main Data Findings

- `Description` and `Customer ID` contain missing values.
- `InvoiceDate` contains both date and time and must be parsed before time analysis.
- Invoices beginning with `C` represent cancellations.
- Negative quantities represent returns or adjustments and should not be treated as ordinary demand.
- `DOT`, `POST`, `M`, and `ADJUST` are operational codes rather than normal products.
- Zero prices and unusually large quantities or prices require quality checks.
- The final observed month is incomplete because the latest date is `2011-12-04`.

## Analytical Views

The cleaned data supports four views:

1. **Data quality:** missing values, duplicates, cancellations, returns, and outliers.
2. **Demand over time:** daily or weekly units, revenue, orders, weekday activity, and hourly patterns.
3. **Customer and product value:** customer RFM, basket behavior, product concentration, and returns.
4. **Model readiness:** product-time aggregates, customer snapshots, and leakage-safe validation datasets.

## Model Readiness and Validation

- Use positive, non-cancelled sales for demand, recommendation, and stockout analysis.
- Keep returns and cancellations separately for return-rate and net-revenue analysis.
- Aggregate to the required prediction grain before splitting the data.
- Use chronological train, validation, and test splits for time-dependent models.
- Build customer features as of each cutoff date; never use future transactions.
- Exclude December 2011 from complete-month comparisons or label it as partial.
- Use MAE and RMSE for forecasts, precision and recall for peak-hour classification, lift and coverage for recommendations, and cluster stability plus business profiles for segmentation.

## Completed Outputs

Cleaning is documented in [data_clean.md](data_clean.md), and feature creation is documented in [feature_engineering.md](feature_engineering.md).

The finalized files are in [data/processed](../data/processed):

- `feature_transactions.csv`: transaction-level features for hourly, basket, and return analysis.
- `daily_product_features.csv`: daily product demand for forecasting and stockout analysis.
- `weekly_product_features.csv`: weekly product demand for forecasting.
- `customer_rfm_asof.csv`: monthly customer RFM snapshots calculated as of each cutoff.

The source data has no cost column or inventory/on-hand quantity. Therefore, profit is estimated with an assumed 30% margin, and true stockout labels cannot be created without additional inventory data.
