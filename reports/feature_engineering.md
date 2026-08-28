# Feature Engineering Process

The feature-engineering process is implemented in [03_feature_engineering.ipynb](../notebooks/03_feature_engineering.ipynb). It reads `data/interim/clean_transactions.csv` and writes model-ready files to `data/processed`.

## Transaction Features

The following features are added to every cleaned transaction:

- `Date` and `Time`: date and clock time extracted from `InvoiceDate`.
- `Hour`: hour of purchase, useful for peak-hour analysis.
- `DayOfWeek`: numeric weekday, where Monday is 0.
- `Month` and `Year`: calendar features for seasonality.
- `TotalPrice`: `Quantity * Price`.
- `IsCancelled`: true when the invoice starts with `C`.
- `IsReturn`: true when `Quantity` is negative.
- `EstimatedProfit`: `TotalPrice * 0.30` using an assumed 30% margin.

The profit value is only an estimate because the raw data has no product cost or actual margin column.

## Product Aggregation

For demand modeling, the notebook uses positive, non-cancelled sales and excludes operational stock codes. It aggregates by time period, `StockCode`, and `Description` before any train/test split.

Each daily or weekly row contains:

- `Units`: total quantity sold.
- `Revenue`: total sales value.
- `EstimatedProfit`: estimated profit value.
- `Orders`: distinct invoice count.
- `AveragePrice`: mean selling price.

## Customer RFM

Monthly customer snapshots are calculated only from transactions on or before each month-end cutoff:

- `Recency`: days since the customer’s last purchase at the cutoff.
- `Frequency`: distinct invoices up to the cutoff.
- `Monetary`: total positive sales value up to the cutoff.

The incomplete December 2011 month is excluded from the RFM cutoff list. This prevents future leakage and avoids treating a partial month as a complete observation period.

## Processed Files and Model Use

| File | Model use |
|---|---|
| `feature_transactions.csv` | Peak-hour classification, basket creation for recommendations, and return analysis. |
| `daily_product_features.csv` | Daily sales forecasting and stockout-risk analysis. |
| `weekly_product_features.csv` | Weekly units, revenue, and estimated-profit forecasting. |
| `customer_rfm_asof.csv` | Customer segmentation using leakage-safe RFM features. |

Time-dependent models must split chronologically. Lag and rolling features must be created using training history only. Recommendations should use positive product sales and exclude operational codes. Actual stockout prediction requires inventory or on-hand data, which is not available in the current dataset.
