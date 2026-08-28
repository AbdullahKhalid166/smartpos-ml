# Data Cleaning Process

The cleaning process is implemented in [02_cleaning.ipynb](../notebooks/02_cleaning.ipynb). It reads the raw retail CSV and writes reusable datasets to `data/interim`.

## Cleaning Steps

1. Parse `InvoiceDate` as datetime and convert `Quantity` and `Price` to numeric values.
2. Normalize text columns by trimming whitespace and keeping invoice and stock-code values as text.
3. Replace missing `Customer ID` values with `Guest`.
4. Fill missing descriptions from a valid description with the same `StockCode`; use `Unknown description` when no match exists.
5. Flag cancelled invoices when `Invoice` starts with `C`.
6. Flag operational stock codes: `DOT`, `POST`, `M`, and `ADJUST`.
7. Remove exact duplicate rows. The executed data contained **34,150** exact duplicates.
8. Keep repeated purchases as valid records; they are not automatically errors.
9. Remove rows where the absolute `Quantity` or `Price` exceeds 1,000. This removed **735** rows.
10. Remove true junk where customer and description are missing, price is zero, and quantity is negative. This removed **2,688** rows.
11. Calculate `TotalPrice = Quantity * Price`.
12. Keep valid cancelled negative-quantity transactions in a separate returns dataset.

## Outputs

After cleaning, the main transaction dataset contains **1,013,408 rows**.

| File | Contents |
|---|---|
| `clean_transactions.csv` | All cleaned transactions, including quality flags and cancellations. |
| `returns.csv` | Negative-quantity cancelled transactions with known customers and descriptions. |
| `modeling_sales.csv` | Positive, non-cancelled sales with operational codes excluded. |
| `complete_month_sales.csv` | Modeling sales excluding the incomplete December 2011 month. |

## Data Use Rules

Use `modeling_sales.csv` for demand-oriented models. Use `returns.csv` for return patterns and keep cancellations available for net-revenue analysis. Do not remove repeated purchases unless a business rule confirms they are duplicates. Do not use the partial final month for full-month comparisons.
