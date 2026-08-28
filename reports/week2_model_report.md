# Week 2 Modeling Report

## Scope

The models use the processed outputs created by the cleaning and feature-engineering notebooks. All time-dependent forecast evaluation uses chronological splits; rows are not randomly shuffled.

## Forecast Results

The weekly product file is aggregated to total weekly demand before the global forecast. The baseline compares Linear Regression and XGBoost using calendar features. The original improved model is XGBoost with `lag_1`, `lag_2`, and a four-week rolling mean. Its multi-step test forecast feeds each prediction back into the next row's lag features, which compounds error over the test horizon.

| Target | Model | RMSE | MAE |
|---|---|---:|---:|
| Units | Linear Regression baseline | 34,654.47 | 26,692.69 |
| Units | XGBoost baseline | 23,499.80 | 19,587.11 |
| Units | XGBoost with lag/rolling features | 40,288.90 | 31,561.82 |
| Units | Direct actual-lag variant (`improved_v2`) | 35,466.38 | 30,934.31 |
| Units | Direct + weekday/price features (`improved_v3`) | 31,087.76 | 27,083.97 |
| Units | Direct + 2-week rolling (`improved_v4`) | 32,586.47 | 25,682.23 |
| Revenue | Linear Regression baseline | 63,324.46 | 48,915.91 |
| Revenue | XGBoost baseline | 41,855.57 | 32,937.87 |
| Revenue | XGBoost with lag/rolling features | 49,403.51 | 36,592.81 |
| Revenue | Direct + weekday/price features (`improved_v3`) | 60,384.15 | 46,806.13 |
| Revenue | Direct + 2-week rolling (`improved_v4`) | 54,646.00 | 43,150.48 |

For this test split, baseline XGBoost remains the best global model for both targets. Switching from recursive to direct one-step features fixed the main implementation problem and materially reduced Units error, but the added features still did not beat the calendar-only baseline. The 2-week rolling variant had the best corrected Units MAE and the best corrected Revenue scores, but remained above baseline. The new variants are saved as `improved_v2_*`, `improved_v3_*`, and `improved_v4_*` for comparison; baseline artifacts remain unchanged.

A per-StockCode seasonal-naive comparison is also implemented in `forecast.py`. It predicts each product’s test demand using its last observed training value. For Units, it produced RMSE **128.75** and MAE **53.93** at the product-row level. The top-100 pooled per-StockCode XGBoost model produced RMSE **290.22** and MAE **174.29** on 1,888 product-period rows. These metrics are not directly comparable with the global total forecast because they evaluate individual product rows rather than weekly totals.

## Forecast Feature Diagnosis

The original recursive function creates test features from model predictions after the first test period. This was verified in `run_improved`: its `history` list starts with training targets and then appends each predicted value. The corrected `run_direct_variant` uses actual values from periods before the current prediction. For test rows, this uses only already-observed history and never the current or future target. Training features are built from training rows only.

The direct feature experiments followed this order:

1. Actual `lag_1`, `lag_2`, and four-week rolling mean.
2. Added `DayOfWeek`, `AveragePrice`, and `price_change`.
3. Replaced the four-week rolling feature with a two-week rolling mean.
4. Evaluated a pooled top-100 per-StockCode model.

Because none of the corrected global variants improved the untouched baseline, baseline XGBoost should be used for the current global forecast. The direct variants remain useful experiments and are safer than the original recursive implementation for rolling one-step predictions.

Charts are saved in `reports/figures/`:

- `baseline_units_predicted_vs_actual.png`
- `baseline_revenue_predicted_vs_actual.png`

Saved joblib models are written to `models/` as `baseline_units.joblib`, `baseline_revenue.joblib`, `improved_units.joblib`, and `improved_revenue.joblib`.

## Stockout Risk

`src/models/stockout.py` reads `daily_product_features.csv` and calculates, for each product:

- Overall average daily units.
- Recent average daily units over the latest 28-day window.
- A risk threshold at the 75th percentile of recent velocity.
- `stockout_risk = True` when recent velocity is at or above that threshold.

The output is `data/processed/stockout_velocity_risk.csv`. The run produced **5,644 product rows**, with **1,412** flagged as high-velocity risk.

This is a velocity-based prioritization signal, not proof of inventory depletion. The source data has no stock-on-hand, replenishment, or inventory-balance column, so actual stockout labels and days-to-stockout cannot be calculated.

## Product Recommendations

`src/models/recommendations.py` groups valid product sales by `Invoice` to create baskets. It excludes cancelled invoices, negative quantities, and operational stock codes: `DOT`, `POST`, `M`, and `ADJUST`. Apriori association rules are ranked by lift and confidence.

The output is `data/processed/association_rules.csv`, containing the top 100 rules with:

- `support`: proportion of invoices containing both item sets.
- `confidence`: probability of the consequent given the antecedent.
- `lift`: strength compared with independent occurrence.

Top observed rules include:

| Antecedent | Consequent | Support | Confidence | Lift |
|---|---|---:|---:|---:|
| `22746` | `22745` | 0.01113 | 0.81655 | 48.06197 |
| `22746` | `22748` | 0.01147 | 0.84173 | 46.84042 |
| `22745` | `22748` | 0.01351 | 0.79509 | 44.24540 |
| `22578` | `22579` | 0.01071 | 0.59054 | 43.40207 |
| `84596F` | `84596B` | 0.01089 | 0.74247 | 42.29826 |

These rules show co-purchase relationships, not causation. They should be evaluated on a later time holdout using recommendation coverage and lift before being used in production.

## Reproducibility

Use the workspace virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

Required packages are listed in `requirements.txt`. The main reusable modules are `src/models/forecast.py`, `src/models/stockout.py`, and `src/models/recommendations.py`.
