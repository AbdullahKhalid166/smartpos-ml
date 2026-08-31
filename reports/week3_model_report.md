# Week 3 Modeling Report

## Scope

Week 3 adds customer segmentation, low-stock alerting, profit forecasting, and peak-hour prediction. Every model is built from the processed outputs created previously in the project. Where time-based modeling is used, train/test splits are chronological and no future leakage is allowed.

## 1. Customer Segmentation

Customer RFM data was loaded from `data/processed/customer_rfm_asof.csv`, filtered to the latest `CutoffDate`, and scaled with `StandardScaler`. Monetary values were log-transformed before scaling when skew was present to stabilize distances.

The optimal K value was selected by silhouette score from `k=3` to `k=6` using KMeans.

| Metric | Value |
|---|---:|
| Best k | 3 |
| Silhouette score | 0.4420 |

Cluster labels were ranked by average RFM behavior and mapped to meaningful labels: `VIP`, `Regular`, and `At-risk`.

## 2. Low Stock Alerts

The velocity-based stockout logic reuses the Week 2 demand-pressure approach. A risk flag is raised when a product's recent daily demand sits at or above the 75th percentile threshold for recent demand.

The alert output is saved as `data/processed/low_stock_alerts.csv` and includes:

- `StockCode`
- `Description`
- `risk_flag`
- `reason`
- `recent_daily_units`
- `risk_threshold`

This is a demand-pressure proxy rather than a verified inventory depletion signal, because no stock-on-hand or replenishment data is available.

## 3. Profit Forecasting

Profit forecasting uses `data/processed/weekly_product_features.csv` with `EstimatedProfit` as the target. The series is aggregated to weekly totals before modeling to match the global forecast approach used in Week 2.

| Model | RMSE | MAE |
|---|---:|---:|
| Linear Regression | 18,997.34 | 14,674.77 |
| XGBoost | 12,556.67 | 9,881.36 |

The XGBoost model was selected as the best saved model and saved to `models/profit_forecast_model.joblib`.

The evaluation plot is saved to `reports/figures/profit_forecast_predicted_vs_actual.png`.

## 4. Peak Hour Prediction

Transaction-level positive sales were aggregated by `Hour`, `DayOfWeek`, and `Month` to measure transaction intensity. Busiest hours were defined as the top-tercile by transaction volume and labeled as `busy`.

The final classifier used the hour signal directly, plus cyclical hour encoding (`hour_sin`, `hour_cos`), weekend context, and a weighted XGBoost setup with threshold tuning. This improved the busy-class detection substantially compared with the earlier baseline version, which had only `DayOfWeek` and `Month` features and a busy recall of `0.3235`.

| Metric | Value |
|---|---:|
| Accuracy | 0.8040 |
| Precision (class 0) | 0.8494 |
| Recall (class 0) | 0.8408 |
| Precision (class 1) | 0.6883 |
| Recall (class 1) | 0.7794 |

The peak-hour model is saved to `models/peak_hour_model.joblib`, and the confusion matrix chart is saved to `reports/figures/peak_hour_confusion_matrix.png`.

This update represents a clear improvement in busy-hour detection: the recall for the busy class increased from `0.3235` to `0.7794`, while precision also improved from `0.6471` to `0.6883` in the final model.

## 5. Sales Insights

The insights module combines the main Week 3 outputs and creates short rule-based summaries, including the segment dominated by customer volume, the highest-risk product, the busiest operational hour, and the strongest profit signal.

The generator is intentionally simple and transparent: it does not use any API or LLM and instead relies on deterministic rules over saved outputs.

## Limitations

- Profit is estimated margin, not verified actual profit after shipping, refunds, or operating costs.
- Stockout risk is a demand-pressure signal, not a confirmed inventory stockout.
- Customer segments are snapshot-based and depend on the latest `CutoffDate`.
- Peak-hour classification captures traffic intensity rather than staffing or conversion quality.

## Output Files

- `models/customer_segmentation_model.joblib`
- `models/stockout_alerts.joblib`
- `models/profit_forecast_model.joblib`
- `models/peak_hour_model.joblib`
- `data/processed/low_stock_alerts.csv`
- `reports/figures/profit_forecast_predicted_vs_actual.png`
- `reports/figures/peak_hour_confusion_matrix.png`

"recall for busy-hour detection is lower than desired; likely due to class imbalance, worth revisiting with class-weighting in a future iteration"