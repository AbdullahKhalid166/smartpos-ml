# Model Documentation Report

## Problem statement
SmartPOS needs a practical decision-support layer for demand planning, profit forecasting, customer segmentation, operational staffing, and product recommendations. The goal is to convert transaction history into a useful set of business signals without leaking future information into model training or evaluation.

## Data source and cleaning
The project uses processed retail data files in `data/processed/`, including weekly product features, daily product features, the latest customer RFM snapshot, and transaction-level sales data. The cleaning stage removed cancelled or invalid transactions, excluded negative-quantity records, and filtered operational stock codes such as `DOT`, `POST`, `M`, and `ADJUST` before modeling.

## Feature engineering
- Weekly forecasting uses calendar features derived from `Period`: year, month, and ISO week.
- Profit forecasting applies the same calendar structure to the `EstimatedProfit` target.
- Customer segmentation uses RFM features (`Recency`, `Frequency`, `Monetary`), with a log transform on Monetary when skew is high.
- Peak-hour classification aggregates positive-sale transactions by `Hour`, `DayOfWeek`, and `Month` and labels the busy class using the top-66th percentile hourly volume.
- Recommendations build invoice-level baskets and mine rules using Apriori association analysis.

## Model approach and results
### Forecasting
The baseline global XGBoost forecast remains the best performing forecast on held-out data with RMSE 23,499.80 and MAE 19,587.11. The recursive improved variant degraded materially on the test horizon and was not kept for production.

### Profit forecasting
The tuned XGBoost profit model achieved RMSE 10,756.62 and MAE 8,620.82 after a light GridSearchCV search. This was retained because it improved hold-out performance over the original baseline logic and is the strongest profit signal in the project.

### Segmentation
KMeans from `k=3` to `k=6` selected `k=3` as the best solution with silhouette score 0.4420. The cluster labels were mapped to `VIP`, `Regular`, and `At-risk` based on average RFM behavior.

### Stockout risk
The stockout module computes recent daily demand against a 75th-percentile threshold. This is a demand-pressure signal rather than a verified inventory stockout because no stock-on-hand, lead-time, or replenishment data is present.

### Peak-hour detection
The original busy-hour recall was 0.3235. After class balancing and threshold tuning, the busy-class recall improved to 0.8971 while precision dropped to 0.4094. This improves operational coverage for rush periods, though the minority class remains difficult to separate cleanly.

### Recommendations
Association rules were mined with Apriori and ranked by lift and confidence. The strongest rules have lift values around 48, but the coverage for rare items remains poor because the method favors frequent co-purchase patterns.

## Limitations
- Estimated profit is a margin estimate, not actual profit after shipping, refunds, or operating costs.
- Velocity-based stockout risk is not an inventory-confirmed stockout and can overstate risk on low-volume products.
- Customer segments are snapshot-based and sensitive to one-transaction profiles.
- Peak-hour classification models traffic intensity, not staffing quality or sales conversion.
- Recommendation coverage is weak for sparse products because frequent-item patterns dominate the rule set.

## Final recommendations
1. Keep the baseline XGBoost demand forecast and the tuned profit forecast as production signals.
2. Treat customer segmentation as a descriptive cohort label rather than a hard operational trigger.
3. Use stockout alerts as prioritization inputs requiring inventory validation.
4. Use the weighted busy-hour model for staffing planning, with manual review when operational urgency is high.
5. Limit recommendations to common products and add a coverage check before suggesting low-frequency SKUs.

to run:
cd /d "C:\Users\HS TRADER\OneDrive\Documents\OneDrive\Desktop\smartpos-ml"
.\.venv\Scripts\python.exe -m streamlit run src/app/demo_app.py --server.headless true --server.port 8501