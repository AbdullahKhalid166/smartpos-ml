# Final Model Evaluation

| model | metric | value |
|---|---|---:|
| forecast_baseline | RMSE | 23499.796765 |
| forecast_baseline | MAE | 19587.109375 |
| forecast_improved | RMSE | 40288.899422 |
| forecast_improved | MAE | 31561.822173 |
| forecast_enhanced_product | RMSE | 81.14 |
| forecast_enhanced_product | MAE | 34.90 |
| profit_forecast | RMSE | 10756.621423 |
| profit_forecast | MAE | 8620.823778 |
| segmentation | silhouette | 0.442038 |
| peak_hour | accuracy | 0.804000 |
| peak_hour | precision_busy | 0.688300 |
| peak_hour | recall_busy | 0.779400 |
| recommendations | top_lift | 48.061966 |
| recommendations | top_confidence | 0.816547 |

## Notes
- Forecasting: baseline XGBoost was the default. The enhanced product-level model (RMSE 81.14, MAE 34.90) operates at per-product-period grain and is not directly comparable to the global weekly forecast (RMSE 23,499.80); it specifically addresses the sparse-product over-prediction issue found in edge case testing.
- Profit forecasting: tuned XGBoost improved hold-out RMSE and MAE versus the earlier baseline and was retained.
- Segmentation: optimal K is 3; silhouette score remains moderate because the snapshot is coarse and customer behavior is varied.
- Peak-hour detection: the improved weighted XGBoost model with cyclical hour features and threshold tuning increased busy-hour recall from 0.3235 to 0.7794 while improving precision to 0.6883.
- Recommendations: top associations remain strong for common items, but coverage is weak for rare products and should be treated as opportunistic rather than universal.
