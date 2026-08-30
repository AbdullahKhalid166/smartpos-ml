# Final Model Evaluation

| model | metric | value |
|---|---|---:|
| forecast_baseline | RMSE | 23499.796765 |
| forecast_baseline | MAE | 19587.109375 |
| forecast_improved | RMSE | 40288.899422 |
| forecast_improved | MAE | 31561.822173 |
| profit_forecast | RMSE | 10756.621423 |
| profit_forecast | MAE | 8620.823778 |
| segmentation | silhouette | 0.442038 |
| peak_hour | precision_busy | 0.409396 |
| peak_hour | recall_busy | 0.897059 |
| recommendations | top_lift | 48.061966 |
| recommendations | top_confidence | 0.816547 |

## Notes
- Forecasting: baseline XGBoost remains the default production forecast because it outperforms the recursive improved variant on the test split.
- Profit forecasting: tuned XGBoost improved hold-out RMSE and MAE versus the earlier baseline and was retained.
- Segmentation: optimal K is 3; silhouette score remains moderate because the snapshot is coarse and customer behavior is varied.
- Peak-hour detection: the weighted XGBoost model with threshold tuning materially improved recall for the busy class.
- Recommendations: top associations remain strong for common items, but coverage is weak for rare products and should be treated as opportunistic rather than universal.
