# Peak Hour Model Evaluation

- Model: xgb_weighted
- Threshold: 0.53
- Features: Hour, DayOfWeek, Month, is_weekend, hour_sin, hour_cos
- Accuracy: 0.8040
- Busy precision: 0.6883
- Busy recall: 0.7794
- Busy F1: 0.7310

## Interpretation
The improved classifier uses the actual hour signal and cyclical hour encoding so it can distinguish peak shopping windows more reliably.
This version is tuned to balance busy-hour recall with precision and is saved to the project artifacts for later reference.
