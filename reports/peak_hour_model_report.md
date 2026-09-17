# Peak Hour Model Evaluation

- Model: xgb_weighted
- Threshold: 0.25
- Features: Hour, DayOfWeek, Month, is_weekend, hour_sin, hour_cos
- Accuracy: 0.6299
- Busy precision: 0.5586
- Busy recall: 0.9492
- Busy F1: 0.7033

## Interpretation
The improved classifier uses the actual hour signal and cyclical hour encoding so it can distinguish peak shopping windows more reliably.
This version is tuned to balance busy-hour recall with precision and is saved to the project artifacts for later reference.
