# Enhanced Product-Level Forecast Model

## Model Configuration
- **Method**: XGBoost with product-level volatility features
- **Target**: Units
- **Train samples**: 147886
- **Test samples**: 37128

## Performance
- **RMSE**: 81.14
- **MAE**: 34.90

## Features
1. Product demand volatility (rolling std of units)
2. Product price volatility (rolling std of average price)
3. Product average price level
4. Per-product lag features (lag_1, lag_2, rolling_4)
5. Calendar features (year, month, week, day_of_week)
6. Product ID encoding

## Rationale
Sparse products (units 1-2) were over-predicted by baseline models (predicted 95 units) because:
- Low-volume products have few training samples
- Models default to mean estimates without product context
- Calendar-only features cannot distinguish product-specific patterns

This model adds product-level signals to help XGBoost learn per-product demand patterns.

## Model Architecture
- n_estimators: 350
- max_depth: 5
- subsample: 0.9 (reduced overfitting for sparse data)
- colsample_bytree: 0.9
