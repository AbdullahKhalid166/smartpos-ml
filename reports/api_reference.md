# SmartPOS ML API Reference

Base URL: `http://localhost:8000`

All prediction responses use this envelope:

```json
{
  "result": {},
  "model_version": "...",
  "notes": "..."
}
```

## Health

### `GET /health`

Readiness check for the API and loaded model registry. No input is required.

Example response:

```json
{
  "status": "ok",
  "models_loaded": [
    "forecast",
    "insights",
    "peakhour",
    "profit",
    "recommendations",
    "segment",
    "stockout"
  ]
}
```

## Forecast

### `POST /predict/forecast`

Input:

```json
{
  "StockCode": "85123A",
  "start_date": "2011-12-05",
  "end_date": "2011-12-19"
}
```

Dates must be ordered and the end date cannot exceed `2012-12-03`.

Example response:

```json
{
  "result": {
    "StockCode": "85123A",
    "periods": ["2011-12-05", "2011-12-12", "2011-12-19"],
    "units": [12.4, 11.8, 12.1],
    "revenue": [31.0, 29.5, 30.3]
  },
  "model_version": "enhanced_product_v1",
  "notes": "Enhanced product-level forecast using volatility and lag features."
}
```

## Profit Forecast

### `POST /predict/profit`

Input:

```json
{
  "start_date": "2011-12-05",
  "end_date": "2011-12-19"
}
```

Example response:

```json
{
  "result": {
    "periods": ["2011-12-05", "2011-12-12", "2011-12-19"],
    "profit": [12500.0, 13200.0, 12950.0]
  },
  "model_version": "profit_forecast_v1",
  "notes": "Weekly profit forecast."
}
```

## Customer Segment

### `POST /predict/segment`

Input:

```json
{
  "CustomerID": "12347"
}
```

Example response:

```json
{
  "result": {
    "CustomerID": "12347",
    "segment": "VIP"
  },
  "model_version": "customer_segmentation_v1",
  "notes": "Segment predicted from the latest customer RFM snapshot."
}
```

## Peak Hour

### `POST /predict/peakhour`

Input uses ISO day-of-week numbering (`1` Monday through `7` Sunday):

```json
{
  "DayOfWeek": 1,
  "Month": 12,
  "Hour": 12
}
```

Example response:

```json
{
  "result": {
    "prediction": "busy",
    "is_busy": true
  },
  "model_version": "peak_hour_v1",
  "notes": "Busy classification from the saved peak-hour model."
}
```

## Stockout Alerts

### `GET /predict/stockout`

No input is required. The result is an array of velocity-based alerts; it is not an inventory-confirmed stockout prediction.

Example response:

```json
{
  "result": [
    {
      "StockCode": "85123A",
      "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
      "risk_flag": true,
      "reason": "Recent average daily demand is 12.50 units versus a 10.00-unit threshold; this indicates unusually high demand pressure.",
      "recent_daily_units": 12.5,
      "risk_threshold": 10.0
    }
  ],
  "model_version": "stockout_velocity_v1",
  "notes": "Current low-stock velocity alerts."
}
```

## Recommendations

### `GET /recommend/{stock_code}`

Replace `{stock_code}` with a valid product code from the processed dataset.

Example response:

```json
{
  "result": {
    "StockCode": "85123A",
    "recommendations": ["85099B", "22423"]
  },
  "model_version": "association_rules_v1",
  "notes": "Top associated products from the saved association rules."
}
```

## Insights

### `POST /insights`

No request body is required.

Example response:

```json
{
  "result": {
    "insights": [
      "Customer base is led by the VIP segment with 100 customers.",
      "The busiest hour is 12:00, with 500 transactions recorded in the positive-sale set."
    ],
    "surprise": {
      "segment": "VIP",
      "product": "85123A",
      "hour": 12,
      "largest_segment_customer_count": 100,
      "top_risk_product_units": 12.5,
      "peak_hour_volume": 500
    }
  },
  "model_version": "insights_summary_v1",
  "notes": "Saved AI Sales Insights summary."
}
```

## Common Errors

- `404`: unknown `StockCode` or `CustomerID`.
- `422`: invalid request fields, reversed dates, or a forecast end date beyond the supported horizon.
