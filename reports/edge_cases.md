# Edge Case Findings

## Forecast and stockout on sparse-history products
- The lowest-volume products in the weekly product table still have total units of 1, such as `79301`, `79309A`, `20885`, `72778`, and `72751A`. On these near-zero histories, the forecast is extremely sensitive to a single weekly swing and can produce unstable or noisy predictions.
- The stockout logic uses recent daily demand relative to a 75th-percentile threshold. For sparse products, a single spike in recent sales can exceed that threshold and trigger a risk flag even when no real inventory shortage exists. This is a demand-pressure alert rather than a confirmed stockout.

## Segmentation on single-transaction customers
- Single-transaction customers are common in the latest snapshot. One example is Customer ID `16312`, which has `Frequency = 1` and a high monetary value. These profiles can distort cluster centroids because a rare one-off customer is treated as a stable segment pattern.
- This is a snapshot-based limitation: the segment assignment can change materially when the cutoff date shifts or a cohort changes dramatically.

## Recommendations for rare or low-frequency products
- Rare items such as `35751C`, `20673`, `35051B`, `20812`, and `21767` appear only a few times in valid invoices. In the mined rules, their coverage is zero because the model is dominated by frequent co-purchase patterns.
- Recommendation quality degrades sharply when support is low; high lift can be driven by a few transactions and does not generalize well to sparse items.

## Summary
- Forecasting degrades on near-zero history.
- Segmentation is sensitive to one-transaction customers.
- Recommendation coverage is poor for rare products and should be treated as a common-item-only feature.
