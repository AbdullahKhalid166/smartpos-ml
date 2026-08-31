#!/usr/bin/env python3
"""Train enhanced product-level forecast model."""

from src.models.forecast import run_enhanced_product_model, FIGURE_DIR
import matplotlib.pyplot as plt
import numpy as np

print("Training enhanced product-level forecast model...")
enhanced = run_enhanced_product_model(target="Units")

print("\n=== Enhanced Product-Level Forecast Results ===")
print(f"RMSE: {round(enhanced['metrics']['RMSE'], 2)}")
print(f"MAE: {round(enhanced['metrics']['MAE'], 2)}")
print(f"Model saved: {enhanced['model_path']}")
print(f"Report saved: {enhanced['report_path']}")

# Create comparison visualization
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Enhanced model actual vs predicted
test_data = enhanced["test"].sort_values("Period")
axes[0, 0].scatter(test_data["Units"], enhanced["prediction"], alpha=0.5, s=20)
axes[0, 0].plot([0, test_data["Units"].max()], [0, test_data["Units"].max()], "r--", lw=2)
axes[0, 0].set_xlabel("Actual Units")
axes[0, 0].set_ylabel("Predicted Units")
axes[0, 0].set_title("Enhanced Model: Actual vs Predicted (Product-Level)")
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals distribution
residuals = test_data["Units"].values - enhanced["prediction"]
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel("Residuals (Actual - Predicted)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Residuals Distribution")
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Time series sample (first 100 test samples)
sample_idx = min(100, len(test_data))
axes[1, 0].plot(range(sample_idx), test_data["Units"].values[:sample_idx], "o-", label="Actual", linewidth=2, markersize=4)
axes[1, 0].plot(range(sample_idx), enhanced["prediction"][:sample_idx], "s--", label="Predicted", linewidth=2, markersize=4, alpha=0.7)
axes[1, 0].set_xlabel("Sample Index")
axes[1, 0].set_ylabel("Units")
axes[1, 0].set_title("Time Series Sample (First 100 Test Periods)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Metrics comparison
metrics_data = {
    "Baseline": [23499.80, 19587.11],
    "Enhanced": [enhanced["metrics"]["RMSE"], enhanced["metrics"]["MAE"]],
}
x = np.arange(len(metrics_data))
width = 0.35
rmse_vals = [metrics_data[k][0] for k in metrics_data.keys()]
mae_vals = [metrics_data[k][1] for k in metrics_data.keys()]
axes[1, 1].bar(x - width/2, rmse_vals, width, label="RMSE", alpha=0.8)
axes[1, 1].bar(x + width/2, mae_vals, width, label="MAE", alpha=0.8)
axes[1, 1].set_ylabel("Error")
axes[1, 1].set_title("Forecast Performance Comparison")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics_data.keys())
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
output_path = FIGURE_DIR / "enhanced_product_forecast_comparison.png"
plt.savefig(output_path, dpi=150)
print(f"\nComparison figure saved: {output_path}")
plt.close()

print("\n=== Summary ===")
print(f"Improvement vs baseline:")
print(f"  Baseline RMSE: 23499.80 → Enhanced RMSE: {round(enhanced['metrics']['RMSE'], 2)}")
print(f"  Baseline MAE: 19587.11 → Enhanced MAE: {round(enhanced['metrics']['MAE'], 2)}")
improvement = (23499.80 - enhanced['metrics']['RMSE']) / 23499.80 * 100
print(f"  RMSE Improvement: {round(improvement, 2)}%")
