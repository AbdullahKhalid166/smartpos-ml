#!/usr/bin/env python3
"""Quick enhanced product forecast training."""
import sys
sys.path.insert(0, r'c:\Users\HS TRADER\OneDrive\Documents\OneDrive\Desktop\smartpos-ml')

from src.models.forecast import run_enhanced_product_model
print("Starting enhanced model training...")
result = run_enhanced_product_model(target="Units")
print(f"\n✓ RMSE: {result['metrics']['RMSE']:.2f}")
print(f"✓ MAE: {result['metrics']['MAE']:.2f}")
print(f"✓ Model: {result['model_path']}")
print(f"✓ Report: {result['report_path']}")
baseline_rmse = 23499.80
improvement = (baseline_rmse - result['metrics']['RMSE']) / baseline_rmse * 100
print(f"✓ Improvement: {improvement:.1f}% better RMSE than baseline")
