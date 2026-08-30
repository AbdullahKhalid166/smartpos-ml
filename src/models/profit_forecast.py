"""Profit forecasting for SmartPOS."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "weekly_product_features.csv"
MODEL_PATH = ROOT / "models" / "profit_forecast_model.joblib"
FIGURE_DIR = ROOT / "reports" / "figures"


def load_profit_data(path=DATA_PATH):
    """Load and sort weekly product data for profit modeling."""
    df = pd.read_csv(path, parse_dates=["Period"])
    return df.sort_values("Period").reset_index(drop=True)


def _aggregate_weekly(data, target="EstimatedProfit"):
    agg = data.groupby("Period", as_index=False)[target].sum().sort_values("Period")
    return agg.reset_index(drop=True)


def regression_metrics(actual, predicted):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
    }


def _calendar_features(data):
    return pd.DataFrame(
        {
            "year": data["Period"].dt.year,
            "month": data["Period"].dt.month,
            "week": data["Period"].dt.isocalendar().week.astype(int),
        },
        index=data.index,
    )


def chronological_split(data, test_fraction=0.2):
    split = max(1, int(len(data) * (1 - test_fraction)))
    return data.iloc[:split].copy(), data.iloc[split:].copy()


def train_profit_model(data=None, target="EstimatedProfit", test_fraction=0.2):
    """Train profit models and keep the best tuned XGBoost variant when it improves hold-out performance."""
    weekly = _aggregate_weekly(load_profit_data() if data is None else data.copy(), target)
    train, test = chronological_split(weekly, test_fraction)

    train_features = _calendar_features(train)
    test_features = _calendar_features(test)

    linear = LinearRegression().fit(train_features, train[target])
    linear_pred = linear.predict(test_features)
    linear_metrics = regression_metrics(test[target], linear_pred)

    param_grid = {
        "n_estimators": [200, 250, 400],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05],
    }
    xgb_search = GridSearchCV(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        ),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=1,
    )
    xgb_search.fit(train_features, train[target])
    best_params = xgb_search.best_params_
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
        **best_params,
    )
    xgb.fit(train_features, train[target])
    xgb_pred = xgb.predict(test_features)
    xgb_metrics = regression_metrics(test[target], xgb_pred)

    best = {"name": "xgb", "model": xgb, "metrics": xgb_metrics, "prediction": xgb_pred, "params": best_params}
    if linear_metrics["RMSE"] < xgb_metrics["RMSE"]:
        best = {"name": "linear", "model": linear, "metrics": linear_metrics, "prediction": linear_pred, "params": None}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best["model"], "metrics": best["metrics"], "target": target, "model_name": best["name"], "params": best["params"]}, MODEL_PATH)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(test["Period"], test[target], label="Actual")
    plt.plot(test["Period"], best["prediction"], label="Predicted")
    plt.title(f"Weekly {target}: actual vs {best['name']} prediction")
    plt.xlabel("Week")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plot_path = FIGURE_DIR / "profit_forecast_predicted_vs_actual.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return {
        "target": target,
        "test": test,
        "linear_model": linear,
        "linear_metrics": linear_metrics,
        "xgb_model": xgb,
        "xgb_metrics": xgb_metrics,
        "best_model": best,
        "plot_path": plot_path,
    }


def predict_profit(new_data):
    """Predict profit using the saved model."""
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features = pd.DataFrame({
        "year": new_data["Period"].dt.year,
        "month": new_data["Period"].dt.month,
        "week": new_data["Period"].dt.isocalendar().week.astype(int),
    })
    return model.predict(features)


def generate():
    """Return a reusable profit forecast result object."""
    return train_profit_model()


if __name__ == "__main__":
    result = train_profit_model()
    print(result["linear_metrics"])
    print(result["xgb_metrics"])
    print(result["best_model"]["metrics"])
