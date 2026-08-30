"""Weekly sales forecasting for SmartPOS.

The functions use chronological splits. Lag features for the improved model are
created recursively, so test-period actual targets are never used as inputs.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "weekly_product_features.csv"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"


def load_weekly_data(path=DATA_PATH):
    """Load and sort weekly product data."""
    data = pd.read_csv(path, parse_dates=["Period"])
    return data.sort_values("Period").reset_index(drop=True)


def regression_metrics(actual, predicted):
    """Return the standard forecast error metrics."""
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
    """Split rows by time, without shuffling."""
    split = max(1, int(len(data) * (1 - test_fraction)))
    return data.iloc[:split].copy(), data.iloc[split:].copy()


def run_baseline(data=None, target="Units", test_fraction=0.2):
    """Train Linear Regression and XGBoost calendar baselines."""
    data = load_weekly_data() if data is None else data.copy()
    data = data.groupby("Period", as_index=False)[target].sum().sort_values("Period")
    train, test = chronological_split(data, test_fraction)
    features_train = _calendar_features(train)
    features_test = _calendar_features(test)

    linear = LinearRegression().fit(features_train, train[target])
    linear_prediction = linear.predict(features_test)
    linear_metrics = regression_metrics(test[target], linear_prediction)

    xgb = XGBRegressor(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    xgb.fit(features_train, train[target])
    xgb_prediction = xgb.predict(features_test)
    xgb_metrics = regression_metrics(test[target], xgb_prediction)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"model": xgb, "features": features_train.columns.tolist(), "target": target}
    explicit_name = MODEL_DIR / f"baseline_xgb_{target.lower()}.joblib"
    legacy_name = MODEL_DIR / f"baseline_{target.lower()}.joblib"
    joblib.dump(artifact, explicit_name)
    joblib.dump(artifact, legacy_name)
    return {
        "target": target,
        "test": test,
        "linear_prediction": linear_prediction,
        "xgb_prediction": xgb_prediction,
        "linear_metrics": linear_metrics,
        "xgb_metrics": xgb_metrics,
    }


def save_prediction_plot(result, target="Units", model_name="baseline"):
    """Save actual versus model predictions using a consistent filename prefix."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    test = result["test"]
    prediction = result.get("xgb_prediction", result.get("prediction"))
    if prediction is None:
        raise ValueError("Result must include either 'xgb_prediction' or 'prediction'.")

    plt.figure(figsize=(10, 5))
    plt.plot(test["Period"], test[target], label="Actual")
    plt.plot(test["Period"], prediction, label="Predicted")
    plt.title(f"Weekly {target}: actual versus {model_name} prediction")
    plt.xlabel("Week")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    output = FIGURE_DIR / f"{model_name}_{target.lower()}_predicted_vs_actual.png"
    plt.savefig(output, dpi=150)
    plt.close()
    return output


def product_seasonal_naive(data=None, target="Units", test_fraction=0.2):
    """Compare a per-product last-value forecast with the global baseline."""
    data = load_weekly_data() if data is None else data.copy()
    data = data.sort_values(["StockCode", "Period"])
    split_period = data["Period"].sort_values().iloc[max(1, int(len(data) * (1 - test_fraction)))]
    train = data[data["Period"] < split_period]
    test = data[data["Period"] >= split_period].copy()
    last_values = train.groupby("StockCode")[target].last()
    test["prediction"] = test["StockCode"].map(last_values).fillna(0)
    return {
        "metrics": regression_metrics(test[target], test["prediction"]),
        "predictions": test,
        "split_period": split_period,
    }


def _recursive_lag_frame(values, periods):
    rows = []
    history = list(values)
    for period in periods:
        rows.append(
            {
                "year": period.year,
                "month": period.month,
                "week": int(period.isocalendar().week),
                "lag_1": history[-1],
                "lag_2": history[-2] if len(history) > 1 else history[-1],
                "rolling_4": float(np.mean(history[-4:])),
            }
        )
        history.append(history[-1])
    return pd.DataFrame(rows)


def _training_lag_frame(data, target):
    """Create lag features from observed training targets only."""
    frame = data.copy()
    frame["year"] = frame["Period"].dt.year
    frame["month"] = frame["Period"].dt.month
    frame["week"] = frame["Period"].dt.isocalendar().week.astype(int)
    frame["lag_1"] = frame[target].shift(1)
    frame["lag_2"] = frame[target].shift(2)
    frame["rolling_4"] = frame[target].shift(1).rolling(4).mean()
    return frame.dropna()


def run_improved(data=None, target="Units", test_fraction=0.2):
    """Train XGBoost with lag and rolling features using training history only."""
    data = load_weekly_data() if data is None else data.copy()
    data = data.groupby("Period", as_index=False)[target].sum().sort_values("Period")
    train, test = chronological_split(data, test_fraction)
    if len(train) < 5:
        raise ValueError("At least five training periods are required for lag features.")

    train_features = _training_lag_frame(train, target)
    model_features = ["year", "month", "week", "lag_1", "lag_2", "rolling_4"]
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(train_features[model_features], train_features[target])
    history = train[target].tolist()
    predictions = []
    for _, row in test.iterrows():
        feature_row = pd.DataFrame(
            [{
                "year": row["Period"].year,
                "month": row["Period"].month,
                "week": int(row["Period"].isocalendar().week),
                "lag_1": history[-1],
                "lag_2": history[-2],
                "rolling_4": np.mean(history[-4:]),
            }]
        )
        prediction = float(model.predict(feature_row[model_features])[0])
        predictions.append(prediction)
        history.append(prediction)
    metrics = regression_metrics(test[target], predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "features": model_features, "target": target, "variant": "recursive_improved"}
    explicit_name = MODEL_DIR / f"improved_recursive_xgb_{target.lower()}.joblib"
    legacy_name = MODEL_DIR / f"improved_{target.lower()}.joblib"
    joblib.dump(artifact, explicit_name)
    joblib.dump(artifact, legacy_name)
    return {"target": target, "metrics": metrics, "test": test, "prediction": predictions}


def _direct_features(data, target, rolling_windows=(4,), include_price_features=False):
    """Build one-step features from observed values before each period."""
    frame = data.copy().sort_values("Period").reset_index(drop=True)
    frame["year"] = frame["Period"].dt.year
    frame["month"] = frame["Period"].dt.month
    frame["week"] = frame["Period"].dt.isocalendar().week.astype(int)
    frame["day_of_week"] = frame["Period"].dt.dayofweek
    frame["lag_1"] = frame[target].shift(1)
    frame["lag_2"] = frame[target].shift(2)
    for window in rolling_windows:
        frame[f"rolling_{window}"] = frame[target].shift(1).rolling(window).mean()
    if include_price_features:
        frame["average_price"] = frame["AveragePrice"]
        frame["price_change"] = frame["AveragePrice"].diff().fillna(0)
    return frame


def run_direct_variant(
    data=None,
    target="Units",
    test_fraction=0.2,
    rolling_windows=(4,),
    include_price_features=False,
    include_extra_features=False,
    model_name="improved_v2",
):
    """Evaluate a non-recursive one-step model using actual prior observations."""
    data = load_weekly_data() if data is None else data.copy()
    data = data.groupby("Period", as_index=False).agg(
        **{target: (target, "sum"), "AveragePrice": ("AveragePrice", "mean")}
    ).sort_values("Period")
    frame = _direct_features(data, target, rolling_windows, include_price_features)
    split = int(len(frame) * (1 - test_fraction))
    train = frame.iloc[:split].dropna().copy()
    test = frame.iloc[split:].dropna().copy()
    features = ["year", "month", "week", "lag_1", "lag_2"]
    features += [f"rolling_{window}" for window in rolling_windows]
    if include_extra_features:
        features += ["day_of_week"]
    if include_price_features:
        features += ["average_price", "price_change"]
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(train[features], train[target])
    prediction = model.predict(test[features])
    metrics = regression_metrics(test[target], prediction)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "features": features, "target": target, "method": "direct_one_step", "variant": model_name}
    explicit_name = MODEL_DIR / f"improved_direct_xgb_{target.lower()}.joblib"
    legacy_name = MODEL_DIR / f"{model_name}_{target.lower()}.joblib"
    joblib.dump(artifact, explicit_name)
    joblib.dump(artifact, legacy_name)
    return {"target": target, "metrics": metrics, "test": test, "prediction": prediction, "features": features}


def run_product_model(data=None, target="Units", test_fraction=0.2, top_n=None):
    """Train one pooled model on product rows with actual per-product lags."""
    data = load_weekly_data() if data is None else data.copy()
    data = data.sort_values(["Period", "StockCode"]).copy()
    if top_n:
        top_products = data.groupby("StockCode")[target].sum().nlargest(top_n).index
        data = data[data["StockCode"].isin(top_products)].copy()
    data["product_id"] = pd.factorize(data["StockCode"])[0]
    data["lag_1"] = data.groupby("StockCode")[target].shift(1)
    data["lag_2"] = data.groupby("StockCode")[target].shift(2)
    data["rolling_4"] = data.groupby("StockCode")[target].shift(1).rolling(4).mean()
    data["year"] = data["Period"].dt.year
    data["month"] = data["Period"].dt.month
    data["week"] = data["Period"].dt.isocalendar().week.astype(int)
    data = data.dropna()
    split_period = data["Period"].sort_values().iloc[max(1, int(len(data) * (1 - test_fraction)))]
    train = data[data["Period"] < split_period]
    test = data[data["Period"] >= split_period]
    features = ["product_id", "year", "month", "week", "lag_1", "lag_2", "rolling_4"]
    model = XGBRegressor(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        objective="reg:squarederror", random_state=42, n_jobs=2,
    )
    model.fit(train[features], train[target])
    prediction = model.predict(test[features])
    return {
        "target": target,
        "metrics": regression_metrics(test[target], prediction),
        "test": test,
        "prediction": prediction,
    }


if __name__ == "__main__":
    baseline = run_baseline(target="Units")
    print("Linear Regression:", baseline["linear_metrics"])
    print("Baseline XGBoost:", baseline["xgb_metrics"])
    print("Saved baseline plot:", save_prediction_plot(baseline, "Units", model_name="baseline_xgb"))

    direct_units = run_direct_variant(
        target="Units",
        include_extra_features=True,
        include_price_features=True,
        model_name="improved_direct_xgb",
    )
    print("Direct improved Units XGBoost:", direct_units["metrics"])
    print("Saved direct improved Units plot:", save_prediction_plot(direct_units, "Units", model_name="improved_direct_xgb"))

    direct_revenue = run_direct_variant(
        target="Revenue",
        rolling_windows=(2,),
        model_name="improved_direct_xgb",
    )
    print("Direct improved Revenue XGBoost:", direct_revenue["metrics"])
    print("Saved direct improved Revenue plot:", save_prediction_plot(direct_revenue, "Revenue", model_name="improved_direct_xgb"))