"""Customer segmentation for the SmartPOS RFM snapshot."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "customer_rfm_asof.csv"
MODEL_PATH = ROOT / "models" / "customer_segmentation.joblib"


class MonetaryLogTransformer:
    """Apply the training-time Monetary transform consistently at prediction."""

    def fit(self, features, y=None):
        self.log_monetary = features["Monetary"].skew() > 1
        return self

    def transform(self, features):
        transformed = features.copy()
        if self.log_monetary:
            transformed["Monetary"] = np.log1p(transformed["Monetary"])
        return transformed


def load_latest_customer_snapshot(path=DATA_PATH):
    """Load the latest customer RFM snapshot only."""
    data = pd.read_csv(path, parse_dates=["CutoffDate"]).copy()
    data = data.sort_values("CutoffDate").reset_index(drop=True)
    latest_date = data["CutoffDate"].max()
    return data[data["CutoffDate"] == latest_date].reset_index(drop=True)


def _prepare_features(frame):
    """Fit the RFM preprocessing pipeline and return transformed features."""
    features = frame[["Recency", "Frequency", "Monetary"]].copy()
    pipeline = Pipeline([
        ("monetary_log", MonetaryLogTransformer()),
        ("scaler", StandardScaler()),
    ])
    scaled = pipeline.fit_transform(features)
    return scaled, pipeline, features.columns.tolist()


def _label_clusters(cluster_summary):
    """Map clusters to human-friendly names by ranking their customer value."""
    summary = cluster_summary.copy()
    summary["value_score"] = summary["Frequency"] + summary["Monetary"] - summary["Recency"]
    ranked = summary.sort_values("value_score", ascending=False).index.tolist()
    mapping = {}
    for rank, cluster_id in enumerate(ranked):
        if rank == 0:
            mapping[cluster_id] = "VIP"
        elif rank == len(ranked) - 1:
            mapping[cluster_id] = "At-risk"
        else:
            mapping[cluster_id] = "Regular"
    if len(ranked) > 3:
        for idx, cluster_id in enumerate(ranked[1:-1]):
            if idx == 0:
                mapping[cluster_id] = "Regular"
            else:
                mapping[cluster_id] = "Low-value"
    return mapping


def fit_customer_segmentation(data=None):
    """Fit a KMeans solution and save the model and preprocessing pipeline."""
    snapshot = load_latest_customer_snapshot() if data is None else data.copy()
    snapshot = snapshot.sort_values("Customer ID").reset_index(drop=True)
    features, preprocessing, feature_names = _prepare_features(snapshot)

    best_k = None
    best_score = -1
    best_model = None
    for k in range(3, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    final_labels = best_model.fit_predict(features)
    snapshot = snapshot.copy()
    snapshot["segment_id"] = final_labels
    cluster_summary = snapshot.groupby("segment_id")[["Recency", "Frequency", "Monetary"]].mean()
    label_map = _label_clusters(cluster_summary)
    snapshot["segment"] = snapshot["segment_id"].map(label_map)

    payload = {
        "model": best_model,
        "preprocessing": preprocessing,
        "feature_names": feature_names,
        "best_k": best_k,
        "silhouette_score": float(best_score),
        "segment_labels": label_map,
        "data": snapshot,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH)
    joblib.dump({"model": best_model, "preprocessing": preprocessing, "segment_labels": label_map, "best_k": best_k}, MODEL_PATH.with_name("customer_segmentation_model.joblib"))
    return payload


def predict_segments(new_data, artifact=None):
    """Predict segment labels for a customer dataframe using the saved model."""
    artifact = joblib.load(MODEL_PATH) if artifact is None else artifact
    features = new_data[["Recency", "Frequency", "Monetary"]].copy()
    preprocessing = artifact.get("preprocessing")
    if preprocessing is None:
        raise KeyError("Segmentation artifact is missing the fitted preprocessing pipeline")
    scaled = preprocessing.transform(features)
    labels = artifact["model"].predict(scaled)
    segment_ids = pd.Series(labels, index=new_data.index)
    segment_names = segment_ids.map(artifact["segment_labels"])
    return segment_names.rename("segment")


def predict_customer_segment(customer_id, artifact=None):
    """Predict one customer's segment from the saved RFM snapshot."""
    artifact = joblib.load(MODEL_PATH) if artifact is None else artifact
    customers = artifact.get("data")
    if customers is None:
        customers = load_latest_customer_snapshot()
    customer = customers[customers["Customer ID"].astype(str) == str(customer_id)]
    if customer.empty:
        raise KeyError(customer_id)
    return str(predict_segments(customer, artifact=artifact).iloc[0])


def generate():
    """Return the fitted segmentation artifact for API use."""
    return fit_customer_segmentation()


if __name__ == "__main__":
    result = fit_customer_segmentation()
    print(f"Best k: {result['best_k']}")
    print(f"Silhouette score: {result['silhouette_score']:.4f}")
    print(result["data"][['Customer ID', 'segment']].head().to_string(index=False))
