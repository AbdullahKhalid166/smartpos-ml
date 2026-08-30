"""Customer segmentation for the SmartPOS RFM snapshot."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "customer_rfm_asof.csv"
MODEL_PATH = ROOT / "models" / "customer_segmentation.joblib"


def load_latest_customer_snapshot(path=DATA_PATH):
    """Load the latest customer RFM snapshot only."""
    data = pd.read_csv(path, parse_dates=["CutoffDate"]).copy()
    data = data.sort_values("CutoffDate").reset_index(drop=True)
    latest_date = data["CutoffDate"].max()
    return data[data["CutoffDate"] == latest_date].reset_index(drop=True)


def _prepare_features(frame):
    """Prepare scaled RFM features with a log-transform on Monetary if needed."""
    features = frame[["Recency", "Frequency", "Monetary"]].copy()
    monetary_skew = features["Monetary"].skew()
    if pd.notna(monetary_skew) and monetary_skew > 1:
        features["Monetary"] = np.log1p(features["Monetary"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler, features.columns.tolist()


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
    """Fit a KMeans solution and save the model and scaler to disk."""
    snapshot = load_latest_customer_snapshot() if data is None else data.copy()
    snapshot = snapshot.sort_values("Customer ID").reset_index(drop=True)
    features, scaler, feature_names = _prepare_features(snapshot)

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
        "scaler": scaler,
        "feature_names": feature_names,
        "best_k": best_k,
        "silhouette_score": float(best_score),
        "segment_labels": label_map,
        "data": snapshot,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH)
    joblib.dump({"model": best_model, "scaler": scaler, "segment_labels": label_map, "best_k": best_k}, MODEL_PATH.with_name("customer_segmentation_model.joblib"))
    return payload


def predict_segments(new_data):
    """Predict segment labels for a customer dataframe using the saved model."""
    artifact = joblib.load(MODEL_PATH)
    scaled = artifact["scaler"].transform(new_data[["Recency", "Frequency", "Monetary"]].copy())
    labels = artifact["model"].predict(scaled)
    segment_ids = pd.Series(labels, index=new_data.index)
    segment_names = segment_ids.map(artifact["segment_labels"])
    return segment_names.rename("segment")


def generate():
    """Return the fitted segmentation artifact for API use."""
    return fit_customer_segmentation()


if __name__ == "__main__":
    result = fit_customer_segmentation()
    print(f"Best k: {result['best_k']}")
    print(f"Silhouette score: {result['silhouette_score']:.4f}")
    print(result["data"][['Customer ID', 'segment']].head().to_string(index=False))
