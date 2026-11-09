"""Unsupervised KMeans clustering and PCA dimensionality reduction."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def evaluate_k_range(
    X_scaled: np.ndarray, k_range: range = range(2, 11), seed: int = 42
) -> Dict[str, List[float]]:
    """Evaluate cluster quality across a range of K using inertia and silhouette score."""
    inertias: List[float] = []
    silhouettes: List[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, labels)))

    return {
        "k_values": list(k_range),
        "inertias": inertias,
        "silhouette_scores": silhouettes,
    }


def fit_clustering_and_segmentation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "stunting",
    k: int = 3,
    seed: int = 42,
) -> Tuple[
    StandardScaler,
    KMeans,
    PCA,
    pd.DataFrame,
    Dict[str, Any],
]:
    """
    Perform descriptive segmentation using StandardScaler, KMeans, and PCA on all districts.
    Ranks clusters strictly by average stunting rate:
        Rank 0 (lowest stunting) -> Tier 1
        Rank 1 (intermediate)    -> Tier 2
        Rank 2 (highest stunting) -> Tier 3
    Returns:
        scaler: StandardScaler fitted on full feature matrix
        kmeans: Fitted KMeans model
        pca: Fitted PCA model
        df_segmented: DataFrame with cluster, tier, and risk_segment columns
        cluster_metrics: Dict of silhouette score and tier distributions
    """
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=50)
    raw_clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    sil_score = float(silhouette_score(X_scaled, raw_clusters))

    df_out = df.copy()
    df_out["raw_cluster"] = raw_clusters
    df_out["pca1"] = X_pca[:, 0]
    df_out["pca2"] = X_pca[:, 1]

    # Rank clusters by mean stunting (lowest mean stunting = Tier 1)
    cluster_mean_stunting = (
        df_out.groupby("raw_cluster")[target_col].mean().sort_values().reset_index()
    )
    cluster_mean_stunting["rank"] = range(len(cluster_mean_stunting))

    tier_short_map = {
        int(c): f"Tier {int(r) + 1}"
        for c, r in zip(cluster_mean_stunting["raw_cluster"], cluster_mean_stunting["rank"])
    }
    tier_full_labels = {
        0: "Tier 1 — Low Risk (Developed)",
        1: "Tier 2 — Moderate Risk",
        2: "Tier 3 — High Risk",
    }
    tier_full_map = {
        int(c): tier_full_labels.get(int(r), f"Tier {int(r) + 1}")
        for c, r in zip(cluster_mean_stunting["raw_cluster"], cluster_mean_stunting["rank"])
    }

    df_out["tier"] = df_out["raw_cluster"].map(tier_short_map)
    df_out["risk_segment"] = df_out["raw_cluster"].map(tier_full_map)

    tier_counts = df_out["tier"].value_counts().to_dict()
    tier_means = df_out.groupby("tier")[target_col].mean().to_dict()

    cluster_metrics = {
        "silhouette_score": round(sil_score, 4),
        "tier_counts": tier_counts,
        "tier_stunting_means": {k: round(float(v), 2) for k, v in tier_means.items()},
        "pca_variance_explained": [round(float(v), 4) for v in pca.explained_variance_ratio_],
    }

    return scaler, kmeans, pca, df_out, cluster_metrics


def save_clustering_artifacts(
    scaler: StandardScaler,
    kmeans: KMeans,
    pca: PCA,
    models_dir: Path,
) -> None:
    """Save clustering and dimensionality reduction artifacts to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "feature_scaler.pkl")
    joblib.dump(kmeans, models_dir / "kmeans_model.pkl")
    joblib.dump(pca, models_dir / "pca_model.pkl")
