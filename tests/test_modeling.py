"""Tests for modeling pipelines, leakage-free scaling, and clustering segmentation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from nfhs.data.loader import load_yaml_config
from nfhs.models.clusterer import fit_clustering_and_segmentation
from nfhs.models.regression import train_and_evaluate_regression
from nfhs.pipeline import clean_stage, engineer_stage


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def feature_defs(project_root: Path) -> dict:
    return load_yaml_config(project_root / "config/feature_defs.yaml")


@pytest.fixture(scope="module")
def prepared_dataset(project_root: Path) -> pd.DataFrame:
    df_clean = clean_stage(project_root)
    return engineer_stage(df_clean, project_root)


def test_scaler_fit_on_train_only(prepared_dataset: pd.DataFrame, feature_defs: dict) -> None:
    """Assert scaler mean differs from full-data mean and strictly equals train mean (zero leakage)."""
    feature_cols = feature_defs["selected_features"]
    pipelines, _, _, split_data = train_and_evaluate_regression(
        df=prepared_dataset,
        feature_cols=feature_cols,
        target_col="stunting",
        test_size=0.2,
        seed=42,
    )

    X_train, X_test, y_train, y_test, train_idx, test_idx = split_data
    X_full = prepared_dataset[feature_cols].values

    # Check across pipelines (OLS, Ridge, Lasso, RandomForest)
    for name, pipe in pipelines.items():
        scaler = pipe.named_steps["scaler"]
        train_mean = X_train.mean(axis=0)
        full_mean = X_full.mean(axis=0)

        # Scaler mean must equal training split mean
        assert np.allclose(
            scaler.mean_, train_mean, atol=1e-6
        ), f"{name}: Scaler mean does not match X_train mean"

        # Scaler mean must differ from full dataset mean (proves test data was not seen)
        assert not np.allclose(
            scaler.mean_, full_mean, atol=1e-4
        ), f"{name}: Data leakage detected! Scaler mean equals full dataset mean"


def test_random_forest_performance(prepared_dataset: pd.DataFrame, feature_defs: dict) -> None:
    """Assert Random Forest test R2 > 0.60."""
    _, metrics, _, _ = train_and_evaluate_regression(
        df=prepared_dataset,
        feature_cols=feature_defs["selected_features"],
        target_col="stunting",
        test_size=0.2,
        seed=42,
    )

    rf_r2 = metrics["RandomForest"]["test_r2"]
    assert rf_r2 > 0.60, f"Random Forest test R2 is {rf_r2}, expected > 0.60"


def test_kmeans_clustering_and_tier_ordering(
    prepared_dataset: pd.DataFrame, feature_defs: dict
) -> None:
    """Assert KMeans returns exactly 3 clusters and tier 1 mean stunting < tier 2 < tier 3."""
    _, kmeans, _, df_segmented, cluster_metrics = fit_clustering_and_segmentation(
        df=prepared_dataset,
        feature_cols=feature_defs["selected_features"],
        target_col="stunting",
        k=3,
        seed=42,
    )

    # Assert exactly 3 clusters
    assert kmeans.n_clusters == 3
    assert len(df_segmented["tier"].unique()) == 3

    # Assert tier stunting monotonicity: Tier 1 < Tier 2 < Tier 3
    t1_mean = df_segmented[df_segmented["tier"] == "Tier 1"]["stunting"].mean()
    t2_mean = df_segmented[df_segmented["tier"] == "Tier 2"]["stunting"].mean()
    t3_mean = df_segmented[df_segmented["tier"] == "Tier 3"]["stunting"].mean()

    assert (
        t1_mean < t2_mean < t3_mean
    ), f"Tier ordering violation: Tier 1 ({t1_mean:.2f}) < Tier 2 ({t2_mean:.2f}) < Tier 3 ({t3_mean:.2f})"
