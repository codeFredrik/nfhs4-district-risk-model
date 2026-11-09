"""SHAP TreeExplainer feature attribution and explainability."""

from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def compute_rf_shap_values(
    rf_pipeline: Pipeline,
    X: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values from the Random Forest model within the pipeline.
    Transforms features using the fitted pipeline scaler before TreeExplainer.
    Returns:
        shap_values: Array of SHAP attribution values
        importance_df: DataFrame of mean absolute SHAP values sorted descending
    """
    scaler = rf_pipeline.named_steps["scaler"]
    rf_model = rf_pipeline.named_steps["model"]

    # Transform features with the pipeline scaler
    X_scaled = scaler.transform(X)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_scaled)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean_SHAP": mean_abs_shap,
            }
        )
        .sort_values("Mean_SHAP", ascending=False)
        .reset_index(drop=True)
    )

    return shap_values, importance_df


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    output_path: Path,
) -> None:
    """Plot and save SHAP summary plot (beeswarm)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X, feature_names=feature_names, show=False, plot_size=(10, 7)
    )
    plt.title(
        "SHAP Summary Plot — Feature Impact on Child Stunting",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_shap_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot and save mean absolute SHAP importance horizontal bar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = importance_df.sort_values("Mean_SHAP", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["Feature"], plot_df["Mean_SHAP"], color="#3498DB", edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value| (average impact on stunting prediction)")
    ax.set_title(
        "Feature Importance via SHAP\n(Random Forest on NFHS-4 District Data)",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
