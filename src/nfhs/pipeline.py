"""End-to-end pipeline execution and orchestration."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from nfhs.data.cleaner import clean_nfhs_data
from nfhs.data.feature_builder import build_features_dataset
from nfhs.data.loader import load_csv_data, load_raw_data, load_yaml_config
from nfhs.models.clusterer import (
    evaluate_k_range,
    fit_clustering_and_segmentation,
    save_clustering_artifacts,
)
from nfhs.models.explainability import compute_rf_shap_values
from nfhs.models.regression import save_regression_artifacts, train_and_evaluate_regression
from nfhs.reporting.policy_matrix import generate_priority_districts
from nfhs.reporting.visualizer import generate_all_figures


def get_default_paths(config: Dict[str, Any], project_root: Path) -> Dict[str, Path]:
    """Resolve all relative paths from config relative to project_root."""
    cfg_paths = config.get("paths", {})
    return {
        "raw_data": project_root / cfg_paths.get("raw_data", "Data/raw/india-nfhs4-final.xlsx"),
        "processed_dir": project_root / cfg_paths.get("processed_dir", "Data/processed"),
        "processed_data": project_root / cfg_paths.get("processed_data", "Data/processed/nfhs4_clean.csv"),
        "final_analysis": project_root / cfg_paths.get("final_analysis", "Outputs/reports/nfhs4_complete_analysis.csv"),
        "priority_districts": project_root / cfg_paths.get("priority_districts", "Outputs/reports/priority_districts.csv"),
        "metrics_json": project_root / cfg_paths.get("metrics_json", "Outputs/reports/metrics.json"),
        "models_dir": project_root / cfg_paths.get("models_dir", "Outputs/models"),
        "figures_dir": project_root / cfg_paths.get("figures_dir", "Outputs/figures"),
        "reports_dir": project_root / cfg_paths.get("reports_dir", "Outputs/reports"),
    }


def clean_stage(
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Execute Data Cleaning stage."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    config = load_yaml_config(project_root / "config/config.yaml")
    feature_defs = load_yaml_config(project_root / "config/feature_defs.yaml")
    paths = get_default_paths(config, project_root)

    raw_df = load_raw_data(paths["raw_data"])
    clean_df = clean_nfhs_data(
        df_raw=raw_df,
        rename_map=feature_defs["rename_map"],
        target_col=config.get("pipeline", {}).get("target", "stunting"),
        missing_threshold=config.get("pipeline", {}).get("missing_threshold", 0.40),
    )

    paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(paths["processed_data"], index=False)
    print(f"Cleaned data saved to {paths['processed_data']} (Shape: {clean_df.shape})")
    return clean_df


def engineer_stage(
    df_clean: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Execute Feature Engineering stage."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    config = load_yaml_config(project_root / "config/config.yaml")
    feature_defs = load_yaml_config(project_root / "config/feature_defs.yaml")
    paths = get_default_paths(config, project_root)

    if df_clean is None:
        df_clean = load_csv_data(paths["processed_data"])

    df_featured = build_features_dataset(
        df_clean=df_clean,
        composites_def=feature_defs["composites"],
        standalone_features=feature_defs["standalone_features"],
        target_col=config.get("pipeline", {}).get("target", "stunting"),
        id_cols=feature_defs.get("id_columns"),
        leakage_cols=feature_defs.get("leakage_columns"),
    )
    print(f"Features dataset built (Shape: {df_featured.shape})")
    return df_featured


def train_stage(
    df_featured: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Execute Model Training and Evaluation stage."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    config = load_yaml_config(project_root / "config/config.yaml")
    feature_defs = load_yaml_config(project_root / "config/feature_defs.yaml")
    paths = get_default_paths(config, project_root)

    if df_featured is None:
        df_clean = load_csv_data(paths["processed_data"])
        df_featured = engineer_stage(df_clean, project_root)

    pipelines, reg_metrics, df_pred, _ = train_and_evaluate_regression(
        df=df_featured,
        feature_cols=feature_defs["selected_features"],
        target_col=config.get("pipeline", {}).get("target", "stunting"),
        test_size=config.get("pipeline", {}).get("test_size", 0.2),
        seed=config.get("pipeline", {}).get("seed", 42),
    )

    save_regression_artifacts(pipelines, paths["models_dir"])
    print("Regression models trained and saved.")
    return pipelines, reg_metrics, df_pred


def segment_stage(
    df_pred: pd.DataFrame,
    project_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Execute Unsupervised Clustering and Segmentation stage."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    config = load_yaml_config(project_root / "config/config.yaml")
    feature_defs = load_yaml_config(project_root / "config/feature_defs.yaml")
    paths = get_default_paths(config, project_root)

    scaler, kmeans, pca, df_segmented, cluster_metrics = fit_clustering_and_segmentation(
        df=df_pred,
        feature_cols=feature_defs["selected_features"],
        target_col=config.get("pipeline", {}).get("target", "stunting"),
        k=config.get("pipeline", {}).get("k", 3),
        seed=config.get("pipeline", {}).get("seed", 42),
    )

    save_clustering_artifacts(scaler, kmeans, pca, paths["models_dir"])

    # Evaluate K=2..10 range
    X_scaled = scaler.transform(df_pred[feature_defs["selected_features"]].values)
    k_eval = evaluate_k_range(X_scaled, k_range=range(2, 11), seed=config.get("pipeline", {}).get("seed", 42))

    print("Clustering and segmentation complete.")
    return df_segmented, cluster_metrics, k_eval


def report_stage(
    df_clean: pd.DataFrame,
    df_segmented: pd.DataFrame,
    pipelines: Dict[str, Any],
    metrics: Dict[str, Any],
    k_eval: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> None:
    """Execute Visualizations and Priority District Reporting stage."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    config = load_yaml_config(project_root / "config/config.yaml")
    feature_defs = load_yaml_config(project_root / "config/feature_defs.yaml")
    paths = get_default_paths(config, project_root)

    # Compute SHAP
    rf_pipe = pipelines["RandomForest"]
    X_full = df_segmented[feature_defs["selected_features"]].values
    shap_vals, imp_df = compute_rf_shap_values(
        rf_pipe, X_full, feature_defs["selected_features"]
    )

    # Export Top 30 Priority Districts
    generate_priority_districts(
        df=df_segmented,
        top_n=30,
        target_col=config.get("pipeline", {}).get("target", "stunting"),
        output_path=paths["priority_districts"],
    )

    # Generate 18 figures
    pca_var = metrics.get("clustering", {}).get("pca_variance_explained", [0.45, 0.10])
    generate_all_figures(
        df_clean=df_clean,
        df_analyzed=df_segmented,
        figures_dir=paths["figures_dir"],
        metrics=metrics.get("regression", {}),
        k_eval=k_eval,
        shap_values=shap_vals,
        importance_df=imp_df,
        feature_cols=feature_defs["selected_features"],
        composite_cols=list(feature_defs["composites"].keys()),
        var_exp=pca_var,
    )

    # Save metrics.json
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    with paths["metrics_json"].open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save final complete analysis dataset
    paths["final_analysis"].parent.mkdir(parents=True, exist_ok=True)
    df_segmented.to_csv(paths["final_analysis"], index=False)
    print(f"Analysis dataset exported to {paths['final_analysis']}")
    print(f"Metrics saved to {paths['metrics_json']}")


def run_all_pipeline(project_root: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute full pipeline end-to-end."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    print("=== [1/5] Executing Data Cleaning ===")
    df_clean = clean_stage(project_root)

    print("=== [2/5] Executing Feature Engineering ===")
    df_featured = engineer_stage(df_clean, project_root)

    print("=== [3/5] Executing Regression Modeling ===")
    pipelines, reg_metrics, df_pred = train_stage(df_featured, project_root)

    print("=== [4/5] Executing Clustering & Risk Segmentation ===")
    df_segmented, cluster_metrics, k_eval = segment_stage(df_pred, project_root)

    combined_metrics = {
        "regression": reg_metrics,
        "clustering": cluster_metrics,
    }

    print("=== [5/5] Executing Reporting & Visualization ===")
    report_stage(
        df_clean=df_clean,
        df_segmented=df_segmented,
        pipelines=pipelines,
        metrics=combined_metrics,
        k_eval=k_eval,
        project_root=project_root,
    )

    print("=== Pipeline Execution Complete ===")
    return df_segmented, combined_metrics
