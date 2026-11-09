"""Data loading, caching, and model access layer for Streamlit dashboard."""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
import streamlit as st
import yaml

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nfhs.reporting.geo_join import perform_geo_join


def get_repo_root() -> Path:
    """Return the absolute Path to the repository root directory."""
    return REPO_ROOT


def load_config() -> Dict[str, Any]:
    """Load config/config.yaml using pathlib."""
    config_path = REPO_ROOT / "config/config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features() -> Dict[str, Any]:
    """Load feature definitions from config/feature_defs.yaml."""
    feat_path = REPO_ROOT / "config/feature_defs.yaml"
    with feat_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metrics() -> Dict[str, Any]:
    """Load evaluation metrics from Outputs/reports/metrics.json."""
    metrics_path = REPO_ROOT / "Outputs/reports/metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_districts() -> pd.DataFrame:
    """Load the final complete district analysis dataset."""
    csv_path = REPO_ROOT / "Outputs/reports/nfhs4_complete_analysis.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"District dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


@st.cache_resource
def load_model() -> Pipeline:
    """Load the fitted Random Forest regression pipeline (scaler + RF)."""
    model_path = REPO_ROOT / "Outputs/models/rf_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


@st.cache_resource
def load_explainer() -> Tuple[shap.TreeExplainer, Any, List[str]]:
    """
    Load SHAP TreeExplainer on the Random Forest regressor inside the pipeline.
    Returns:
        explainer: Fitted SHAP TreeExplainer
        scaler: The pipeline's fitted StandardScaler
        feature_names: List of 15 selected feature names
    """
    pipeline = load_model()
    scaler = pipeline.named_steps["scaler"]
    rf = pipeline.named_steps["model"]
    feat_defs = load_features()
    feature_names = feat_defs["selected_features"]
    explainer = shap.TreeExplainer(rf)
    return explainer, scaler, feature_names


@st.cache_data
def load_geo_data() -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load district boundary GeoJSON and join with district data.
    Downloads the boundary GeoJSON on first run if not present locally.
    Returns:
        geojson: GeoJSON dict
        df_geo: DataFrame with matched 'geo_feature_id'
        stats: Match statistics dict
    """
    geojson_path = REPO_ROOT / "Data/geo/india_districts.geojson"
    overrides_path = REPO_ROOT / "Data/geo/name_overrides.csv"
    unmatched_path = REPO_ROOT / "Outputs/reports/unmatched_districts.csv"

    try:
        from nfhs.reporting.geo_join import download_geojson_if_missing
        geojson_path = download_geojson_if_missing(geojson_path)
    except Exception as e:
        if not geojson_path.exists():
            return None, None, {"error": f"Failed to retrieve boundary GeoJSON: {e}"}

    if not geojson_path.exists():
        return None, None, {"error": "GeoJSON file not found"}

    with geojson_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    # Assign integer id to each feature for robust location matching in Plotly
    for idx, feature in enumerate(geojson.get("features", [])):
        feature["id"] = idx

    df = load_districts()
    df_joined, stats, _ = perform_geo_join(
        df=df,
        geojson_path=geojson_path,
        overrides_path=overrides_path,
        unmatched_output_path=unmatched_path,
    )
    return geojson, df_joined, stats
