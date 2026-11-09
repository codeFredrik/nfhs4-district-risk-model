"""Tests for data cleaning, suppression marker handling, and feature construction."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from nfhs.data.cleaner import (
    DEFAULT_SUPPRESSION_MARKERS,
    clean_nfhs_data,
    impute_features,
    replace_suppression_markers,
)
from nfhs.data.feature_builder import build_composites, build_features_dataset
from nfhs.data.loader import load_raw_data, load_yaml_config


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def feature_defs(project_root: Path) -> dict:
    return load_yaml_config(project_root / "config/feature_defs.yaml")


def test_suppression_markers_become_nan() -> None:
    """Verify that survey suppression markers are coerced to NaN."""
    df_dummy = pd.DataFrame(
        {
            "col1": ["10.5", "*", "**", "na", "n/a", "N/A", "–", "-", "...", "ns", "NS", " "],
            "col2": ["normal", "text", "*", "42", "na", "", "   ", "valid", "ns", "–", "99", "ok"],
        }
    )
    df_cleaned = replace_suppression_markers(df_dummy, DEFAULT_SUPPRESSION_MARKERS)

    # Check known suppression cells are np.nan
    assert np.isnan(df_cleaned.loc[1, "col1"])
    assert np.isnan(df_cleaned.loc[2, "col1"])
    assert np.isnan(df_cleaned.loc[3, "col1"])
    assert np.isnan(df_cleaned.loc[4, "col1"])
    assert np.isnan(df_cleaned.loc[5, "col1"])
    assert np.isnan(df_cleaned.loc[6, "col1"])
    assert np.isnan(df_cleaned.loc[7, "col1"])
    assert np.isnan(df_cleaned.loc[8, "col1"])
    assert np.isnan(df_cleaned.loc[9, "col1"])
    assert np.isnan(df_cleaned.loc[10, "col1"])
    assert np.isnan(df_cleaned.loc[11, "col1"])


def test_no_nans_after_imputation(project_root: Path, feature_defs: dict) -> None:
    """Verify that after state median and fallback national median imputation, no NaNs remain in features."""
    raw_path = project_root / "Data/raw/india-nfhs4-final.xlsx"
    raw_df = load_raw_data(raw_path)
    clean_df = clean_nfhs_data(
        df_raw=raw_df,
        rename_map=feature_defs["rename_map"],
        target_col="stunting",
    )

    feature_cols = [
        c
        for c in clean_df.columns
        if c not in ["state", "district", "rural_urban", "households", "women", "men"]
    ]
    nan_count = clean_df[feature_cols].isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaNs remaining in features after imputation"


def test_composite_scores_within_bounds(project_root: Path, feature_defs: dict) -> None:
    """Verify all 7 composite domain scores fall strictly within [0, 100]."""
    raw_path = project_root / "Data/raw/india-nfhs4-final.xlsx"
    raw_df = load_raw_data(raw_path)
    clean_df = clean_nfhs_data(
        df_raw=raw_df,
        rename_map=feature_defs["rename_map"],
        target_col="stunting",
    )
    df_featured = build_features_dataset(
        df_clean=clean_df,
        composites_def=feature_defs["composites"],
        standalone_features=feature_defs["standalone_features"],
        target_col="stunting",
    )

    for comp_name in feature_defs["composites"]:
        assert comp_name in df_featured.columns
        min_val = df_featured[comp_name].min()
        max_val = df_featured[comp_name].max()
        assert min_val >= 0.0, f"Composite {comp_name} has value below 0: {min_val}"
        assert max_val <= 100.0, f"Composite {comp_name} has value above 100: {max_val}"


def test_target_never_imputed() -> None:
    """Verify that rows with missing target are dropped and the target is never imputed."""
    df_dummy = pd.DataFrame(
        {
            "state": ["A", "A", "B", "B"],
            "district": ["D1", "D2", "D3", "D4"],
            "stunting": [35.0, np.nan, 40.0, 45.0],
            "feature1": [10.0, np.nan, 20.0, np.nan],
        }
    )
    imputed = impute_features(df_dummy, target_col="stunting", id_cols=["state", "district"])
    assert len(imputed) == 3, "Row with missing target was not dropped"
    assert "D2" not in imputed["district"].values
    assert not imputed["feature1"].isna().any()
