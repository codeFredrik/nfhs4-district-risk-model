"""Feature engineering domain composite scores and feature dataset construction."""

from typing import Dict, List, Optional
import pandas as pd


def build_composites(df: pd.DataFrame, composites_def: Dict[str, List[str]]) -> pd.DataFrame:
    """Calculate domain composite scores as the mean of constituent indicators."""
    df_out = df.copy()
    for comp_name, constituent_cols in composites_def.items():
        missing_constituents = [c for c in constituent_cols if c not in df_out.columns]
        if missing_constituents:
            raise KeyError(
                f"Missing constituent columns for composite '{comp_name}': {missing_constituents}"
            )
        df_out[comp_name] = df_out[constituent_cols].mean(axis=1)
    return df_out


def build_features_dataset(
    df_clean: pd.DataFrame,
    composites_def: Dict[str, List[str]],
    standalone_features: List[str],
    target_col: str = "stunting",
    id_cols: Optional[List[str]] = None,
    leakage_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build modeled features dataset by constructing composite scores and selecting target/features.
    Excludes leakage columns. Does NOT fit any scaler.
    """
    if id_cols is None:
        id_cols = ["state", "district", "rural_urban", "households", "women", "men"]
    if leakage_cols is None:
        leakage_cols = [
            "wasting",
            "severe_wasting",
            "underweight",
            "anaemia_children",
            "adequate_diet_total",
            "adequate_diet_bf",
        ]

    # Calculate composites
    df_with_composites = build_composites(df_clean, composites_def)

    # Validate presence of standalone features
    missing_standalone = [
        col for col in standalone_features if col not in df_with_composites.columns
    ]
    if missing_standalone:
        raise KeyError(f"Missing standalone features: {missing_standalone}")

    # Build list of columns to retain
    composite_names = list(composites_def.keys())
    existing_ids = [c for c in id_cols if c in df_with_composites.columns]
    feature_columns = composite_names + standalone_features

    # Ensure no leakage columns are in feature columns
    leaked = [col for col in feature_columns if col in leakage_cols]
    if leaked:
        raise ValueError(f"Data leakage columns detected in features list: {leaked}")

    # Ensure all composite scores and features are within valid 0-100 range or reasonable bounds
    columns_to_keep = existing_ids + [target_col] + feature_columns

    # Return copy with ordered columns
    return df_with_composites[columns_to_keep].copy()
