"""Data cleaning and preprocessing pipeline."""

import re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

DEFAULT_SUPPRESSION_MARKERS = [
    "*",
    "**",
    "***",
    "na",
    "n/a",
    "N/A",
    "NA",
    "–",
    "-",
    "...",
    " ",
    "",
    "ns",
    "NS",
]


def clean_col_name(name: str) -> str:
    """Standardize column names to clean snake_case."""
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def deduplicate_column_names(columns: List[str]) -> List[str]:
    """Append incrementing suffixes (_1, _2...) to duplicate column names."""
    counts: Dict[str, int] = {}
    deduped: List[str] = []
    for c in columns:
        if c in counts:
            counts[c] += 1
            deduped.append(f"{c}_{counts[c]}")
        else:
            counts[c] = 0
            deduped.append(c)
    return deduped


def extract_headers_and_slice(
    df_raw: pd.DataFrame, header_row: int = 2, data_row: int = 3
) -> pd.DataFrame:
    """Extract column headers and slice data rows from raw NFHS Excel structure."""
    raw_headers = df_raw.iloc[header_row].tolist()
    cleaned_headers = [clean_col_name(c) for c in raw_headers]
    deduped_headers = deduplicate_column_names(cleaned_headers)

    df = df_raw.iloc[data_row:].copy().reset_index(drop=True)
    df.columns = deduped_headers
    return df


def replace_suppression_markers(
    df: pd.DataFrame, markers: Optional[List[str]] = None
) -> pd.DataFrame:
    """Convert survey suppression and missingness strings to np.nan."""
    if markers is None:
        markers = DEFAULT_SUPPRESSION_MARKERS
    df_out = df.copy()
    df_out.replace(markers, np.nan, inplace=True)
    df_out = df_out.map(
        lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
    )
    return df_out


def filter_missingness(
    df: pd.DataFrame,
    threshold: float = 0.40,
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Coerce non-ID columns to numeric and drop rows/cols exceeding missing threshold."""
    if id_cols is None:
        id_cols = ["state", "district", "ruralurban_classification"]

    df_out = df.copy()
    valid_id_cols = [c for c in id_cols if c in df_out.columns]
    feature_cols = [c for c in df_out.columns if c not in valid_id_cols]

    for col in feature_cols:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    # Drop columns exceeding missing threshold
    col_missing_frac = df_out[feature_cols].isna().mean()
    cols_to_drop = col_missing_frac[col_missing_frac > threshold].index.tolist()
    df_out.drop(columns=cols_to_drop, inplace=True)
    feature_cols = [c for c in feature_cols if c not in cols_to_drop]

    # Drop rows exceeding missing threshold
    row_missing_frac = df_out[feature_cols].isna().mean(axis=1)
    df_out = df_out[row_missing_frac <= threshold].copy().reset_index(drop=True)

    return df_out


def drop_subgroup_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Total district block by dropping Urban/Rural split duplicate columns (_2, _3)."""
    cols_to_keep = [c for c in df.columns if not c.endswith(("_2", "_3"))]
    return df[cols_to_keep].copy()


def rename_indicators(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    """Rename raw indicator column names to standardized semantic names."""
    return df.rename(columns=rename_map)


def impute_features(
    df: pd.DataFrame,
    target_col: str = "stunting",
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Impute missing feature values using state medians with national median fallback.
    Never impute target column; rows with missing target are dropped prior to feature imputation.
    """
    if id_cols is None:
        id_cols = ["state", "district", "rural_urban", "households", "women", "men"]

    df_out = df.copy()

    # Drop rows where target is missing
    if target_col in df_out.columns:
        df_out = df_out[df_out[target_col].notna()].copy().reset_index(drop=True)

    existing_id_cols = [c for c in id_cols if c in df_out.columns]
    features_to_impute = [
        c for c in df_out.columns if c not in existing_id_cols and c != target_col
    ]

    # State-level median imputation
    df_out[features_to_impute] = df_out.groupby("state")[features_to_impute].transform(
        lambda s: s.fillna(s.median())
    )

    # National/global median fallback for any remaining NaNs
    df_out[features_to_impute] = df_out[features_to_impute].fillna(
        df_out[features_to_impute].median()
    )

    return df_out


def clean_nfhs_data(
    df_raw: pd.DataFrame,
    rename_map: Dict[str, str],
    target_col: str = "stunting",
    id_cols: Optional[List[str]] = None,
    missing_threshold: float = 0.40,
) -> pd.DataFrame:
    """Orchestrate end-to-end data cleaning pipeline returning cleaned, imputed DataFrame."""
    df_sliced = extract_headers_and_slice(df_raw, header_row=2, data_row=3)
    df_clean_na = replace_suppression_markers(df_sliced)
    df_filtered = filter_missingness(df_clean_na, threshold=missing_threshold)
    df_total_block = drop_subgroup_duplicates(df_filtered)
    df_renamed = rename_indicators(df_total_block, rename_map=rename_map)
    df_imputed = impute_features(df_renamed, target_col=target_col, id_cols=id_cols)
    return df_imputed
