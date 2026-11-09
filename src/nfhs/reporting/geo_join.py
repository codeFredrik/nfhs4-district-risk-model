"""Geospatial matching between NFHS district data and boundary GeoJSON."""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from rapidfuzz import fuzz, process

STATE_CODE_TO_GEOJSON_NAMES: Dict[str, List[str]] = {
    "AN": ["Andaman and Nicobar"],
    "AP": ["Andhra Pradesh"],
    "AR": ["Arunachal Pradesh"],
    "AS": ["Assam"],
    "BR": ["Bihar"],
    "CT": ["Chhattisgarh"],
    "DD": ["Daman and Diu", "Dadra and Nagar Haveli"],
    "GA": ["Goa"],
    "GJ": ["Gujarat"],
    "HP": ["Himachal Pradesh"],
    "HR": ["Haryana"],
    "JH": ["Jharkhand"],
    "JK": ["Jammu and Kashmir"],
    "KA": ["Karnataka"],
    "KL": ["Kerala"],
    "MH": ["Maharashtra"],
    "ML": ["Meghalaya"],
    "MN": ["Manipur"],
    "MP": ["Madhya Pradesh"],
    "MZ": ["Mizoram"],
    "NL": ["Nagaland"],
    "OR": ["Orissa"],
    "PB": ["Punjab"],
    "PY": ["Puducherry"],
    "RJ": ["Rajasthan"],
    "SK": ["Sikkim"],
    "TG": ["Andhra Pradesh"],  # In pre-bifurcation geography, Telangana was part of AP
    "TN": ["Tamil Nadu"],
    "TR": ["Tripura"],
    "UP": ["Uttar Pradesh"],
    "UT": ["Uttaranchal"],
    "WB": ["West Bengal"],
}


def normalize_geo_name(name: Any) -> str:
    """Normalize district or state string: lowercase, remove punctuation, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    name_clean = name.lower().strip()
    name_clean = re.sub(r"[^\w\s]", "", name_clean)
    name_clean = re.sub(r"\s+", " ", name_clean)
    return name_clean.strip()


GEOJSON_UPSTREAM_URL: str = (
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
)


def download_geojson_if_missing(
    geojson_path: Path, url: str = GEOJSON_UPSTREAM_URL
) -> Path:
    """Download boundary GeoJSON on first run if not present locally."""
    geojson_path = Path(geojson_path)
    if geojson_path.exists() and geojson_path.stat().st_size > 0:
        return geojson_path

    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    print(f"Downloading boundary GeoJSON from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        content = resp.read()
        with geojson_path.open("wb") as f:
            f.write(content)
    print(f"Successfully downloaded GeoJSON to {geojson_path} ({len(content)} bytes)")
    return geojson_path


def load_overrides(overrides_path: Path) -> Dict[Tuple[str, str], str]:
    """Load manual name overrides CSV into a dictionary: (state, district) -> target_name."""
    if not overrides_path.exists():
        return {}
    df_overrides = pd.read_csv(overrides_path)
    overrides = {}
    for _, row in df_overrides.iterrows():
        st = str(row["state"]).strip()
        dist = str(row["district"]).strip()
        geo_dist = str(row["geo_district"]).strip()
        overrides[(st, dist)] = geo_dist
    return overrides


def perform_geo_join(
    df: pd.DataFrame,
    geojson_path: Path,
    overrides_path: Optional[Path] = None,
    unmatched_output_path: Optional[Path] = None,
    fuzzy_threshold: float = 90.0,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Join NFHS district metrics with boundary GeoJSON.
    Returns:
        df_matched: Copy of df with 'geo_feature_id' and 'matched_geo_name'
        match_stats: Summary dict (total, matched, unmatched, match_rate)
        unmatched_records: List of unmatched district dicts
    """
    geojson_path = download_geojson_if_missing(geojson_path)
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    with geojson_path.open("r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    # Build GeoJSON lookup table
    geo_records = []
    for idx, feature in enumerate(geojson_data.get("features", [])):
        props = feature.get("properties", {})
        st_name = props.get("NAME_1", "")
        dist_name = props.get("NAME_2", "")
        geo_records.append(
            {
                "geo_id": idx,
                "geo_state": st_name,
                "geo_district": dist_name,
                "norm_dist": normalize_geo_name(dist_name),
            }
        )
    geo_df = pd.DataFrame(geo_records)

    overrides = {}
    if overrides_path is not None and overrides_path.exists():
        overrides = load_overrides(overrides_path)

    df_out = df.copy()
    df_out["geo_feature_id"] = None
    df_out["matched_geo_name"] = None

    matched_count = 0
    unmatched_list: List[Dict[str, Any]] = []

    for idx, row in df_out.iterrows():
        st = str(row["state"]).strip()
        dist = str(row["district"]).strip()
        norm_d = normalize_geo_name(dist)

        candidate_states = STATE_CODE_TO_GEOJSON_NAMES.get(st, [])
        cand_geo = geo_df[geo_df["geo_state"].isin(candidate_states)]

        matched_geo_id = None
        matched_name = None

        # 1. Manual override lookup
        if (st, dist) in overrides:
            target_norm = normalize_geo_name(overrides[(st, dist)])
            match_row = cand_geo[cand_geo["norm_dist"] == target_norm]
            if len(match_row) > 0:
                matched_geo_id = match_row.iloc[0]["geo_id"]
                matched_name = match_row.iloc[0]["geo_district"]

        # 2. Exact match within candidate state
        if matched_geo_id is None and len(cand_geo) > 0:
            exact_match = cand_geo[cand_geo["norm_dist"] == norm_d]
            if len(exact_match) > 0:
                matched_geo_id = exact_match.iloc[0]["geo_id"]
                matched_name = exact_match.iloc[0]["geo_district"]

        # 3. RapidFuzz match within state
        if matched_geo_id is None and len(cand_geo) > 0:
            choices = cand_geo["norm_dist"].tolist()
            best = process.extractOne(norm_d, choices, scorer=fuzz.ratio)
            if best and best[1] >= fuzzy_threshold:
                match_row = cand_geo[cand_geo["norm_dist"] == best[0]]
                if len(match_row) > 0:
                    matched_geo_id = match_row.iloc[0]["geo_id"]
                    matched_name = match_row.iloc[0]["geo_district"]

        if matched_geo_id is not None:
            df_out.at[idx, "geo_feature_id"] = int(matched_geo_id)
            df_out.at[idx, "matched_geo_name"] = matched_name
            matched_count += 1
        else:
            unmatched_list.append(
                {
                    "state": st,
                    "district": dist,
                    "stunting": row.get("stunting", None),
                    "tier": row.get("tier", None),
                }
            )

    match_rate = (matched_count / len(df_out)) * 100.0 if len(df_out) > 0 else 0.0
    match_stats = {
        "total_districts": len(df_out),
        "matched_districts": matched_count,
        "unmatched_districts": len(unmatched_list),
        "match_rate_pct": round(match_rate, 2),
    }

    print(
        f"Geospatial Join: {matched_count}/{len(df_out)} matched ({match_rate:.2f}%)"
    )

    if unmatched_output_path is not None:
        unmatched_output_path = Path(unmatched_output_path)
        unmatched_output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(unmatched_list).to_csv(unmatched_output_path, index=False)

    return df_out, match_stats, unmatched_list
