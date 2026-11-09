"""Geospatial choropleth map of district stunting rates."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plotly.express as px
import streamlit as st

from app.utils.data import load_geo_data

st.set_page_config(page_title="District Stunting Map", page_icon="🗺️", layout="wide")

st.title("🗺️ District-Level Child Stunting Choropleth")
st.caption(
    "Interactive spatial distribution of child stunting rates based on a boundary file with 594 polygons that predates some district splits."
)

# Load boundary and joined data
geojson, df_geo, stats = load_geo_data()

if geojson is None or df_geo is None:
    st.error("Boundary GeoJSON could not be loaded. Please verify Data/geo/india_districts.geojson.")
    st.stop()

# Sidebar filters
st.sidebar.header("Map Filters")

all_states = sorted(df_geo["state"].unique())
selected_states = st.sidebar.multiselect(
    "Filter by State / UT",
    options=all_states,
    default=[],
    help="Leave empty to display all States and UTs",
)

all_tiers = sorted(df_geo["tier"].dropna().unique())
selected_tiers = st.sidebar.multiselect(
    "Filter by Risk Tier",
    options=all_tiers,
    default=all_tiers,
    help="Filter districts by socioeconomic risk segmentation tier",
)

# Filter dataframe
filtered_df = df_geo.copy()
if selected_states:
    filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]
if selected_tiers:
    filtered_df = filtered_df[filtered_df["tier"].isin(selected_tiers)]

# Separate mapped from unmapped
mapped_df = filtered_df[filtered_df["geo_feature_id"].notna()].copy()
unmapped_df = filtered_df[filtered_df["geo_feature_id"].isna()].copy()
mapped_df["geo_feature_id"] = mapped_df["geo_feature_id"].astype(int)

# Choropleth visualization
if len(mapped_df) > 0:
    fig = px.choropleth_map(
        mapped_df,
        geojson=geojson,
        locations="geo_feature_id",
        featureidkey="id",
        color="stunting",
        color_continuous_scale="Reds",
        range_color=(10, 65),
        map_style="carto-positron",
        center={"lat": 22.5, "lon": 82.0},
        zoom=3.8,
        hover_name="district",
        hover_data={
            "state": True,
            "stunting": ":.1f%",
            "tier": True,
            "predicted": ":.1f%",
            "residual": ":.1f%",
            "geo_feature_id": False,
        },
        labels={"stunting": "Stunting (%)"},
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=650,
        coloraxis_colorbar=dict(
            title="Stunting (%)",
            ticks="outside",
            tickformat=".0f",
            lenmode="fraction",
            len=0.75,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No districts matching current filter selections have geographic coordinates.")

# Reporting unmatched districts (never silently drop)
unmatched_count = len(unmapped_df)
total_selected = len(filtered_df)

col_stat1, col_stat2 = st.columns([1, 2])
with col_stat1:
    st.info(f"**{unmatched_count} of {total_selected}** filtered districts not mapped (using a boundary file with 594 polygons that predates some district splits).")

with col_stat2:
    if "Yadgir" in unmapped_df["district"].values:
        st.warning(
            "📍 **Note on Yadgir**: Yadgir (Karnataka, 55.5% stunting, Tier 1 anomaly) was formed in 2009 from Gulbarga "
            "and does not have an individual polygon in this boundary file with 594 polygons that predates some district splits, so it cannot be plotted on the choropleth map. "
            "It is fully tracked and analyzed in tabular reports and the **🚨 Priority & Anomalies** module."
        )

if unmatched_count > 0:
    with st.expander(f"View list of all {unmatched_count} unmapped districts"):
        st.dataframe(
            unmapped_df[["state", "district", "stunting", "tier"]].sort_values(
                "stunting", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )
