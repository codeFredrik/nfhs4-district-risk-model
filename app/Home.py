"""NFHS-4 District Child Stunting Analysis & Risk Segmentation Dashboard - Home Page."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from app.utils.data import load_districts, load_metrics

st.set_page_config(
    page_title="NFHS-4 Child Stunting Intelligence",
    page_icon="👶",
    layout="wide",
)

st.title("👶 NFHS-4 District Child Stunting Intelligence")

st.markdown(
    """
    **An empirical policy analytics platform modeling child stunting across India's districts.**
    Built on the National Family Health Survey Round 4 (NFHS-4) across 607 districts.
    Combines supervised non-linear regression with multidimensional unsupervised risk tiering.
    """
)

# Load data & metrics dynamically (strictly no hardcoded numbers)
df = load_districts()
metrics = load_metrics()

total_districts = len(df)
mean_stunting = df["stunting"].mean()

rf_metrics = metrics.get("regression", {}).get("RandomForest", {})
rf_r2 = rf_metrics.get("test_r2", 0.0)

tier_counts = metrics.get("clustering", {}).get("tier_counts", {})
tier3_count = tier_counts.get("Tier 3", (df["tier"] == "Tier 3").sum())

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Districts Analyzed",
        value=f"{total_districts:,}",
        help="Total coverage of districts across India's States and Union Territories",
    )

with col2:
    st.metric(
        label="National Mean Stunting",
        value=f"{mean_stunting:.1f}%",
        help="District-level average percentage of children under 5 stunted",
    )

with col3:
    st.metric(
        label="Tier 3 (High-Risk) Districts",
        value=f"{tier3_count:,}",
        delta=f"{(tier3_count / total_districts) * 100:.1f}% of India",
        delta_color="inverse",
        help="Districts clustered in the highest multidimensional deprivation tier",
    )

with col4:
    st.metric(
        label="Random Forest Test R²",
        value=f"{rf_r2:.3f}",
        help="Predictive power on held-out test split evaluated without data leakage",
    )

st.divider()

st.subheader("Explore the Modules")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
        - 🗺️ **[1_Map](/Map)**: Geospatial choropleth of child stunting with state and risk-tier filtering.
        - 🔍 **[2_District_Lookup](/District_Lookup)**: Drill down into individual district profiles, SHAP feature attribution waterfall, and state/national benchmark comparisons.
        """
    )

with col_b:
    st.markdown(
        """
        - 🎛️ **[3_What_If](/What_If)**: Counterfactual scenario simulation adjusting domain composite scores to observe model-predicted impacts.
        - 🚨 **[4_Priority_and_Anomalies](/Priority_and_Anomalies)**: Top priority intervention targets, developmental anomalies (such as Koppal & Yadgir), and cluster profiles.
        """
    )
