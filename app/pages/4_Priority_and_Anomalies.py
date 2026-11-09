"""Priority districts, developmental anomalies, and risk tier summary."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from app.utils.data import load_districts, load_features

st.set_page_config(page_title="Priority & Anomalies", page_icon="🚨", layout="wide")

st.title("🚨 Priority Targeting & Developmental Anomalies")

df = load_districts()
feat_defs = load_features()
composites = list(feat_defs["composites"].keys())

tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Priority Intervention Districts",
        "⚠️ Developmental Anomalies",
        "📊 Risk Tier Profiles",
    ]
)

with tab1:
    st.subheader("Top Deprivation Districts Requiring Priority Deployment")
    top_n = st.slider("Select Top-N Districts to Display:", min_value=10, max_value=50, value=30, step=5)

    display_cols = [
        "state",
        "district",
        "stunting",
        "tier",
        "empowerment_score",
        "wash_score",
        "wealth_proxy",
        "nutrition_deprivation",
        "fertility_risk",
        "predicted",
        "residual",
    ]
    avail_cols = [c for c in display_cols if c in df.columns]

    priority_df = df[avail_cols].sort_values("stunting", ascending=False).head(top_n).reset_index(drop=True)
    priority_df.index += 1  # 1-based rank

    st.dataframe(priority_df, use_container_width=True)

    csv_data = priority_df.to_csv(index=True, index_label="Rank").encode("utf-8")
    st.download_button(
        label=f"📥 Download Top {top_n} Priority Districts CSV",
        data=csv_data,
        file_name=f"priority_districts_top_{top_n}.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Developmental Anomalies in Developed Tiers")
    # Rule computed dynamically: Tier 1 AND stunting > national mean + 1 standard deviation
    nat_mean = df["stunting"].mean()
    nat_std = df["stunting"].std()
    cutoff_threshold = nat_mean + nat_std

    st.caption(
        f"**Anomaly Detection Rule**: Districts segmented into **Tier 1 (Low Risk / Developed)** "
        f"that nonetheless suffer from child stunting exceeding the National Mean + 1 SD "
        f"({nat_mean:.1f}% + {nat_std:.1f}% = **>{cutoff_threshold:.1f}%**)."
    )

    anomaly_mask = (df["tier"] == "Tier 1") & (df["stunting"] > cutoff_threshold)
    anomalies_df = (
        df[anomaly_mask]
        .sort_values("stunting", ascending=False)[
            [
                "state",
                "district",
                "stunting",
                "tier",
                "predicted",
                "residual",
                "empowerment_score",
                "wash_score",
                "wealth_proxy",
            ]
        ]
        .reset_index(drop=True)
    )
    anomalies_df.index += 1

    col_anom1, col_anom2 = st.columns([1, 2])
    with col_anom1:
        st.metric("Total Anomalous Districts", len(anomalies_df))
        st.info(
            "These districts exhibit high overall institutional, educational, and wealth infrastructure, "
            "yet suffer from acute localized stunting pockets (e.g. marginalized tribal or migrant clusters)."
        )
    with col_anom2:
        st.dataframe(anomalies_df, use_container_width=True)

with tab3:
    st.subheader("Comprehensive Segment Profiles Across Tiers")
    st.caption("Aggregated empirical means across all 7 composite domain indicators.")

    tier_order = ["Tier 1", "Tier 2", "Tier 3"]
    tier_summary = df.groupby("tier").agg(
        Districts=("district", "count"),
        Mean_Stunting=("stunting", lambda x: round(float(x.mean()), 1)),
        **{comp: (comp, lambda x: round(float(x.mean()), 1)) for comp in composites},
    ).reindex(tier_order)

    st.dataframe(tier_summary, use_container_width=True)
