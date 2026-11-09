"""District deep-dive, SHAP feature attribution waterfall, and benchmark table."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from app.utils.data import load_districts, load_explainer, load_features

st.set_page_config(page_title="District Lookup", page_icon="🔍", layout="wide")

st.title("🔍 District Profile & Attribution Lookup")

df = load_districts()
feat_defs = load_features()
features = feat_defs["selected_features"]
explainer, scaler, feature_names = load_explainer()

# State selectbox, then district selectbox (keyed by (state, district))
states_list = sorted(df["state"].unique())
selected_state = st.selectbox("Select State / Union Territory:", options=states_list)

state_districts = df[df["state"] == selected_state].sort_values("district")
districts_list = state_districts["district"].unique().tolist()
selected_district = st.selectbox("Select District:", options=districts_list)

# Get the unique district row using (state, district)
district_mask = (df["state"] == selected_state) & (df["district"] == selected_district)
district_row = df[district_mask].iloc[0]

# Headline district metadata cards
st.subheader(f"📍 {selected_district}, {selected_state}")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(
        "Risk Segmentation",
        value=district_row.get("tier", "N/A"),
        help=f"Full label: {district_row.get('risk_segment', 'N/A')}",
    )
with col2:
    st.metric("Actual Stunting", f"{district_row['stunting']:.1f}%")
with col3:
    st.metric("Predicted Stunting", f"{district_row.get('predicted', 0.0):.1f}%")
with col4:
    res = district_row.get("residual", 0.0)
    st.metric("Residual (Actual - Pred)", f"{res:+.1f}%")
with col5:
    split_flag = district_row.get("split", "N/A").upper()
    st.metric("Dataset Split", split_flag, help="Whether district was in training or held-out test split")

st.divider()

# Layout: Left column SHAP waterfall, Right column Benchmark table
col_shap, col_bench = st.columns([1.1, 1.0])

with col_shap:
    st.subheader("Model Feature Attribution (SHAP)")
    row_values = district_row[features].values.reshape(1, -1)
    row_scaled = scaler.transform(row_values)
    shap_explanation = explainer(row_scaled)
    shap_explanation.feature_names = features

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_explanation[0], show=False)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close("all")
    st.caption("Explains the model's prediction, not causes.")

with col_bench:
    st.subheader("Socioeconomic Benchmark Comparison")
    st.caption("Comparing district values against its State Median and the National Median.")

    state_subset = df[df["state"] == selected_state]

    bench_data = []
    for f in features:
        dist_val = district_row[f]
        state_med = state_subset[f].median()
        nat_med = df[f].median()
        bench_data.append(
            {
                "Indicator": f,
                f"{selected_district}": round(float(dist_val), 1),
                f"{selected_state} Median": round(float(state_med), 1),
                "National Median": round(float(nat_med), 1),
            }
        )

    bench_df = pd.DataFrame(bench_data)
    st.dataframe(bench_df, use_container_width=True, hide_index=True)
