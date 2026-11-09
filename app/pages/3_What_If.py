"""Counterfactual what-if policy scenario simulator."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils.data import load_districts, load_features, load_model

st.set_page_config(page_title="What-If Simulator", page_icon="🎛️", layout="wide")

st.title("🎛️ Counterfactual What-If Policy Simulator")

st.warning(
    "⚠️ **Associational model.** Composite scores move together in real data, so a single-slider change "
    "is not a realistic intervention or a causal effect."
)

df = load_districts()
feat_defs = load_features()
pipeline = load_model()
features = feat_defs["selected_features"]

# District selection keyed by (state, district)
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    state_list = sorted(df["state"].unique())
    selected_state = st.selectbox("Select State:", options=state_list)
with col_sel2:
    districts_list = df[df["state"] == selected_state]["district"].sort_values().unique().tolist()
    selected_district = st.selectbox("Select District:", options=districts_list)

district_mask = (df["state"] == selected_state) & (df["district"] == selected_district)
district_row = df[district_mask].iloc[0]

# Slider features: 7 composite scores + oop_delivery_cost
slider_features = [
    "wash_score",
    "empowerment_score",
    "healthcare_score",
    "immunization_score",
    "nutrition_deprivation",
    "wealth_proxy",
    "fertility_risk",
    "oop_delivery_cost",
]

# Session state reset mechanism
reset_clicked = st.button("🔄 Reset Sliders to District Baseline")
if reset_clicked:
    for sf in slider_features:
        st.session_state[f"slider_{selected_state}_{selected_district}_{sf}"] = float(district_row[sf])
    st.rerun()

st.subheader(f"Adjust Domain Indicators for {selected_district} ({selected_state})")
st.caption("Sliders are bounded to observed empirical minimum and maximum across India.")

col_left, col_right = st.columns(2)
scenario_values = {}

for idx, sf in enumerate(slider_features):
    target_col = col_left if idx % 2 == 0 else col_right
    obs_min = float(df[sf].min())
    obs_max = float(df[sf].max())
    cur_val = float(district_row[sf])

    slider_key = f"slider_{selected_state}_{selected_district}_{sf}"
    with target_col:
        val = st.slider(
            label=f"{sf.replace('_', ' ').title()}",
            min_value=round(obs_min, 1),
            max_value=round(obs_max, 1),
            value=float(st.session_state.get(slider_key, cur_val)),
            step=0.5 if sf != "oop_delivery_cost" else 25.0,
            key=slider_key,
        )
        scenario_values[sf] = val

# Assemble full 15-feature vectors
baseline_vector = np.array([float(district_row[f]) for f in features]).reshape(1, -1)
scenario_vector = np.array(
    [scenario_values[f] if f in scenario_values else float(district_row[f]) for f in features]
).reshape(1, -1)

# Predictions pass through saved pipeline
pred_baseline = float(pipeline.predict(baseline_vector)[0])
pred_scenario = float(pipeline.predict(scenario_vector)[0])
delta = pred_scenario - pred_baseline

st.divider()
st.subheader("Simulation Outcome")

res_col1, res_col2, res_col3, res_col4 = st.columns(4)
with res_col1:
    st.metric("Actual Observed Stunting", f"{district_row['stunting']:.1f}%")
with res_col2:
    st.metric("Model Baseline Prediction", f"{pred_baseline:.1f}%")
with res_col3:
    st.metric("Scenario Prediction", f"{pred_scenario:.1f}%")
with res_col4:
    st.metric(
        "Estimated Delta",
        f"{delta:+.2f}% pts",
        delta=f"{delta:.2f}% pts",
        delta_color="inverse",
    )

# Bar chart of the comparison
chart_data = pd.DataFrame(
    {
        "Condition": ["Actual Observed", "Model Baseline", "Scenario"],
        "Stunting Rate (%)": [
            round(district_row["stunting"], 2),
            round(pred_baseline, 2),
            round(pred_scenario, 2),
        ],
    }
)
fig = px.bar(
    chart_data,
    x="Condition",
    y="Stunting Rate (%)",
    color="Condition",
    color_discrete_map={
        "Actual Observed": "#7F8C8D",
        "Model Baseline": "#3498DB",
        "Scenario": "#E74C3C" if delta > 0 else "#2ECC71",
    },
    text="Stunting Rate (%)",
)
fig.update_layout(height=400, showlegend=False)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig, use_container_width=True)
