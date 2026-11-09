"""Visualization suite replicating all 18 analytical and publication figures."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from nfhs.models.explainability import plot_shap_importance, plot_shap_summary


def plot_01_target_distribution(df: pd.DataFrame, figures_dir: Path, target_col: str = "stunting") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].hist(df[target_col].dropna(), bins=30, color="#E07B54", edgecolor="white", linewidth=0.6)
    axes[0].axvline(df[target_col].mean(), color="#2C3E50", lw=2, linestyle="--", label=f"Mean: {df[target_col].mean():.1f}%")
    axes[0].axvline(df[target_col].median(), color="#27AE60", lw=2, linestyle=":", label=f"Median: {df[target_col].median():.1f}%")
    axes[0].set_title("Distribution of Child Stunting Rate", fontweight="bold")
    axes[0].set_xlabel("Stunting Rate (%)")
    axes[0].set_ylabel("Number of Districts")
    axes[0].legend()

    if "rural_urban" in df.columns and df["rural_urban"].nunique() > 1:
        plot_data = [df[df["rural_urban"] == cat][target_col].dropna() for cat in df["rural_urban"].dropna().unique()]
        labels = df["rural_urban"].dropna().unique().tolist()
        axes[1].boxplot(plot_data, labels=labels, patch_artist=True, boxprops=dict(facecolor="#AED6F1"), medianprops=dict(color="#E74C3C", linewidth=2))
        axes[1].set_title("Stunting Rate by Rural/Urban", fontweight="bold")
    else:
        sns.boxplot(y=df[target_col], ax=axes[1], color="#AED6F1")
        axes[1].set_title("District Stunting Spread", fontweight="bold")

    axes[1].set_ylabel("Stunting Rate (%)")
    plt.suptitle("Target Variable: Child Stunting Rate across Districts", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(figures_dir / "01_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_02_state_stunting(df: pd.DataFrame, figures_dir: Path, target_col: str = "stunting") -> None:
    state_stunting = df.groupby("state")[target_col].mean().sort_values(ascending=False).reset_index()
    state_stunting.columns = ["state", "avg_stunting"]

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = [
        "#E74C3C" if v >= 40 else "#E67E22" if v >= 30 else "#F1C40F" if v >= 20 else "#2ECC71"
        for v in state_stunting["avg_stunting"]
    ]
    bars = ax.barh(state_stunting["state"], state_stunting["avg_stunting"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(df[target_col].mean(), color="#2C3E50", linestyle="--", linewidth=1.5, label=f"National Mean: {df[target_col].mean():.1f}%")

    for bar, val in zip(bars, state_stunting["avg_stunting"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8.5)

    ax.set_xlabel("Average Stunting Rate (%)")
    ax.set_title("Average Child Stunting Rate by State/UT — NFHS-4", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor="#E74C3C", label="≥ 40% (Critical)"),
        Patch(facecolor="#E67E22", label="30–40% (High)"),
        Patch(facecolor="#F1C40F", label="20–30% (Moderate)"),
        Patch(facecolor="#2ECC71", label="< 20% (Low)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(figures_dir / "02_state_stunting.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_03_district_extremes(df: pd.DataFrame, figures_dir: Path, target_col: str = "stunting") -> None:
    top15 = df.nlargest(15, target_col)[["state", "district", target_col]]
    bottom15 = df.nsmallest(15, target_col)[["state", "district", target_col]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    axes[0].barh(top15["district"] + " (" + top15["state"] + ")", top15[target_col], color="#E74C3C", edgecolor="white")
    axes[0].set_title("Top 15 Most Stunted Districts", fontweight="bold")
    axes[0].set_xlabel("Stunting Rate (%)")
    axes[0].invert_yaxis()
    for i, v in enumerate(top15[target_col]):
        axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    axes[1].barh(bottom15["district"] + " (" + bottom15["state"] + ")", bottom15[target_col], color="#2ECC71", edgecolor="white")
    axes[1].set_title("Top 15 Least Stunted Districts", fontweight="bold")
    axes[1].set_xlabel("Stunting Rate (%)")
    axes[1].invert_yaxis()
    for i, v in enumerate(bottom15[target_col]):
        axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    plt.suptitle("District-Level Stunting Extremes — NFHS-4", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "03_district_extremes.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_04_correlation_stunting(
    df: pd.DataFrame,
    figures_dir: Path,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "stunting",
) -> None:
    exclude = [
        "wasting",
        "severe_wasting",
        "underweight",
        "anaemia_children",
        "adequate_diet_total",
        "adequate_diet_bf",
        "state",
        "district",
        "rural_urban",
        "households",
        "women",
        "men",
    ]
    if feature_cols is None:
        avail_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude and c != target_col
        ]
    else:
        avail_cols = [c for c in feature_cols if c in df.columns and c != target_col]

    corr = df[avail_cols + [target_col]].corr()[target_col].drop(target_col).dropna().sort_values()
    top_corr = pd.concat([corr.head(10), corr.tail(10)])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in top_corr.values]
    bars = ax.barh(top_corr.index, top_corr.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Stunting Rate")
    ax.set_title("Top Correlated Features with Child Stunting\n(Red = positive risk, Blue = protective)", fontweight="bold")

    for bar, val in zip(bars, top_corr.values):
        xpos = bar.get_width() + 0.005 if val >= 0 else bar.get_width() - 0.005
        align = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", ha=align, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(figures_dir / "04_correlation_stunting.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_05_correlation_heatmap(df: pd.DataFrame, figures_dir: Path) -> None:
    key_features = [
        "stunting", "women_literacy", "child_marriage", "teenage_pregnancy",
        "hh_sanitation", "hh_clean_water", "hh_clean_fuel", "hh_electricity",
        "institutional_births", "full_anc", "anc_4visits",
        "fully_immunized", "women_low_bmi", "anaemia_women_all",
        "ifa_100days", "postnatal_care", "excl_breastfeed",
    ]
    available_keys = [c for c in key_features if c in df.columns]
    corr_matrix = df[available_keys].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
        center=0, linewidths=0.4, annot_kws={"size": 7}, ax=ax
    )
    ax.set_title("Correlation Heatmap — Key Policy Features vs Stunting", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_06_scatter_predictors(df: pd.DataFrame, figures_dir: Path, target_col: str = "stunting") -> None:
    key_pairs = [
        ("women_literacy", "Women Literacy Rate (%)"),
        ("hh_sanitation", "HH Improved Sanitation (%)"),
        ("hh_clean_fuel", "HH Clean Fuel for Cooking (%)"),
        ("institutional_births", "Institutional Births (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (feat, label) in enumerate(key_pairs):
        if feat not in df.columns:
            continue
        valid_mask = df[feat].notna() & df[target_col].notna()
        x_vals = df.loc[valid_mask, feat]
        y_vals = df.loc[valid_mask, target_col]

        axes[i].scatter(x_vals, y_vals, alpha=0.4, s=18, color="#3498DB", edgecolors="none")
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        axes[i].plot(x_line, m * x_line + b, color="#E74C3C", linewidth=2, label=f"r = {np.corrcoef(x_vals, y_vals)[0,1]:.2f}")
        axes[i].set_xlabel(label, fontsize=10)
        axes[i].set_ylabel("Stunting Rate (%)", fontsize=10)
        axes[i].set_title(f"Stunting vs {label}", fontweight="bold", fontsize=10)
        axes[i].legend(fontsize=9)

    plt.suptitle("Key Socioeconomic Predictors vs Child Stunting Rate", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "06_scatter_predictors.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_07_risk_tiers(df: pd.DataFrame, figures_dir: Path, target_col: str = "stunting") -> None:
    def stunting_tier_func(val: float) -> str:
        if val >= 50:
            return "Critical (≥50%)"
        if val >= 40:
            return "High (40–50%)"
        if val >= 30:
            return "Moderate (30–40%)"
        return "Low (<30%)"

    tiers = df[target_col].apply(stunting_tier_func)
    tier_order = ["Critical (≥50%)", "High (40–50%)", "Moderate (30–40%)", "Low (<30%)"]
    tier_colors = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71"]
    tier_counts = tiers.value_counts().reindex(tier_order, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].bar(tier_counts.index, tier_counts.values, color=tier_colors, edgecolor="white", linewidth=0.6)
    axes[0].set_title("Districts by Stunting Risk Tier", fontweight="bold")
    axes[0].set_ylabel("Number of Districts")
    axes[0].set_xticks(range(len(tier_counts)))
    axes[0].set_xticklabels(tier_counts.index, rotation=15, ha="right")
    for i, v in enumerate(tier_counts.values):
        axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")

    axes[1].pie(tier_counts.values, labels=tier_counts.index, colors=tier_colors, autopct="%1.1f%%", startangle=140, pctdistance=0.8, wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[1].set_title("Share of Districts by Risk Tier", fontweight="bold")
    plt.suptitle("Stunting Risk Tier Distribution — NFHS-4 Districts", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "07_risk_tiers.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_08_composite_correlations(
    df: pd.DataFrame, figures_dir: Path, composite_cols: List[str], target_col: str = "stunting"
) -> None:
    comp_corr = df[composite_cols + [target_col]].corr()[target_col].drop(target_col).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in comp_corr.values]
    bars = ax.barh(comp_corr.index, comp_corr.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Stunting")
    ax.set_title("Composite Domain Scores vs Child Stunting Rate", fontweight="bold")

    for bar, val in zip(bars, comp_corr.values):
        xpos = bar.get_width() + 0.005 if val >= 0 else bar.get_width() - 0.005
        align = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", ha=align, fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / "08_composite_correlations.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_09_vif_analysis(df: pd.DataFrame, figures_dir: Path, feature_cols: List[str]) -> None:
    X_vif = df[feature_cols].dropna().copy()
    vif_values = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    vif_df = pd.DataFrame({"Feature": feature_cols, "VIF": vif_values}).sort_values("VIF", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#E74C3C" if v > 10 else "#F39C12" if v > 5 else "#2ECC71" for v in vif_df["VIF"]]
    ax.barh(vif_df["Feature"], vif_df["VIF"], color=colors, edgecolor="white")
    ax.axvline(10, color="#E74C3C", linestyle="--", linewidth=1.5, label="VIF = 10 (threshold)")
    ax.axvline(5, color="#F39C12", linestyle=":", linewidth=1.5, label="VIF = 5 (caution)")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title("Multicollinearity Check — Selected Features", fontweight="bold")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(figures_dir / "09_vif_analysis.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_10_model_comparison(metrics: Dict[str, Dict[str, float]], figures_dir: Path) -> None:
    models = list(metrics.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    palette = ["#2ECC71", "#3498DB", "#E67E22", "#E74C3C"]

    for ax, metric, metric_key in zip(axes, ["R²", "RMSE", "MAE"], ["test_r2", "test_rmse", "test_mae"]):
        vals = [metrics[m][metric_key] for m in models]
        bars = ax.bar(models, vals, color=palette[:len(models)], edgecolor="white")
        ax.set_title(f"Model Comparison — {metric}", fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.suptitle("Regression Model Performance Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "10_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_11_actual_vs_predicted(df_pred: pd.DataFrame, figures_dir: Path, target_col: str = "stunting") -> None:
    test_df = df_pred[df_pred["split"] == "test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = [("Ridge Regression", "predicted_ridge"), ("Random Forest", "predicted_randomforest")]
    for ax, (title, pred_col) in zip(axes, models):
        if pred_col not in test_df.columns:
            continue
        y_act = test_df[target_col]
        y_pred = test_df[pred_col]
        ax.scatter(y_act, y_pred, alpha=0.5, s=25, color="#3498DB", edgecolors="none")
        lims = [min(y_act.min(), y_pred.min()) - 2, max(y_act.max(), y_pred.max()) + 2]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Stunting Rate (%)")
        ax.set_ylabel("Predicted Stunting Rate (%)")
        ax.set_title(f"Actual vs Predicted\n{title}", fontweight="bold")
        r2 = np.corrcoef(y_act, y_pred)[0, 1] ** 2
        ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(fontsize=9)

    plt.suptitle("Model Fit: Actual vs Predicted Stunting Rate (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "11_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_14_elbow_silhouette(k_eval: Dict[str, List[float]], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    k_vals = k_eval["k_values"]
    inertias = k_eval["inertias"]
    sil_scores = k_eval["silhouette_scores"]

    axes[0].plot(k_vals, inertias, "bo-", linewidth=2, markersize=7)
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia (Within-cluster SSE)")
    axes[0].set_title("Elbow Method — Optimal K", fontweight="bold")
    axes[0].set_xticks(k_vals)

    axes[1].plot(k_vals, sil_scores, "go-", linewidth=2, markersize=7)
    best_idx = int(np.argmax(sil_scores))
    best_k = k_vals[best_idx]
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k} (sil={sil_scores[best_idx]:.3f})")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score — Optimal K", fontweight="bold")
    axes[1].set_xticks(k_vals)
    axes[1].legend()

    plt.suptitle("Choosing Optimal Number of Clusters", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "14_elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_15_cluster_profiles(df: pd.DataFrame, figures_dir: Path) -> None:
    display_features = [
        "stunting", "empowerment_score", "wealth_proxy", "wash_score",
        "healthcare_score", "nutrition_deprivation", "fertility_risk",
        "immunization_score", "oop_delivery_cost", "unmet_need_total",
        "ifa_100days", "excl_breastfeed",
    ]
    avail_features = [f for f in display_features if f in df.columns]
    segment_col = "risk_segment" if "risk_segment" in df.columns else "tier"

    profile = df.groupby(segment_col)[avail_features].mean().round(1)
    if "stunting" in profile.columns:
        profile = profile.loc[profile["stunting"].sort_values(ascending=False).index]

    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        profile_norm, annot=profile.values, fmt=".1f", cmap="RdYlGn_r",
        linewidths=0.5, annot_kws={"size": 9}, ax=ax,
        cbar_kws={"label": "Normalized Score (0=best, 1=worst)"}
    )
    ax.set_title("Cluster Profiles — Mean Feature Values by Risk Segment", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(figures_dir / "15_cluster_profiles.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_16_pca_clusters(df: pd.DataFrame, figures_dir: Path, var_exp: List[float]) -> None:
    segment_colours = {
        "Tier 3 — High Risk": "#E74C3C",
        "Tier 2 — Moderate Risk": "#F1C40F",
        "Tier 1 — Low Risk (Developed)": "#2ECC71",
        "Tier 3": "#E74C3C",
        "Tier 2": "#F1C40F",
        "Tier 1": "#2ECC71",
    }
    seg_col = "risk_segment" if "risk_segment" in df.columns else "tier"

    fig, ax = plt.subplots(figsize=(11, 7))
    for seg in df[seg_col].unique():
        mask = df[seg_col] == seg
        color = segment_colours.get(seg, "#3498DB")
        ax.scatter(df.loc[mask, "pca1"], df.loc[mask, "pca2"], label=f"{seg} (n={mask.sum()})", color=color, alpha=0.7, s=35, edgecolors="white", linewidth=0.4)

    pc1_str = f"{var_exp[0]*100:.1f}%" if len(var_exp) > 0 else ""
    pc2_str = f"{var_exp[1]*100:.1f}%" if len(var_exp) > 1 else ""
    ax.set_xlabel(f"PC1 ({pc1_str} variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pc2_str} variance)", fontsize=10)
    ax.set_title("District Risk Segmentation — PCA 2D View\nKMeans Clusters on Socioeconomic Features (NFHS-4)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "16_pca_clusters.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_17_segment_profiles(df: pd.DataFrame, figures_dir: Path) -> None:
    profile_features = {
        "empowerment_score": "Women's Empowerment",
        "wealth_proxy": "Household Wealth",
        "wash_score": "WASH Access",
        "healthcare_score": "Healthcare Access",
        "nutrition_deprivation": "Nutrition Deprivation",
        "fertility_risk": "Fertility Risk",
        "immunization_score": "Immunization Coverage",
        "ifa_100days": "Iron Folic Acid (100 days)",
        "oop_delivery_cost": "OOP Delivery Cost (Rs.)",
        "unmet_need_total": "Unmet Family Planning Need",
    }
    avail = {k: v for k, v in profile_features.items() if k in df.columns}
    tier_order = ["Tier 3", "Tier 2", "Tier 1"]
    tier_col = "tier" if "tier" in df.columns else "risk_segment"
    tier_colors = {"Tier 3": "#E74C3C", "Tier 2": "#F1C40F", "Tier 1": "#2ECC71"}

    seg_profile = df.groupby(tier_col)[list(avail.keys())].mean().round(1).reindex(tier_order).rename(columns=avail).T

    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(seg_profile))
    width = 0.25

    for i, tier in enumerate(tier_order):
        if tier not in seg_profile.columns:
            continue
        vals = seg_profile[tier].values.copy().astype(float)
        # Normalize OOP cost
        if "OOP Delivery Cost (Rs.)" in list(avail.values()):
            oop_idx = list(avail.values()).index("OOP Delivery Cost (Rs.)")
            vals[oop_idx] = vals[oop_idx] / 315.0
        ax.bar(x + i * width, vals, width, label=tier, color=tier_colors.get(tier, "#999999"), edgecolor="white", alpha=0.9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(list(avail.values()), rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Score (0–100 scale)")
    ax.set_title("Socioeconomic Profiles Across Risk Tiers", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "17_segment_profiles.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_18_segment_distribution(df: pd.DataFrame, figures_dir: Path, target_col: str = "stunting") -> None:
    tier_col = "tier" if "tier" in df.columns else "risk_segment"
    tier_order = ["Tier 3", "Tier 2", "Tier 1"]
    tier_colors = {"Tier 3": "#E74C3C", "Tier 2": "#F1C40F", "Tier 1": "#2ECC71"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.violinplot(data=df, x=tier_col, y=target_col, hue=tier_col, order=tier_order, palette=tier_colors, inner="box", ax=axes[0], legend=False)
    axes[0].set_title("Stunting Distribution by Risk Segment", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Stunting Rate (%)")

    top_states = df["state"].value_counts().head(12).index
    df_top = df[df["state"].isin(top_states)]
    state_seg = df_top.groupby(["state", tier_col]).size().unstack(fill_value=0).reindex(columns=tier_order, fill_value=0)

    state_seg.plot(
        kind="bar", stacked=True,
        color=[tier_colors.get(t, "#999999") for t in tier_order],
        edgecolor="white", linewidth=0.5, ax=axes[1]
    )
    axes[1].set_title("Risk Tier Composition by State (Top 12 States)", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Number of Districts")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right", fontsize=9)
    axes[1].legend(fontsize=8, loc="upper right")

    plt.suptitle("Risk Segmentation — NFHS-4 District Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "18_segment_distribution.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def generate_all_figures(
    df_clean: pd.DataFrame,
    df_analyzed: pd.DataFrame,
    figures_dir: Path,
    metrics: Dict[str, Dict[str, float]],
    k_eval: Dict[str, List[float]],
    shap_values: np.ndarray,
    importance_df: pd.DataFrame,
    feature_cols: List[str],
    composite_cols: List[str],
    var_exp: List[float],
) -> None:
    """Generate all 18 analytical and presentation figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_01_target_distribution(df_clean, figures_dir)
    plot_02_state_stunting(df_clean, figures_dir)
    plot_03_district_extremes(df_clean, figures_dir)
    plot_04_correlation_stunting(df_clean, figures_dir, feature_cols=None)
    plot_05_correlation_heatmap(df_clean, figures_dir)
    plot_06_scatter_predictors(df_clean, figures_dir)
    plot_07_risk_tiers(df_clean, figures_dir)
    plot_08_composite_correlations(df_analyzed, figures_dir, composite_cols)
    plot_09_vif_analysis(df_analyzed, figures_dir, feature_cols)
    plot_10_model_comparison(metrics, figures_dir)
    plot_11_actual_vs_predicted(df_analyzed, figures_dir)
    plot_shap_summary(shap_values, df_analyzed[feature_cols].values, feature_cols, figures_dir / "12_shap_summary.png")
    plot_shap_importance(importance_df, figures_dir / "13_shap_importance.png")
    plot_14_elbow_silhouette(k_eval, figures_dir)
    plot_15_cluster_profiles(df_analyzed, figures_dir)
    plot_16_pca_clusters(df_analyzed, figures_dir, var_exp)
    plot_17_segment_profiles(df_analyzed, figures_dir)
    plot_18_segment_distribution(df_analyzed, figures_dir)
