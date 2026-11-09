"""Policy action matrix generation and priority district reporting."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


def generate_priority_districts(
    df: pd.DataFrame,
    top_n: int = 30,
    target_col: str = "stunting",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Identify and rank top priority districts requiring urgent intervention based on stunting.
    Exports to CSV if output_path is provided.
    """
    preferred_cols = [
        "state",
        "district",
        target_col,
        "tier",
        "risk_segment",
        "empowerment_score",
        "wash_score",
        "wealth_proxy",
        "nutrition_deprivation",
        "fertility_risk",
        "healthcare_score",
        "predicted",
        "residual",
    ]
    available_cols = [c for c in preferred_cols if c in df.columns]

    priority_df = (
        df[available_cols]
        .sort_values(target_col, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    priority_df.index += 1  # 1-based Rank

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        priority_df.to_csv(output_path, index_label="Rank")

    return priority_df


def generate_policy_recommendations(df: pd.DataFrame, target_col: str = "stunting") -> Dict[str, Any]:
    """Compile structured cross-ministerial recommendations tailored to each risk tier."""
    recommendations = {
        "Tier 3 — High Risk": {
            "districts_count": int((df["tier"] == "Tier 3").sum()),
            "mean_stunting": float(round(df[df["tier"] == "Tier 3"][target_col].mean(), 2)),
            "core_deficit": "Depressed female literacy, high fertility risk, severe WASH deficits",
            "priority_actions": [
                "Scale up girls' secondary education and female adult literacy campaigns",
                "Strict enforcement of Prohibition of Child Marriage Act",
                "ICDS infrastructure expansion with saturated Anganwadi coverage",
                "Accelerated tap water and sanitation delivery under Jal Jeevan & Swachh Bharat Missions",
                "Conditional cash transfers linked to complete ANC attendance and IFA adherence",
            ],
            "lead_ministries": ["WCD", "Health & Family Welfare", "Education", "Jal Shakti"],
        },
        "Tier 2 — Moderate Risk": {
            "districts_count": int((df["tier"] == "Tier 2").sum()),
            "mean_stunting": float(round(df[df["tier"] == "Tier 2"][target_col].mean(), 2)),
            "core_deficit": "Transitional districts with high out-of-pocket delivery costs and feeding gaps",
            "priority_actions": [
                "Strengthen JSY and JSSK to eliminate out-of-pocket institutional delivery expenses",
                "Intensive exclusive breastfeeding and complementary feeding counseling via ASHA workers",
                "Target unmet family planning need and birth spacing",
                "Cold-chain maintenance and full immunization tracking",
            ],
            "lead_ministries": ["Health & Family Welfare", "WCD", "Finance"],
        },
        "Tier 1 — Low Risk": {
            "districts_count": int((df["tier"] == "Tier 1").sum()),
            "mean_stunting": float(round(df[df["tier"] == "Tier 1"][target_col].mean(), 2)),
            "core_deficit": "Developed baseline with isolated pockets of high stunting in tribal or peri-urban areas",
            "priority_actions": [
                "Micro-surveillance and disaggregated data tracking in vulnerable sub-pockets",
                "Dietary diversity campaigns targeting micronutrient deficiencies and residual anaemia",
                "Special tribal health and nutrition action plans",
            ],
            "lead_ministries": ["Health & Family Welfare", "Tribal Affairs", "NITI Aayog"],
        },
    }
    return recommendations
