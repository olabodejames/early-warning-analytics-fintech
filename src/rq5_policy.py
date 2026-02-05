# src/rq5_policy.py
import numpy as np
import pandas as pd

def generate_scenarios():
    """
    Defines policy/control scenarios.
    You can interpret these as:
      - stronger KYC/ID checks (reduces fraud loss ratio)
      - agent expansion (increases payment growth / inclusion)
      - user awareness campaigns (reduces social engineering losses)
    """
    # Example grid: agent expansion vs fraud-control strength
    agent_expansion = np.array([0.00, 0.05, 0.10, 0.15])   # +0% to +15% growth boost
    fraud_control = np.array([0.00, 0.10, 0.20, 0.30])     # 0% to 30% reduction in fraud intensity

    rows = []
    sid = 0
    for a in agent_expansion:
        for f in fraud_control:
            sid += 1
            rows.append({
                "scenario_id": sid,
                "agent_expansion_boost": float(a),
                "fraud_control_reduction": float(f),
            })
    return pd.DataFrame(rows)

def apply_scenario_to_features(base_row: pd.Series, scenario: pd.Series) -> pd.Series:
    """
    Applies counterfactual shocks to a single period's features (latest year or baseline).
    This keeps the approach transparent and reproducible.
    """
    x = base_row.copy()

    # Boost payment growth (proxy for access expansion / agent network)
    if "pay_value_g" in x.index:
        x["pay_value_g"] = x["pay_value_g"] * (1.0 + scenario["agent_expansion_boost"])
    if "pay_volume_g" in x.index:
        x["pay_volume_g"] = x["pay_volume_g"] * (1.0 + scenario["agent_expansion_boost"])

    # Reduce fraud intensity / ratio
    if "fraud_loss_ratio" in x.index and pd.notna(x["fraud_loss_ratio"]):
        x["fraud_loss_ratio"] = x["fraud_loss_ratio"] * (1.0 - scenario["fraud_control_reduction"])

    # You may extend with additional knobs: reporting quality, identity coverage, etc.
    return x

def simulate_tradeoffs(feature_matrix: pd.DataFrame, model, scenarios: pd.DataFrame, time_col="year") -> pd.DataFrame:
    """
    Uses trained RQ4 early-warning model to compute EW-FRI under counterfactual scenarios.
    Also returns a simple inclusion proxy score based on payment-growth (for tradeoff curves).
    """
    df = feature_matrix.sort_values(time_col).copy()
    base = df.iloc[-1]  # latest available period
    feat_cols = [c for c in df.columns if c not in ("year", "period", "y_high_risk")]

    rows = []
    for _, sc in scenarios.iterrows():
        x_cf = apply_scenario_to_features(base[feat_cols], sc)

        # Model probability
        p = float(model.predict_proba(pd.DataFrame([x_cf.values], columns=feat_cols))[:, 1][0])
        ew_fri = 100.0 * p

        # Inclusion proxy: scaled payment growth indicator
        # (replace later with RQ2 estimated inclusion response if available)
        incl_proxy = float(np.nan_to_num(x_cf.get("pay_value_g", 0.0)) + np.nan_to_num(x_cf.get("pay_volume_g", 0.0)))

        rows.append({
            "scenario_id": int(sc["scenario_id"]),
            "agent_expansion_boost": float(sc["agent_expansion_boost"]),
            "fraud_control_reduction": float(sc["fraud_control_reduction"]),
            "p_high_risk": p,
            "ew_fri": ew_fri,
            "inclusion_proxy": incl_proxy,
        })

    return pd.DataFrame(rows)

def build_policy_memo_table(tradeoffs: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a decision-ready table: best scenarios by low EW-FRI and high inclusion proxy.
    """
    t = tradeoffs.copy()
    # Pareto-like ranking
    t["risk_rank"] = t["ew_fri"].rank(ascending=True)
    t["incl_rank"] = t["inclusion_proxy"].rank(ascending=False)
    t["combined_rank"] = t["risk_rank"] + t["incl_rank"]
    return t.sort_values("combined_rank").head(10)
