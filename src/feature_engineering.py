# src/feature_engineering.py
import numpy as np
import pandas as pd

def build_annual_totals(epay_long: pd.DataFrame) -> pd.DataFrame:
    """
    Converts channel-level long data into annual aggregate totals.
    Produces: year, total_epayment_volume, total_epayment_value_ngn
    """
    df = epay_long.copy()

    # Prefer the explicit total if present; otherwise sum channels
    total_mask = df["channel"].str.lower().str.contains("total")
    totals = df[total_mask].copy()

    if len(totals) > 0:
        totals = totals.groupby(["year", "period"], as_index=False).agg(
            total_epayment_volume=("volume", "sum"),
            total_epayment_value_ngn=("value", "sum"),
        )
        # If multiple periods (e.g., Jan-Jun, Jan-Dec), keep both; later you can filter
        return totals

    # Fallback: sum all channels
    agg = df.groupby(["year", "period"], as_index=False).agg(
        total_epayment_volume=("volume", "sum"),
        total_epayment_value_ngn=("value", "sum"),
    )
    return agg

def merge_payments_fraud(annual_pay: pd.DataFrame, fraud: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join fraud onto payments at year level.
    """
    out = annual_pay.merge(fraud, on="year", how="left")
    return out

def add_growth_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Adds log-diff growth rates and ratios needed for RQ3/RQ4.
    """
    df = panel.sort_values(["year", "period"]).copy()

    # Use safe log for positive values
    def log1p_safe(x):
        return np.log1p(np.where(x < 0, np.nan, x))

    df["log_pay_value"] = log1p_safe(df["total_epayment_value_ngn"])
    df["log_pay_volume"] = log1p_safe(df["total_epayment_volume"])

    df["pay_value_g"] = df["log_pay_value"].diff()
    df["pay_volume_g"] = df["log_pay_volume"].diff()

    # Fraud losses in NGN (convert from billions)
    if "fraud_losses_ngn_billion" in df.columns:
        df["fraud_losses_ngn"] = df["fraud_losses_ngn_billion"] * 1e9
        df["log_fraud_losses"] = log1p_safe(df["fraud_losses_ngn"])
        df["fraud_losses_g"] = df["log_fraud_losses"].diff()

        # Loss ratio (system-level)
        df["fraud_loss_ratio"] = df["fraud_losses_ngn"] / df["total_epayment_value_ngn"]

    # Basic intensity
    if "fraud_cases" in df.columns:
        df["fraud_cases_per_1m_txn"] = (df["fraud_cases"] / df["total_epayment_volume"]) * 1e6

    return df

def add_lags(df: pd.DataFrame, cols: list[str], lags: list[int] = [1, 2]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for L in lags:
            out[f"{c}_lag{L}"] = out[c].shift(L)
    return out

def build_rq4_feature_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a model-ready matrix for early warning modeling (RQ4).
    """
    df = panel.copy()

    candidate_features = [
        "pay_value_g",
        "pay_volume_g",
        "fraud_losses_g",
        "fraud_loss_ratio",
        "fraud_cases_per_1m_txn",
    ]
    existing = [c for c in candidate_features if c in df.columns]

    df = add_lags(df, cols=existing, lags=[1, 2])

    # Keep a clean set
    keep_cols = ["year", "period"] + existing + [c for c in df.columns if c.endswith("_lag1") or c.endswith("_lag2")]
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df.columns]))

    return df[keep_cols]
