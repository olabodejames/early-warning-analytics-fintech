# src/cleaning.py
import pandas as pd
from .io_utils import safe_to_numeric

def clean_epayments_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns (long format):
    year, period, channel, volume, value
    """
    df = df.copy()

    # Normalize labels
    df["channel"] = df["channel"].astype(str).str.strip()
    df["period"] = df["period"].astype(str).str.strip()

    # Numeric cleaning
    df["year"] = safe_to_numeric(df["year"]).astype("Int64")
    df["volume"] = safe_to_numeric(df["volume"])
    df["value"] = safe_to_numeric(df["value"])

    # Remove obvious totals if you want channel-level modeling; keep if you want aggregate features
    # df = df[df["channel"].str.lower().ne("total e-payment transactions")]

    return df

def clean_fraud_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns:
    year, fraud_cases, fraud_losses_ngn_billion
    """
    df = df.copy()
    df["year"] = safe_to_numeric(df["year"]).astype("Int64")
    if "fraud_cases" in df.columns:
        df["fraud_cases"] = safe_to_numeric(df["fraud_cases"])
    if "fraud_losses_ngn_billion" in df.columns:
        df["fraud_losses_ngn_billion"] = safe_to_numeric(df["fraud_losses_ngn_billion"])
    return df
