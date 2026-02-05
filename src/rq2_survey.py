# src/rq2_survey.py
import numpy as np
import pandas as pd
import statsmodels.api as sm

def prepare_survey_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns (minimum):
      weight, account_owner (0/1), digital_usage (0/1),
      plus controls like age, gender, education, income_quintile, urban

    This function:
      - cleans types,
      - builds simple interaction terms for subgroup effects,
      - returns model-ready frame.
    """
    out = df.copy()

    # standardize column names you plan to use
    required = ["weight", "account_owner", "digital_usage"]
    for c in required:
        if c not in out.columns:
            raise ValueError(f"Missing required survey column: {c}")

    # basic numeric conversion
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out["account_owner"] = pd.to_numeric(out["account_owner"], errors="coerce")
    out["digital_usage"] = pd.to_numeric(out["digital_usage"], errors="coerce")

    # controls are optional; keep only those present
    controls = [c for c in ["age", "urban", "female", "education_years", "income_quintile"] if c in out.columns]
    for c in controls:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # subgroup example: urban x digital_usage
    if "urban" in out.columns:
        out["digital_x_urban"] = out["digital_usage"] * out["urban"]

    return out

def wls_with_robust_se(y: pd.Series, X: pd.DataFrame, w: pd.Series):
    """
    Survey-weighted linear probability model (LPM) with robust SE.
    Simple and transparent for capstone; can be complemented by logit as robustness.
    """
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.WLS(y, Xc, weights=w)
    res = model.fit(cov_type="HC3")
    return res

def rq2_model_account_ownership(df: pd.DataFrame) -> dict:
    """
    RQ2: Effect of payment-growth proxies / digital usage on inclusion outcomes.
    Here: account_owner ~ digital_usage + controls (+ subgroup interaction).
    """
    d = df.dropna(subset=["weight", "account_owner", "digital_usage"]).copy()
    y = d["account_owner"]

    feature_cols = ["digital_usage"]
    for c in ["age", "urban", "female", "education_years", "income_quintile", "digital_x_urban"]:
        if c in d.columns:
            feature_cols.append(c)

    X = d[feature_cols]
    w = d["weight"]

    res = wls_with_robust_se(y, X, w)

    # Effect sizes and CI
    params = res.params.to_dict()
    conf = res.conf_int().rename(columns={0: "ci_low", 1: "ci_high"}).to_dict("index")

    return {
        "n": int(res.nobs),
        "params": params,
        "conf_int": conf,
        "r2": float(res.rsquared),
        "notes": "Survey-weighted linear probability model (WLS) with HC3 robust SE.",
    }

def rq2_model_digital_usage(df: pd.DataFrame) -> dict:
    """
    Another inclusion dimension: digital_usage as outcome (optional).
    """
    if "digital_usage" not in df.columns:
        raise ValueError("digital_usage is required for this model.")

    d = df.dropna(subset=["weight", "digital_usage"]).copy()
    y = d["digital_usage"]

    feature_cols = []
    for c in ["age", "urban", "female", "education_years", "income_quintile"]:
        if c in d.columns:
            feature_cols.append(c)

    if not feature_cols:
        raise ValueError("No controls found; add at least one control variable for RQ2.")

    X = d[feature_cols]
    w = d["weight"]

    res = wls_with_robust_se(y, X, w)

    params = res.params.to_dict()
    conf = res.conf_int().rename(columns={0: "ci_low", 1: "ci_high"}).to_dict("index")

    return {
        "n": int(res.nobs),
        "params": params,
        "conf_int": conf,
        "r2": float(res.rsquared),
        "notes": "Survey-weighted LPM with robust SE.",
    }
