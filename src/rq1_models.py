# src/rq1_models.py
import json
import numpy as np
import pandas as pd
import ruptures as rpt
from hmmlearn.hmm import GaussianHMM

def prepare_rq1_series(epay_long: pd.DataFrame, channel: str = "POS Transactions") -> pd.DataFrame:
    """
    Builds a clean yearly series for a chosen channel (e.g., POS Transactions).
    Input: long data with [year, period, channel, volume, value]
    Output: year, vol, val, log_val, log_vol
    """
    df = epay_long.copy()
    df = df[df["channel"].str.lower() == channel.lower()].copy()

    # Prefer full-year (Jan-Dec) if present; otherwise keep all and user filters later
    # We'll keep period column for transparency
    df = df.sort_values(["year", "period"])

    df["log_value"] = np.log1p(df["value"])
    df["log_volume"] = np.log1p(df["volume"])
    return df[["year", "period", "volume", "value", "log_volume", "log_value"]]

def bai_perron_like_breaks(
    series: pd.Series,
    model: str = "l2",
    max_breaks: int = 3,
    min_size: int = 2
) -> dict:
    """
    Uses ruptures to detect multiple breakpoints (Bai–Perron-like).
    Returns break indices and basic metadata.

    Note: This is a practical approximation for capstone use (transparent & reproducible).
    """
    y = series.dropna().values.reshape(-1, 1)
    n = len(y)
    if n < (min_size * 2 + 1):
        return {"break_indices": [], "n_obs": n, "warning": "Too few observations for break detection."}

    algo = rpt.Binseg(model=model).fit(y)
    bkps = algo.predict(n_bkps=min(max_breaks, max(0, n // min_size - 1)))
    # ruptures returns final point n as a breakpoint—remove it for reporting
    bkps = [b for b in bkps if b < n]

    return {"break_indices": bkps, "n_obs": n, "model": model, "max_breaks": max_breaks, "min_size": min_size}

def fit_hmm_regimes(series: pd.Series, n_states: int = 2, random_state: int = 42) -> dict:
    """
    Fits a Gaussian HMM to the (log) series changes to detect regimes.
    Returns state sequence and state means.
    """
    s = series.dropna()
    if len(s) < 6:
        return {"warning": "Too few observations to fit HMM reliably.", "states": None}

    # Work with first differences for stationarity
    y = np.diff(s.values).reshape(-1, 1)

    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=random_state)
    model.fit(y)
    states = model.predict(y)

    return {
        "n_states": n_states,
        "states": states.tolist(),
        "state_means": model.means_.flatten().tolist(),
        "state_vars": model.covars_.flatten().tolist(),
        "log_likelihood": float(model.score(y)),
    }

def rq1_run_all(series_df: pd.DataFrame, target_col: str = "log_value") -> dict:
    """
    Runs break detection + regime detection on the chosen series.
    """
    out = {}
    out["breaks"] = bai_perron_like_breaks(series_df[target_col])
    out["hmm"] = fit_hmm_regimes(series_df[target_col], n_states=2)
    return out

def export_rq1_outputs(series_df: pd.DataFrame, results: dict, outputs_dir) -> None:
    """
    Writes CSV + JSON outputs to data/outputs/rq1/
    """
    series_df.to_csv(outputs_dir / "rq1_series.csv", index=False)
    with open(outputs_dir / "rq1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
