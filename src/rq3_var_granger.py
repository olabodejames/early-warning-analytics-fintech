# src/rq3_var_granger.py
import json
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

def adf_test(series: pd.Series, autolag: str = "AIC") -> dict:
    """
    Augmented Dickey-Fuller stationarity test.
    Returns p-value and test stat for reporting.
    """
    s = series.dropna()
    if len(s) < 8:
        return {"n": int(len(s)), "adf_stat": None, "p_value": None, "warning": "Too few observations for ADF."}

    stat, p, usedlag, nobs, crit, icbest = adfuller(s.values, autolag=autolag)
    return {
        "n": int(nobs),
        "adf_stat": float(stat),
        "p_value": float(p),
        "used_lag": int(usedlag),
        "crit_values": {k: float(v) for k, v in crit.items()},
        "icbest": float(icbest),
    }

def difference_if_needed(df: pd.DataFrame, cols: list[str], p_thresh: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """
    Checks stationarity via ADF. If non-stationary (p > p_thresh),
    differences the series once.
    """
    out = df.copy()
    report = {}
    for c in cols:
        test = adf_test(out[c])
        report[c] = {"adf_before": test}
        if test["p_value"] is not None and test["p_value"] > p_thresh:
            out[c] = out[c].diff()
            test2 = adf_test(out[c])
            report[c]["adf_after"] = test2
            report[c]["transformation"] = "diff(1)"
        else:
            report[c]["transformation"] = "none"
    return out, report

def fit_var(df: pd.DataFrame, cols: list[str], maxlags: int = 4, ic: str = "aic") -> dict:
    """
    Fits a VAR model with lag selection by IC.
    Returns summary stats and fitted lag order.
    """
    data = df[cols].dropna()
    if len(data) < 10:
        return {"warning": "Too few observations for VAR.", "n": int(len(data))}

    model = VAR(data)
    sel = model.select_order(maxlags=maxlags)
    # choose based on IC
    if ic.lower() == "aic":
        lag_order = int(sel.aic) if sel.aic is not None else 1
    elif ic.lower() == "bic":
        lag_order = int(sel.bic) if sel.bic is not None else 1
    else:
        lag_order = int(sel.aic) if sel.aic is not None else 1

    lag_order = max(1, min(lag_order, maxlags))
    res = model.fit(lag_order)

    return {
        "n": int(res.nobs),
        "k_ar": int(res.k_ar),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "hqic": float(res.hqic),
        "sigma_u": res.sigma_u.tolist(),
        "params": res.params.to_dict(),
    }

def run_granger(df: pd.DataFrame, x: str, y: str, maxlag: int = 4) -> dict:
    """
    Tests if x Granger-causes y.
    Returns p-values for each lag, plus min p-value.
    """
    data = df[[y, x]].dropna()
    if len(data) < (maxlag + 8):
        return {"warning": "Too few observations for Granger.", "n": int(len(data))}

    tests = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    pvals = {}
    for lag, out in tests.items():
        # ssr_ftest p-value
        pvals[lag] = float(out[0]["ssr_ftest"][1])

    return {
        "n": int(len(data)),
        "maxlag": int(maxlag),
        "pvals_ssr_ftest": pvals,
        "min_p": float(min(pvals.values())) if pvals else None,
    }

def rq3_var_granger_pipeline(panel: pd.DataFrame, cols: list[str], maxlags: int = 4) -> dict:
    """
    Full RQ3 pipeline:
      - stationarity checks & differencing
      - VAR fit
      - Granger tests (pairwise)
    """
    # stationarity transform
    transformed, adf_report = difference_if_needed(panel, cols=cols, p_thresh=0.05)

    # VAR model (on transformed)
    var_result = fit_var(transformed, cols=cols, maxlags=maxlags, ic="aic")

    # Granger (pairwise)
    granger_results = {}
    for x in cols:
        for y in cols:
            if x == y:
                continue
            granger_results[f"{x} -> {y}"] = run_granger(transformed, x=x, y=y, maxlag=maxlags)

    return {
        "adf_report": adf_report,
        "var_result": var_result,
        "granger_results": granger_results,
        "notes": "Series were differenced if ADF p>0.05 to improve stationarity. VAR lag chosen by AIC.",
    }

def export_rq3_outputs(results: dict, out_json_path):
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
