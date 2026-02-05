# src/labels.py

# W label for RQ4; adaptable for inclusion-risk later
import numpy as np
import pandas as pd

def make_quantile_label(
    df: pd.DataFrame,
    target_col: str,
    q: float = 0.75,
    label_col: str = "y_high_risk"
) -> pd.DataFrame:
    """
    Creates binary high-risk label:
      y=1 if target_col > historical q-quantile, else 0.
    """
    out = df.copy()
    if target_col not in out.columns:
        raise ValueError(f"target_col '{target_col}' not found in df")

    vals = out[target_col].dropna()
    if len(vals) < 4:
        # avoid unstable threshold with too few points
        out[label_col] = np.nan
        return out

    thr = vals.quantile(q)
    out[label_col] = (out[target_col] > thr).astype("Int64")
    out.attrs["label_threshold"] = float(thr)
    return out

def make_threshold_label(
    df: pd.DataFrame,
    target_col: str,
    threshold: float,
    label_col: str = "y_high_risk"
) -> pd.DataFrame:
    out = df.copy()
    out[label_col] = (out[target_col] > threshold).astype("Int64")
    out.attrs["label_threshold"] = float(threshold)
    return out
