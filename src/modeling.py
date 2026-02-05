# src/modeling.py

# calibrated EW probability → EW-FRI

import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report

@dataclass
class ModelResult:
    model: object
    metrics: dict
    predictions: pd.DataFrame

def train_ew_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "y_high_risk",
    time_col: str = "year",
    calibrate: bool = True,
    random_state: int = 42
) -> ModelResult:
    """
    Trains a regularized logistic EWS with optional probability calibration.
    Uses time-series splits to respect temporal ordering.
    """
    data = df.sort_values(time_col).copy()
    data = data.dropna(subset=[label_col])

    X = data[feature_cols]
    y = data[label_col].astype(int)

    base = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            random_state=random_state
        ))
    ])

    # With very small samples, calibration can be unstable; still useful if data allows.
    if calibrate:
        # time-series CV
        n_splits = min(3, max(2, len(data) - 1))
        cv = TimeSeriesSplit(n_splits=n_splits)
        model = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    else:
        model = base

    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc_in_sample": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        "brier_in_sample": float(brier_score_loss(y, proba)),
        "classification_report": classification_report(y, pred, output_dict=True),
    }

    preds = data[[time_col, "period"]].copy() if "period" in data.columns else data[[time_col]].copy()
    preds["p_high_risk"] = proba
    preds["ew_fri"] = 100.0 * preds["p_high_risk"]  # EW-FRI definition

    return ModelResult(model=model, metrics=metrics, predictions=preds)

def save_model(result: ModelResult, model_path, metrics_path, predictions_path):
    joblib.dump(result.model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result.metrics, f, indent=2)

    result.predictions.to_csv(predictions_path, index=False)
