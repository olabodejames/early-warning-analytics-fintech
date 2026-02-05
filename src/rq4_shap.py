# src/rq4_shap.py
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import GradientBoostingClassifier

import shap

def train_tree_model(df: pd.DataFrame, feature_cols: list[str], label_col: str, time_col: str = "year") -> dict:
    """
    Trains a tree-based classifier and returns in-sample metrics + fitted model pipeline.
    Uses time ordering for splitting decisions (no shuffle).
    """
    data = df.sort_values(time_col).dropna(subset=[label_col]).copy()
    X = data[feature_cols]
    y = data[label_col].astype(int)

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])

    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]

    metrics = {
        "roc_auc_in_sample": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        "brier_in_sample": float(brier_score_loss(y, proba)),
        "n": int(len(data))
    }
    return {"model": pipe, "metrics": metrics, "data": data, "proba": proba}

def compute_shap_values(model_pipe: Pipeline, X: pd.DataFrame) -> dict:
    """
    Computes SHAP values for the tree model.
    Uses SHAP TreeExplainer on the underlying estimator.
    """
    # Transform with pipeline preprocessing (imputer only here)
    imputer = model_pipe.named_steps["imputer"]
    clf = model_pipe.named_steps["clf"]

    X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_imp)

    # For binary classification, shap_values is (n, p)
    return {
        "X_imputed": X_imp,
        "explainer": explainer,
        "shap_values": shap_values
    }

def shap_global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Mean absolute SHAP value per feature (global importance).
    """
    imp = np.mean(np.abs(shap_values), axis=0)
    out = pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

def export_rq4_shap_outputs(model_pipe, metrics: dict, preds: pd.DataFrame,
                            shap_importance: pd.DataFrame,
                            model_path, metrics_path, preds_path, shap_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipe, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    preds.to_csv(preds_path, index=False)
    shap_importance.to_csv(shap_path, index=False)
