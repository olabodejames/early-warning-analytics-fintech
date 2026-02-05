# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
OUTPUTS_DIR = DATA_DIR / "outputs"

FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Core dataset filenames (edit to your actual filenames)
EPAYMENTS_CSV = PROCESSED_DIR / "cbn_epayments_long.csv"
FRAUD_SUMMARY_CSV = PROCESSED_DIR / "fraud_summary_annual.csv"
MERGED_PANEL_CSV = PROCESSED_DIR / "panel_payments_fraud.csv"

FEATURE_MATRIX_CSV = FEATURES_DIR / "rq4_feature_matrix.csv"

MODEL_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

for d in [PROCESSED_DIR, FEATURES_DIR, OUTPUTS_DIR, FIGURES_DIR, MODEL_DIR, METRICS_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# src/config.py  (append these)
RQ1_OUTPUTS_DIR = OUTPUTS_DIR / "rq1"
RQ2_OUTPUTS_DIR = OUTPUTS_DIR / "rq2"
RQ5_OUTPUTS_DIR = OUTPUTS_DIR / "rq5"

for d in [RQ1_OUTPUTS_DIR, RQ2_OUTPUTS_DIR, RQ5_OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# RQ2 survey microdata (you will add later, e.g., Findex microdata extract)
SURVEY_MICRODATA_CSV = PROCESSED_DIR / "survey_microdata.csv"  # <-- create/export this
SURVEY_FEATURES_CSV = FEATURES_DIR / "rq2_survey_features.csv"

# RQ1 panel (from earlier)
RQ1_PANEL_CSV = PROCESSED_DIR / "rq1_payments_panel.csv"

# RQ5 scenario outputs
RQ5_SCENARIOS_CSV = OUTPUTS_DIR / "rq5" / "rq5_scenarios.csv"
RQ5_TRADEOFF_CSV = OUTPUTS_DIR / "rq5" / "rq5_tradeoff_curves.csv"

