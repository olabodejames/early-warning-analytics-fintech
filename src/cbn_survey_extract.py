# src/cbn_survey_extract.py
import pandas as pd
from .config import RAW_DIR
from .io_utils import write_csv

def export_cbn_fintech_survey_insights():
    """
    The CBN Fintech Report (2025) contains survey results as aggregated percentages.
    This script stores them as a structured CSV for transparency and optional use as priors/scenarios.

    NOTE: These are not respondent-level microdata.
    """
    rows = [
        # Regulatory environment perception (50/50 split noted)
        {"theme":"regulation_perception","metric":"supportive_percent","value":50.0},
        {"theme":"regulation_perception","metric":"restrictive_percent","value":50.0},

        # Time-to-market distribution (chart shows 37.5% over 1 year etc.)
        {"theme":"time_to_market","metric":"over_1_year_percent","value":37.5},
        {"theme":"time_to_market","metric":"6_12_months_percent","value":12.5},
        {"theme":"time_to_market","metric":"3_6_months_percent","value":25.0},
        {"theme":"time_to_market","metric":"less_than_3_months_percent","value":25.0},

        # Compliance costs impact
        {"theme":"compliance_costs","metric":"significantly_impacts_innovation_percent","value":87.5},

        # AI use cases
        {"theme":"ai_use","metric":"fraud_detection_percent","value":87.5},
        {"theme":"ai_use","metric":"chatbots_customer_service_percent","value":62.5},
        {"theme":"ai_use","metric":"credit_scoring_risk_modeling_percent","value":37.5},
        {"theme":"ai_use","metric":"customer_onboarding_kyc_percent","value":37.5},

        # Inclusion constraints
        {"theme":"inclusion_constraints","metric":"lack_digital_id_or_credit_history_percent","value":37.5},
        {"theme":"inclusion_constraints","metric":"limited_mobile_data_penetration_percent","value":25.0},
        {"theme":"inclusion_constraints","metric":"cost_of_last_mile_delivery_percent","value":25.0},
        {"theme":"inclusion_constraints","metric":"agent_network_limitations_percent","value":12.5},
    ]

    df = pd.DataFrame(rows)
    out_path = RAW_DIR / "cbn_fintech_report_2025_survey_aggregates.csv"
    write_csv(df, out_path, index=False)
    return out_path
