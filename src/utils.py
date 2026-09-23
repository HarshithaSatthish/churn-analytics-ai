"""Shared paths and constants for the churn analytics project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "telco_churn.csv"
DATA_CLEAN = ROOT / "data" / "clean" / "telco_clean.csv"
QUALITY_LOG = ROOT / "data" / "clean" / "quality_log.md"

FIGURES_DIR = ROOT / "figures"
EDA_FINDINGS = FIGURES_DIR / "eda_findings.md"

MODEL_DIR = ROOT / "models"
MODEL_PKL = MODEL_DIR / "churn_model.pkl"
METRICS_JSON = MODEL_DIR / "metrics.json"
FEATURE_INFO = MODEL_DIR / "feature_info.json"
FEATURE_IMPORTANCE = MODEL_DIR / "feature_importance.csv"

RANDOM_STATE = 42
TARGET = "churn_binary"
MIN_TARGET_RECALL = 0.70

SERVICE_YES_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Included for documentation/validation of the nine service-related fields.
SERVICE_COLS = ["PhoneService", "MultipleLines", "InternetService", *SERVICE_YES_COLS[2:]]

# Model features deliberately avoid exact duplicate indicators such as
# is_fiber + InternetService or is_month_to_month + Contract. tenure_band is
# retained alongside tenure because it gives a simple non-linear tenure effect.
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_band",
]

NUMERIC_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "num_services",
]

MODEL_FEATURES = CATEGORICAL_COLS + NUMERIC_COLS

RISK_LOW = 0.20
RISK_VERY_HIGH = 0.60
