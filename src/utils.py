"""Shared paths and constants for the churn analytics pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "telco_churn.csv"
DATA_CLEAN = ROOT / "data" / "clean" / "telco_clean.csv"
QUALITY_LOG = ROOT / "data" / "clean" / "quality_log.md"

FIGURES_DIR = ROOT / "figures"
EDA_FINDINGS = ROOT / "figures" / "eda_findings.md"

MODEL_PKL = ROOT / "models" / "churn_model.pkl"
METRICS_JSON = ROOT / "models" / "metrics.json"
FEATURE_INFO = ROOT / "models" / "feature_info.json"

RANDOM_STATE = 42
TARGET = "churn_binary"

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "tenure_band",
]

NUMERIC_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "num_services", "is_fiber", "is_month_to_month", "is_electronic_check",
]

# NOTE (pandas 3.x gotcha): text columns infer as `str` dtype, not `object`.
# Never check `df[col].dtype == object`; use select_dtypes(include="str")
# or OneHotEncoder, which handles the new dtype natively.
