"""Central configuration for the churn project.

Everything that a mentor might want to change (paths, seed, split sizes,
business assumptions, feature lists) lives here so the pipeline stays
reproducible and decisions are traceable.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- paths -----------------------------------------------------------------
RAW_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
CANDIDATES_DIR = MODELS_DIR / "candidates"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FINAL_MODEL_PATH = MODELS_DIR / "churn_model.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

# ---- reproducibility ---------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20   # final hold-out, touched once
VAL_SIZE = 0.20    # used for model + threshold selection

# ---- schema ------------------------------------------------------------------
TARGET = "Churn"
ID_COL = "customerID"

RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
REQUIRED_COLUMNS = RAW_NUMERIC + RAW_CATEGORICAL

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# engineered features (created in src/features.py)
ENGINEERED_NUMERIC = [
    "avg_monthly_spend", "num_services", "charge_per_service",
    "is_month_to_month", "autopay",
]
ENGINEERED_CATEGORICAL = ["tenure_group"]

NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL

# ---- business assumptions (SIMULATED - state these in your presentation) -----
# A retention offer costs OFFER_COST per contacted customer. If the customer
# would have churned and accepts the offer (ACCEPT_RATE), the company keeps
# SAVE_VALUE of future margin.
OFFER_COST = 20.0
SAVE_VALUE = 200.0
ACCEPT_RATE = 0.30
