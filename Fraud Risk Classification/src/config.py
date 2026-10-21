"""Central configuration for the fraud-risk project.

Paths, seed, split sizes, feature lists and the (SIMULATED) business
assumptions all live here so the pipeline stays reproducible and every
decision is traceable.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- paths -----------------------------------------------------------------
RAW_PATH = ROOT / "data" / "raw" / "transactions.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FINAL_MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

# ---- reproducibility ---------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# ---- schema ------------------------------------------------------------------
TARGET = "is_fraud"
ID_COL = "transaction_id"
AMOUNT_COL = "amount"

RAW_NUMERIC = [
    "amount", "hour", "day_of_week", "distance_from_home_km", "txn_count_24h",
    "avg_amount_30d", "card_age_days", "account_age_days",
    "is_international", "is_new_merchant", "device_changed",
]
RAW_CATEGORICAL = ["merchant_category", "channel"]
REQUIRED_COLUMNS = RAW_NUMERIC + RAW_CATEGORICAL

ENGINEERED_NUMERIC = ["amount_ratio", "log_amount", "is_night", "intl_online", "new_card"]
NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = RAW_CATEGORICAL

# ---- imbalance handling --------------------------------------------------------
# Target ratio positives:negatives in the TRAINING data after resampling.
RESAMPLE_RATIO = 0.2
SMOTE_NEIGHBORS = 5

# ---- business assumptions (SIMULATED - state these in your presentation) -----
# Cost of a missed fraud  = the transaction amount (full loss).
# Cost of any alert       = REVIEW_COST (analyst time / customer friction),
#                           whether or not the alert was real.
# A correctly caught fraud is blocked, so its loss is avoided.
REVIEW_COST = 5.0

# Alert budgets (share of transactions the fraud team can review) to report on.
ALERT_BUDGETS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
