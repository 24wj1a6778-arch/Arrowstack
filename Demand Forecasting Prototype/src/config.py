"""Central configuration for the demand-forecasting project.

Paths, seed, forecasting set-up and the (SIMULATED) inventory-cost assumptions
live here so the pipeline stays reproducible and every decision is traceable.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- paths -----------------------------------------------------------------
RAW_PATH = ROOT / "data" / "raw" / "sales.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PANEL_PATH = PROCESSED_DIR / "demand_daily.csv"
SPLIT_PATH = PROCESSED_DIR / "split_info.json"
MODELS_DIR = ROOT / "models"
SELECTED_MODEL_PATH = MODELS_DIR / "selected_model.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---- reproducibility ---------------------------------------------------------
SEED = 42

# ---- schema ------------------------------------------------------------------
DATE_COL = "date"
PRODUCT_COL = "product_id"
NAME_COL = "product_name"
TARGET = "units_sold"
PROMO_COL = "promo"
REQUIRED_COLUMNS = [DATE_COL, PRODUCT_COL, NAME_COL, TARGET, PROMO_COL]

# ---- forecasting set-up --------------------------------------------------------
SEASON = 7                 # weekly seasonality
HORIZON = 14               # days forecast ahead (= lead time + review period)
STRIDE = 7                 # a new forecast is issued every 7 days (weekly planning)
VAL_DAYS = 84              # validation period: 12 weeks
TEST_DAYS = 84             # test period: last 12 weeks, touched once
MIN_HISTORY_DAYS = 120     # below this, predict.py falls back to a simpler model
GBM_ORIGIN_STEP = 2        # GBM trains on every 2nd forecast origin (speed)

# ---- inventory policy (SIMULATED assumptions - state them in your presentation) ---
LEAD_TIME = 7              # days between placing an order and receiving it
REVIEW_PERIOD = 7          # days between orders
SERVICE_LEVEL = 0.95       # target probability of not stocking out in a cycle
HOLDING_COST = 0.10        # cost per unit per day held in stock
STOCKOUT_COST = 3.00       # cost per unit of lost sales (lost margin + goodwill)

# sensible 90% prediction interval
INTERVAL_LEVEL = 0.90
