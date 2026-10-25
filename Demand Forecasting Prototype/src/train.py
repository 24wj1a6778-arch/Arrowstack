"""Compare all forecasting models on the VALIDATION period and pick one.

Steps
  1. Rolling-origin backtest of 6 models over the 84 validation days
     (weekly origins, 14-day horizon, models see only past data).
  2. Rank models by mean MASE (scale-free, so a 120-unit/day product and a
     2-unit/day product count equally).  MASE < 1 means better than seasonal naive.
  3. Measure each model's daily error spread (RMSE) per product on validation -
     this sizes the safety stock and the prediction intervals later.
  4. Save the selection.  The TEST period is not touched here.

Run:  python -m src.train
"""
import json
import time

import pandas as pd

from src import config
from src.backtest import origins_for, rolling_origin_backtest
from src.data_prep import load_panel, load_split_info, product_series
from src.metrics import seasonal_naive_scale, summarize
from src.models import BASELINE, MODEL_FACTORIES


def training_scales(panel, before) -> dict:
    """MASE denominators from data strictly before `before`."""
    scales = {}
    for pid in sorted(panel[config.PRODUCT_COL].unique()):
        y, _ = product_series(panel, pid)
        scales[pid] = seasonal_naive_scale(y.loc[:pd.Timestamp(before) - pd.Timedelta(days=1)])
    return scales


def main() -> None:
    panel, split = load_panel(), load_split_info()
    origins = origins_for(split["val_start"], split["val_end"])
    print(f"Validation: {split['val_start']} to {split['val_end']} | "
          f"{len(origins)} weekly forecast origins x {config.HORIZON}-day horizon x "
          f"{panel[config.PRODUCT_COL].nunique()} products")

    t0 = time.time()
    bt = rolling_origin_backtest(panel, MODEL_FACTORIES, origins)
    print(f"Backtest finished in {time.time() - t0:.0f}s")
    scales = training_scales(panel, split["val_start"])
    overall, per_product = summarize(bt, scales)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bt.to_csv(config.REPORTS_DIR / "val_backtest.csv", index=False)
    overall.to_csv(config.REPORTS_DIR / "model_comparison_val.csv", index=False)
    per_product.to_csv(config.REPORTS_DIR / "val_product_metrics.csv", index=False)

    print("\nValidation comparison (lower is better; MASE < 1 beats the seasonal naive baseline):")
    print(overall.round(3).to_string(index=False))
    print("\nMASE by product:")
    print(per_product.pivot(index="model", columns="product_id", values="mase").round(2).to_string())

    best = overall.iloc[0]
    sigma = {m: {pid: float(r.rmse) for pid, r in g.set_index("product_id").iterrows()}
             for m, g in per_product.groupby("model")}
    base_mase = overall.set_index("model").loc[BASELINE, "mean_mase"]
    print(f"\nSelected model: {best['model']}  (mean MASE {best['mean_mase']:.3f}; "
          f"baseline {BASELINE} {base_mase:.3f})")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SELECTED_MODEL_PATH, "w") as f:
        json.dump({"selected_model": best["model"], "val_mean_mase": float(best["mean_mase"]),
                   "baseline": BASELINE, "daily_error_rmse_by_model": sigma,
                   "assumptions": {"service_level": config.SERVICE_LEVEL,
                                   "lead_time": config.LEAD_TIME,
                                   "review_period": config.REVIEW_PERIOD,
                                   "horizon": config.HORIZON}}, f, indent=2)
    print(f"Saved {config.SELECTED_MODEL_PATH.name}")


if __name__ == "__main__":
    main()
