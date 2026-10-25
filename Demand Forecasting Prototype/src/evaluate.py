"""Final evaluation on the untouched TEST period (the last 84 days).

Walk-forward backtest of every model, exactly like validation, then:
  * accuracy table (MAE, RMSE, WAPE, bias, MASE) and improvement over the baseline
  * how well the 90% prediction interval covers the real demand
  * forecast-vs-actual charts

Run:  python -m src.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.backtest import origins_for, rolling_origin_backtest
from src.data_prep import load_panel, load_split_info, product_series
from src.inventory import z_value
from src.metrics import summarize
from src.models import BASELINE, MODEL_FACTORIES
from src.train import training_scales


def interval_coverage(bt_model: pd.DataFrame, sigma: dict, level=config.INTERVAL_LEVEL):
    z = z_value(0.5 + level / 2)
    s = bt_model["product_id"].map(sigma)
    lo, hi = (bt_model.forecast - z * s).clip(lower=0), bt_model.forecast + z * s
    inside = (bt_model.actual >= lo) & (bt_model.actual <= hi)
    return float(inside.mean())


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panel, split = load_panel(), load_split_info()
    with open(config.SELECTED_MODEL_PATH) as f:
        sel = json.load(f)
    chosen = sel["selected_model"]

    origins = origins_for(split["test_start"], split["test_end"])
    bt = rolling_origin_backtest(panel, MODEL_FACTORIES, origins)
    scales = training_scales(panel, split["test_start"])
    overall, per_product = summarize(bt, scales)
    bt.to_csv(config.REPORTS_DIR / "test_backtest.csv", index=False)
    overall.to_csv(config.REPORTS_DIR / "model_comparison_test.csv", index=False)
    per_product.to_csv(config.REPORTS_DIR / "test_product_metrics.csv", index=False)

    ov = overall.set_index("model")
    base, best = ov.loc[BASELINE], ov.loc[chosen]
    coverage = interval_coverage(bt[bt.model == chosen], sel["daily_error_rmse_by_model"][chosen])
    results = {
        "selected_model": chosen, "baseline": BASELINE,
        "test_period": [split["test_start"], split["test_end"]],
        "n_origins": len(origins), "horizon": config.HORIZON,
        "selected": best.to_dict(), "baseline_metrics": base.to_dict(),
        "mae_improvement_vs_baseline": float(1 - best["mae"] / base["mae"]),
        "rmse_improvement_vs_baseline": float(1 - best["rmse"] / base["rmse"]),
        "interval_90_coverage": coverage,
        "val_mean_mase": sel["val_mean_mase"],
        "all_models": overall.to_dict(orient="records"),
    }
    with open(config.REPORTS_DIR / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Test period {split['test_start']} to {split['test_end']}: {len(origins)} weekly origins x "
          f"{config.HORIZON}-day horizon x {panel[config.PRODUCT_COL].nunique()} products")
    print(overall.round(3).to_string(index=False))
    print(f"\nSelected model: {chosen}")
    print(f"  MAE  {best['mae']:.2f} vs baseline {base['mae']:.2f}  ({results['mae_improvement_vs_baseline']:.1%} better)")
    print(f"  RMSE {best['rmse']:.2f} vs baseline {base['rmse']:.2f}  ({results['rmse_improvement_vs_baseline']:.1%} better)")
    print(f"  WAPE {best['wape']:.1%} | bias {best['bias']:+.2f} units/day | mean MASE {best['mean_mase']:.3f} "
          f"(validation {sel['val_mean_mase']:.3f})")
    print(f"  90% prediction interval covered {coverage:.1%} of actual days")
    print("\nMASE by product (test):")
    print(per_product.pivot(index="model", columns="product_id", values="mase").round(2).to_string())

    # ---- figures ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2b6cb0" if m == chosen else "#a0aec0" for m in overall["model"]]
    ax.barh(overall["model"][::-1], overall["mean_mase"][::-1], color=colors[::-1])
    ax.axvline(1.0, color="red", ls="--", lw=1, label="seasonal-naive level (in-sample)")
    ax.set_xlabel("Mean MASE on test (lower is better)"); ax.set_title("Model comparison (test)")
    ax.legend(); fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_comparison_test.png", dpi=150); plt.close(fig)

    g = bt[bt.model == chosen]
    pids = sorted(panel[config.PRODUCT_COL].unique())
    names = panel.groupby(config.PRODUCT_COL)[config.NAME_COL].first()
    fig, axes = plt.subplots(len(pids), 1, figsize=(11, 2.4 * len(pids)), sharex=True)
    for ax, pid in zip(axes, pids):
        y, _ = product_series(panel, pid)
        t = y.loc[split["test_start"]:split["test_end"]]
        stitched = g[(g.product_id == pid) & (g.h <= config.STRIDE)].sort_values("date")
        ax.plot(t.index, t.values, color="#4a5568", lw=1.2, label="Actual")
        ax.plot(stitched["date"], stitched["forecast"], color="#c53030", lw=1.2, label=f"Forecast ({chosen})")
        ax.set_ylabel(f"{pid}\n{names[pid]}", fontsize=8)
    axes[0].legend(loc="upper left", ncol=2); axes[0].set_title("Test period: actual vs forecast (first week of each weekly forecast)")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "forecast_vs_actual.png", dpi=150); plt.close(fig)
    print(f"Figures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
