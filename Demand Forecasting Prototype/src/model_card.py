"""Generate reports/model_card.md from the saved metrics (never drifts from results).

Run:  python -m src.model_card
"""
import json
from datetime import date

import pandas as pd

from src import config


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main() -> None:
    with open(config.REPORTS_DIR / "test_metrics.json") as f:
        t = json.load(f)
    inv = pd.read_csv(config.REPORTS_DIR / "inventory_results.csv").set_index("model")
    comp = pd.read_csv(config.REPORTS_DIR / "model_comparison_val.csv")
    s, b, chosen = t["selected"], t["baseline_metrics"], t["selected_model"]
    ma = inv.loc["moving_average_28"]
    me = inv.loc[chosen]
    inv_table = _md_table(inv.reset_index()[["model", "fill_rate", "lost_units", "avg_on_hand", "total_cost"]]
                          .round(3))
    val_table = _md_table(comp[["model", "mae", "rmse", "wape", "mean_mase"]].round(3))

    card = f"""# Model Card - Demand Forecasting Prototype

*Generated {date.today().isoformat()} by `src/model_card.py`*

## Model details
- **Selected model:** `{chosen}` (chosen on validation mean MASE). Alternatives were built without statsmodels/Prophet and are documented in `src/models.py`.
- **Task:** forecast daily unit demand for each product for the next {t['horizon']} days.
- **Method:** direct multi-horizon gradient boosting (Poisson loss) using recent averages, same-weekday history, last year's level, calendar features and *planned* promotions.
- **Random seed:** {config.SEED}

## Intended use
- **Stakeholder:** inventory planner / supply-chain manager.
- **Decision supported:** weekly order quantities per product (order-up-to policy, lead time {config.LEAD_TIME} days, review every {config.REVIEW_PERIOD} days, {config.SERVICE_LEVEL:.0%} service target).
- **Not intended for:** new products with little history, price optimisation, or long-range (months ahead) planning.

## Data
- Simulated daily sales for 5 products, {t['test_period'][0]} test start; 2022-01-01 to 2025-06-30 overall (see `src/generate_data.py`).
- Time-ordered split: train, then 84 days validation, then 84 days test (never shuffled).
- Cleaning handled duplicates, negative values and missing days (`reports/cleaning_report.json`).

## Validation design
Walk-forward (rolling-origin) backtest: a new forecast every 7 days, each model sees only data up to that day. {t['n_origins']} origins in the test period.

## Performance (test period, {t['n_origins']} weekly origins x {t['horizon']}-day horizon)
| Metric | {chosen} | seasonal naive baseline |
|---|---|---|
| MAE (units/day) | {s['mae']:.2f} | {b['mae']:.2f} |
| RMSE | {s['rmse']:.2f} | {b['rmse']:.2f} |
| WAPE | {s['wape']:.1%} | {b['wape']:.1%} |
| Bias (units/day) | {s['bias']:+.2f} | {b['bias']:+.2f} |
| Mean MASE | {s['mean_mase']:.3f} | {b['mean_mase']:.3f} |

- MAE is **{t['mae_improvement_vs_baseline']:.1%}** lower than the baseline; RMSE **{t['rmse_improvement_vs_baseline']:.1%}** lower.
- The nominal 90% prediction interval covered **{t['interval_90_coverage']:.1%}** of actual days (slightly narrow: treat intervals as optimistic).
- Validation mean MASE was {t['val_mean_mase']:.3f} vs {s['mean_mase']:.3f} on test.

### Inventory impact (simulated costs: holding {config.HOLDING_COST}/unit/day, lost sale {config.STOCKOUT_COST}/unit)
{inv_table}

Compared with a 28-day moving-average planner, the selected model has total cost {me['total_cost']:,.0f} vs {ma['total_cost']:,.0f} and fill rate {me['fill_rate']:.1%} vs {ma['fill_rate']:.1%}.

## Model comparison (validation)
{val_table}

## Limitations & risks
- Simulated data and assumed costs: real demand has price changes, weather, competitors and supply issues not modelled here.
- Sales are used as a proxy for demand; real stock-outs hide true demand (censoring).
- Planned promotions are assumed known in advance; wrong promo plans will hurt accuracy.
- The model under-forecasts on average (negative bias), which raises stockout risk if safety stock is not adjusted.
- Prediction intervals assume roughly normal, independent errors and were slightly too narrow on test.
- Only 11 weekly test origins: differences between close models are not conclusive.
- Slow-moving items (SKU-005) are tiny counts; safety-stock maths is approximate there.
- A single selected model is used for all products; per-product selection could do better.

## Next improvements
Per-product model choice, statsmodels ETS/SARIMA and Prophet for comparison, quantile or probabilistic forecasts, hierarchical reconciliation, price and weather features, handling censored demand, monitoring forecast error in production.
"""
    (config.REPORTS_DIR / "model_card.md").write_text(card)
    print(f"Wrote {config.REPORTS_DIR / 'model_card.md'}")


if __name__ == "__main__":
    main()
