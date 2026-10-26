# Demand Forecasting Prototype 

A time-series workflow that forecasts daily product demand 14 days ahead and **turns the forecast into
weekly order quantities** for an inventory planner. It covers every handbook requirement: time-ordered
train/validation/test discipline, baselines, metric comparison, error analysis, reproducible pipeline and
model card.

> **Data note:** sales are *simulated* (`src/generate_data.py`): 5 products, 2022-01-01 to 2025-06-30, with
> trend, weekly and yearly seasonality, promotions, holidays, a slow-moving item, and injected data problems
> (duplicates, negative values, missing days). Costs are assumptions.
>
> **Tooling note:** the handbook suggests "statsmodels/Prophet alternatives". The models here are written
> from scratch (and unit-tested) so the project runs with only pandas, numpy, scikit-learn and matplotlib.

## Project structure

```
demand_forecasting/
├── README.md, requirements.txt, run_all.py
├── data/raw/sales.csv                  <- simulated daily sales (long format)
├── data/processed/                     <- demand_daily.csv, split_info.json, demo inputs
├── src/
│   ├── config.py           settings: horizon, split lengths, inventory costs, seed
│   ├── generate_data.py    simulated sales with realistic problems
│   ├── data_prep.py        validate, clean, complete calendar, TIME-ORDERED split
│   ├── features.py         calendar/holiday features + leakage-safe feature builder
│   ├── models.py           6 forecasters: naive, moving average, seasonal naive/mean, Holt-Winters, GBM
│   ├── backtest.py         rolling-origin (walk-forward) backtest
│   ├── metrics.py          MAE, RMSE, WAPE, bias, MASE
│   ├── train.py            compare models on validation, pick one, measure error spread
│   ├── evaluate.py         one-time test evaluation, interval coverage, charts
│   ├── inventory.py        order-up-to policy, safety stock, simulation
│   ├── inventory_sim.py    what each model is worth in cost and service
│   ├── error_analysis.py   errors by horizon, weekday, promo, product
│   ├── eda.py              exploratory analysis (training period only)
│   ├── model_card.py       builds reports/model_card.md from real results
│   └── predict.py          forecast the next 14 days + recommend orders
├── tests/test_pipeline.py  34 unit tests (including "no peeking at the future")
├── models/selected_model.json
├── reports/                metrics, figures, model card, analyses
└── docs/                   requirements, decision log, work log, presentation outline, viva prep
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py                    # everything in one go (about 1 minute)

# or step by step
python -m src.generate_data --force  # 1. simulate sales
python -m src.data_prep              # 2. clean + time-ordered split
python -m src.eda                    # 3. explore (training period only)
python -m src.train                  # 4. compare 6 models on validation
python -m src.evaluate               # 5. final test-period results
python -m src.inventory_sim          # 6. what the forecast is worth in inventory terms
python -m src.error_analysis         # 7. where it fails
python -m src.model_card             # 8. model card
python -m src.predict --inventory-file data/processed/sample_inventory.csv --promo-file data/processed/sample_future_promos.csv

python -m unittest discover -s tests -v
```

## Method in brief

1. **Time-ordered split.** Train, then 84 days validation, then 84 days test. Never shuffled.
2. **Walk-forward backtest.** Every week we stand on a "forecast origin", show models only the past, and forecast the
   next 14 days. 11 origins per period, 5 products, 6 models.
3. **Baselines first.** Naive, 28-day moving average and **seasonal naive** (same weekday last week) are the bars to clear.
4. **Models.** Seasonal mean; Holt-Winters (damped trend + weekly seasonality); direct gradient boosting with
   recent averages, same-weekday history, last year's level, calendar, holidays and planned promotions.
5. **Metrics.** MAE, RMSE, WAPE, bias and MASE (selection metric: scale-free, so a 120-unit item and a 2-unit item count equally).
   MAPE is not used because it breaks on zero-sales days.
6. **Inventory decision.** Weekly order-up-to policy (lead time 7 days, review 7 days): order-up-to level =
   forecast for the next 14 days + safety stock (z x forecast error x sqrt(14)), with the error measured on validation.
7. **Business simulation.** Same policy and costs for every model on the test period: fill rate, stock held, total cost.

## Results (test period 2025-04-08 to 2025-06-30, 11 weekly origins x 14 days x 5 products)

| Metric | gbm_direct (selected) | seasonal naive baseline |
|---|---|---|
| MAE (units/day) | **12.36** | 17.21 |
| RMSE | **20.65** | 28.97 |
| WAPE | 19.9% | 27.7% |
| Mean MASE | **0.709** | 0.976 |
| Bias | -2.02 units/day | -1.29 |

- MAE is **28.2% lower** and RMSE **28.7% lower** than the seasonal-naive baseline. Validation shows the same ordering (MASE 0.651 vs 0.843).
- The 90% prediction interval covered **85.6%** of actual days (a little too narrow).

| Inventory simulation (test) | Fill rate | Avg units in stock | Total cost |
|---|---|---|---|
| **gbm_direct** | **97.9%** | **1,452** | **13,855** |
| seasonal naive | 97.4% | 1,642 | 15,802 |
| 28-day moving average | 96.2% | 1,540 | 15,870 |
| naive (last value) | 89.1% | 1,680 | 22,621 |

Versus a 28-day moving-average planner the selected model costs **12.7% less** while serving more demand.

**Honest findings**
- The model **under-forecasts promotion days** (bias -9.1 units vs -1.4 on normal days) and the high-volume
  product SKU-001 (bias -11.6 units/day), which is where most of its lost sales happen.
- Error is **roughly flat across the 14 days ahead**: day 1 MAE 11.3, day 14 MAE 11.8. It beats the baseline at every horizon.
- Safety stock sized from validation errors over-protected SKU-002 (the validation period was harder for it) and
  under-protected SKU-001. Safety stock is only as good as the past error is representative.
- Only 11 weekly origins with overlapping windows: close models (Holt-Winters, moving average, seasonal mean) cannot be separated reliably.

## Handbook acceptance criteria -> evidence

| Criterion | Where to find it |
|---|---|
| Core requirements complete and mapped to evidence | this table + `docs/requirements.md` |
| Reproducible by a mentor | `requirements.txt`, fixed seed, `run_all.py`, this README |
| Validation documented, issues disclosed | `reports/model_comparison_val.csv`, `test_metrics.json`, `test_results.txt`, model card limitations |
| Professionally organised | folder structure above |
| Can explain decisions and trade-offs | `docs/decision_log.md`, `docs/viva_prep.md` |

## Limitations

- Simulated data and assumed costs; real demand has prices, weather, competitors and supply problems.
- Sales are a proxy for demand (stock-outs hide true demand). Planned promotions are assumed known in advance.
- One model for all products; per-product selection could help. The slow-moving item (SKU-005) is tiny counts, so safety-stock maths is approximate.
- Prediction intervals assume near-normal, independent errors.
- Hand-written Holt-Winters is simpler than statsmodels' ETS (no automatic initialisation or parameter optimiser).
- Cleaning fills the 26 missing days by linear interpolation between the days on either side. For a gap that sits at a
  split boundary this borrows one or two values from the next period. The effect is negligible here (26 days of 6,365 rows),
  but a production pipeline would fill gaps using only past data.

## Next improvements

statsmodels ETS/SARIMA and Prophet as extra benchmarks, per-product model choice, quantile/probabilistic forecasts,
price and weather features, censored-demand handling, hierarchical reconciliation, forecast-error monitoring.
