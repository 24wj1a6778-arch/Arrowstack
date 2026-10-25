# Model Card - Demand Forecasting Prototype

*Generated 2026-10-25 by `src/model_card.py`*

## Model details
- **Selected model:** `gbm_direct` (chosen on validation mean MASE). Alternatives were built without statsmodels/Prophet and are documented in `src/models.py`.
- **Task:** forecast daily unit demand for each product for the next 14 days.
- **Method:** direct multi-horizon gradient boosting (Poisson loss) using recent averages, same-weekday history, last year's level, calendar features and *planned* promotions.
- **Random seed:** 42

## Intended use
- **Stakeholder:** inventory planner / supply-chain manager.
- **Decision supported:** weekly order quantities per product (order-up-to policy, lead time 7 days, review every 7 days, 95% service target).
- **Not intended for:** new products with little history, price optimisation, or long-range (months ahead) planning.

## Data
- Simulated daily sales for 5 products, 2025-04-08 test start; 2022-01-01 to 2025-06-30 overall (see `src/generate_data.py`).
- Time-ordered split: train, then 84 days validation, then 84 days test (never shuffled).
- Cleaning handled duplicates, negative values and missing days (`reports/cleaning_report.json`).

## Validation design
Walk-forward (rolling-origin) backtest: a new forecast every 7 days, each model sees only data up to that day. 11 origins in the test period.

## Performance (test period, 11 weekly origins x 14-day horizon)
| Metric | gbm_direct | seasonal naive baseline |
|---|---|---|
| MAE (units/day) | 12.36 | 17.21 |
| RMSE | 20.65 | 28.97 |
| WAPE | 19.9% | 27.7% |
| Bias (units/day) | -2.02 | -1.29 |
| Mean MASE | 0.709 | 0.976 |

- MAE is **28.2%** lower than the baseline; RMSE **28.7%** lower.
- The nominal 90% prediction interval covered **85.6%** of actual days (slightly narrow: treat intervals as optimistic).
- Validation mean MASE was 0.651 vs 0.709 on test.

### Inventory impact (simulated costs: holding 0.1/unit/day, lost sale 3.0/unit)
| model | fill_rate | lost_units | avg_on_hand | total_cost |
|---|---|---|---|---|
| gbm_direct | 0.979 | 552.000 | 1452.274 | 13855.100 |
| holt_winters | 0.963 | 964.000 | 1517.857 | 15642.000 |
| seasonal_naive | 0.974 | 671.000 | 1641.595 | 15802.400 |
| moving_average_28 | 0.962 | 979.000 | 1539.667 | 15870.200 |
| seasonal_mean | 0.961 | 1006.000 | 1539.155 | 15946.900 |
| naive_last | 0.891 | 2836.000 | 1680.167 | 22621.400 |

Compared with a 28-day moving-average planner, the selected model has total cost 13,855 vs 15,870 and fill rate 97.9% vs 96.2%.

## Model comparison (validation)
| model | mae | rmse | wape | mean_mase |
|---|---|---|---|---|
| gbm_direct | 10.828 | 17.105 | 0.192 | 0.651 |
| holt_winters | 11.422 | 18.270 | 0.202 | 0.676 |
| seasonal_mean | 11.719 | 18.832 | 0.207 | 0.695 |
| moving_average_28 | 11.906 | 18.575 | 0.211 | 0.704 |
| seasonal_naive | 14.301 | 22.618 | 0.253 | 0.843 |
| naive_last | 15.052 | 23.186 | 0.266 | 0.894 |

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
