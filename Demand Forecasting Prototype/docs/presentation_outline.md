# Presentation Outline (10 slides, ~10 minutes)

1. **Problem and stakeholder** - an inventory planner orders weekly; too little loses sales, too much ties up cash.
2. **Data** - 5 simulated products, 3.5 years of daily sales, promotions and holidays; problems found and fixed (duplicates, negatives, gaps).
3. **What demand looks like** - `eda_series.png`, `eda_seasonality.png`: weekly rhythm, yearly peaks, promotion uplift, a slow-moving item.
4. **Why time series is different** - never shuffle; train, validation, test in time order; walk-forward backtest.
5. **Baselines and models** - naive, moving average, seasonal naive, seasonal mean, Holt-Winters, gradient boosting.
6. **Model comparison** - `model_comparison_test.png`; gbm_direct MASE 0.71 vs 0.98 baseline; MAE 28% lower.
7. **Forecast vs actual** - `forecast_vs_actual.png`; tracks level and promotions, misses daily noise (and that is fine).
8. **From forecast to order** - safety stock, order-up-to policy; `inventory_comparison.png`: 12.7% lower cost than a moving-average planner with a higher fill rate.
9. **Where it fails** - under-forecasts promotion days (bias -9) and the biggest product; intervals slightly narrow (85.6% vs 90%).
10. **Limitations and next steps** - simulated data, censored demand, per-product models, statsmodels/Prophet benchmarks, monitoring; demo `python -m src.predict`.

**Tip:** one takeaway sentence per slide, evidence underneath, code stays in the repo.
