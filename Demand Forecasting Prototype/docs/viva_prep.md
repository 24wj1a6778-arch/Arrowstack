# Viva / Interview Preparation - Project 3

Use these as starting points; answer in your own words.

**What problem does it solve and how did you define success?**
A planner must order stock every week. Success = a forecast clearly better than a simple seasonal baseline (MAE/RMSE, MASE) that also lowers inventory cost at a similar or better fill rate.

**Why can't you split time-series data randomly?**
A random split puts future days in training, so the model effectively sees the answer. I split in time order and used a walk-forward backtest where every forecast only sees the past.

**How did you prevent leakage?**
The backtest slices history with `.loc[:origin]`; features use only data up to the origin; only planned promotions and calendar facts (known in advance) look ahead. A unit test corrupts all later data and checks the forecasts do not change.

**What is MASE and why did you pick it?**
MAE divided by the MAE of a seasonal-naive forecast on the training data. It is scale-free, so big and small products count equally, and below 1 means better than the baseline. MAPE fails when sales are zero.

**Which model won and is that trustworthy?**
Gradient boosting: test MAE 12.4 vs 17.2 for the seasonal naive baseline (28% lower), consistent with validation. Holt-Winters, moving average and seasonal mean are close to each other; with 11 overlapping origins I would not rank those reliably.

**How does the forecast become an order?**
Order-up-to level = forecast for the next 14 days (lead time + review period) + safety stock. Safety stock = z x daily forecast error x sqrt(14), z = 1.645 for 95%. Order quantity = that level minus stock on hand and on order.

**What did the inventory simulation show?**
Same policy for every model: gradient boosting reached a 97.9% fill rate with about 1,450 units in stock, versus 96.2% with 1,540 for a 28-day moving average. Total cost 12.7% lower. A bad forecast (naive last value) fell to 89% fill.

**Where does the model fail?**
It under-forecasts promotion days (bias about -9 units) and the largest product, so lost sales concentrate there. Prediction intervals were a bit narrow (85.6% coverage vs 90%).

**Edge cases and risks?**
Duplicates, negative values, missing days, a slow-moving item, short history (fallback model), missing promo plan, unknown products, negative stock. Risks: sales are not true demand (stock-outs hide demand), promo plans may change, safety stock relies on past errors.

**Why not statsmodels or Prophet?**
Not installable everywhere; the handbook allows documented alternatives. I wrote Holt-Winters and a GBM forecaster and tested them. statsmodels ETS/SARIMA and Prophet are the next benchmarks.

**What would change at 10x the data (thousands of products)?**
One global model across products, hierarchical forecasting and reconciliation, automated retraining, error monitoring, a feature store, and probabilistic (quantile) forecasts instead of normal-error safety stock.

**Explain it to a non-technical manager.**
"We predict how many units each product will sell over the next two weeks and order just enough, plus a cushion sized to how wrong our predictions usually are. In testing this cost about 13% less than a simple average-based method and served more of the demand."

**Terms:** time-ordered split, walk-forward backtest, forecast origin, horizon, seasonality, baseline, MAE, RMSE, WAPE, MASE, bias, lead time, safety stock, service level, order-up-to policy, fill rate, leakage.
