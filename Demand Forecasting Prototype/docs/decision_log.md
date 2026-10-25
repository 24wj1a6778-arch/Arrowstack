# Decision Log

| # | Decision | Alternatives | Reason |
|---|---|---|---|
| 1 | Simulated multi-product daily sales | Public retail datasets | Handbook allows simulation; known patterns make results explainable and the project runs offline. |
| 2 | Time-ordered split (train / 84d validation / 84d test) | Random split, k-fold | Shuffling lets a model see the future. Time order mirrors real use. |
| 3 | Walk-forward backtest, weekly origins, 14-day horizon | One single train/test cut | One cut gives one noisy number; walk-forward matches weekly planning and gives 11 forecasts per product. |
| 4 | Horizon = 14 days | 7 or 28 | Equals lead time (7) + review period (7), the window an order must protect. |
| 5 | Seasonal naive as the baseline | Naive last value | It is the standard hard-to-beat baseline for weekly-seasonal data. |
| 6 | Select on mean MASE | MAE, RMSE, MAPE | Scale-free: a 120-unit and a 2-unit product count equally. MAPE breaks on zero days. |
| 7 | Write Holt-Winters and GBM ourselves | statsmodels, Prophet | Not installable everywhere; allowed as documented alternatives. statsmodels is listed as a next step. |
| 8 | Direct multi-horizon GBM with horizon as a feature | Recursive forecasting | Avoids error build-up and leakage from feeding forecasts back in. |
| 9 | Poisson loss for GBM | Squared error | Sales are counts; keeps forecasts non-negative and handles the slow-moving item. |
| 10 | Planned promotions treated as known inputs | Ignore promotions | Planners know promos in advance; the baselines cannot use them, which is disclosed. |
| 11 | Safety stock = z x validation RMSE x sqrt(14) | Quantile forecasts | Standard, transparent and easy to explain; its weaknesses (independent errors, past = future) are listed as limitations. |
| 12 | Compare every model with the same inventory policy | Compare accuracy only | Accuracy numbers do not tell the planner what a forecast is worth; cost and fill rate do. |
| 13 | Fixed Black Friday rule to "Friday after the 4th Thursday" | 4th Friday | Found while reviewing the calendar code; data was regenerated and results re-run. |

*Add your own entries as you work - mentors read this log.*
