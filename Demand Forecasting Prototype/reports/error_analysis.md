# Error analysis (test period)

- Model: **gbm_direct** | rows analysed: 770 (product-days across overlapping weekly forecasts)

## By forecast horizon

- Day 1 MAE 11.3, day 14 MAE 11.8 units: error is roughly flat across the 14 days (trend under 2% per day).
- The model beats the seasonal-naive baseline at every horizon (14 of 14 days ahead).

## By condition

- Promo days: MAE 16.4, bias -9.1 | non-promo days: MAE 12.0, bias -1.4
- Holidays: MAE 16.2 (only 10 product-days: treat with caution)

## By product

- SKU-001: MAE 30.8, WAPE 18%, bias -11.6
- SKU-002: MAE 15.9, WAPE 21%, bias +1.1
- SKU-003: MAE 8.9, WAPE 23%, bias +0.4
- SKU-004: MAE 4.9, WAPE 25%, bias -0.4
- SKU-005: MAE 1.2, WAPE 52%, bias +0.4

- Largest under-forecast bias: SKU-001 (-11.6 units/day). Under-forecasting causes stockouts.

> Interpretation (write your own): which products, days or conditions are hardest, and why? What data would help (price changes, weather, competitor actions, stock-outs hiding demand)?
