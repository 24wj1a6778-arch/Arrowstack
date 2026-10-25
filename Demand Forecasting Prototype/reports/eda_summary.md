# EDA summary (training period)

- Period: 2022-01-01 to 2025-01-13 | products: 5

| product | mean/day | weekend vs weekday | promo uplift | zero-sales days | peak month |
|---|---|---|---|---|---|
| SKU-001 Bottled Water | 138.1 | x1.15 | x1.41 | 0% | 8 |
| SKU-002 Instant Noodles | 76.3 | x1.07 | x1.44 | 0% | 12 |
| SKU-003 School Notebook | 39.0 | x0.58 | x1.27 | 0% | 7 |
| SKU-004 Phone Charger | 22.0 | x1.02 | x1.67 | 0% | 12 |
| SKU-005 Specialty Spice | 2.4 | x1.08 | x1.51 | 10% | 6 |

Note: the 'promo uplift' compares average sales on promo and non-promo days; promos are not randomly timed, so treat it as indicative, not causal.

Autocorrelation of SKU-001 at lags (days): 1: 0.56, 7: 0.45, 14: 0.47, 28: 0.43, 364: 0.52
(strong at 7, 14, 28 = weekly pattern; at 364 = yearly pattern)
