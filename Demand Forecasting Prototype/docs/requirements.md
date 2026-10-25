# Requirements Document - Demand Forecasting Prototype

## 1. Problem and stakeholder
- **Stakeholder:** inventory planner / supply-chain manager at a retailer.
- **Decision:** how many units of each product to order every week.
- **Problem:** ordering too little causes lost sales; ordering too much ties up cash and stock. A better demand forecast lets the planner hold less stock for the same service level.

## 2. Inputs and outputs
- **Input:** daily units sold per product (history), a planned-promotion calendar for the next 14 days, and current stock on hand and on order.
- **Output:** 14-day demand forecast per product with a 90% interval, and a recommended order quantity (order-up-to policy).

## 3. Success metrics (defined before modelling)
| Metric | Target |
|---|---|
| MAE / RMSE vs seasonal-naive baseline on test | at least 15% lower |
| Mean MASE | below 1 (better than in-sample seasonal naive) |
| Inventory simulation | lower total cost than a 28-day moving-average planner at similar or better fill rate |
| 90% interval coverage | within 5 points of 90% |
| Reproducibility | one command, same numbers |

## 4. Scope and constraints
- **In scope:** 14-day daily forecasts for existing products, order recommendations, error analysis.
- **Out of scope:** new products without history, price optimisation, multi-echelon supply chains, live deployment.
- Simulated data only; no personal data. Fixed seed 42.
- Forecast horizon 14 days = lead time 7 + review period 7.

## 5. Edge cases handled
| Case | Handling |
|---|---|
| Duplicate rows | dropped, counted in `reports/cleaning_report.json` |
| Negative units (returns glitch) | clipped to 0, counted |
| Missing days | complete calendar built, gaps interpolated, counted |
| Slow-moving item with zero-sales days | MASE/WAPE instead of MAPE; Poisson loss in GBM |
| Short history (< 120 days) | predict.py falls back to a simpler model and says so |
| Missing promo plan | assumes no promotion |
| Unknown product / negative stock in inventory file | flagged, no order produced |
| Missing columns in input files | clear error naming them |
| Data leakage | time-ordered split; model sees only history up to the origin; tested |

## 6. Acceptance criteria
See the mapping table in `README.md`.

## 7. Assumptions (simulated; state them in the presentation)
Service target 95%; lead time 7 days; weekly review; holding cost 0.10 per unit per day; lost sale cost 3.00 per unit; unmet demand is lost (no backorders); planned promotions are known in advance.
