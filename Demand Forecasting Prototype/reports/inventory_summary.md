# Inventory simulation (test period)

- Policy: order-up-to, review every 7 days, lead time 7 days, target service 95%
- Costs (assumed): holding 0.1/unit/day, lost sale 3.0/unit

| model | fill rate | lost units | avg stock | total cost |
|---|---|---|---|---|
| gbm_direct | 97.9% | 552 | 1,452 | 13,855 |
| holt_winters | 96.3% | 964 | 1,518 | 15,642 |
| seasonal_naive | 97.4% | 671 | 1,642 | 15,802 |
| moving_average_28 | 96.2% | 979 | 1,540 | 15,870 |
| seasonal_mean | 96.1% | 1,006 | 1,539 | 15,947 |
| naive_last | 89.1% | 2,836 | 1,680 | 22,621 |

> Write your own reading: which model achieves the target service with the least stock? Which models miss the target, and why (hint: forecast bias)?
