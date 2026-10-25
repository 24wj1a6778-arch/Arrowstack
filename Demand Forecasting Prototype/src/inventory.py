"""Turn forecasts into inventory decisions.

Policy: periodic-review, order-up-to ("base stock")
  * every REVIEW_PERIOD (7) days we place an order
  * it arrives LEAD_TIME (7) days later
  * so each order must protect against demand over LEAD_TIME + REVIEW_PERIOD = 14 days
  * order-up-to level   S = forecast(next 14 days) + safety stock
  * safety stock        z * sigma_daily * sqrt(14)      (z from the service level, 95% -> 1.645)
  * order quantity      max(0, S - (on hand + on order)), rounded up

sigma_daily is the model's daily forecast error (RMSE) measured on VALIDATION data.
A better forecast has a smaller sigma, so it needs less safety stock for the same service.

Unmet demand is treated as lost sales (the customer walks away).
"""
import math
from statistics import NormalDist

import numpy as np

from src import config

PROTECTION_DAYS = config.LEAD_TIME + config.REVIEW_PERIOD


def z_value(service_level: float = config.SERVICE_LEVEL) -> float:
    if not 0.5 <= service_level < 1:
        raise ValueError("service_level must be in [0.5, 1)")
    return NormalDist().inv_cdf(service_level)


def safety_stock(sigma_daily: float, service_level: float = config.SERVICE_LEVEL,
                 days: int = PROTECTION_DAYS) -> float:
    return z_value(service_level) * sigma_daily * math.sqrt(days)


def order_up_to_level(forecast, sigma_daily: float, service_level: float = config.SERVICE_LEVEL) -> float:
    f = np.asarray(forecast, dtype=float)
    if len(f) < PROTECTION_DAYS:
        raise ValueError(f"need a {PROTECTION_DAYS}-day forecast, got {len(f)} days")
    return float(f[:PROTECTION_DAYS].sum() + safety_stock(sigma_daily, service_level))


def order_quantity(order_up_to: float, on_hand: float, on_order: float = 0.0) -> int:
    if on_hand < 0 or on_order < 0:
        raise ValueError("on_hand and on_order cannot be negative")
    return int(max(0, math.ceil(order_up_to - (on_hand + on_order))))


def simulate_policy(demand, forecasts, sigma_daily: float,
                    service_level: float = config.SERVICE_LEVEL) -> dict:
    """Simulate the policy day by day.

    demand     actual daily demand over the simulated period
    forecasts  list of 14-day forecasts, one per weekly review (review k happens on day 7k)
    """
    demand = np.asarray(demand, dtype=float)
    levels = [order_up_to_level(f, sigma_daily, service_level) for f in forecasts]
    on_hand = float(math.ceil(levels[0]))        # start fully stocked for the first cycle
    pipeline = []                                # (arrival_day, quantity)
    lost = held = 0.0
    for d in range(len(demand)):
        arrived = sum(q for a, q in pipeline if a == d)
        pipeline = [(a, q) for a, q in pipeline if a != d]
        on_hand += arrived
        if d % config.REVIEW_PERIOD == 0:
            k = d // config.REVIEW_PERIOD
            if k < len(levels):
                qty = order_quantity(levels[k], on_hand, sum(q for _, q in pipeline))
                if qty > 0:
                    pipeline.append((d + config.LEAD_TIME, qty))
        sold = min(on_hand, demand[d])
        lost += demand[d] - sold
        on_hand -= sold
        held += on_hand
    total = float(demand.sum())
    return {
        "total_demand": total,
        "lost_units": float(lost),
        "fill_rate": 1 - lost / total if total > 0 else 1.0,
        "avg_on_hand": held / len(demand),
        "holding_cost": config.HOLDING_COST * held,
        "stockout_cost": config.STOCKOUT_COST * lost,
        "total_cost": config.HOLDING_COST * held + config.STOCKOUT_COST * lost,
    }
