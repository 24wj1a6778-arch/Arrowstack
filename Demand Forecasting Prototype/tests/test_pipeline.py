"""Unit tests. Run with either:

    python -m unittest discover -s tests -v
    pytest -v            (if pytest is installed)
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.backtest import origins_for, rolling_origin_backtest  # noqa: E402
from src.data_prep import clean, make_split_info, product_series, validate_schema  # noqa: E402
from src.features import direct_feature_matrix, holiday_flags  # noqa: E402
from src.generate_data import generate  # noqa: E402
from src.inventory import (order_quantity, order_up_to_level, safety_stock,  # noqa: E402
                           simulate_policy, z_value)
from src.metrics import bias, mae, mase, rmse, seasonal_naive_scale, summarize, wape  # noqa: E402
from src.models import (MODEL_FACTORIES, GBMDirect, HoltWinters, MovingAverage,  # noqa: E402
                        NaiveLast, SeasonalMean, SeasonalNaive)
from src.predict import choose_model_for, forecast_next, read_promo_file, recommend_orders  # noqa: E402


def make_series(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(np.asarray(values, dtype=float), index=idx), pd.Series(0, index=idx)


# built once: full simulated data, cleaned
RAW = generate(seed=1)
PANEL, REPORT = clean(RAW)


class TestData(unittest.TestCase):
    def test_generation_is_reproducible(self):
        pd.testing.assert_frame_equal(generate(seed=7), generate(seed=7))

    def test_raw_data_contains_the_injected_problems(self):
        self.assertTrue(RAW.duplicated(subset=["date", "product_id"]).any())
        self.assertTrue((RAW["units_sold"] < 0).any())

    def test_clean_fixes_each_problem(self):
        self.assertGreater(REPORT["duplicates_removed"], 0)
        self.assertGreater(REPORT["negative_values_clipped_to_0"], 0)
        self.assertGreater(REPORT["missing_days_filled_by_interpolation"], 0)
        self.assertFalse(PANEL.duplicated(subset=["date", "product_id"]).any())
        self.assertTrue((PANEL["units"] >= 0).all())
        self.assertFalse(PANEL["units"].isna().any())

    def test_clean_gives_complete_daily_calendar_per_product(self):
        for _, g in PANEL.groupby("product_id"):
            self.assertEqual(len(g), REPORT["days_per_product"])
            self.assertTrue((g["date"].diff().dropna() == pd.Timedelta(days=1)).all())

    def test_validate_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            validate_schema(RAW.drop(columns=["units_sold"]))
        bad = RAW.copy(); bad.loc[0, "promo"] = 3
        with self.assertRaises(ValueError):
            validate_schema(bad)
        bad = RAW.copy(); bad["date"] = bad["date"].astype(object); bad.loc[0, "date"] = "not a date"
        with self.assertRaises(ValueError):
            validate_schema(bad)

    def test_split_is_time_ordered_and_non_overlapping(self):
        s = {k: pd.Timestamp(v) for k, v in make_split_info(PANEL).items()}
        self.assertLess(s["train_end"], s["val_start"])
        self.assertLess(s["val_end"], s["test_start"])
        self.assertEqual((s["val_end"] - s["val_start"]).days + 1, config.VAL_DAYS)
        self.assertEqual((s["test_end"] - s["test_start"]).days + 1, config.TEST_DAYS)
        self.assertEqual(s["test_end"], PANEL["date"].max())


class TestFeatures(unittest.TestCase):
    def test_holiday_flags(self):
        d = pd.to_datetime(["2024-01-01", "2024-03-05", "2024-11-29", "2024-11-22", "2025-11-28"])
        self.assertEqual(list(holiday_flags(d)), [1, 0, 1, 0, 1])      # Black Friday = day after 4th Thursday

    def test_features_use_only_data_up_to_origin(self):
        y = np.arange(1, 500, dtype=float)
        dates = pd.date_range("2023-01-01", periods=len(y) + 14)
        promo = np.zeros(len(y) + 14)
        o = np.array([400, 400]); h = np.array([1, 14])
        a = direct_feature_matrix(y, promo, dates, o, h)
        y2 = y.copy(); y2[401:] = 99999.0                              # change the "future"
        b = direct_feature_matrix(y2, promo, dates, o, h)
        pd.testing.assert_frame_equal(a, b)

    def test_features_have_no_unexpected_nans(self):
        y = np.random.default_rng(0).poisson(20, 500).astype(float)
        dates = pd.date_range("2023-01-01", periods=514)
        X = direct_feature_matrix(y, np.zeros(514), dates, np.full(14, 499), np.arange(1, 15))
        self.assertFalse(X.drop(columns=["last_year_mean7"]).isna().any().any())
        self.assertFalse(X["last_year_mean7"].isna().any())            # 500 days of history is enough


class TestModels(unittest.TestCase):
    def test_naive_and_moving_average(self):
        y, p = make_series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3)
        self.assertTrue((NaiveLast().fit(y, p).predict(3) == 10).all())
        self.assertAlmostEqual(MovingAverage(window=28).fit(y, p).predict(1)[0], y.iloc[-28:].mean())

    def test_seasonal_naive_repeats_last_week(self):
        week = [10, 20, 30, 40, 50, 60, 70]
        y, p = make_series(week * 4)
        self.assertEqual(list(SeasonalNaive().fit(y, p).predict(14)), week * 2)

    def test_seasonal_mean_averages_same_weekdays(self):
        weeks = [[1] * 7, [2] * 7, [3] * 7, [4] * 7]
        y, p = make_series(sum(weeks, []))
        self.assertTrue(np.allclose(SeasonalMean(weeks=4).fit(y, p).predict(7), 2.5))

    def test_holt_winters_learns_weekly_pattern(self):
        pattern = np.array([100, 110, 105, 95, 120, 150, 140], dtype=float)
        rng = np.random.default_rng(0)
        y, p = make_series(np.tile(pattern, 40) + rng.normal(0, 2, 280))
        f = HoltWinters().fit(y, p).predict(7)
        self.assertLess(np.abs(f - pattern).max(), 8)

    def test_every_model_gives_finite_nonnegative_forecasts(self):
        y, p = product_series(PANEL, "SKU-005")
        y, p = y.iloc[:600], p.iloc[:600]
        for name, factory in MODEL_FACTORIES.items():
            f = factory().fit(y, p).predict(config.HORIZON, np.zeros(config.HORIZON))
            self.assertEqual(len(f), config.HORIZON, name)
            self.assertTrue(np.isfinite(f).all() and (f >= 0).all(), name)

    def test_gbm_uses_planned_promotions(self):
        y, p = product_series(PANEL, "SKU-001")
        y, p = y.iloc[:700], p.iloc[:700]
        m = GBMDirect().fit(y, p)
        no_promo = m.predict(config.HORIZON, np.zeros(config.HORIZON)).sum()
        with_promo = m.predict(config.HORIZON, np.ones(config.HORIZON)).sum()
        self.assertGreater(with_promo, no_promo)

    def test_short_history_and_long_horizon_rejected(self):
        y, p = make_series(range(1, 21))                    # 20 days: too short for GBM (150) and HW (28)
        with self.assertRaises(ValueError):
            GBMDirect().fit(y, p)
        with self.assertRaises(ValueError):
            HoltWinters().fit(y, p)
        with self.assertRaises(ValueError):
            SeasonalNaive().fit(make_series([1, 2, 3])[0], None)
        y, p = product_series(PANEL, "SKU-001")
        with self.assertRaises(ValueError):
            GBMDirect().fit(y.iloc[:400], p.iloc[:400]).predict(config.HORIZON + 1)


class TestBacktest(unittest.TestCase):
    def test_origins_fit_inside_period(self):
        start, end = pd.Timestamp("2025-04-08"), pd.Timestamp("2025-06-30")
        o = origins_for(start, end)
        self.assertEqual(o[0], start - pd.Timedelta(days=1))
        self.assertEqual(len(o), 11)
        self.assertLessEqual(o[-1] + pd.Timedelta(days=config.HORIZON), end)
        self.assertTrue(all((b - a).days == config.STRIDE for a, b in zip(o, o[1:])))

    def test_backtest_never_sees_the_future(self):
        origin = pd.Timestamp("2024-06-30")
        factories = {k: MODEL_FACTORIES[k] for k in ("seasonal_naive", "holt_winters", "gbm_direct")}
        a = rolling_origin_backtest(PANEL, factories, [origin], products=["SKU-001"])
        tampered = PANEL.copy()
        tampered.loc[tampered["date"] > origin, "units"] = 123456            # corrupt every later day
        b = rolling_origin_backtest(tampered, factories, [origin], products=["SKU-001"])
        np.testing.assert_allclose(a["forecast"].values, b["forecast"].values)
        self.assertFalse(np.allclose(a["actual"].values, b["actual"].values))   # actuals do change

    def test_backtest_columns_and_shape(self):
        bt = rolling_origin_backtest(PANEL, {"seasonal_naive": SeasonalNaive},
                                     [pd.Timestamp("2024-06-30")], products=["SKU-002"])
        self.assertEqual(len(bt), config.HORIZON)
        for col in ("model", "product_id", "origin", "h", "date", "actual", "forecast", "promo"):
            self.assertIn(col, bt.columns)


class TestMetrics(unittest.TestCase):
    def test_formulas(self):
        a, f = np.array([10, 20, 30]), np.array([12, 18, 36])
        self.assertAlmostEqual(mae(a, f), (2 + 2 + 6) / 3)
        self.assertAlmostEqual(rmse(a, f), math.sqrt((4 + 4 + 36) / 3))
        self.assertAlmostEqual(wape(a, f), 10 / 60)
        self.assertAlmostEqual(bias(a, f), (2 - 2 + 6) / 3)

    def test_wape_with_zero_demand_is_nan_not_inf(self):
        self.assertTrue(math.isnan(wape([0, 0], [1, 1])))

    def test_mase_scale_and_value(self):
        y = [10, 12, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12, 10, 12]
        self.assertAlmostEqual(seasonal_naive_scale(y, m=2), 0.0 or 1.0)          # perfectly seasonal -> fallback 1
        self.assertAlmostEqual(seasonal_naive_scale([1, 2, 3, 4], m=1), 1.0)
        self.assertAlmostEqual(mase([10], [12], 4.0), 0.5)

    def test_summarize_ranks_better_model_first(self):
        rows = []
        for m, err in (("good", 1), ("bad", 5)):
            for i in range(20):
                rows.append({"model": m, "product_id": "A", "actual": 10 + i % 3, "forecast": 10 + i % 3 + err})
        overall, per_product = summarize(pd.DataFrame(rows), {"A": 2.0})
        self.assertEqual(overall.iloc[0]["model"], "good")
        self.assertAlmostEqual(per_product.set_index("model").loc["good", "mase"], 0.5)


class TestInventory(unittest.TestCase):
    def test_z_value_and_safety_stock(self):
        self.assertAlmostEqual(z_value(0.95), 1.645, places=3)
        self.assertAlmostEqual(safety_stock(10, 0.95, days=14), z_value(0.95) * 10 * math.sqrt(14), places=6)
        with self.assertRaises(ValueError):
            z_value(1.0)

    def test_order_up_to_and_quantity(self):
        level = order_up_to_level([10] * 14, sigma_daily=0)
        self.assertEqual(level, 140)
        self.assertEqual(order_quantity(level, on_hand=50, on_order=30), 60)
        self.assertEqual(order_quantity(level, on_hand=500), 0)
        with self.assertRaises(ValueError):
            order_quantity(level, on_hand=-1)
        with self.assertRaises(ValueError):
            order_up_to_level([10] * 5, sigma_daily=1)

    def test_perfect_forecast_never_stocks_out(self):
        demand = np.full(84, 10.0)
        res = simulate_policy(demand, [[10.0] * 14] * 11, sigma_daily=0.0)
        self.assertEqual(res["lost_units"], 0)
        self.assertEqual(res["fill_rate"], 1.0)

    def test_under_forecasting_causes_lost_sales(self):
        demand = np.full(84, 10.0)
        res = simulate_policy(demand, [[5.0] * 14] * 11, sigma_daily=0.0)
        self.assertGreater(res["lost_units"], 0)
        self.assertLess(res["fill_rate"], 1.0)

    def test_more_safety_stock_means_more_inventory_and_fewer_lost_sales(self):
        rng = np.random.default_rng(0)
        demand = rng.poisson(50, 84).astype(float)
        fc = [[50.0] * 14] * 11
        low = simulate_policy(demand, fc, sigma_daily=1.0)
        high = simulate_policy(demand, fc, sigma_daily=15.0)
        self.assertGreater(high["avg_on_hand"], low["avg_on_hand"])
        self.assertLessEqual(high["lost_units"], low["lost_units"])

    def test_zero_demand_costs_only_holding(self):
        res = simulate_policy(np.zeros(84), [[0.0] * 14] * 11, sigma_daily=0.0)
        self.assertEqual(res["lost_units"], 0)
        self.assertEqual(res["stockout_cost"], 0)


class TestPredictEdgeCases(unittest.TestCase):
    def test_short_history_falls_back(self):
        self.assertEqual(choose_model_for(500, "gbm_direct"), ("gbm_direct", ""))
        name, note = choose_model_for(60, "gbm_direct")
        self.assertEqual(name, "seasonal_mean"); self.assertIn("short history", note)
        self.assertEqual(choose_model_for(10, "gbm_direct")[0], "moving_average_28")
        with self.assertRaises(ValueError):
            choose_model_for(0, "gbm_direct")

    def test_forecast_next_with_short_history_product(self):
        panel = PANEL[PANEL.date >= PANEL.date.max() - pd.Timedelta(days=60)]
        fc = forecast_next(panel, "gbm_direct", {})
        self.assertEqual(len(fc), 5 * config.HORIZON)
        self.assertTrue((fc["model_used"] == "seasonal_mean").all())
        self.assertTrue(fc["note"].str.contains("short history").all())
        self.assertTrue((fc["lower_90"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper_90"]).all())

    def test_promo_file_errors(self):
        tmp = Path(__file__).parent / "_tmp_promo.csv"
        try:
            pd.DataFrame({"date": ["2025-07-01"], "product_id": ["SKU-001"]}).to_csv(tmp, index=False)
            with self.assertRaises(ValueError) as ctx:
                read_promo_file(tmp, "SKU-001", pd.date_range("2025-07-01", periods=3))
            self.assertIn("promo", str(ctx.exception))
            pd.DataFrame({"date": ["2025-07-01"], "product_id": ["SKU-001"], "promo": [5]}).to_csv(tmp, index=False)
            with self.assertRaises(ValueError):
                read_promo_file(tmp, "SKU-001", pd.date_range("2025-07-01", periods=3))
            pd.DataFrame({"date": ["2025-07-02"], "product_id": ["SKU-001"], "promo": [1]}).to_csv(tmp, index=False)
            self.assertEqual(list(read_promo_file(tmp, "SKU-001", pd.date_range("2025-07-01", periods=3))), [0, 1, 0])
        finally:
            tmp.unlink(missing_ok=True)

    def test_order_recommendations_handle_bad_rows(self):
        panel = PANEL[PANEL.date >= PANEL.date.max() - pd.Timedelta(days=60)]
        fc = forecast_next(panel, "seasonal_mean", {"seasonal_mean": {"SKU-001": 10.0}})
        inv = pd.DataFrame({"product_id": ["SKU-001", "SKU-002", "SKU-999"],
                            "on_hand": [100, -5, 10], "on_order": [0, 0, 0]})
        out = recommend_orders(fc, inv).set_index("product_id")
        self.assertEqual(out.loc["SKU-001", "status"], "ok")
        self.assertGreater(out.loc["SKU-001", "order_quantity"], 0)
        self.assertEqual(out.loc["SKU-002", "status"], "invalid_input")
        self.assertTrue(np.isnan(out.loc["SKU-002", "order_quantity"]))
        self.assertEqual(out.loc["SKU-999", "status"], "unknown_product")

    def test_inventory_file_missing_column(self):
        panel = PANEL[PANEL.date >= PANEL.date.max() - pd.Timedelta(days=60)]
        fc = forecast_next(panel, "seasonal_mean", {})
        with self.assertRaises(ValueError) as ctx:
            recommend_orders(fc, pd.DataFrame({"product_id": ["SKU-001"], "on_hand": [10]}))
        self.assertIn("on_order", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
