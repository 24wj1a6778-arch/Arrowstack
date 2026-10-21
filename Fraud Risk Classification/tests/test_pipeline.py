"""Unit tests. Run with either:

    python -m unittest discover -s tests -v
    pytest -v            (if pytest is installed)
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression  # noqa: E402

from src import config  # noqa: E402
from src.data_prep import clean, split, validate_schema  # noqa: E402
from src.explain import local_explanation  # noqa: E402
from src.features import add_features, build_pipeline  # noqa: E402
from src.generate_data import generate  # noqa: E402
from src.metrics import (best_threshold_cost, classification_metrics,  # noqa: E402
                         recall_at_precision, threshold_for_alert_budget, total_cost)
from src.predict import predict_df, prepare_input  # noqa: E402
from src.resampling import ResampledClassifier, random_undersample, smote  # noqa: E402


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = generate(n=6000, seed=1)
        cls.df = clean(cls.raw)

    def test_generation_is_reproducible(self):
        pd.testing.assert_frame_equal(generate(500, seed=7), generate(500, seed=7))

    def test_fraud_is_rare(self):
        rate = self.raw[config.TARGET].mean()
        self.assertTrue(0.005 < rate < 0.025)

    def test_schema_complete(self):
        for col in config.REQUIRED_COLUMNS + [config.TARGET]:
            self.assertIn(col, self.raw.columns)

    def test_clean_drops_id_and_keeps_binary_target(self):
        self.assertNotIn(config.ID_COL, self.df.columns)
        self.assertSetEqual(set(self.df[config.TARGET].unique()), {0, 1})

    def test_missing_values_kept_for_pipeline_to_impute(self):
        self.assertGreater(int(self.df["distance_from_home_km"].isna().sum()), 0)

    def test_validate_rejects_missing_column(self):
        with self.assertRaises(ValueError):
            validate_schema(self.df.drop(columns=["channel"]))

    def test_validate_rejects_negative_amount_and_bad_hour(self):
        bad = self.df.copy(); bad.loc[0, "amount"] = -5
        with self.assertRaises(ValueError):
            validate_schema(bad)
        bad = self.df.copy(); bad.loc[0, "hour"] = 30
        with self.assertRaises(ValueError):
            validate_schema(bad)

    def test_split_sizes_stratified(self):
        tr, va, te = split(self.df)
        self.assertEqual(len(tr) + len(va) + len(te), len(self.df))
        base = self.df[config.TARGET].mean()
        for part in (tr, va, te):
            self.assertAlmostEqual(part[config.TARGET].mean(), base, delta=0.004)
        self.assertAlmostEqual(len(te) / len(self.df), config.TEST_SIZE, delta=0.01)


class TestResampling(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(2000, 4))
        self.y = np.zeros(2000, dtype=int)
        self.y[:40] = 1
        self.X[:40] += 3                       # make frauds a separate cluster

    def test_undersample_ratio_and_keeps_all_positives(self):
        X, y = random_undersample(self.X, self.y, ratio=0.2, seed=1)
        self.assertEqual(int(y.sum()), 40)
        self.assertEqual(int((y == 0).sum()), 200)

    def test_smote_reaches_ratio_and_stays_in_fraud_region(self):
        X, y = smote(self.X, self.y, ratio=0.2, seed=1)
        self.assertEqual(int(y.sum()), int(0.2 * 1960))
        synth = X[y == 1][40:] if len(X) > 2000 else X[y == 1]
        real_pos = self.X[:40]
        lo, hi = real_pos.min(axis=0), real_pos.max(axis=0)
        # synthetic points are interpolations, so they lie inside the bounding box of real frauds
        new_pos = X[y == 1]
        self.assertTrue(((new_pos >= lo - 1e-9) & (new_pos <= hi + 1e-9)).all())
        self.assertGreater(len(synth), 0)

    def test_smote_does_not_change_original_data(self):
        X0, y0 = self.X.copy(), self.y.copy()
        smote(self.X, self.y, seed=1)
        np.testing.assert_array_equal(X0, self.X)
        np.testing.assert_array_equal(y0, self.y)

    def test_resampling_only_in_fit_never_in_predict(self):
        clf = ResampledClassifier(LogisticRegression(max_iter=500), strategy="smote").fit(self.X, self.y)
        self.assertEqual(len(clf.predict_proba(self.X[:7])), 7)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            ResampledClassifier(LogisticRegression(), strategy="magic").fit(self.X, self.y)


class TestFeaturesAndModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = clean(generate(n=6000, seed=3))
        cls.X = cls.df.drop(columns=[config.TARGET])
        cls.y = cls.df[config.TARGET]

    def test_add_features(self):
        out = add_features(self.X)
        for col in config.ENGINEERED_NUMERIC:
            self.assertIn(col, out.columns)
        self.assertTrue((out["amount_ratio"].dropna() >= 0).all())

    def test_add_features_does_not_mutate_input(self):
        before = self.X.copy()
        add_features(self.X)
        pd.testing.assert_frame_equal(before, self.X)

    def test_pipeline_with_each_strategy(self):
        for strategy in ("none", "undersample", "smote"):
            pipe = build_pipeline(ResampledClassifier(LogisticRegression(max_iter=1000), strategy=strategy))
            pipe.fit(self.X, self.y)
            p = pipe.predict_proba(self.X)[:, 1]
            self.assertFalse(np.isnan(p).any())
            self.assertTrue(((p >= 0) & (p <= 1)).all())
            self.assertGreater(classification_metrics(self.y, p, 0.5)["roc_auc"], 0.7)

    def test_unseen_category_does_not_crash(self):
        pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(self.X, self.y)
        row = self.X.head(3).copy()
        row["merchant_category"] = "crypto_exchange"
        self.assertEqual(len(pipe.predict_proba(row)), 3)

    def test_local_explanation_finds_the_driver(self):
        pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(self.X, self.y)
        bg = self.X.sample(100, random_state=0)
        row = self.X.iloc[0].copy()
        row["amount"], row["avg_amount_30d"] = 3000.0, 20.0     # absurdly large vs usual
        expl = local_explanation(pipe, row, bg)
        self.assertIn("amount", list(expl["feature"].head(3)))
        self.assertGreater(expl.loc[expl.feature == "amount", "contribution"].iloc[0], 0)


class TestMetrics(unittest.TestCase):
    def test_total_cost_formula(self):
        y = np.array([1, 1, 0, 0]); pred = np.array([1, 0, 1, 0]); amt = np.array([100, 50, 10, 10])
        # missed fraud (row 1) costs 50; two alerts cost 2 * REVIEW_COST
        self.assertAlmostEqual(total_cost(y, pred, amt), 50 + 2 * config.REVIEW_COST)

    def test_best_threshold_beats_alert_everything_and_nothing(self):
        rng = np.random.default_rng(0)
        y = (rng.random(5000) < 0.02).astype(int)
        proba = np.clip(0.05 + 0.6 * y + rng.normal(0, 0.15, 5000), 0, 1)
        amt = np.full(5000, 100.0)
        thr, cost = best_threshold_cost(y, proba, amt)
        self.assertLess(cost, total_cost(y, np.zeros(5000, dtype=int), amt))
        self.assertLess(cost, total_cost(y, np.ones(5000, dtype=int), amt))

    def test_alert_budget_threshold(self):
        proba = np.linspace(0, 1, 1001)
        thr = threshold_for_alert_budget(proba, 0.10)
        self.assertAlmostEqual((proba >= thr).mean(), 0.10, delta=0.01)

    def test_recall_at_precision(self):
        y = np.array([1, 1, 0, 0, 0, 0]); p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        self.assertAlmostEqual(recall_at_precision(y, p, 1.0), 1.0)
        self.assertEqual(recall_at_precision(np.array([0, 0, 1]), np.array([0.9, 0.8, 0.1]), 0.9), 0.0)

    def test_metrics_keys_and_counts(self):
        m = classification_metrics([0, 1, 1, 0], [0.1, 0.8, 0.6, 0.4], 0.5, amount=[1, 2, 3, 4])
        for k in ("precision", "recall", "pr_auc", "alerts_per_1000", "total_cost"):
            self.assertIn(k, m)
        self.assertEqual(m["tp"] + m["fp"] + m["tn"] + m["fn"], 4)


class TestPredictEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = clean(generate(n=6000, seed=5))
        cls.X = df.drop(columns=[config.TARGET])
        cls.model = build_pipeline(LogisticRegression(max_iter=1000)).fit(cls.X, df[config.TARGET])

    def test_missing_column_gives_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            prepare_input(self.X.drop(columns=["amount"]))
        self.assertIn("amount", str(ctx.exception))

    def test_invalid_rows_flagged_not_scored(self):
        bad = self.X.head(5).copy()
        bad.loc[bad.index[0], "amount"] = -20
        bad.loc[bad.index[1], "hour"] = 27
        out = predict_df(bad, model=self.model, threshold=0.3)
        self.assertEqual(list(out["decision"].iloc[:2]), ["invalid_input", "invalid_input"])
        self.assertTrue(out["fraud_score"].iloc[:2].isna().all())
        self.assertTrue((out["decision"].iloc[2:] != "invalid_input").all())

    def test_missing_values_are_imputed(self):
        row = self.X.head(2).copy()
        row.loc[row.index[0], "distance_from_home_km"] = np.nan
        row.loc[row.index[0], "merchant_category"] = None
        out = predict_df(row, model=self.model, threshold=0.3)
        self.assertFalse(out["fraud_score"].isna().any())

    def test_decision_and_band_always_agree(self):
        out = predict_df(self.X.head(300), model=self.model, threshold=0.2)
        self.assertTrue(((out["decision"] == "review") == (out["risk_band"] == "alert")).all())


if __name__ == "__main__":
    unittest.main()
