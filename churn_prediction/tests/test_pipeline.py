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
from src.features import add_features, build_pipeline  # noqa: E402
from src.generate_data import generate  # noqa: E402
from src.metrics import best_threshold, business_value, classification_metrics  # noqa: E402
from src.predict import predict_df, prepare_input  # noqa: E402


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = generate(n=1500, seed=1)
        cls.df = clean(cls.raw)

    def test_generation_is_reproducible(self):
        pd.testing.assert_frame_equal(generate(200, seed=7), generate(200, seed=7))

    def test_schema_and_churn_rate(self):
        for col in config.REQUIRED_COLUMNS + [config.TARGET]:
            self.assertIn(col, self.raw.columns)
        self.assertTrue(0.15 < (self.raw["Churn"] == "Yes").mean() < 0.45)

    def test_clean_handles_blank_total_charges(self):
        self.assertTrue(pd.api.types.is_float_dtype(self.df["TotalCharges"]))
        self.assertEqual(int(self.df["TotalCharges"].isna().sum()), 11)

    def test_clean_encodes_target_and_drops_id(self):
        self.assertSetEqual(set(self.df[config.TARGET].unique()), {0, 1})
        self.assertNotIn(config.ID_COL, self.df.columns)

    def test_validate_rejects_missing_column(self):
        with self.assertRaises(ValueError):
            validate_schema(self.df.drop(columns=["Contract"]))

    def test_validate_rejects_bad_target(self):
        bad = self.df.copy()
        bad.loc[0, config.TARGET] = 5
        with self.assertRaises(ValueError):
            validate_schema(bad)

    def test_split_sizes_stratified_and_disjoint(self):
        tr, va, te = split(self.df)
        self.assertEqual(len(tr) + len(va) + len(te), len(self.df))
        self.assertAlmostEqual(len(te) / len(self.df), config.TEST_SIZE, delta=0.01)
        self.assertAlmostEqual(len(va) / len(self.df), config.VAL_SIZE, delta=0.01)
        base = self.df[config.TARGET].mean()
        for part in (tr, va, te):
            self.assertAlmostEqual(part[config.TARGET].mean(), base, delta=0.02)


class TestFeaturesAndModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = clean(generate(n=1200, seed=3))
        cls.X = cls.df.drop(columns=[config.TARGET])
        cls.y = cls.df[config.TARGET]

    def test_add_features_creates_columns(self):
        out = add_features(self.X)
        for col in config.ENGINEERED_NUMERIC + config.ENGINEERED_CATEGORICAL:
            self.assertIn(col, out.columns)
        self.assertTrue(out["num_services"].between(0, len(config.SERVICE_COLUMNS)).all())

    def test_add_features_does_not_mutate_input(self):
        before = self.X.copy()
        add_features(self.X)
        pd.testing.assert_frame_equal(before, self.X)

    def test_pipeline_fits_and_predicts_probabilities(self):
        pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(self.X, self.y)
        proba = pipe.predict_proba(self.X)[:, 1]
        self.assertEqual(proba.shape, (len(self.X),))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())
        self.assertFalse(np.isnan(proba).any())

    def test_pipeline_beats_random_on_train(self):
        pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(self.X, self.y)
        m = classification_metrics(self.y, pipe.predict_proba(self.X)[:, 1])
        self.assertGreater(m["roc_auc"], 0.7)

    def test_unseen_category_does_not_crash(self):
        pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(self.X, self.y)
        row = self.X.head(3).copy()
        row["PaymentMethod"] = "Crypto"
        self.assertEqual(len(pipe.predict_proba(row)), 3)


class TestMetrics(unittest.TestCase):
    def test_business_value_formula(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])           # TP=1, FP=1
        expected = 1 * config.ACCEPT_RATE * config.SAVE_VALUE - 2 * config.OFFER_COST
        self.assertAlmostEqual(business_value(y_true, y_pred), expected)

    def test_best_threshold_beats_extremes(self):
        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.3).astype(int)
        proba = np.clip(0.3 + 0.35 * (y - 0.3) + rng.normal(0, 0.15, 2000), 0, 1)
        thr, val = best_threshold(y, proba)
        self.assertGreaterEqual(val, business_value(y, (proba >= 0.05).astype(int)))
        self.assertGreaterEqual(val, business_value(y, (proba >= 0.95).astype(int)))
        self.assertTrue(0.05 <= thr <= 0.95)

    def test_metrics_keys(self):
        m = classification_metrics([0, 1, 1, 0], [0.1, 0.8, 0.6, 0.4])
        for k in ("precision", "recall", "f1", "roc_auc", "pr_auc", "business_value"):
            self.assertIn(k, m)
        self.assertEqual(m["tp"] + m["fp"] + m["tn"] + m["fn"], 4)


class TestPredictEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = clean(generate(n=800, seed=5))
        cls.X = df.drop(columns=[config.TARGET])
        cls.model = build_pipeline(LogisticRegression(max_iter=1000)).fit(cls.X, df[config.TARGET])

    def test_missing_column_gives_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            prepare_input(self.X.drop(columns=["tenure"]))
        self.assertIn("tenure", str(ctx.exception))

    def test_invalid_rows_are_flagged_not_scored(self):
        bad = self.X.head(4).copy()
        bad.loc[bad.index[0], "tenure"] = -5
        out = predict_df(bad, model=self.model, threshold=0.5)
        self.assertEqual(out["decision"].iloc[0], "invalid_input")
        self.assertTrue(np.isnan(out["churn_probability"].iloc[0]))
        self.assertTrue((out["decision"].iloc[1:] != "invalid_input").all())

    def test_blank_total_charges_is_imputed(self):
        row = self.X.head(2).copy().astype({"TotalCharges": object})
        row.loc[row.index[0], "TotalCharges"] = " "
        out = predict_df(row, model=self.model, threshold=0.5)
        self.assertFalse(out["churn_probability"].isna().any())


if __name__ == "__main__":
    unittest.main()
