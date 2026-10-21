"""Imbalance handling without extra dependencies.

The handbook suggests imbalanced-learn.  To keep the project installable
everywhere we implement the two techniques we need ourselves (small and
fully unit-tested).  If you prefer the library, `imblearn.pipeline.Pipeline`
with SMOTE / RandomUnderSampler is a drop-in alternative - document it if you
switch.

Strategies
  none          train on the data as it is
  class_weight  the model penalises mistakes on fraud more (set on the model)
  undersample   keep all frauds, randomly drop legitimate rows
  smote         create synthetic frauds by interpolating between real ones

IMPORTANT: resampling happens only inside `fit` and only on TRAINING data.
Validation and test data are never resampled, so metrics stay honest.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors

from src import config

STRATEGIES = ("none", "class_weight", "undersample", "smote")


def random_undersample(X, y, ratio=config.RESAMPLE_RATIO, seed=config.SEED):
    """Keep all positives; sample negatives so that positives/negatives = ratio."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_keep = min(len(neg), int(round(len(pos) / ratio)))
    keep_neg = rng.choice(neg, size=n_keep, replace=False)
    idx = rng.permutation(np.concatenate([pos, keep_neg]))
    return X[idx], y[idx]


def smote(X, y, ratio=config.RESAMPLE_RATIO, k=config.SMOTE_NEIGHBORS, seed=config.SEED):
    """Oversample positives with SMOTE until positives/negatives = ratio.

    Each synthetic point = a real fraud + a random fraction of the way towards
    one of its k nearest fraud neighbours.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    n_neg = int((y == 0).sum())
    n_new = int(ratio * n_neg) - len(pos_idx)
    if n_new <= 0 or len(pos_idx) < 2:
        return X, y
    X_pos = X[pos_idx]
    k = min(k, len(pos_idx) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_pos)
    neighbours = nn.kneighbors(X_pos, return_distance=False)[:, 1:]   # drop self
    base = rng.integers(0, len(X_pos), n_new)
    other = neighbours[base, rng.integers(0, k, n_new)]
    gap = rng.random((n_new, 1))
    synthetic = X_pos[base] + gap * (X_pos[other] - X_pos[base])
    X_out = np.vstack([X, synthetic])
    y_out = np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
    idx = rng.permutation(len(y_out))
    return X_out[idx], y_out[idx]


class ResampledClassifier(ClassifierMixin, BaseEstimator):
    """Wraps any classifier; resamples the training data inside `fit` only."""

    def __init__(self, estimator=None, strategy="none", ratio=config.RESAMPLE_RATIO,
                 k_neighbors=config.SMOTE_NEIGHBORS, random_state=config.SEED):
        self.estimator = estimator
        self.strategy = strategy
        self.ratio = ratio
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if self.strategy == "undersample":
            X, y = random_undersample(X, y, self.ratio, self.random_state)
        elif self.strategy == "smote":
            X, y = smote(X, y, self.ratio, self.k_neighbors, self.random_state)
        elif self.strategy not in ("none", "class_weight"):
            raise ValueError(f"Unknown strategy {self.strategy!r}")
        self.model_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))
