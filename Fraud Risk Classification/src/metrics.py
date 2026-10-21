"""Metrics for rare-event classification, including a cost metric.

Accuracy is useless here (a model that never flags anything is ~98.8%
accurate), so we use PR-AUC, precision, recall, alerts per 1,000
transactions and a monetary cost.
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

from src import config


def total_cost(y_true, y_pred, amount) -> float:
    """Missed fraud costs its amount; every alert costs REVIEW_COST."""
    y_true, y_pred, amount = map(np.asarray, (y_true, y_pred, amount))
    missed = (y_true == 1) & (y_pred == 0)
    return float(amount[missed].sum() + config.REVIEW_COST * (y_pred == 1).sum())


def candidate_thresholds(proba, lo=0.0005, hi=0.20, n=200):
    """Thresholds that alert on between lo and hi of all transactions."""
    fracs = np.linspace(lo, hi, n)
    return np.unique(np.quantile(proba, 1 - fracs))


def best_threshold_cost(y_true, proba, amount):
    """Return (threshold, cost) minimising total cost. Ties -> higher threshold."""
    proba = np.asarray(proba)
    best_t, best_c = None, np.inf
    for t in candidate_thresholds(proba):
        c = total_cost(y_true, (proba >= t).astype(int), amount)
        if c < best_c or (c == best_c and best_t is not None and t > best_t):
            best_t, best_c = float(t), c
    return best_t, float(best_c)


def threshold_for_alert_budget(proba, budget: float) -> float:
    """Threshold that alerts on roughly `budget` (e.g. 0.01 = 1%) of transactions."""
    return float(np.quantile(np.asarray(proba), 1 - budget))


def recall_at_precision(y_true, proba, min_precision: float) -> float:
    """Highest recall achievable while keeping precision >= min_precision."""
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def classification_metrics(y_true, proba, threshold: float, amount=None) -> dict:
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    out = {
        "threshold": float(threshold),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if both else float("nan"),
        "pr_auc": average_precision_score(y_true, proba) if both else float("nan"),
        "brier": brier_score_loss(y_true, np.clip(proba, 0, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "alerts_per_1000": 1000.0 * (tp + fp) / len(y_true),
        "recall_at_precision_50": recall_at_precision(y_true, proba, 0.5) if both else float("nan"),
    }
    if amount is not None:
        out["total_cost"] = total_cost(y_true, pred, amount)
    return out
