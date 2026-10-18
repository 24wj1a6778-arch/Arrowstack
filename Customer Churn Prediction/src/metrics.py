"""Metrics, including a simple business-value metric."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from src import config


def business_value(y_true, y_pred) -> float:
    """Net value of contacting every customer predicted to churn.

    value = TP * ACCEPT_RATE * SAVE_VALUE  -  (TP + FP) * OFFER_COST
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tp * config.ACCEPT_RATE * config.SAVE_VALUE - (tp + fp) * config.OFFER_COST


def best_threshold(y_true, proba, grid=None):
    """Return (threshold, value) that maximises business value."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    values = [business_value(y_true, (proba >= t).astype(int)) for t in grid]
    i = int(np.argmax(values))
    return float(grid[i]), float(values[i])


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y_true, proba) if both_classes else float("nan"),
        "brier": brier_score_loss(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "business_value": business_value(y_true, pred),
    }


def top_decile_lift(y_true, proba) -> float:
    """How many times more churners the top 10% scored customers contain."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    k = max(1, int(len(y_true) * 0.10))
    top = np.argsort(-proba)[:k]
    base = y_true.mean()
    return float(y_true[top].mean() / base) if base > 0 else float("nan")
