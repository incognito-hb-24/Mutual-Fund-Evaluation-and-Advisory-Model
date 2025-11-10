from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, confusion_matrix,
    mean_absolute_error
)


def pr_auc_and_sweep(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    pr = average_precision_score(y, s)
    prec, rec, th = precision_recall_curve(y, s)
    sweep = pd.DataFrame({"threshold": list(th) + [np.inf], "precision": prec, "recall": rec})
    return float(pr), sweep


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    prec, rec, th = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    idx = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    return float(th[idx]) if idx < len(th) else 0.5


def business_threshold(y_true: np.ndarray, scores: np.ndarray,
                       target_precision: float = 0.70, min_recall: float = 0.30) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    prec, rec, th = precision_recall_curve(y, s)
    for i in range(len(th)):
        if prec[i] >= target_precision and rec[i] >= min_recall:
            return float(th[i])
    return best_f1_threshold(y, s)


def confusion_counts(y_true: np.ndarray, y_pred_bin: np.ndarray) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred_bin.astype(int)).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def bootstrap_ci_metric(metric_fn, y: np.ndarray, yhat: np.ndarray,
                        B: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    n = len(y)
    if n == 0: return np.nan, np.nan, np.nan
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        stats.append(float(metric_fn(y[idx], yhat[idx])))
    mean = float(np.mean(stats))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return mean, float(lo), float(hi)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def mae_ci(y_true: np.ndarray, y_pred: np.ndarray, **kw):
    return bootstrap_ci_metric(mean_absolute_error, y_true, y_pred, **kw)


def smape_ci(y_true: np.ndarray, y_pred: np.ndarray, **kw):
    return bootstrap_ci_metric(smape, y_true, y_pred, **kw)
