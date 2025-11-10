from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def plot_pr_curves(y_true: np.ndarray, score_dict: dict[str, np.ndarray], out_path: str | None = None):
    plt.figure(figsize=(6, 4))
    for name, s in score_dict.items():
        prec, rec, _ = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(s).astype(float))
        plt.plot(rec, prec, label=name)
    plt.title("Precision–Recall Curves (OOF)")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160)
    return plt.gca()


def plot_calibration(y_true: np.ndarray, scores: np.ndarray, out_path: str | None = None, n_bins: int = 15):
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(np.asarray(y_true).astype(int),
                                             np.asarray(scores).astype(float),
                                             n_bins=n_bins)
    plt.figure(figsize=(5, 4))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Predicted probability"); plt.ylabel("True positive rate")
    plt.title("Calibration Curve"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160)
    return plt.gca()


def plot_equity(curve_df: pd.DataFrame, out_path: str | None = None, title: str = "Equity Curve"):
    plt.figure(figsize=(7, 4))
    plt.plot(curve_df["decision_date"], curve_df["equity"])
    plt.title(title)
    plt.xlabel("Decision Date"); plt.ylabel("Equity (excess, cum)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160)
    return plt.gca()


def plot_lift(lift_df: pd.DataFrame, out_path: str | None = None, title: str = "Lift by Score Decile"):
    plt.figure(figsize=(6, 4))
    plt.plot(lift_df["bucket"], lift_df["positive_rate"], marker="o")
    plt.xticks(range(0, 10)); plt.xlabel("Score Decile (0=low, 9=high)")
    plt.ylabel("Positive Rate"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160)
    return plt.gca()
