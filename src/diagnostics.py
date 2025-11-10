from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from .metrics_eval import bootstrap_ci_metric, smape


def pr_auc_ci(y_true: np.ndarray, scores: np.ndarray, B: int = 1000, seed: int = 42):
    def _prauc(a, b): return average_precision_score(a.astype(int), b.astype(float))
    return bootstrap_ci_metric(_prauc, y_true, scores, B=B, seed=seed)


def threshold_sweep_with_cost(y_true: np.ndarray,
                              scores: np.ndarray,
                              realized_excess: np.ndarray | None = None,
                              tx: float = 0.002) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    prec, rec, th = precision_recall_curve(y, s)
    sweep = pd.DataFrame({"threshold": list(th) + [np.inf], "precision": prec, "recall": rec})

    if realized_excess is None:
        # proxy payoff
        def payoff(t):
            pred = (s >= t).astype(int)
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            return float(tp * 0.05 - fp * 0.02 - tx * pred.sum())
        sweep["cost"] = sweep["threshold"].apply(payoff)
        return sweep

    re = np.asarray(realized_excess).astype(float)
    def payoff_real(t):
        pred = (s >= t).astype(int)
        tp = (pred == 1) & (y == 1)
        fp = (pred == 1) & (y == 0)
        return float(np.nansum(re[tp]) - tx * tp.sum() + np.nansum(re[fp]) - tx * fp.sum())
    sweep["cost"] = sweep["threshold"].apply(payoff_real)
    return sweep


def lift_by_decile(df: pd.DataFrame, label_col: str, score_col: str) -> pd.DataFrame:
    m = df[[label_col, score_col]].dropna().copy()
    if m.empty: return pd.DataFrame(columns=["bucket","positive_rate"])
    m["score_q"] = m[score_col].rank(pct=True)
    m["bucket"] = (m["score_q"] * 10).astype(int).clip(0, 9)
    return (m.groupby("bucket", as_index=False)[label_col]
              .mean()
              .rename(columns={label_col: "positive_rate"}))


def precision_by_year_topdecile(df: pd.DataFrame, date_col: str, label_col: str, score_col: str) -> pd.DataFrame:
    m = df[[date_col, label_col, score_col]].dropna().copy()
    if m.empty: return pd.DataFrame(columns=["year","precision_top10"])
    m["score_q"] = m[score_col].rank(pct=True)
    m = m[m["score_q"] >= 0.90].copy()
    m["year"] = pd.to_datetime(m[date_col]).dt.year
    return (m.groupby("year", as_index=False)[label_col]
              .mean()
              .rename(columns={label_col: "precision_top10"}))


def ablation_lightgbm(df: pd.DataFrame, y_col: str, feature_sets: dict,
                      n_splits: int = 3, seed: int = 42) -> pd.DataFrame:
    try:
        from lightgbm import LGBMClassifier
    except Exception as e:
        raise ImportError("lightgbm not installed") from e
    from sklearn.model_selection import TimeSeriesSplit

    rows = []
    mask = df[[y_col]].notna().all(axis=1)
    y = df.loc[mask, y_col].astype(int)

    for tag, feat_cols in feature_sets.items():
        X = df.loc[mask, feat_cols].astype(float)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        ap = []
        for tr, te in tscv.split(X):
            m = LGBMClassifier(
                n_estimators=600, learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=seed, n_jobs=-1
            )
            m.fit(X.iloc[tr], y.iloc[tr])
            s = m.predict_proba(X.iloc[te])[:, 1]
            ap.append(average_precision_score(y.iloc[te], s))
        rows.append({"ablation": tag, "features": len(feat_cols), "PR_AUC": float(np.mean(ap)) if ap else np.nan})
    return pd.DataFrame(rows)
