from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from .metrics_eval import pr_auc_and_sweep


def timeseries_splits(X: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X))


def train_oof_logistic(X: pd.DataFrame, y: pd.Series, splits, class_weight="balanced", seed: int = 42):
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=500, class_weight=class_weight, random_state=seed)
    )
    oof = np.full(len(y), np.nan, float)
    fold_rows = []
    for k, (tr, te) in enumerate(splits, 1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        s = pipe.predict_proba(X.iloc[te])[:, 1]
        oof[te] = s
        pr, _ = pr_auc_and_sweep(y.iloc[te], s)
        fold_rows.append({"fold": k, "PR_AUC": pr, "test_n": len(te)})
    return oof, pd.DataFrame(fold_rows), pipe


def train_oof_xgb(X: pd.DataFrame, y: pd.Series, splits, seed: int = 42):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise ImportError("xgboost not installed") from e

    model = XGBClassifier(
        n_estimators=700, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, eval_metric="logloss",
    )
    oof = np.full(len(y), np.nan, float)
    fold_rows = []
    for k, (tr, te) in enumerate(splits, 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        s = model.predict_proba(X.iloc[te])[:, 1]
        oof[te] = s
        pr, _ = pr_auc_and_sweep(y.iloc[te], s)
        fold_rows.append({"fold": k, "PR_AUC": pr, "test_n": len(te)})
    return oof, pd.DataFrame(fold_rows), model


def train_oof_lgb(X: pd.DataFrame, y: pd.Series, splits, seed: int = 42):
    try:
        from lightgbm import LGBMClassifier
    except Exception as e:
        raise ImportError("lightgbm not installed") from e

    model = LGBMClassifier(
        n_estimators=1200, learning_rate=0.03,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, n_jobs=-1
    )
    oof = np.full(len(y), np.nan, float)
    fold_rows = []
    for k, (tr, te) in enumerate(splits, 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        s = model.predict_proba(X.iloc[te])[:, 1]
        oof[te] = s
        pr, _ = pr_auc_and_sweep(y.iloc[te], s)
        fold_rows.append({"fold": k, "PR_AUC": pr, "test_n": len(te)})
    return oof, pd.DataFrame(fold_rows), model


def build_oof_table(meta_idx: pd.DataFrame,
                    y: pd.Series,
                    preds: Dict[str, np.ndarray]) -> pd.DataFrame:
    out = meta_idx.copy()
    out["y_true"] = y.values
    for name, arr in preds.items():
        out[name] = arr
    return out


def weighted_ensemble(oof: pd.DataFrame, cv_tables: Dict[str, pd.DataFrame], cols=("logit", "xgb", "lgb")) -> pd.Series:
    weights = {}
    w_sum = 0.0
    for name in cols:
        if name in cv_tables:
            w = float(cv_tables[name]["PR_AUC"].mean())
            weights[name] = w
            w_sum += w
    if w_sum == 0:
        return oof[list(cols)].mean(axis=1)  # fallback simple mean
    for k in weights:
        weights[k] /= w_sum
    s = 0.0
    for name in cols:
        if name in oof.columns and name in weights:
            s = s + weights[name] * oof[name]
    return pd.Series(s, index=oof.index, name="ens_weighted")


def train_quantile_lgb(X: pd.DataFrame, y: np.ndarray, alpha: float = 0.5, seed: int = 42):
    try:
        from lightgbm import LGBMRegressor
    except Exception as e:
        raise ImportError("lightgbm not installed") from e

    model = LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=800, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1
    )
    model.fit(X, y)
    return model
