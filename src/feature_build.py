from __future__ import annotations
import numpy as np
import pandas as pd


def align_with_calendar_ffill_bfill(calendar_df: pd.DataFrame,
                                    series_df: pd.DataFrame,
                                    value_col: str,
                                    bfill_head_days: int = 10) -> pd.DataFrame:
    out = calendar_df.merge(series_df, on="date", how="left").sort_values("date")
    out[value_col] = out[value_col].ffill()
    out[value_col] = out[value_col].bfill(limit=bfill_head_days)
    return out[["date", value_col]]


def _rolling_group_stat(df: pd.DataFrame, group: str, col: str, win: int, fn: str) -> pd.Series:
    # no groupby.apply (avoids deprecation warnings)
    gb = df.groupby(group)[col]
    if fn == "std":
        s = gb.rolling(win).std().reset_index(level=0, drop=True)
    elif fn == "mean":
        s = gb.rolling(win).mean().reset_index(level=0, drop=True)
    elif fn == "var":
        s = gb.rolling(win).var().reset_index(level=0, drop=True)
    else:
        raise ValueError("Unsupported fn")
    return s


def build_feature_panel(master_union: pd.DataFrame,
                        rf_annual: float = 0.06) -> pd.DataFrame:
    df = master_union.copy()
    df.columns = df.columns.str.lower()
    df = df.sort_values(["fund_id", "date"]).reset_index(drop=True)

    # daily returns
    df["ret_1d"] = df.groupby("fund_id")["nav"].pct_change()
    df["bmk_1d"] = df["tri"].pct_change()

    # multi-horizon momentum/excess
    for h in (7, 14, 21, 63):
        df[f"ret_{h}d"] = df.groupby("fund_id")["nav"].pct_change(h)
        df[f"bmk_{h}d"] = df["tri"].pct_change(h)
        df[f"excess_{h}d"] = df[f"ret_{h}d"] - df[f"bmk_{h}d"]

    # volatility & Sharpe (126d)
    rf_daily = rf_annual / 252.0
    win = 126
    roll_std = _rolling_group_stat(df, "fund_id", "ret_1d", win, "std")
    roll_mean = _rolling_group_stat(df, "fund_id", "ret_1d", win, "mean")
    df["vol_126d"] = roll_std * np.sqrt(252)
    df["sharpe_126d"] = (roll_mean - rf_daily) / roll_std.replace(0, np.nan)

    # beta & annualized alpha (126d)
    win_beta = 126
    # rolling cov/var without apply
    cov = (df.groupby("fund_id")["ret_1d"]
             .rolling(win_beta).cov(df["bmk_1d"])
             .reset_index(level=0, drop=True))
    var = df["bmk_1d"].rolling(win_beta).var()
    beta = cov / var.replace(0, np.nan)
    mean_r = _rolling_group_stat(df, "fund_id", "ret_1d", win_beta, "mean")
    mean_m = df["bmk_1d"].rolling(win_beta).mean()
    alpha_daily = (mean_r - rf_daily) - beta * (mean_m - rf_daily)
    df["beta_126d"] = beta
    df["alpha_ann_126d"] = alpha_daily * 252

    # rolling MDD (252d) & consistency of positive days (63d)
    win_dd, win_cons = 252, 63
    roll_peak = (df.groupby("fund_id")["nav"]
                   .rolling(win_dd, min_periods=1).max()
                   .reset_index(level=0, drop=True))
    dd = df["nav"] / roll_peak - 1.0
    df["mdd_252d"] = (dd.groupby(df["fund_id"])
                        .rolling(win_dd, min_periods=1).min()
                        .reset_index(level=0, drop=True))

    is_pos = (df["ret_1d"] > 0).astype(float)
    df["consistency_pos_63d"] = (is_pos.groupby(df["fund_id"])
                                   .rolling(win_cons, min_periods=1).mean()
                                   .reset_index(level=0, drop=True))

    # rolling corr with benchmark (63d)
    win_reg = 63
    # corr = cov(x,y)/sqrt(varx*vary)
    cov_63 = (df.groupby("fund_id")["ret_1d"]
                .rolling(win_reg).cov(df["bmk_1d"])
                .reset_index(level=0, drop=True))
    var_x = _rolling_group_stat(df, "fund_id", "ret_1d", win_reg, "var")
    var_y = df["bmk_1d"].rolling(win_reg).var()
    df["corr_bmk_63d"] = cov_63 / (np.sqrt(var_x * var_y) + 1e-12)

    # regime flags from 63d benchmark
    df["regime_bull_63d"] = (df["bmk_63d"] > 0).astype(int)
    df["regime_bear_63d"] = (df["bmk_63d"] < 0).astype(int)

    return df


def add_cross_section_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "excess_21d" in out.columns:
        out["xs_rank_excess_21d"] = out.groupby("date")["excess_21d"].rank(pct=True, ascending=False)
    if "sharpe_126d" in out.columns:
        out["xs_rank_sharpe_126d"] = out.groupby("date")["sharpe_126d"].rank(pct=True, ascending=False)
    if "mdd_252d" in out.columns:
        out["xs_rank_mdd_252d"] = out.groupby("date")["mdd_252d"].rank(pct=True, ascending=True)
    return out
