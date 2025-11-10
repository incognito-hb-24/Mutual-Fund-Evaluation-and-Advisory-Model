from __future__ import annotations
import numpy as np
import pandas as pd


def decision_calendar(df_dates: pd.Series, freq: str) -> pd.DatetimeIndex:
    cal = (pd.DataFrame({"date": pd.to_datetime(df_dates)})
             .dropna().drop_duplicates()
             .set_index("date").sort_index())
    if freq.upper() in ["BM", "BME"]:
        return cal.resample("BME").last().dropna().index  # use BME to avoid BM deprecation
    return cal.resample(freq).last().dropna().index


def select_topk(day_slice: pd.DataFrame, score_col: str,
                K: int, use_threshold: bool, thr: float, max_w: float) -> pd.DataFrame:
    g = day_slice.dropna(subset=[score_col]).copy()
    if use_threshold:
        g = g.loc[g[score_col] >= thr]
    g = g.sort_values(score_col, ascending=False).head(K)
    if g.empty:
        return g
    w0 = min(1.0 / max(len(g), 1), max_w)
    g = g.assign(weight=w0)
    s = g["weight"].sum()
    if s > 0:
        g["weight"] = g["weight"] / s
    return g[["fund_id", score_col, "weight"]].copy()


def build_trades(oof_df: pd.DataFrame, score_col: str, freq: str,
                 K: int, use_threshold: bool, thr: float, max_w: float) -> pd.DataFrame:
    o = oof_df[["date","fund_id",score_col]].dropna().copy()
    dates = decision_calendar(o["date"], freq=freq)
    picks = []
    for d in dates:
        day = o.loc[o["date"] == d]
        if len(day) == 0:
            continue
        sel = select_topk(day, score_col, K=K, use_threshold=use_threshold, thr=thr, max_w=max_w)
        if sel.empty:
            continue
        sel = sel.assign(decision_date=d)
        picks.append(sel[["decision_date","fund_id","weight",score_col]])
    if not picks:
        return o.iloc[0:0][["date","fund_id"]].rename(columns={"date":"decision_date"}).assign(weight=np.nan)
    trades = pd.concat(picks, ignore_index=True)
    trades = trades.sort_values(["decision_date","fund_id"]).reset_index(drop=True)
    return trades


class PortfolioSimulator:
    """
    Simulates tranche P&L at a fixed horizon H using realized excess returns y_excess_63d.
    """
    def __init__(self, mst: pd.DataFrame, holding_days: int = 63, tx_cost: float = 0.002):
        self.H = int(holding_days)
        self.tx = float(tx_cost)
        self.tranche_log = pd.DataFrame(columns=[
            "decision_date","n_pos","ret_excess","ret_excess_net","equity"
        ])
        self.position_log = []
        self.metrics = pd.Series(dtype=float)

        # lookup realized horizon excess
        y = mst[["date","fund_id","y_excess_63d"]].dropna().copy()
        self._y_map = y.set_index(["date","fund_id"])["y_excess_63d"]

    def _realized_excess(self, d, f):
        try:
            return float(self._y_map.loc[(d, f)])
        except KeyError:
            return np.nan

    def run(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df.empty:
            raise ValueError("No trades to simulate.")

        decisions = sorted(trades_df["decision_date"].drop_duplicates())
        equity = 1.0
        rows = []

        prev_set = set()
        for d in decisions:
            basket = trades_df.loc[trades_df["decision_date"] == d].copy()
            if basket.empty:
                continue

            basket["ret_excess"] = basket.apply(lambda r: self._realized_excess(d, r["fund_id"]), axis=1)
            basket["ret_excess_net"] = basket["ret_excess"] - 2.0 * self.tx * basket["weight"]

            r_raw = float((basket["weight"] * basket["ret_excess"]).sum())
            r_net = float((basket["weight"] * basket["ret_excess_net"]).sum())
            equity *= (1.0 + r_net)

            cur_set = set(basket["fund_id"])
            if prev_set:
                added = len(cur_set - prev_set)
                dropped = len(prev_set - cur_set)
                denom = max((len(cur_set) + len(prev_set)) / 2.0, 1.0)
                turnover = (added + dropped) / denom
            else:
                turnover = np.nan
            prev_set = cur_set

            hits = int((basket["ret_excess_net"] > 0).sum())
            self.position_log.append(basket.assign(hit=(basket["ret_excess_net"] > 0).astype(int)))

            rows.append(dict(
                decision_date=d, n_pos=int(len(basket)),
                ret_excess=r_raw, ret_excess_net=r_net,
                turnover=turnover, hits=hits, equity=equity
            ))

        self.tranche_log = pd.DataFrame(rows).sort_values("decision_date").reset_index(drop=True)

        tr = self.tranche_log.copy()
        if len(tr):
            bdays = tr["decision_date"].diff().dt.days.dropna()
            avg_gap = float(np.nanmean(bdays)) if len(bdays) else 21.0
            ann_factor = 252.0 / max(avg_gap, 1.0)

            r = tr["ret_excess_net"].fillna(0.0).values
            mu, sd = np.mean(r), np.std(r, ddof=1) if len(r) > 1 else 0.0
            sharpe = (mu * ann_factor) / (sd * np.sqrt(ann_factor) + 1e-12) if sd > 0 else np.nan

            eq = tr["equity"].values
            peak = np.maximum.accumulate(eq)
            dd = (eq / peak) - 1.0
            mdd = float(np.min(dd)) if len(dd) else np.nan

            hit_rate = float((r > 0).mean())
            n_obs = int(len(r))

            self.metrics = pd.Series({
                "annualized_excess_return": (np.prod(1 + r) ** ann_factor) - 1.0 if len(r) else np.nan,
                "sharpe": sharpe,
                "max_drawdown": mdd,
                "hit_rate": hit_rate,
                "observations": n_obs,
                "avg_gap_days": avg_gap,
                "avg_turnover": float(np.nanmean(tr["turnover"])) if "turnover" in tr else np.nan
            })
        return self.tranche_log
