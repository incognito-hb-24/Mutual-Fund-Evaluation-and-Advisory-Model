"""
MLBA Project - Python package

Convenience imports so notebooks/scripts can do:
    from src import io, metrics, features, models, diag, sim, plots
"""

from .utils_io import (
    ProjectPaths, ensure_dirs, save_csv, load_csv, save_json, load_json, set_seed
)
from .metrics_eval import (
    pr_auc_and_sweep, best_f1_threshold, business_threshold,
    confusion_counts, bootstrap_ci_metric, smape, mae_ci, smape_ci
)
from .feature_build import (
    align_with_calendar_ffill_bfill, build_feature_panel, add_cross_section_ranks
)
from .models_train import (
    timeseries_splits, train_oof_logistic, train_oof_xgb, train_oof_lgb,
    build_oof_table, weighted_ensemble, train_quantile_lgb
)
from .diagnostics import (
    pr_auc_ci, threshold_sweep_with_cost, lift_by_decile, precision_by_year_topdecile,
    ablation_lightgbm
)
from .portfolio_simulator import PortfolioSimulator, build_trades, decision_calendar, select_topk
from .plot_utils import (
    plot_pr_curves, plot_calibration, plot_equity, plot_lift
)

__all__ = [
    # io
    "ProjectPaths", "ensure_dirs", "save_csv", "load_csv", "save_json", "load_json", "set_seed",
    # metrics
    "pr_auc_and_sweep", "best_f1_threshold", "business_threshold",
    "confusion_counts", "bootstrap_ci_metric", "smape", "mae_ci", "smape_ci",
    # features
    "align_with_calendar_ffill_bfill", "build_feature_panel", "add_cross_section_ranks",
    # models
    "timeseries_splits", "train_oof_logistic", "train_oof_xgb", "train_oof_lgb",
    "build_oof_table", "weighted_ensemble", "train_quantile_lgb",
    # diagnostics
    "pr_auc_ci", "threshold_sweep_with_cost", "lift_by_decile",
    "precision_by_year_topdecile", "ablation_lightgbm",
    # sim
    "PortfolioSimulator", "build_trades", "decision_calendar", "select_topk",
    # plots
    "plot_pr_curves", "plot_calibration", "plot_equity", "plot_lift",
]
