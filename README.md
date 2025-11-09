# Mutual-Fund-Evaluation-and-Advisory-Model
---

##  Overview <br>
The **Mutual Fund Advisory & Evaluation Model (MLBA)** is a comprehensive end-to-end machine learning framework designed to evaluate, forecast, and simulate mutual fund performance using real-world data and economic indicators. <br>
It combines data engineering, predictive modeling, diagnostic validation, and backtesting — providing an analytical foundation for fund recommendation and investment decision-making. <br>

The model is built in phases to ensure transparency, interpretability, and full reproducibility. Each stage transforms data into deeper intelligence — starting from raw NAVs and ending with actionable portfolio-level insights. <br>

---

##  Repository Structure <br>
<br>
Mutual-Fund-Evaluation-and-Advisory-Model/   <br>
│ <br>
├── README.md   <br>
├── LICENSE  <br>
├── .gitignore <br>
├── environment.yml   <br> 
├── requirements.txt    <br>
├── make_reproduce.sh    <br>
│   <br>
├── data/   <br>
│ ├── raw/     <br>
│ ├── processed/    <br>
│ │ ├── clean_master_union.csv      <br>
│ │ ├── clean_with_features.csv     <br>
│ │ ├── clf_oof_predictions.csv     <br>
│ │ ├── results_summary_classification.csv     <br>
│ │ ├── bt_tranche_log.csv       <br>
│ │ ├── bt_robustness_grid.csv      <br>
│ │ └── ...      <br>
│ └── README.md       <br>
│             <br>
├── notebooks/       <br>
│ ├── phase1_feature_engineering.ipynb        <br>
│ ├── phase2_feature_selection.ipynb          <br>
│ ├── phase3_model_training.ipynb             <br>
│ ├── phase4_diagnostics.ipynb                <br>
│ └── phase5_backtesting.ipynb                <br>
│              <br>
├── src/          <br>
│ ├── init.py         <br>
│ ├── utils_io.py            <br>
│ ├── metrics_eval.py           <br>
│ ├── feature_build.py            <br>
│ ├── models_train.py            <br>
│ ├── diagnostics.py              <br>
│ ├── portfolio_simulator.py          <br>
│ └── plot_utils.py           <br>
│        <br>
├── reports/     <br>
│ ├── figures/        <br>
│ │ ├── pr_auc_ci.png           <br>
│ │ ├── cost_curve_ensemble.png         <br>
│ │ ├── lift_by_bucket_ensemble.png          <br>
│ │ ├── bt_equity_base.png       <br>
│ │ ├── bt_equity_vs_zero.png              <br>
│ │ └── bt_robust_equity.png         <br>
│ ├── pr_auc_ci.csv          <br>
│ ├── ablation_window_macro.csv            <br>
│ ├── bt_metrics.json             <br>
│ ├── bt_tranche_log.csv            <br>
│ ├── bt_turnover_stats.csv          <br>
│ ├── bt_robustness_grid.csv           <br>
│ └── summary_dashboard.csv              <br>
│     <br>
├── models/        <br>
│ ├── logit.pkl     <br>
│ ├── xgb.pkl      <br>
│ ├── lgb.pkl       <br>
│ └── meta.json         <br>
│     <br>
└── docs/         <br>
├── methodology_diagram.png        <br>
├── results_summary_table.png        <br>
└── paper_appendix.pdf         <br>
<br>
Each folder and file has a clear purpose — ensuring that anyone reviewing or reproducing this project can follow the exact flow of data, modeling, and analysis from raw ingestion to portfolio outcomes. <br>

---

##  Walkthrough of the Project <br>

### Phase 1 — Data Ingestion & Master Creation <br>
All raw NAVs, TRI, and macroeconomic indicators (such as VIX, USD/INR, G-sec yield, Gold, Brent, etc.) are cleaned, aligned, and merged chronologically. <br>
The final **`clean_master_union.csv`** acts as the unified time-series base for the model. <br>

### Phase 2 — Feature Engineering <br>
Generates fund-level and market-level rolling metrics such as returns, volatility, Sharpe ratio, beta, and drawdown. <br>
Macro variables are lagged and normalized to avoid lookahead bias. <br>
Sparse columns and incomplete rows are removed, resulting in **`clean_with_features.csv`** — the primary feature dataset. <br>

### Phase 3 — Model Training & Ensemble Building <br>
A binary classifier predicts whether each fund will outperform the TRI benchmark over the next 63 days. <br>
Models used include Logistic Regression, XGBoost, LightGBM, and a weighted Ensemble. <br>
Outputs include PR-AUC scores, confusion matrices, and per-fold evaluation summaries. <br>

### Phase 4 — Diagnostics & Validation <br>
Analyzes precision-recall curves, cost optimization, ablation tests, and feature impact. <br>
Each model’s thresholds are tuned to align with practical business trade-offs between precision and recall. <br>
This phase also measures sensitivity to hyperparameters and generates interpretability plots for key drivers of fund alpha. <br>

### Phase 5 — Portfolio Simulation & Backtesting <br>
Transforms model predictions into real fund selection decisions under a realistic no-lookahead scenario. <br>
Simulates rebalancing weekly (W-FRI) and calculates equity curves, Sharpe ratios, hit rates, and turnover. <br>
The model demonstrates consistent positive alpha with controlled drawdowns and diversified fund exposure. <br>

### Results Interpretation <br>
- Strong PR-AUC (~0.72) indicates high discriminative performance. <br>
- Sharpe ratios above 1.5 with a 65% hit rate show robust profitability. <br>
- Turnover around 0.44 ensures sustainable rebalancing frequency. <br>
- Ablation studies confirm macroeconomic indicators significantly enhance accuracy. <br>

---



## Outputs Generated <br>
Processed datasets under /data/processed/ <br>

Model weights under /models/ <br>

Diagnostic plots and performance figures under /reports/figures/ <br>

Comprehensive CSV and JSON logs summarizing all results and metrics <br>

These outputs demonstrate model accuracy, robustness, and portfolio-level profitability. <br>

📈 Visual Summaries <br>
| Visualization | Description | <br>
|:--|:--| <br>
| bt_equity_base.png | Strategy equity curve vs zero-excess baseline | <br>
| bt_robust_equity.png | Robustness of returns across portfolio sizes and thresholds | <br>
| cost_curve_ensemble.png | Cost curve showing precision-recall trade-off | <br>
| lift_by_bucket_ensemble.png | Lift analysis showing performance across score deciles | <br>

These visuals form the analytical backbone of the project’s evaluation and validation. <br>

## Data Overview <br>
| Type | Description | Source | <br>
|:--|:--|:--| <br>
| NAV / TRI | Mutual Fund NAVs and Total Return Index | AMFI / Yahoo Finance | <br>
| Macroeconomic Indicators | VIX, USD/INR, Gold, Brent, G-sec yield | RBI, FRED, Investing.com | <br>
| Benchmarks | NIFTY 50 TRI and Sectoral Benchmarks | NSE India | <br>

**All data was collected for academic purposes and preprocessed for consistency and continuity.** <br>

## Highlights <br>
Chronological walk-forward validation ensures realistic testing without leakage. <br>

Ensemble modeling provides stability and improved recall. <br>

Macro and market factors contribute meaningfully to predictive accuracy. <br>

Portfolio backtesting confirms the viability of model-driven fund selection. <br>

Fully reproducible pipeline with environment files and fixed seeds. <br>
## Future Extensions <br>
Integrate LSTM or Transformer architectures for sequential NAV analysis. <br>

Extend portfolio simulation to ETFs and hybrid assets. <br>

Build an interactive Streamlit dashboard for real-time fund recommendation. <br>

Add explainable AI layers (e.g., SHAP) for fund-level interpretability. <br>

## Conceptual Flow <br>

Raw NAV / Macro Data
      ↓
Data Cleaning & Union (Phase 1)
      ↓
Feature Engineering (Phase 2)
      ↓
Classification & Ensemble Modeling (Phase 3)
      ↓
Diagnostics & Cost Analysis (Phase 4)
      ↓
Portfolio Backtesting & Performance Tracking (Phase 5)
      ↓
Reports & Research-Ready Results

## License <br>

This project is open-sourced under the MIT License. You are free to use, modify, and reference this repository for educational or research purposes. <br>
