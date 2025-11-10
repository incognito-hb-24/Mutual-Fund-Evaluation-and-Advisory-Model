## Data Directory — Mutual Fund Advisory & Evaluation Model (MLBA) <br>

This folder contains all data files used and generated throughout the MLBA workflow. <br>
Each subfolder corresponds to a stage in the end-to-end pipeline — from raw collection to processed, feature-enriched datasets ready for modeling and evaluation. <br>

---

##  Folder Structure <br>
data/   <br>
├── raw/               <br>             
├── processed/                <br>
└── README.md              <br>
<br>

---

##  raw/ Folder <br>

This folder stores the **original datasets** before any cleaning, transformation, or feature engineering. <br>
These files are typically downloaded, collected, or merged from public data sources. <br>

**Typical contents include:** <br>

| **File**               | **Description**                                                       | **Source**                    |
|------------------------|-----------------------------------------------------------------------|-------------------------------|
| `nav_largecap.csv`     | Daily NAV (Net Asset Value) of each mutual fund scheme                | AMFI / Yahoo Finance          |
| `benchmark_tri.csv`    | Total Return Index used as the fund benchmark                         | NSE / Yahoo Finance           |
| `macro_indicators.csv` | Macroeconomic indicators (VIX, USD/INR, G-sec yield, Gold, Brent)     | RBI / Investing.com / FRED    |
| `fund_master.xlsx`     | Metadata linking fund IDs, categories, and launch dates               | Manually curated              |



 **Notes:** <br>
- These files are not uploaded to GitHub (excluded via `.gitignore`) due to data size and licensing. <br>
- Only small synthetic or sample slices can be included for demonstration. <br>

---

##  processed/ Folder <br>

This folder stores **intermediate and final datasets** used for feature generation, model training, diagnostics, and backtesting. <br>
Every file in this directory is the direct result of one or more transformation phases within the MLBA pipeline. <br>

**Key files and their purposes:** <br>
| **File**                              | **Description**                                                                                           |
|-------------------------------        |-------------------------------------------------------------------------------------------------------    |
| `clean_master_union.csv`              | Unified master dataset created in Phase 1 (aligned NAV, TRI, and macro data)                              |
| `clean_with_features.csv`             | Feature-engineered dataset containing rolling returns, volatility, Sharpe, beta, drawdown, and macro lags |
| `clf_oof_predictions.csv`             | Out-of-fold model predictions from Phase 3 (classification ensemble)                                      |
| `results_summary_classification.csv`  | Summary of model performance metrics such as PR-AUC, precision, recall, and F1 score                      |
| `bt_tranche_log.csv`                  | Transaction-level log from portfolio backtesting                                                          |
| `bt_turnover_stats.csv`               | Per-period turnover and performance summary                                                               |
| `bt_robustness_grid.csv`              | Robustness test results across portfolio size and threshold combinations                                  |
| `bt_metrics.json`                     | Final backtest metrics (Sharpe, drawdown, hit rate, turnover)                                             |

 **Each processed file corresponds to one or more MLBA Phases:** <br>
- **Phase 1:** `clean_master_union.csv` <br>
- **Phase 2:** `clean_with_features.csv` <br>
- **Phase 3:** `clf_oof_predictions.csv`, `results_summary_classification.csv` <br>
- **Phase 4:** Diagnostics outputs (PR-AUC tables, cost curves) <br>
- **Phase 5:** Backtesting outputs (equity curves, trade logs, metrics) <br>

---

##  Data Privacy & Ethics <br>

All datasets used in this project are either: <br>
- Publicly available from government or exchange data portals, <br>
- Or used strictly for academic and research purposes within educational fair-use limits. <br>

**No confidential, proprietary, or personally identifiable data is included.** <br>

---

##  Data Flow Summary <br>
raw/     <br>
↓ (Phase 1)    <br>
clean_master_union.csv    <br>
↓ (Phase 2)    <br>
clean_with_features.csv    <br>
↓ (Phase 3)    <br>
model_training_results + ensemble predictions   <br>
↓ (Phase 4)    <br>
diagnostics, ablation & PR-AUC results    <br>
↓ (Phase 5)    <br>
backtesting logs, equity curves, performance metrics    <br>


**This flow ensures full transparency, version control, and the ability to reproduce results across environments.** <br>

---

##  Best Practices <br>

- Keep **raw/** data immutable (never overwrite). <br>
- Use **processed/** for all derived datasets. <br>
- Add a short note in this file when new data sources are integrated. <br>
- Store large files on Google Drive, Kaggle Datasets, or Zenodo, and reference them in this README. <br>

---
