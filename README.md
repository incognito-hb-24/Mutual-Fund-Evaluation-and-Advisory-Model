# Mutual-Fund-Evaluation-and-Advisory-Model

# Mutual Fund Evaluation & Advisory Model

Predict whether a mutual fund will **outperform the benchmark** over the next **63 trading days**, then convert forecasts into **Buy / Hold / Sell** actions under uncertainty. The project is built for reproducibility and app-readiness.

---

## Phases (roadmap)

**Phase 0 — Repo & Reproducibility (this step)**
- Clean structure, pinned `requirements.txt`, `Makefile`, and a sample data policy.
- No raw data in Git; full data lives in Google Drive.

**Phase 1 — Trading Calendar & Strict Merge**
- Build `trading_calendar_strict.csv` from NAV dates.
- Align TRI and macros to the NAV trading calendar via forward-fill.
- Output: `data/processed/clean_master_strict.csv`, merge coverage report.

**Phase 2 — Feature Engineering (leak-safe)**
- Momentum: `ret_{7,14,21,63}` and `excess_{7,14,21,63}`
- Risk/quality: `vol_126d`, `sharpe_126d`, `up_vol_63d`, `down_vol_63d`
- Beta/alpha: `beta_126d`, `alpha_ann_126d`
- Path: `mdd_252d`, `days_since_peak`
- Regimes: `corr_bmk_63d`, `regime_{bull,bear}_63d`
- Macro lags: `{india_vix, usd_inr, gsec_10y_yield, gold_inr, brent_usd}_{lag5,lag10}`
- Cross-section ranks per date
- Output: `data/processed/clean_with_features.csv`, `reports/data_card.csv`, `reports/fig_feature_missing.png`

**Phase 3 — Modelling (classification + LSTM)**
- Label: future 63d **excess** return `> 0` (binary)
- Models: Logistic (balanced), RandomForest (balanced), XGBoost, LightGBM, **LSTM sequence classifier**
- Validation: TimeSeriesSplit (walk-forward)
- Metrics: PR-AUC (primary), F1, Precision, Recall; 95% bootstrap CIs
- Threshold sweep & **cost curve** → choose operating point
- Outputs: OOF predictions, results table, PR curve, cost curve, confusion matrix, models saved

**Phase 4 — Policy Simulation (decisions)**
- Probability policy: Buy if score ≥ τ (from cost curve)
- Interval policy: quantile model (p10/p50/p90 of excess) → Buy/Hold/Sell
- Output: `data/processed/policy_simulation.csv`, policy summary figure

**Phase 5 — Error Analysis & Ablations**
- Representative FP/FN cases with feature snapshots
- Ablations: windows, macro toggles, hyperparams
- Baselines: seasonal-naive / constant-prob

**Phase 6 — Documentation & Governance**
- Data Card & Model Card
- One key figure with captions (PR curve or PR-AUC bars with CI)
- README run steps; reproducibility instructions

**Phase 7 — App Scaffolding (deploy later)**
- Streamlit app to visualize leaderboard, PR/threshold, policy
- `app/config.yaml` holds threshold & bucket cut-points

**Phase 8 — Deployment**
- Streamlit Cloud / HF Spaces with a tiny demo dataset and saved models

---
## Repo Structure


## Data Usage

- Full raw & processed data are stored in **Google Drive** (`MyDrive/MLBA_Project/data/`).
- This repo only includes a tiny **`data/sample/`** for smoke testing.
- Do **not** commit private or large data.


## How to Run

**A) Quick demo (on sample data)**

pip install -r requirements.txt
make reproduce

**B) Full pipeline (on Google Drive, Colab)**

Mount Drive and set:

BASE = "/content/drive/MyDrive/MLBA_Project"

Place full CSVs in:

BASE/data/raw/ (NAVs, TRI, macros)

Run phase scripts/notebooks:

Phase 1 → Phase 2 → Phase 3

Outputs:

BASE/data/processed/, BASE/models/, BASE/reports/

