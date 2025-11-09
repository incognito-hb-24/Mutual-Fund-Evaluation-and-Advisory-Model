#!/bin/bash
# ============================================================
# Mutual Fund Advisory & Evaluation Model (MLBA)
# End-to-End Reproduction Script
# Author: Himanshu
# ============================================================

echo " Starting Mutual-Fund-Evaluation-and-Advisory-Model full pipeline execution..."

# Phase 1 — Data ingestion & master creation
echo "▶ Running Phase 1: Data Ingestion & Master Creation..."
python notebooks/phase1_feature_engineering.ipynb

# Phase 2 — Feature engineering
echo "▶ Running Phase 2: Feature Engineering..."
python notebooks/phase2_feature_selection.ipynb

# Phase 3 — Model training & ensemble
echo "▶ Running Phase 3: Model Training & Ensemble..."
python notebooks/phase3_model_training.ipynb

# Phase 4 — Diagnostics & validation
echo "▶ Running Phase 4: Diagnostics & Ablation..."
python notebooks/phase4_diagnostics.ipynb

# Phase 5 — Portfolio simulation & backtesting
echo "▶ Running Phase 5: Backtesting & Performance Simulation..."
python notebooks/phase5_backtesting.ipynb

echo " Mutual-Fund-Evaluation-and-Advisory-Model pipeline completed successfully."
echo "All results are available under: /reports and /data/processed"
