#!/bin/bash
# ==========================================
# CEO Sentiment Pipeline Orchestrator
# ==========================================

echo "[1/4] Running Day 1: XLM-RoBERTa Sentiment Analysis..."
python3 day1_roberta_sentiment.py

echo "[2/4] Running Day 1.5: Translate-to-English (T2E) Microservice..."
python3 setup_scripts/day1.5_translate.py

echo "[3/4] Running Day 2: BERTopic & Zero-Shot Emotion Extraction..."
python3 day2_aspects.py

echo "[4/4] Running Day 3: ROI Simulation & Business Matrix..."
python3 day3_matrix.py

echo "Pipeline execution complete. Launching Executive Dashboard..."
streamlit run gui/app.py
