#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting backend processing pipeline..."

# 1. Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# 1.5 Stream the dataset (flagship + competitor discovery) if not already present
if [ ! -f "data/clean/subcategory_sample.csv" ]; then
    echo "📡 Discovering flagship product + competitors from HuggingFace..."
    python3 setup_scripts/stream_data.py
fi

# 2. Resolve product identity FIRST (brand/title for flagship + competitors) —
# must run before the competitor benchmark so it picks up resolved names
# instead of "Unknown"
echo "🏷️ Resolving product identity..."
python3 day3.7_product_identity.py

# 3. Run the pipeline step-by-step
echo "🔍 Running RoBERTa sentiment analysis..."
python3 day1_roberta_sentiment.py

echo "🌐 Running translation..."
# Ensure the script is in the expected directory or provide the full path
python3 day1.5_translate.py

echo "🏷️ Running aspect extraction..."
python3 day2_aspects.py

echo "💼 Running business logic..."
python3 day3_business_logic.py

echo "🤖 Running ML model suite (XGBoost churn, anomaly, forecast, semantic search)..."
python3 day4_ml_models.py

echo "📊 Running competitor benchmark..."
python3 day3.5_competitor_benchmark.py

echo "🤖 Generating AI complaint summary..."
python3 day3.6_ai_complaint_summary.py

echo "✅ Backend processing complete!"

# 3. Launch the Streamlit GUI
echo "📊 Launching Streamlit GUI..."
streamlit run gui/app.py
