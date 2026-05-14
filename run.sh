#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting backend processing pipeline..."

# 1. Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# 2. Run the pipeline step-by-step
echo "🔍 Running RoBERTa sentiment analysis..."
python3 day1_roberta_sentiment.py

echo "🌐 Running translation..."
# Ensure the script is in the expected directory or provide the full path
python3 day1.5_translate.py

echo "🏷️ Running aspect extraction..."
python3 day2_aspects.py

echo "💼 Running business logic..."
python3 day3_business_logic.py

echo "✅ Backend processing complete!"

# 3. Launch the Streamlit GUI
echo "📊 Launching Streamlit GUI..."
streamlit run gui/app.py
