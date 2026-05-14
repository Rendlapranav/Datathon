# 📊 Executive Sentiment & ROI Pipeline

An enterprise-grade NLP pipeline designed to detect "Silent Killers"—customers who leave 4-5 star ratings but express deep linguistic frustration (anger/sadness)—and project the financial ROI of fixing specific product features.

## 🏗️ Architecture
1. **Multilingual Ingestion:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`
2. **T2E Microservice:** `Helsinki-NLP/opus-mt-mul-en` handles translation to bypass CPU bottlenecks.
3. **Aspect Mining:** Multilingual `BERTopic` clustering with custom c-TF-IDF stop-word penalization.
4. **Emotion Zero-Shot:** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` maps text to 7 core emotions.
5. **Presentation Layer:** Live Plotly + Streamlit Dashboard.

## 🚀 Quickstart
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end pipeline and launch the GUI
./run.sh
