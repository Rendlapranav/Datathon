# 🚨 The Product Rescue Mission

> **Product Rescue Mission Datathon 2026** · Enterprise NLP Pipeline detecting Silent Killers and quantifying revenue at risk.

## 🏆 Key Results
| Metric | Value |
|--------|-------|
| Reviews Analyzed | 50,000 |
| Silent Killers Detected | 420 (8.4%) |
| Revenue at Risk | **$173,250** |
| Top Issue | Battery & Charging |
| Sentiment Boost if Fixed | **+52.9%** |

---

## 🏗️ Architecture

```
Raw Reviews (50K)
    │
    ▼
[Day 1] RoBERTa Sentiment  (cardiffnlp/twitter-xlm-roberta-base-sentiment)
    │   → prob_negative / positive / neutral
    │   → Silent Killer flag (data-driven p10 threshold)
    │   → Revenue at Risk (AOV × 5x Bain multiplier × sk_weight)
    │
    ▼
[Day 1.5] Helsinki-NLP Translation  (opus-mt-mul-en)
    │   → 25 languages → English
    │
    ▼
[Day 2] BERTopic Aspect Mining + mDeBERTa Emotion
    │   → Feature clusters: Battery, Connectivity, Sound, Value
    │   → Emotions: Anger / Disappointment / Joy
    │
    ▼
[Day 3] Strategic Business Logic
    │   → Priority Matrix (Frequency × Emotional Intensity)
    │   → ROI Simulation (Scenario A/B)
    │   → CEO JSON export
    │
    ▼
[GUI] Streamlit Dashboard  +  Jupyter Notebook  +  HTML Slides
```

---

## 🚀 Quickstart (Windows)

```powershell
# Clone the repo
git clone https://github.com/Rendlapranav/Datathon.git
cd Datathon

# One-command: streams data, runs pipeline, launches dashboard
.\run.ps1
```

## 🚀 Quickstart (Linux/Mac)

```bash
git clone https://github.com/Rendlapranav/Datathon.git
cd Datathon
pip install -r requirements.txt
./run.sh
```

---

## 📁 Repository Structure

```
Datathon/
├── data/
│   ├── raw/                    # Original dataset (gitignored)
│   └── clean/                  # Pipeline outputs (gitignored)
│       ├── subcategory_sample.csv
│       ├── day1_reviews_roberta.csv
│       ├── day1.5_translated.csv
│       ├── day2_aspects.csv
│       ├── day2_representative_quotes.csv
│       └── ceo_summary.json
├── notebooks/
│   ├── make_notebook.py        # Generates the .ipynb
│   └── product_rescue_analysis.ipynb
├── gui/
│   ├── app.py                  # Streamlit dashboard (5 tabs)
│   └── dash_app.py             # Dash alternative
├── slides/
│   └── deck.html               # 12-slide CEO presentation
├── figures/
│   ├── day2_heatmap.png
│   ├── day2_heatmap_urgency.png
│   ├── day2_emotional_intensity.png
│   └── day3_priority_matrix.png
├── setup_scripts/
│   ├── stream_data.py          # HuggingFace streaming (50K reviews)
│   ├── load_data.py
│   ├── verify_data.py
│   └── fix_data.py
├── day1_roberta_sentiment.py
├── day1.5_translate.py
├── day2_aspects.py
├── day3_business_logic.py
├── requirements.txt
├── run.sh                      # Linux/Mac launcher
├── run.ps1                     # Windows launcher
└── README.md
```

---

## 📊 Dashboard Tabs

| Tab | Content |
|-----|---------|
| 🎯 Priority Matrix & Roadmap | CEO bubble chart · Strategic roadmap · **Interactive ROI Calculator** |
| 🧠 Emotion Deep Dive | Feature × Emotion heatmap · Urgency signal · Stacked bar |
| 🗣️ Voice of Customer | Auto-extracted representative quotes per aspect |
| 📈 **Temporal Trend** *(NEW)* | Monthly sentiment drift · Silent Killer rate · Changepoint detection |
| 🔍 **Silent Killer Drill-Down** *(NEW)* | Filterable table of all flagged reviews + revenue by aspect |

---

## 🧠 Model Justifications

**Why RoBERTa over VADER?**
VADER is English-only and rule-based. Our dataset has 25 languages. `cardiffnlp/twitter-xlm-roberta-base-sentiment` achieves SOTA cross-lingual sentiment.

**Why p10 threshold for Silent Killers?**
We ask: what's the minimum confidence the model shows on confirmed-bad (1-2★) reviews? The p10 of that distribution = our bar. It recalibrates per dataset — no magic number.

**Why Bain 5x churn multiplier?**
Conservative lower bound of the 5-25x range from Bain & Company / HBR research on customer retention.

---

## 👥 Team

Built for The Product Rescue Mission Datathon 2026.
