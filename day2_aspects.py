# day2_aspects.py
# ─────────────────────────────────────────────────────────────────
# RUN:    python day2_aspects.py
# INPUT:  data/clean/day1_reviews_roberta.csv
# OUTPUT: data/clean/day2_aspects.csv
#         figures/day2_heatmap.png
#         figures/day2_topic_wordclouds.png
# ─────────────────────────────────────────────────────────────────
# Install if needed:
#   pip install bertopic sentence-transformers hdbscan umap-learn wordcloud
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings, re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────
INPUT_CSV      = "data/clean/day1_reviews_roberta.csv"
OUTPUT_CSV     = "data/clean/day2_aspects.csv"
FIG_DIR        = "figures"
EMOTION_MODEL  = "j-hartmann/emotion-english-distilroberta-base"
BATCH_SIZE     = 16
N_TOPICS       = 5        # ask BERTopic for exactly 5 product aspects
MIN_TOPIC_SIZE = 20       # raised to 20 — filters out tiny irrelevant clusters

os.makedirs(FIG_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────
print(f"Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
df['review_text'] = df['review_text'].astype(str)
print(f"Loaded {len(df):,} reviews | product: {df['product_id'].iloc[0]}")

# ── Text Preprocessing for BERTopic ──────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)    # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\nCleaning review text...")
df['clean_text'] = df['review_text'].apply(clean_text)

# Drop very short reviews (less than 5 words) — not useful for topic modeling
before = len(df)
df = df[df['clean_text'].apply(lambda x: len(x.split()) >= 5)].copy()
print(f"Dropped {before - len(df)} reviews with < 5 words. Remaining: {len(df):,}")

# ── Hardware ──────────────────────────────────────────────────────
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"\n[INFO] Device: {device.upper()}")

# ── Step 1: BERTopic — Aspect Mining ─────────────────────────────
print("\n[STEP 1] Running BERTopic for aspect extraction...")
print("         (This may take 3–5 minutes on CPU)")

# Custom vectorizer — remove common English stopwords + product-generic words
custom_stopwords = [
    "headphone", "headphones", "earphone", "earphones", "earbud", "earbuds",
    "product", "amazon", "buy", "bought", "purchase", "use", "using", "used",
    "one", "get", "got", "like", "also", "would", "really", "just", "very",
    "good", "great", "nice", "bad", "well", "make", "made", "way", "even",
    "still", "much", "lot", "little", "best", "better", "worst"
]

vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=1.0
)

# Filter out custom stopwords post-hoc via BERTopic's vectorizer model
bertopic_model = BERTopic(
    vectorizer_model=vectorizer,
    nr_topics=N_TOPICS,
    min_topic_size=MIN_TOPIC_SIZE,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = bertopic_model.fit_transform(df['clean_text'].tolist())

df['topic_id']   = topics
df['topic_prob'] = [p.max() if hasattr(p, '__len__') else p for p in probs]

# ── Topic Info & Manual Labelling ────────────────────────────────
topic_info = bertopic_model.get_topic_info()
print("\n[TOPIC INFO]")
print(topic_info[['Topic', 'Count', 'Name']].to_string(index=False))

print("\n[TOP WORDS PER TOPIC]")

# ── Expanded keyword map with synonyms ───────────────────────────
# Each entry: (label, [synonyms]) — first match wins
LABEL_RULES = [
    ("Battery & Charging",  ["battery", "charge", "charging", "life", "juice",
                              "drain", "power", "hour", "hours", "dies", "dead"]),
    ("Sound Quality",       ["sound", "bass", "audio", "quality", "noise",
                              "cancel", "cancellation", "treble", "eq", "volume",
                              "loud", "clarity", "distort"]),
    ("Comfort & Fit",       ["comfort", "fit", "ear", "wear", "pain", "tight",
                              "loose", "cushion", "foam", "tip", "tips", "ergonomic"]),
    ("Connectivity",        ["connect", "bluetooth", "pair", "pairing", "signal",
                              "cord", "plug", "wire", "wireless", "disconnect",
                              "drop", "lag", "latency", "range"]),
    ("Microphone & Calls",  ["mic", "microphone", "call", "voice", "phone",
                              "siri", "alexa", "assistant", "mute", "pickup"]),
    ("Value for Money",     ["price", "value", "money", "worth", "cost",
                              "cheap", "expensive", "afford", "deal", "budget"]),
    ("Build Quality",       ["build", "durable", "durability", "material", "plastic",
                              "hinge", "break", "broke", "crack", "flimsy",
                              "sturdy", "robust", "quality"]),
]

topic_labels = {}
for topic_id in sorted(df['topic_id'].unique()):
    if topic_id == -1:
        topic_labels[-1] = "Outlier"
        continue
    words = bertopic_model.get_topic(topic_id)
    top_words = [w for w, _ in words[:10]]   # use top 10 for better synonym coverage
    print(f"  Topic {topic_id}: {top_words[:6]}")
    joined = " ".join(top_words)
    label = None
    for candidate_label, synonyms in LABEL_RULES:
        if any(syn in joined for syn in synonyms):
            label = candidate_label
            break
    if label is None:
        label = f"Aspect {topic_id + 1}"
    topic_labels[topic_id] = label
    print(f"         → Auto-labelled: '{label}'")

# ── Representative Reviews per Topic (for slide quotes) ──────────
print("\n[REPRESENTATIVE REVIEWS] (3 most central per topic — use as slide quotes)")
rep_docs = bertopic_model.get_representative_docs()   # dict: topic_id → [doc1, doc2, doc3]
representative_records = []
for topic_id, docs in rep_docs.items():
    if topic_id == -1:
        continue
    label = topic_labels.get(topic_id, f"Aspect {topic_id + 1}")
    print(f"\n  {label} (Topic {topic_id}):")
    for i, doc in enumerate(docs[:3]):
        snippet = doc[:200].replace('\n', ' ')
        print(f"    Quote {i+1}: \"{snippet}\"")
        representative_records.append({
            "topic_id":    topic_id,
            "topic_label": label,
            "quote_rank":  i + 1,
            "quote_text":  doc
        })

# Save quotes to CSV for easy copy-paste into slides
quotes_df = pd.DataFrame(representative_records)
quotes_path = "data/clean/day2_representative_quotes.csv"
quotes_df.to_csv(quotes_path, index=False)
print(f"\nRepresentative quotes saved → {quotes_path}  (use for Slides 45–48)")

df['topic_label'] = df['topic_id'].map(topic_labels)

# ── Step 2: Emotion Detection ─────────────────────────────────────
print(f"\n[STEP 2] Running emotion detection: {EMOTION_MODEL}")
print("         7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise")

emo_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
emo_model     = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL).to(device)
emo_model.eval()

# Read emotion label order from config
emo_id2label = emo_model.config.id2label
emo_labels   = [emo_id2label[i].lower() for i in range(len(emo_id2label))]
print(f"Emotion label order: {emo_labels}")

all_emo_probs = []
print(f"Running on {len(df):,} rows...")

for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch = df['review_text'].iloc[i:i + BATCH_SIZE].tolist()
    enc   = emo_tokenizer(
        batch, padding=True, truncation=True,
        max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = emo_model(**enc).logits
    probs = softmax(logits.cpu().numpy(), axis=1)
    all_emo_probs.extend(probs)

emo_arr = np.array(all_emo_probs)
for idx, label in enumerate(emo_labels):
    df[f'emo_{label}'] = emo_arr[:, idx]

# Dominant emotion per review
df['dominant_emotion'] = [emo_labels[i] for i in emo_arr.argmax(axis=1)]

print("\nEmotion distribution across all reviews:")
print(df['dominant_emotion'].value_counts().to_string())

# ── Step 3: Feature × Emotion Heatmap ────────────────────────────
print("\n[STEP 3] Building Feature × Emotion Heatmap...")

# Only use reviews assigned to a real topic (not outliers)
df_topics = df[df['topic_id'] != -1].copy()

# ── Heatmap A: Anger + Sadness only (CEO urgency view) ───────────
# These are the primary drivers of the "Silent Quitter" segment
urgency_emotions = ['anger', 'sadness']
cols_present = [e for e in urgency_emotions if f'emo_{e}' in df_topics.columns]

heatmap_urgency = (
    df_topics.groupby('topic_label')[[f'emo_{e}' for e in cols_present]]
    .mean() * 100
)
heatmap_urgency.columns = [e.capitalize() for e in cols_present]

# ── Heatmap B: All non-neutral emotions (full picture) ────────────
emotions_to_plot = [e for e in emo_labels if e != 'neutral']
heatmap_data = (
    df_topics
    .groupby(['topic_label', 'dominant_emotion'])
    .size()
    .unstack(fill_value=0)
)
heatmap_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
cols_present_full = [e for e in emotions_to_plot if e in heatmap_pct.columns]
heatmap_pct = heatmap_pct[cols_present_full]

print("\nFeature × Emotion matrix (%):")
print(heatmap_pct.round(1).to_string())

# ── Plot Heatmap A: Anger + Sadness (primary slide chart) ─────────
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    heatmap_urgency,
    ax=ax,
    cmap='YlOrRd',
    annot=True,
    fmt='.1f',
    linewidths=0.5,
    linecolor='#1e293b',
    cbar_kws={'label': 'Mean probability (%)'},
)
ax.set_title(
    f'Urgency Lever: Anger & Sadness Per Aspect\n'
    f'Product: {df["product_id"].iloc[0]}  |  Higher = stronger silent quitter signal',
    fontsize=12, pad=12
)
ax.set_xlabel('Negative Emotion', fontsize=11)
ax.set_ylabel('Product Aspect', fontsize=11)
ax.tick_params(axis='x', rotation=0)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
urgency_heatmap_path = os.path.join(FIG_DIR, 'day2_heatmap_urgency.png')
plt.savefig(urgency_heatmap_path, dpi=300, bbox_inches='tight')
print(f"\nUrgency heatmap saved → {urgency_heatmap_path}")

# ── Plot Heatmap B: Full emotion view ─────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

sns.heatmap(
    heatmap_pct,
    ax=ax,
    cmap='RdYlGn_r',
    annot=True,
    fmt='.1f',
    linewidths=0.5,
    linecolor='#1e293b',
    cbar_kws={'label': '% of reviews (dominant emotion)'},
    vmin=0,
    vmax=heatmap_pct.values.max()
)

ax.set_title(
    f'Product Aspect × Emotion Heatmap  |  Product: {df["product_id"].iloc[0]}\n'
    f'(% of reviews per aspect where that emotion dominates)',
    fontsize=13, pad=15
)
ax.set_xlabel('Dominant Emotion', fontsize=11)
ax.set_ylabel('Product Aspect', fontsize=11)
ax.tick_params(axis='x', rotation=30)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
heatmap_path = os.path.join(FIG_DIR, 'day2_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved → {heatmap_path}")

# ── Plot: Mean Anger + Sadness per Aspect (input for Day 3) ──────
# This is the Y-axis of the Day 3 Priority Matrix
fig, ax = plt.subplots(figsize=(9, 5))

aspect_emotions = df_topics.groupby('topic_label').agg(
    mean_anger   = ('emo_anger',   'mean'),
    mean_sadness = ('emo_sadness', 'mean'),
    review_count = ('topic_label', 'count'),
).reset_index()

aspect_emotions['emotional_intensity'] = (
    aspect_emotions['mean_anger'] + aspect_emotions['mean_sadness']
)
aspect_emotions = aspect_emotions.sort_values('emotional_intensity', ascending=True)

bars = ax.barh(
    aspect_emotions['topic_label'],
    aspect_emotions['emotional_intensity'],
    color='#ef4444', alpha=0.8
)
ax.set_xlabel('Emotional Intensity (mean anger + sadness probability)', fontsize=11)
ax.set_title('Emotional Pain Per Product Aspect\n(Preview of Day 3 Priority Matrix Y-axis)',
             fontsize=12)
ax.grid(True, alpha=0.15, axis='x')

# Annotate with review counts
for bar, (_, row) in zip(bars, aspect_emotions.iterrows()):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"n={int(row['review_count'])}", va='center', fontsize=9, color='white')

plt.tight_layout()
intensity_path = os.path.join(FIG_DIR, 'day2_emotional_intensity.png')
plt.savefig(intensity_path, dpi=300, bbox_inches='tight')
print(f"Emotional intensity chart saved → {intensity_path}")

# ── Save Enriched CSV ─────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nEnriched dataset saved → {OUTPUT_CSV}")
print(f"New columns: topic_id, topic_prob, topic_label, dominant_emotion, emo_*")
print(f"Representative quotes → data/clean/day2_representative_quotes.csv")

# ── Summary for Day 3 ─────────────────────────────────────────────
print("\n" + "="*60)
print("DAY 2 SUMMARY — Pass to day3_matrix.py")
print("="*60)
print(f"  Topics found       : {df['topic_id'].nunique() - (1 if -1 in df['topic_id'].values else 0)}")
print(f"  Outlier reviews    : {(df['topic_id'] == -1).sum()} (topic_id = -1, excluded from heatmap)")
print(f"  Dominant emotions  :")
print(df['dominant_emotion'].value_counts().to_string())
print(f"\n  Top aspect by anger+sadness:")
top = aspect_emotions.sort_values('emotional_intensity', ascending=False).iloc[0]
print(f"  → '{top['topic_label']}' (intensity={top['emotional_intensity']:.3f}, n={int(top['review_count'])})")
# 1. Get a summary of all topics found
topic_info = bertopic_model.get_topic_info()

# 2. Extract and print the Top 5 words for the Top 5 aspects (excluding outlier -1)
print("\n--- TOP 5 WORDS PER ASPECT ---")
for topic_id in topic_info['Topic'].tolist()[1:6]: # Skip Topic -1 (outliers)
    words = bertopic_model.get_topic(topic_id)
    top_5 = [word for word, score in words[:5]]
    print(f"Aspect {topic_id}: {top_5}")
print("\n✓ Day 2 complete. Pass data/clean/day2_aspects.csv to day3_matrix.py")

