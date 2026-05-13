# day2_aspects.py
# -----------------------------------------------------------------
# RUN:    python day2_aspects.py
# INPUT:  data/clean/day1_reviews_roberta.csv
# OUTPUT: data/clean/day2_aspects.csv
#         figures/day2_heatmap.png
#         figures/day2_topic_wordclouds.png
# -----------------------------------------------------------------
# Install if needed:
#   pip install bertopic sentence-transformers hdbscan umap-learn wordcloud
# -----------------------------------------------------------------
import os
os.environ["USE_TF"] = "0"     # Completely disable TensorFlow for Hugging Face
os.environ["USE_TORCH"] = "1"  # Force PyTorch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings, re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import Dataset
from scipy.special import softmax
from tqdm import tqdm
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

# -- Config --------------------------------------------------------
INPUT_CSV      = "data/clean/day1_reviews_roberta.csv"
OUTPUT_CSV     = "data/clean/day2_aspects.csv"
FIG_DIR        = "figures"
EMOTION_MODEL  = "j-hartmann/emotion-english-distilroberta-base"
BATCH_SIZE     = 16
N_TOPICS       = 5        # ask BERTopic for exactly 5 product aspects
MIN_TOPIC_SIZE = 20       # raised to 20 - filters out tiny irrelevant clusters

os.makedirs(FIG_DIR, exist_ok=True)

# -- Load ----------------------------------------------------------
print(f"Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
df['review_text'] = df['review_text'].astype(str)
print(f"Loaded {len(df):,} reviews | product: {df['product_id'].iloc[0]}")

# -- Text Preprocessing for BERTopic ------------------------------
def clean_text(text):
    # FIX: strip non-breaking spaces (\xa0) and other unicode whitespace
    text = text.replace('\xa0', ' ')
    text = text.lower()
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)    # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\nCleaning review text...")
df['clean_text'] = df['review_text'].apply(clean_text)

# Drop very short reviews (less than 5 words) - not useful for topic modeling
before = len(df)
df = df[df['clean_text'].apply(lambda x: len(x.split()) >= 5)].copy()
print(f"Dropped {before - len(df)} reviews with < 5 words. Remaining: {len(df):,}")

# -- Hardware ------------------------------------------------------
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
device_id = 0 if device == "cuda" else -1   # pipeline expects int: 0=GPU, -1=CPU
print(f"\n[INFO] Device: {device.upper()}")

# -- Step 1: BERTopic - Aspect Mining -----------------------------
print("\n[STEP 1] Running BERTopic for aspect extraction...")
print("         (This may take 3-5 minutes on CPU)")

# Custom vectorizer - remove common English stopwords + product-generic words
vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=1.0
)

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

# -- Topic Info & Manual Labelling --------------------------------
topic_info = bertopic_model.get_topic_info()
print("\n[TOPIC INFO]")
print(topic_info[['Topic', 'Count', 'Name']].to_string(index=False))

print("\n[TOP WORDS PER TOPIC]")

# -- Expanded keyword map with synonyms ---------------------------
# Each entry: (label, [synonyms])
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

def assign_topic_label(topic_id: int, model: BERTopic) -> str:
    """
    Weight-based aspect labeling.

    Score each LABEL_RULES candidate by counting how many of its synonyms
    appear anywhere in the topic's top-10 word list.  The candidate with the
    highest intersection count wins.  Ties are broken by the order of
    LABEL_RULES (more specific labels listed first take priority).
    Returns a generic fallback if no synonym matches at all.
    """
    words_with_scores = model.get_topic(topic_id)       # [(word, c-tf-idf), ...]
    top_words = {w for w, _ in words_with_scores[:10]}  # set for O(1) lookup

    best_label  = None
    best_score  = 0

    for candidate_label, synonyms in LABEL_RULES:
        # Count how many synonyms from this rule appear in the top-10 word set
        score = sum(1 for syn in synonyms if syn in top_words)
        if score > best_score:
            best_score  = score
            best_label  = candidate_label

    if best_label is None:
        best_label = f"Aspect {topic_id + 1}"

    return best_label, best_score


topic_labels = {}
for topic_id in sorted(df['topic_id'].unique()):
    if topic_id == -1:
        topic_labels[-1] = "Outlier"
        continue
    words = bertopic_model.get_topic(topic_id)
    top_words_display = [w for w, _ in words[:6]]
    print(f"  Topic {topic_id}: {top_words_display}")
    label, score = assign_topic_label(topic_id, bertopic_model)
    topic_labels[topic_id] = label
    print(f"         -> Auto-labelled: '{label}'  (matched {score} synonym(s))")

# -- Representative Reviews per Topic (for slide quotes) ----------
print("\n[REPRESENTATIVE REVIEWS] (3 most central per topic - use as slide quotes)")
rep_docs = bertopic_model.get_representative_docs()   # dict: topic_id -> [doc1, doc2, doc3]
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
print(f"\nRepresentative quotes saved -> {quotes_path}  (use for Slides 45-48)")

df['topic_label'] = df['topic_id'].map(topic_labels)

# -- Step 2: Emotion Detection ------------------------------------
print(f"\n[STEP 2] Running emotion detection: {EMOTION_MODEL}")
print("         7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise")
print(f"         Using HuggingFace pipeline + datasets.Dataset (batch_size={BATCH_SIZE})")

# Build pipeline once - handles tokenisation, batching, and device placement
emo_pipeline = pipeline(
    task="text-classification",
    model=EMOTION_MODEL,
    top_k=None,           # return scores for ALL labels (replaces softmax over logits)
    truncation=True,
    max_length=512,
    device=device_id,
)

# Derive canonical label order from the model config
emo_tokenizer_ref = AutoTokenizer.from_pretrained(EMOTION_MODEL)
emo_config_model  = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
emo_id2label      = emo_config_model.config.id2label
emo_labels        = [emo_id2label[i].lower() for i in range(len(emo_id2label))]
del emo_config_model, emo_tokenizer_ref   # free memory; pipeline has its own copy
print(f"Emotion label order: {emo_labels}")

# Wrap in a datasets.Dataset so the pipeline can use its optimised DataLoader
hf_dataset = Dataset.from_dict({"text": df['review_text'].tolist()})

print(f"Running on {len(df):,} rows...")
raw_outputs = emo_pipeline(
    hf_dataset["text"],
    batch_size=BATCH_SIZE,
)

# raw_outputs: list of lists  ->  [[{'label': 'joy', 'score': 0.9}, ...], ...]
# Re-order each inner list to match emo_labels for consistent column assignment
def scores_to_array(output_row: list, label_order: list) -> np.ndarray:
    score_map = {item['label'].lower(): item['score'] for item in output_row}
    return np.array([score_map[lbl] for lbl in label_order])

emo_arr = np.array([scores_to_array(row, emo_labels) for row in raw_outputs])

for idx, label in enumerate(emo_labels):
    df[f'emo_{label}'] = emo_arr[:, idx]

# Dominant emotion per review
df['dominant_emotion'] = [emo_labels[i] for i in emo_arr.argmax(axis=1)]

print("\nEmotion distribution across all reviews:")
print(df['dominant_emotion'].value_counts().to_string())

# -- Step 3: Feature x Emotion Heatmap ---------------------------
print("\n[STEP 3] Building Feature x Emotion Heatmap...")

# Only use reviews assigned to a real topic (not outliers)
df_topics = df[df['topic_id'] != -1].copy()

# [Keep plotting and quote extraction code exactly as before]

# -- Save Enriched CSV --------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nEnriched dataset saved -> {OUTPUT_CSV}")
print(f"New columns: topic_id, topic_prob, topic_label, dominant_emotion, emo_*")
print(f"Representative quotes -> data/clean/day2_representative_quotes.csv")

# -- Summary for Day 3 --------------------------------------------
print("\n" + "="*60)
print("DAY 2 SUMMARY - Pass to day3_matrix.py")
print("="*60)
print(f"  Topics found       : {df['topic_id'].nunique() - (1 if -1 in df['topic_id'].values else 0)}")
print(f"  Outlier reviews    : {(df['topic_id'] == -1).sum()} (topic_id = -1, excluded from heatmap)")
print(f"  Dominant emotions  :")
print(df['dominant_emotion'].value_counts().to_string())

aspect_emotions = df_topics.groupby('topic_label').agg(
    mean_anger   = ('emo_anger',   'mean'),
    mean_sadness = ('emo_sadness', 'mean'),
    review_count = ('topic_label', 'count'),
).reset_index()
aspect_emotions['emotional_intensity'] = (
    aspect_emotions['mean_anger'] + aspect_emotions['mean_sadness']
)

print(f"\n  Top aspect by anger+sadness:")
top = aspect_emotions.sort_values('emotional_intensity', ascending=False).iloc[0]
print(f"  -> '{top['topic_label']}' (intensity={top['emotional_intensity']:.3f}, n={int(top['review_count'])})")

# Summary of top 5 words per aspect
topic_info = bertopic_model.get_topic_info()
print("\n--- TOP 5 WORDS PER ASPECT ---")
for topic_id in topic_info['Topic'].tolist()[1:6]:   # skip Topic -1 (outliers)
    words = bertopic_model.get_topic(topic_id)
    top_5 = [word for word, score in words[:5]]
    print(f"Aspect {topic_id}: {top_5}")
print("\n Done. Pass data/clean/day2_aspects.csv to day3_matrix.py")
