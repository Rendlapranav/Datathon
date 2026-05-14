import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, re
import torch
from transformers import pipeline
from datasets import Dataset
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

# -- Config --------------------------------------------------------
INPUT_CSV      = "data/clean/day1.5_translated.csv"
OUTPUT_CSV     = "data/clean/day2_aspects.csv"
FIG_DIR        = "figures"

EMOTION_MODEL  = "j-hartmann/emotion-english-distilroberta-base"

# Full model output labels -- model always returns all 7, we cannot change this.
# This is a fixed-head classifier, NOT a zero-shot NLI model.
MODEL_EMO_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# The 3 signals that are semantically meaningful for hardware product reviews.
# "sadness" is retained as the model label; renamed to "disappointment" in plots only.
# Rationale: "fear" and "surprise" carry no actionable signal for a CEO reviewing
# hardware complaints. "disgust" is too ambiguous to separate from "anger" reliably.
ACTIVE_EMOTIONS  = ["anger", "sadness", "joy"]
DISPLAY_LABELS   = {
    "anger"  : "Anger",
    "sadness": "Disappointment",   # semantic rename for executive communication
    "joy"    : "Joy",
}

BATCH_SIZE     = 16
N_TOPICS       = 5
MIN_TOPIC_SIZE = 20

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("data/clean", exist_ok=True)

# -- Load ----------------------------------------------------------
print(f"Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
df["review_text"] = df["review_text"].astype(str)
print(f"Loaded {len(df):,} reviews | product: {df['product_id'].iloc[0]}")

# -- Text Preprocessing for BERTopic ------------------------------
def clean_text(text):
    text = text.replace("\xa0", " ")
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("\nCleaning review text...")
df["clean_text"] = df["review_text"].apply(clean_text)

before = len(df)
df = df[df["clean_text"].apply(lambda x: len(x.split()) >= 5)].copy()
df = df.reset_index(drop=True)
print(f"Dropped {before - len(df)} reviews with < 5 words. Remaining: {len(df):,}")

# -- Hardware ------------------------------------------------------
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
device_id = 0 if device == "cuda" else -1
print(f"\n[INFO] Device: {device.upper()}")

# -- Step 1: BERTopic - Aspect Mining -----------------------------
print("\n[STEP 1] Running BERTopic for aspect extraction...")

vectorizer   = CountVectorizer(stop_words="english", ngram_range=(1, 2))
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

bertopic_model = BERTopic(
    vectorizer_model        = vectorizer,
    ctfidf_model            = ctfidf_model,
    nr_topics               = N_TOPICS,
    min_topic_size          = MIN_TOPIC_SIZE,
    calculate_probabilities = True,
    verbose                 = True,
)

topics, probs = bertopic_model.fit_transform(df["clean_text"].tolist())

df["topic_id"]   = topics
df["topic_prob"] = [p.max() if hasattr(p, "__len__") else p for p in probs]

topic_info = bertopic_model.get_topic_info()
print("\n[TOPIC INFO]")
print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))

LABEL_RULES = [
    ("Battery & Charging", ["battery", "charge", "charging", "life", "juice",
                             "drain", "power", "hour", "hours", "dies", "dead"]),
    ("Sound Quality",      ["sound", "bass", "audio", "quality", "noise",
                             "cancel", "cancellation", "treble", "eq", "volume",
                             "loud", "clarity", "distort"]),
    ("Comfort & Fit",      ["comfort", "fit", "ear", "wear", "pain", "tight",
                             "loose", "cushion", "foam", "tip", "tips", "ergonomic"]),
    ("Connectivity",       ["connect", "bluetooth", "pair", "pairing", "signal",
                             "cord", "plug", "wire", "wireless", "disconnect",
                             "drop", "lag", "latency", "range"]),
    ("Microphone & Calls", ["mic", "microphone", "call", "voice", "phone",
                             "siri", "alexa", "assistant", "mute", "pickup"]),
    ("Value for Money",    ["price", "value", "money", "worth", "cost", "cheap",
                             "expensive", "afford", "deal", "budget"]),
    ("Build Quality",      ["build", "durable", "durability", "material", "plastic",
                             "hinge", "break", "broke", "crack", "flimsy",
                             "sturdy", "robust", "quality"]),
]

def assign_topic_label(topic_id, model):
    words_with_scores = model.get_topic(topic_id)
    top_words = {w for w, _ in words_with_scores[:10]}
    best_label, best_score = None, 0
    for candidate_label, synonyms in LABEL_RULES:
        score = sum(1 for syn in synonyms if syn in top_words)
        if score > best_score:
            best_score = score
            best_label = candidate_label
    if best_label is None:
        best_label = f"Aspect {topic_id + 1}"
    return best_label, best_score

topic_labels = {}
for topic_id in sorted(df["topic_id"].unique()):
    if topic_id == -1:
        topic_labels[-1] = "Outlier"
        continue
    words = bertopic_model.get_topic(topic_id)
    top_words_display = [w for w, _ in words[:6]]
    print(f"  Topic {topic_id}: {top_words_display}")
    label, score = assign_topic_label(topic_id, bertopic_model)
    topic_labels[topic_id] = label
    print(f"         -> Auto-labelled: '{label}'  (matched {score} synonym(s))")

df["topic_label"] = df["topic_id"].map(topic_labels)

# -- Representative Reviews Extraction ----------------------------
print("\n[REPRESENTATIVE REVIEWS] (3 most central per topic)")
rep_docs = bertopic_model.get_representative_docs()
representative_records = []
for topic_id, docs in rep_docs.items():
    if topic_id == -1:
        continue
    label = topic_labels.get(topic_id, f"Aspect {topic_id + 1}")
    print(f"\n  {label} (Topic {topic_id}):")
    for i, doc in enumerate(docs[:3]):
        snippet = doc[:200].replace("\n", " ")
        print(f"    Quote {i+1}: \"{snippet}\"")
        representative_records.append({
            "topic_id"   : topic_id,
            "topic_label": label,
            "quote_rank" : i + 1,
            "quote_text" : doc,
        })

quotes_df   = pd.DataFrame(representative_records)
quotes_path = "data/clean/day2_representative_quotes.csv"
quotes_df.to_csv(quotes_path, index=False)
print(f"\nRepresentative quotes saved -> {quotes_path}")

# -- Step 2: Emotion Detection ------------------------------------
# Model: j-hartmann/emotion-english-distilroberta-base
# This is a FIXED-HEAD classifier trained on 7 classes.
# It is NOT a zero-shot NLI model -- candidate labels cannot be injected.
# We run full inference (all 7 scores), then select only the 3 columns
# that carry actionable signal for hardware product analysis.
# Discarded: disgust, fear, surprise, neutral
#   - "fear" and "surprise" appear on hardware reviews as noise artifacts
#   - "disgust" correlates too strongly with "anger" to add independent signal
#   - "neutral" is not actionable for a CEO prioritization pitch
print(f"\n[STEP 2] Running emotion detection: {EMOTION_MODEL}")
print(f"Model outputs 7 classes. Retaining 3 for downstream analysis: "
      f"{ACTIVE_EMOTIONS}")
print(f"Display rename: 'sadness' -> 'Disappointment' (executive communication only)")

emo_pipeline = pipeline(
    task      = "text-classification",
    model     = EMOTION_MODEL,
    framework = "pt",
    top_k     = None,
    device    = device_id,
    truncation= True,
    max_length= 512,
)

model_labels = set(emo_pipeline.model.config.id2label.values())
print(f"Model emotion labels confirmed: {model_labels}")

missing = [e for e in ACTIVE_EMOTIONS if e not in model_labels]
if missing:
    raise ValueError(
        f"ACTIVE_EMOTIONS contains labels not in model output: {missing}\n"
        f"Valid labels are: {model_labels}"
    )

hf_dataset  = Dataset.from_dict({"text": df["review_text"].tolist()})
print(f"Running on {len(df):,} rows (batch_size={BATCH_SIZE})...")

raw_outputs = emo_pipeline(
    hf_dataset["text"],
    batch_size = BATCH_SIZE,
    truncation = True,
    max_length = 512,
)

# Unpack all 7 scores, then immediately select only ACTIVE_EMOTIONS columns.
# Storing only the 3 active columns keeps the output CSV clean and prevents
# downstream code from accidentally using the discarded emotion signals.
records = []
for result in raw_outputs:
    score_map = {item["label"]: item["score"] for item in result}
    records.append({emo: score_map[emo] for emo in ACTIVE_EMOTIONS})

emo_df = pd.DataFrame(records, index=df.index)
for emo in ACTIVE_EMOTIONS:
    df[f"emo_{emo}"] = emo_df[emo]

# dominant_emotion is resolved only within ACTIVE_EMOTIONS.
# A review where "fear" would have scored highest will be assigned
# its highest-scoring active emotion instead. This is intentional --
# "fear" on a battery review is almost always model noise.
emo_matrix       = emo_df[ACTIVE_EMOTIONS].values
dominant_indices = emo_matrix.argmax(axis=1)
df["dominant_emotion"]         = [ACTIVE_EMOTIONS[i] for i in dominant_indices]
df["dominant_emotion_display"] = df["dominant_emotion"].map(DISPLAY_LABELS)

print("\nEmotion distribution across all reviews (active emotions only):")
print(df["dominant_emotion_display"].value_counts().to_string())

# -- Step 3: Visualizations ---------------------------------------
print("\n[STEP 3] Building Feature x Emotion Visualizations...")
df_topics = df[df["topic_id"] != -1].copy()

# Heatmap A: Anger + Disappointment (Urgency) -- 2-column, clean signal
heatmap_urgency = (
    df_topics.groupby("topic_label")[["emo_anger", "emo_sadness"]].mean() * 100
)
heatmap_urgency.columns = ["Anger", "Disappointment"]

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    heatmap_urgency, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
    linewidths=0.5, linecolor="#1e293b",
    cbar_kws={"label": "Mean probability (%)"},
)
ax.set_title(
    f"Urgency Signal: Anger & Disappointment Per Aspect\n"
    f"Product: {df['product_id'].iloc[0]}  |  Higher = stronger silent quitter signal",
    fontsize=12, pad=12,
)
ax.set_xlabel("Negative Emotion", fontsize=11)
ax.set_ylabel("Product Aspect", fontsize=11)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "day2_heatmap_urgency.png"),
    dpi=300, bbox_inches="tight",
)

# Heatmap B: Full 3-emotion view (dominant emotion % per aspect)
heatmap_data = (
    df_topics.groupby(["topic_label", "dominant_emotion_display"])
             .size()
             .unstack(fill_value=0)
)
heatmap_pct  = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
display_col_order = [DISPLAY_LABELS[e] for e in ACTIVE_EMOTIONS
                     if DISPLAY_LABELS[e] in heatmap_pct.columns]
heatmap_pct  = heatmap_pct[display_col_order]

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    heatmap_pct, ax=ax, cmap="RdYlGn_r", annot=True, fmt=".1f",
    linewidths=0.5, linecolor="#1e293b",
    cbar_kws={"label": "% of reviews (dominant emotion)"},
    vmin=0, vmax=heatmap_pct.values.max() if not heatmap_pct.empty else 100,
)
ax.set_title(
    f"Product Aspect x Emotion Heatmap  |  Product: {df['product_id'].iloc[0]}\n"
    f"(% of reviews per aspect where that emotion dominates — 3 actionable signals)",
    fontsize=13, pad=15,
)
ax.set_xlabel("Dominant Emotion", fontsize=11)
ax.set_ylabel("Product Aspect", fontsize=11)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "day2_heatmap.png"),
    dpi=300, bbox_inches="tight",
)

# Bar Chart: Emotional Intensity (Anger + Disappointment per aspect)
aspect_emotions = (
    df_topics.groupby("topic_label")
             .agg(
                 mean_anger        = ("emo_anger",   "mean"),
                 mean_disappointment = ("emo_sadness", "mean"),
                 review_count      = ("topic_label", "count"),
             )
             .reset_index()
)
aspect_emotions["emotional_intensity"] = (
    aspect_emotions["mean_anger"] + aspect_emotions["mean_disappointment"]
)
aspect_emotions = aspect_emotions.sort_values("emotional_intensity", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(
    aspect_emotions["topic_label"],
    aspect_emotions["emotional_intensity"],
    color="#ef4444", alpha=0.8,
)
ax.set_xlabel(
    "Emotional Intensity (mean anger + disappointment probability)",
    fontsize=11,
)
ax.set_title("Emotional Pain Per Product Aspect", fontsize=12)
ax.grid(True, alpha=0.15, axis="x")
for bar, (_, row) in zip(bars, aspect_emotions.iterrows()):
    ax.text(
        bar.get_width() + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"n={int(row['review_count'])}",
        va="center", fontsize=9, color="white",
    )
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "day2_emotional_intensity.png"),
    dpi=300, bbox_inches="tight",
)

# -- Save ----------------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nEnriched dataset saved -> {OUTPUT_CSV}")
print(f"Emotion columns written: {[f'emo_{e}' for e in ACTIVE_EMOTIONS]}")
print(f"Discarded (not written): disgust, fear, surprise, neutral")

print("\n--- TOP 5 WORDS PER ASPECT ---")
topic_info = bertopic_model.get_topic_info()
for topic_id in topic_info["Topic"].tolist()[1:6]:
    words  = bertopic_model.get_topic(topic_id)
    top_5  = [word for word, score in words[:5]]
    label  = topic_labels.get(topic_id, f"Aspect {topic_id}")
    print(f"  {label}: {top_5}")

print("\nDay 2 complete. Pass data/clean/day2_aspects.csv to day3_matrix.py")
