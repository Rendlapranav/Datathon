import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, re, torch
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

# PRUNED EMOTIONS: Focused on hardware actionable signals
ACTIVE_EMOTIONS  = ["anger", "sadness", "joy"]
DISPLAY_LABELS   = {
    "anger"  : "Anger",
    "sadness": "Disappointment", 
    "joy"    : "Joy",
}

BATCH_SIZE     = 16
N_TOPICS       = 5
MIN_TOPIC_SIZE = 20

os.makedirs(FIG_DIR, exist_ok=True)

# -- Load & Clean --------------------------------------------------
df = pd.read_csv(INPUT_CSV)
df["review_text"] = df["review_text"].astype(str)

def clean_text(text):
    text = text.replace("\xa0", " ").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df["clean_text"] = df["review_text"].apply(clean_text)
df = df[df["clean_text"].apply(lambda x: len(x.split()) >= 5)].reset_index(drop=True)

# -- Step 1: Aspect Mining (BERTopic) -----------------------------
print("\n[STEP 1] Running BERTopic...")
vectorizer   = CountVectorizer(stop_words="english", ngram_range=(1, 2))
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

bertopic_model = BERTopic(
    vectorizer_model = vectorizer,
    ctfidf_model    = ctfidf_model,
    nr_topics       = N_TOPICS,
    min_topic_size  = MIN_TOPIC_SIZE,
    calculate_probabilities = True
)

topics, probs = bertopic_model.fit_transform(df["clean_text"].tolist())
df["topic_id"] = topics

# -- Labeling Logic ------------------------------------------------
LABEL_RULES = [
    ("Battery & Charging", ["battery", "charge", "charging", "life", "power", "dead"]),
    ("Sound Quality",      ["sound", "bass", "audio", "quality", "noise", "volume"]),
    ("Comfort & Fit",      ["comfort", "fit", "ear", "wear", "pain", "tight"]),
    ("Connectivity",       ["connect", "bluetooth", "pair", "wireless", "drop"]),
    ("Build Quality",      ["build", "durable", "material", "plastic", "broke"])
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
    return (best_label if best_label else f"Aspect {topic_id + 1}"), best_score

topic_labels = {}
for topic_id in sorted(df["topic_id"].unique()):
    if topic_id == -1:
        topic_labels[-1] = "Outlier"; continue
    label, score = assign_topic_label(topic_id, bertopic_model)
    topic_labels[topic_id] = label
    print(f"  Topic {topic_id} auto-labelled as: {label} (Score: {score})")

# ==========================================
# EMERGENCY OVERRIDE: Fix "Aspect N" names
# ==========================================
MANUAL_OVERRIDES = {
    # Example: "Aspect 2": "Microphone Quality"
}

for tid, current_label in topic_labels.items():
    if current_label in MANUAL_OVERRIDES:
        topic_labels[tid] = MANUAL_OVERRIDES[current_label]
    elif current_label.startswith("Aspect "):
        print(f"  ⚠️ WARNING: Topic {tid} is unlabelled ({current_label}). Update MANUAL_OVERRIDES!")

df["topic_label"] = df["topic_id"].map(topic_labels)
# -- Representative Reviews Extraction ----------------------------
print("\n[REPRESENTATIVE REVIEWS] (3 most central per topic)")
rep_docs = bertopic_model.get_representative_docs()
representative_records = []
for topic_id, docs in rep_docs.items():
    if topic_id == -1:
        continue
    label = topic_labels.get(topic_id, f"Aspect {topic_id}")
    for i, doc in enumerate(docs[:3]):
        representative_records.append({
            "topic_id"   : topic_id,
            "topic_label": label,
            "quote_rank" : i + 1,
            "quote_text" : doc,
        })

quotes_df   = pd.DataFrame(representative_records)
quotes_path = "data/clean/day2_representative_quotes.csv"
quotes_df.to_csv(quotes_path, index=False)
print(f"Representative quotes saved -> {quotes_path}")
# -- Step 2: Emotion Detection ------------------------------------
print(f"\n[STEP 2] Running Emotion Detection (GPU optimized)...")
device_id = 0 if torch.cuda.is_available() else -1
emo_pipeline = pipeline("text-classification", model=EMOTION_MODEL, top_k=None, device=device_id)

hf_dataset = Dataset.from_dict({"text": df["review_text"].tolist()})
raw_outputs = emo_pipeline(hf_dataset["text"], batch_size=BATCH_SIZE, truncation=True)

records = []
for result in raw_outputs:
    score_map = {item["label"]: item["score"] for item in result}
    records.append({emo: score_map[emo] for emo in ACTIVE_EMOTIONS})

emo_df = pd.DataFrame(records, index=df.index)
for emo in ACTIVE_EMOTIONS:
    df[f"emo_{emo}"] = emo_df[emo]

df["dominant_emotion"] = [ACTIVE_EMOTIONS[i] for i in emo_df.values.argmax(axis=1)]
df["dominant_emotion_display"] = df["dominant_emotion"].map(DISPLAY_LABELS)

# -- Step 3: Visualizations ---------------------------------------
plt.style.use("dark_background")
df_topics = df[df["topic_id"] != -1].copy()

# Heatmap: Urgency
heatmap_urgency = (df_topics.groupby("topic_label")[["emo_anger", "emo_sadness"]].mean() * 100)
heatmap_urgency.columns = ["Anger", "Disappointment"]
plt.figure(figsize=(7, 5))
sns.heatmap(heatmap_urgency, cmap="YlOrRd", annot=True, fmt=".1f")
plt.title("Urgency: Anger & Disappointment per Aspect")
plt.savefig(os.path.join(FIG_DIR, "day2_heatmap_urgency.png"))

# -- Save ----------------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSuccess! Dataset saved to {OUTPUT_CSV}")
