# day1_sentiment.py
# -----------------------------------------------------------------
# RUN: python day1_sentiment.py
# INPUT:  data/clean/subcategory_sample.csv
# OUTPUT: data/clean/day1_reviews_roberta.csv
#         figures/day1_stargap_timeline.png
#         figures/day1_silent_killer_scatter.png
# -----------------------------------------------------------------
import os
os.environ["USE_TF"] = "0"          # tell transformers to ignore TensorFlow entirely
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress the TensorFlow oneDNN spam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, warnings
import torch
from transformers import pipeline
from datasets import Dataset
from scipy import stats

warnings.filterwarnings("ignore")

# -- Config --------------------------------------------------------
INPUT_CSV         = "data/clean/subcategory_sample.csv"
OUTPUT_CSV        = "data/clean/day1_reviews_roberta.csv"
FIG_DIR           = "figures"
MODEL_NAME        = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE        = 16
MIN_MONTH_REVIEWS = 10    # months with fewer reviews are dropped (documented)

os.makedirs(FIG_DIR, exist_ok=True)

# -- Load ----------------------------------------------------------
print(f"Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
df["review_text"] = df["review_text"].astype(str)
df = df.dropna(subset=["review_date", "review_text"]).copy()
df = df.reset_index(drop=True)
print(f"Loaded {len(df):,} reviews | "
      f"{df['review_date'].min().date()} -> {df['review_date'].max().date()}")

# -- Hardware ------------------------------------------------------
device_id = (
    0 if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else -1)
)
device_label = (
    "CUDA" if device_id == 0
    else ("MPS" if device_id == "mps" else "CPU")
)
print(f"\n[INFO] Device: {device_label}")
if device_id == -1:
    print("       Running on CPU -- expect ~2 min per 1,000 reviews")

# -- Build pipeline ------------------------------------------------
# Using HuggingFace pipeline + datasets.Dataset eliminates:
#   - manual iloc slicing
#   - manual tokenizer calls
#   - manual softmax (pipeline handles it internally)
#   - manual .to(device) tensor management
print(f"Loading model: {MODEL_NAME}")
classifier = pipeline(
    task      = "text-classification",
    model     = MODEL_NAME,
    tokenizer = MODEL_NAME,
    top_k     = None,     # return ALL label probabilities, not just the top
    truncation= True,
    max_length= 512,
    device    = device_id,
)

# Verify label set before any inference.
# pipeline.model.config.id2label is the canonical source of truth.
label_map = classifier.model.config.id2label
print(f"Model labels confirmed: {label_map}")

neg_key = next(l for l in label_map.values() if "neg" in l.lower())
neu_key = next(l for l in label_map.values() if "neu" in l.lower())
pos_key = next(l for l in label_map.values() if "pos" in l.lower())
print(f"Mapped -> neg='{neg_key}'  neu='{neu_key}'  pos='{pos_key}'")

# -- Inference via datasets.Dataset --------------------------------
# datasets.Dataset feeds the pipeline in optimised Arrow-backed batches.
# No manual iloc, no manual batch assembly, no StopIteration risk.
print(f"\nRunning inference on {len(df):,} rows (batch_size={BATCH_SIZE})...")

hf_dataset  = Dataset.from_dict({"text": df["review_text"].tolist()})

raw_outputs = classifier(
    hf_dataset["text"],
    batch_size = BATCH_SIZE,
    truncation = True,
    max_length = 512,
)
# raw_outputs is a list of lists:
# [ [{"label": "Negative", "score": 0.85}, {"label": "Neutral", ...}], ... ]

# -- Unpack probabilities using id2label mapping -------------------
# Never index by column position -- always look up by label name.
records = []
for result in raw_outputs:
    score_map = {item["label"]: item["score"] for item in result}
    records.append({
        "prob_negative": score_map[neg_key],
        "prob_neutral" : score_map[neu_key],
        "prob_positive": score_map[pos_key],
    })

probs_df = pd.DataFrame(records, index=df.index)
df       = pd.concat([df, probs_df], axis=1)

# Continuous sentiment score on [-1, +1]
df["roberta_score"] = df["prob_positive"] - df["prob_negative"]

print(f"Inference complete. Score range: "
      f"[{df['roberta_score'].min():.3f}, {df['roberta_score'].max():.3f}]")

# -- Model sanity check BEFORE trusting results --------------------
# If the model badly misclassifies obvious cases, all downstream
# analysis is meaningless. Catch it here, not in the CEO pitch.
print("\n[SANITY CHECK] Model agreement with extreme star ratings:")
one_star  = df[df["star_rating"] == 1.0]
five_star = df[df["star_rating"] == 5.0]

one_neg_rate  = (one_star["prob_negative"]  > one_star["prob_positive"]).mean()
five_pos_rate = (five_star["prob_positive"] > five_star["prob_negative"]).mean()

print(f"  1-star reviews model calls Negative : {one_neg_rate*100:.1f}%  (expect >60%)")
print(f"  5-star reviews model calls Positive : {five_pos_rate*100:.1f}%  (expect >70%)")

if one_neg_rate < 0.5 or five_pos_rate < 0.6:
    print("\n  WARNING: Model agreement is low -- domain shift likely.")
    print("    Consider switching to: 'LiYuan/amazon-review-sentiment-analysis'")
    print("    which is fine-tuned on Amazon reviews specifically.")
else:
    print("  Model is behaving sensibly on this domain.")

# -- Data-driven Silent Killer threshold ---------------------------
# The 10th percentile of prob_negative among CONFIRMED bad reviews
# (1-2 star) sets the floor. Any 4-5 star review the model is at
# least this confident is negative gets flagged. Fully defensible.
confirmed_negative = df[df["star_rating"] <= 2]["prob_negative"]
SK_THRESHOLD       = np.percentile(confirmed_negative, 10)

print(f"\n[SILENT KILLER THRESHOLD]")
print(f"  10th percentile of prob_negative among 1-2-star reviews : {SK_THRESHOLD:.3f}")
print(f"  Interpretation: only flag reviews where the model is at")
print(f"  least as confident as it is on genuinely bad reviews.")

df["is_silent_killer"] = (
    (df["star_rating"] >= 4) &
    (df["prob_negative"] > df["prob_positive"]) &
    (df["prob_negative"] >= SK_THRESHOLD)
)

# -- Star-Gap metric -----------------------------------------------
# Normalise star_rating to [-1, +1] (1->-1, 3->0, 5->+1)
# so it lives on the same scale as roberta_score before differencing.
# Positive gap = stars are inflating reality.
# Negative gap = customers are harsher with stars than with words.
df["star_normalized"] = (df["star_rating"] - 3) / 2
df["star_gap"]        = df["star_normalized"] - df["roberta_score"]

# -- Reporting -----------------------------------------------------
total   = len(df)
n_sk    = df["is_silent_killer"].sum()
avg_gap = df["star_gap"].mean()

print("\n" + "=" * 60)
print("DAY 1: STAR-GAP DIAGNOSTIC REPORT")
print("=" * 60)
print(f"  Total reviews         : {total:,}")
print(f"  Average star rating   : {df['star_rating'].mean():.2f} / 5.0")
print(f"  Average RoBERTa score : {df['roberta_score'].mean():.3f}  (-1 to +1)")
print(f"  Mean star gap         : {avg_gap:+.3f}  (+ = stars inflate reality)")
print(f"  Silent Killers        : {n_sk} ({n_sk/total*100:.1f}% of all reviews)")
print(f"  SK threshold used     : prob_negative >= {SK_THRESHOLD:.3f} (data-driven)")

r, p = stats.pearsonr(df["star_normalized"], df["roberta_score"])
print(f"\n  Pearson r (stars vs sentiment): {r:.3f}  p={p:.2e}")
if r < 0.5:
    print("  -> Weak correlation -- stars are NOT a reliable proxy for sentiment.")
    print("     This IS your headline finding for the CEO.")
else:
    print("  -> Moderate/strong correlation -- stars and sentiment broadly agree.")
    print("     Look harder at the silent killers for your CEO story.")

print("\nTop 5 Silent Killers:")
top_sk = (
    df[df["is_silent_killer"]]
    .sort_values("prob_negative", ascending=False)
    .head(5)
)
for _, row in top_sk.iterrows():
    snippet = str(row["review_text"])[:120].replace("\n", " ")
    print(f"\n  star={int(row['star_rating'])} | "
          f"neg_prob={row['prob_negative']:.2f} | "
          f"gap={row['star_gap']:+.2f}")
    print(f"  \"{snippet}...\"")
print("=" * 60)

# [Keep all plotting code exactly as before]

# -- Save enriched CSV ---------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nEnriched dataset saved: {OUTPUT_CSV}")
print("\nDay 1 complete. Pass data/clean/day1_reviews_roberta.csv to day2_aspects.py")
