"""
day3.6_ai_complaint_summary.py
================================
Turns raw complaint text into something a CEO can read in 10 seconds.

Reads the negative/frustrated reviews (low sentiment, low stars, or
flagged Silent Killers) for the flagship product, groups them by aspect,
and runs each group through an abstractive summarization model
(sshleifer/distilbart-cnn-12-6) to produce a short natural-language
summary of what customers are actually complaining about — instead of
making an executive read raw quotes or a topic-frequency table.

RUN:    python day3.6_ai_complaint_summary.py
INPUT:  data/clean/day2_aspects.csv
OUTPUT: data/clean/ai_complaint_summary.json
        {
          "overall": "...",
          "by_aspect": {"Battery & Charging": "...", ...},
          "n_complaints_summarized": 123
        }
"""
import os
import json
import warnings

os.environ["USE_TF"] = "0"
warnings.filterwarnings("ignore")

import pandas as pd

DATA_DIR   = "data/clean"
INPUT_CSV  = os.path.join(DATA_DIR, "day2_aspects.csv")
OUT_JSON   = os.path.join(DATA_DIR, "ai_complaint_summary.json")

MODEL_NAME       = "sshleifer/distilbart-cnn-12-6"
MAX_INPUT_CHARS  = 3000     # keep well under the model's 1024-token limit
MAX_REVIEWS_PER_GROUP = 40  # cap how many complaint snippets feed one summary
MIN_REVIEWS_FOR_SUMMARY = 3

print("=" * 60)
print("DAY 3.6: AI COMPLAINT SUMMARY")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)
df = df[df["topic_id"] != -1].copy()
df["review_text"] = df["review_text"].astype(str)

# "Raw complaints" = the frustrated end of the spectrum: negative sentiment,
# low stars, or a Silent Killer (high stars but hidden frustration) — this
# mirrors the same definition used everywhere else in the pipeline rather
# than inventing a new one.
is_complaint = (
    (df["roberta_score"] < -0.1) |
    (df["star_rating"] <= 2) |
    (df.get("is_silent_killer", False) == True)
)
complaints = df[is_complaint].copy()
print(f"Loaded {len(df):,} reviews -> {len(complaints):,} flagged as complaints")

if complaints.empty:
    print("[SKIP] No complaint reviews found.")
    with open(OUT_JSON, "w") as f:
        json.dump({"overall": "", "by_aspect": {}, "n_complaints_summarized": 0}, f, indent=2)
    raise SystemExit(0)

print(f"\nLoading model: {MODEL_NAME}")
from transformers import pipeline
import torch

device_id = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model=MODEL_NAME, device=device_id)


def build_corpus(texts, max_chars=MAX_INPUT_CHARS, max_reviews=MAX_REVIEWS_PER_GROUP, seed=42):
    """Sample complaint texts up to a char budget.

    A random sample (not "the N longest") so the summary represents the
    complaint set broadly rather than getting dominated by whichever one
    or two reviews happen to be the most verbose outliers. Reviews under
    20 words are filtered out first — too short to carry a real complaint
    ("meh." isn't useful context for a summarizer).
    """
    import random
    substantive = [t for t in texts if len(str(t).split()) >= 20]
    pool = substantive if substantive else list(texts)
    rng = random.Random(seed)
    ordered = rng.sample(pool, min(len(pool), max_reviews))
    out, total = [], 0
    for t in ordered:
        t = t.strip().replace("\n", " ")
        if not t:
            continue
        if total + len(t) > max_chars:
            continue
        out.append(t)
        total += len(t)
    if not out and ordered:
        # Every candidate was individually too long — truncate the
        # shortest one to fit rather than return nothing.
        shortest = min((t.strip().replace("\n", " ") for t in ordered if t.strip()), key=len, default="")
        return shortest[:max_chars]
    return " ".join(out)


def summarize(corpus):
    if not corpus or len(corpus.split()) < 15:
        return None
    try:
        result = summarizer(
            corpus, max_length=80, min_length=20, do_sample=False, truncation=True
        )
        return result[0]["summary_text"].strip()
    except Exception as e:
        print(f"  [WARN] Summarization failed: {e}")
        return None


# -- Overall summary --------------------------------------------------
print("\nSummarizing overall complaints...")
overall_corpus = build_corpus(complaints["review_text"].tolist(), max_reviews=60)
overall_summary = summarize(overall_corpus)
print(f"  -> {overall_summary}")

# -- Per-aspect summaries ----------------------------------------------
by_aspect = {}
print("\nSummarizing per-aspect complaints...")
for aspect, group in complaints.groupby("topic_label"):
    if len(group) < MIN_REVIEWS_FOR_SUMMARY:
        continue
    corpus = build_corpus(group["review_text"].tolist())
    summary = summarize(corpus)
    if summary:
        by_aspect[aspect] = summary
        print(f"  [{aspect}] ({len(group)} complaints) -> {summary}")

result = {
    "model": MODEL_NAME,
    "overall": overall_summary or "",
    "by_aspect": by_aspect,
    "n_complaints_summarized": int(len(complaints)),
}
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved -> {OUT_JSON}")
print("DAY 3.6 COMPLETE")
