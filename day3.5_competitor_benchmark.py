"""
day3.5_competitor_benchmark.py
===============================
Benchmarks the flagship product against its direct competitors (same
category, different brand — selected by setup_scripts/stream_data.py) on
the metric that actually matters for this mission: sentiment, not just
stars. A competitor with the same 4.1-star average could be running a
much healthier (or much worse) real-sentiment score.

Uses the SAME multilingual RoBERTa model as Day 1, scored directly on
each competitor's raw review text — no translation needed for sentiment
(per the Day 1 model justification: cardiffnlp/twitter-xlm-roberta-base
is cross-lingual by design). The Silent Killer threshold is recomputed
on the pooled flagship + competitor confirmed-negative (1-2 star)
distribution, so every product is held to the exact same bar — the
point of a benchmark is a shared ruler, not each product grading its
own homework.

RUN:    python day3.5_competitor_benchmark.py
INPUT:  data/clean/competitor_reviews.csv    (from stream_data.py)
        data/clean/competitor_products.csv   (from stream_data.py)
        data/clean/day1_reviews_roberta.csv  (flagship, already scored)
OUTPUT: data/clean/competitor_comparison.csv
        data/clean/competitor_summary.json
        figures/competitor_benchmark.png
"""
import os
import sys
import json
import warnings

os.environ["USE_TF"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data/clean"
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

COMPETITOR_REVIEWS_CSV  = os.path.join(DATA_DIR, "competitor_reviews.csv")
COMPETITOR_PRODUCTS_CSV = os.path.join(DATA_DIR, "competitor_products.csv")
FLAGSHIP_SCORED_CSV     = os.path.join(DATA_DIR, "day1_reviews_roberta.csv")
OUT_CSV                 = os.path.join(DATA_DIR, "competitor_comparison.csv")
OUT_JSON                = os.path.join(DATA_DIR, "competitor_summary.json")
OUT_FIG                 = os.path.join(FIG_DIR, "competitor_benchmark.png")

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE = 16

print("=" * 60)
print("DAY 3.5: COMPETITOR BENCHMARK")
print("=" * 60)

if not (os.path.exists(COMPETITOR_REVIEWS_CSV) and os.path.exists(COMPETITOR_PRODUCTS_CSV)):
    print(f"[SKIP] {COMPETITOR_REVIEWS_CSV} or {COMPETITOR_PRODUCTS_CSV} not found.")
    print("       Run setup_scripts/stream_data.py first — it discovers competitor products.")
    sys.exit(0)

if not os.path.exists(FLAGSHIP_SCORED_CSV):
    print(f"[SKIP] {FLAGSHIP_SCORED_CSV} not found. Run day1_roberta_sentiment.py first.")
    sys.exit(0)

products_df = pd.read_csv(COMPETITOR_PRODUCTS_CSV)
comp_df     = pd.read_csv(COMPETITOR_REVIEWS_CSV)
flagship_df = pd.read_csv(FLAGSHIP_SCORED_CSV)

if comp_df.empty:
    print("[SKIP] No competitor reviews collected. Nothing to benchmark.")
    sys.exit(0)

comp_df["review_text"] = comp_df["review_text"].astype(str)
print(f"Loaded {len(comp_df):,} competitor reviews across "
      f"{comp_df['product_id'].nunique()} products")

# -- Score competitor reviews with the same multilingual model ----------
print(f"\nLoading model: {MODEL_NAME}")
import torch
from transformers import pipeline

device_id = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    task="text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME,
    top_k=None, truncation=True, max_length=512, device=device_id,
)
label_map = classifier.model.config.id2label
neg_key = next(l for l in label_map.values() if "neg" in l.lower())
pos_key = next(l for l in label_map.values() if "pos" in l.lower())

print(f"Scoring {len(comp_df):,} competitor reviews (batch_size={BATCH_SIZE})...")
raw_outputs = classifier(comp_df["review_text"].tolist(), batch_size=BATCH_SIZE,
                          truncation=True, max_length=512)
scores = [{"prob_negative": {i["label"]: i["score"] for i in r}[neg_key],
           "prob_positive": {i["label"]: i["score"] for i in r}[pos_key]} for r in raw_outputs]
scores_df = pd.DataFrame(scores, index=comp_df.index)
comp_df = pd.concat([comp_df, scores_df], axis=1)
comp_df["roberta_score"] = comp_df["prob_positive"] - comp_df["prob_negative"]

# -- Shared Silent Killer threshold across flagship + competitors -------
pooled_negative = pd.concat([
    flagship_df.loc[flagship_df["star_rating"] <= 2, "prob_negative"],
    comp_df.loc[comp_df["star_rating"] <= 2, "prob_negative"],
])
SK_THRESHOLD = float(np.percentile(pooled_negative, 10)) if len(pooled_negative) >= 10 else 0.5
print(f"\nShared Silent Killer threshold (pooled p10, same rule as Day 1): {SK_THRESHOLD:.3f}")

comp_df["is_silent_killer"] = (
    (comp_df["star_rating"] >= 4) &
    (comp_df["prob_negative"] > comp_df["prob_positive"]) &
    (comp_df["prob_negative"] >= SK_THRESHOLD)
)

# -- Per-product summary --------------------------------------------------
def summarize(g):
    return pd.Series({
        "n_reviews": len(g),
        "avg_star": g["star_rating"].mean(),
        "avg_sentiment": g["roberta_score"].mean(),
        "silent_killer_rate_pct": g["is_silent_killer"].mean() * 100,
    })

comp_summary = comp_df.groupby("product_id").apply(summarize).reset_index()
comp_summary = comp_summary.merge(
    products_df[["product_id", "brand", "title", "category"]], on="product_id", how="left"
)
comp_summary["is_flagship"] = False

flagship_sk = (
    (flagship_df["star_rating"] >= 4) &
    (flagship_df["prob_negative"] > flagship_df["prob_positive"]) &
    (flagship_df["prob_negative"] >= SK_THRESHOLD)
)
flagship_row = pd.DataFrame([{
    "product_id": flagship_df["product_id"].iloc[0] if "product_id" in flagship_df.columns else "flagship",
    "n_reviews": len(flagship_df),
    "avg_star": flagship_df["star_rating"].mean(),
    "avg_sentiment": flagship_df["roberta_score"].mean(),
    "silent_killer_rate_pct": flagship_sk.mean() * 100,
    "brand": products_df.loc[products_df["is_flagship"] == True, "brand"].iloc[0]
             if (products_df["is_flagship"] == True).any() else "Flagship",
    "title": products_df.loc[products_df["is_flagship"] == True, "title"].iloc[0]
             if (products_df["is_flagship"] == True).any() else "Flagship Product",
    "category": products_df["category"].iloc[0] if not products_df.empty else "Electronics",
    "is_flagship": True,
}])

comparison = pd.concat([flagship_row, comp_summary], ignore_index=True)
comparison = comparison.sort_values("is_flagship", ascending=False).reset_index(drop=True)
comparison.to_csv(OUT_CSV, index=False)

print("\nBenchmark table:")
print(comparison[["brand", "n_reviews", "avg_star", "avg_sentiment",
                   "silent_killer_rate_pct", "is_flagship"]].to_string(index=False))

# -- Headline insight -----------------------------------------------------
flagship_sent = flagship_row["avg_sentiment"].iloc[0]
rival_sent    = comp_summary["avg_sentiment"].mean() if not comp_summary.empty else None
flagship_star = flagship_row["avg_star"].iloc[0]
rival_star    = comp_summary["avg_star"].mean() if not comp_summary.empty else None

insight = None
if rival_sent is not None:
    sent_gap = flagship_sent - rival_sent
    star_gap = flagship_star - rival_star if rival_star is not None else None
    direction = "trails" if sent_gap < 0 else "leads"
    insight = (
        f"The flagship {direction} competitors on real sentiment by {abs(sent_gap):.3f} "
        f"points (RoBERTa score), despite a star-rating gap of only {star_gap:+.2f}. "
        f"Stars alone would not have surfaced this competitive gap."
    )
    print(f"\n[INSIGHT] {insight}")

summary_json = {
    "sk_threshold_shared": round(SK_THRESHOLD, 4),
    "n_competitors": int(comp_summary.shape[0]),
    "flagship_avg_sentiment": round(float(flagship_sent), 4),
    "competitor_avg_sentiment": round(float(rival_sent), 4) if rival_sent is not None else None,
    "flagship_avg_star": round(float(flagship_star), 3),
    "competitor_avg_star": round(float(rival_star), 3) if rival_star is not None else None,
    "insight": insight,
}
with open(OUT_JSON, "w") as f:
    json.dump(summary_json, f, indent=2)
print(f"\nSaved -> {OUT_JSON}")

# -- Chart: star rating vs real sentiment, flagship highlighted ---------
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

plot_df = comparison.copy()
plot_df["label"] = plot_df["brand"].fillna("Unknown") + plot_df["is_flagship"].map(
    {True: " (Flagship)", False: ""}
)
colors = ["#FF4C4C" if f else "#4C9EFF" for f in plot_df["is_flagship"]]

axes[0].barh(plot_df["label"], plot_df["avg_star"], color=colors)
axes[0].set_xlabel("Average Star Rating")
axes[0].set_title("Star Rating (what management sees)")
axes[0].set_xlim(0, 5)

axes[1].barh(plot_df["label"], plot_df["avg_sentiment"], color=colors)
axes[1].set_xlabel("Average Sentiment Score (-1 to +1)")
axes[1].set_title("Real Sentiment (what customers feel)")
axes[1].set_xlim(-1, 1)
axes[1].axvline(0, color="white", lw=0.8, alpha=0.4)

fig.suptitle("Flagship vs. Direct Competitors: Stars Hide What Sentiment Reveals", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
print(f"Chart saved -> {OUT_FIG}")

print("\nDAY 3.5 COMPLETE")
