"""
=============================================================
 DATATHON – DAY 1: The Diagnostic & The "Star-Gap"
 Product : Amazon Fire TV Stick (B075X8471B)
 Model   : cardiffnlp/twitter-roberta-base-sentiment-latest
=============================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import warnings, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# ── 1. Load Dataset ───────────────────────────────────────────────────────────
CSV_PATH = "/home/pranav/Datathon/datathon_target_B075X8471B.csv"

df = pd.read_csv(CSV_PATH)
df["date"]  = pd.to_datetime(df["timestamp"], unit="ms")
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.to_period("M")

# Keep only rows with meaningful text (>5 chars)
df = df[df["text"].str.len() > 5].reset_index(drop=True)

print(f"✅  Loaded {len(df):,} reviews  |  {df['date'].min().date()} → {df['date'].max().date()}")
print(f"    Rating dist:\n{df['rating'].value_counts().sort_index().to_string()}\n")


# ── 2. Text Preprocessing ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Light cleaning: strip HTML, normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", str(text))   # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# RoBERTa max token limit is 512; truncate long reviews to ~500 words
def truncate(text, max_words=100):
    """Keep the first max_words words so the tokeniser never overflows."""
    return " ".join(text.split()[:max_words])

df["input_text"] = df["clean_text"].apply(truncate)


# ── 3. Load RoBERTa Model ─────────────────────────────────────────────────────
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
print(f"⏳  Loading model: {MODEL_NAME}  (first run downloads ~500 MB) …")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

LABELS = ["negative", "neutral", "positive"]   # model output order

print("✅  Model loaded.\n")


# ── 4. Batch Inference ────────────────────────────────────────────────────────
BATCH_SIZE = 32

def predict_batch(texts):
    """Return (neg, neu, pos) probability arrays for a list of texts."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
    probs = softmax(logits.numpy(), axis=1)
    return probs   # shape (N, 3)

all_probs = []
n = len(df)
for start in range(0, n, BATCH_SIZE):
    batch = df["input_text"].iloc[start : start + BATCH_SIZE].tolist()
    all_probs.append(predict_batch(batch))
    if (start // BATCH_SIZE) % 5 == 0:
        pct = min(start + BATCH_SIZE, n) / n * 100
        print(f"   …inference {min(start+BATCH_SIZE, n)}/{n}  ({pct:.0f}%)")

probs_array = np.vstack(all_probs)   # (N, 3)

df["prob_neg"] = probs_array[:, 0]
df["prob_neu"] = probs_array[:, 1]
df["prob_pos"] = probs_array[:, 2]

# Compound score in [-1, +1]:  pos − neg  (same intuition as VADER compound)
df["sentiment_score"] = df["prob_pos"] - df["prob_neg"]

# Dominant label
df["sentiment_label"] = [LABELS[i] for i in probs_array.argmax(axis=1)]

print(f"\n✅  Inference complete.")
print(df[["rating","sentiment_score","sentiment_label"]].describe().round(3))


# ── 5. "Silent Killer" Detection ──────────────────────────────────────────────
#   Definition: star ≥ 4  AND  sentiment_score < 0.2  (text is NOT clearly positive)

df["is_silent_killer"] = (df["rating"] >= 4.0) & (df["sentiment_score"] < 0.2)

high_star      = df[df["rating"] >= 4.0]
silent_killers = df[df["is_silent_killer"]]

pct_sk = len(silent_killers) / len(high_star) * 100
print(f"\n🔴 Silent Killers  :  {len(silent_killers)} reviews")
print(f"   Out of {len(high_star)} high-star (≥4★) reviews → {pct_sk:.1f}% are misleadingly positive")


# ── 6. Save Enriched CSV ──────────────────────────────────────────────────────
OUT_CSV = "day1_reviews_with_sentiment.csv"
df.to_csv(OUT_CSV, index=False)
print(f"\n💾  Enriched CSV saved → {OUT_CSV}")


# ── 7. Visualisations ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"  : "DejaVu Sans",
    "figure.facecolor": "#0f1117",
    "axes.facecolor"  : "#1a1d27",
    "axes.edgecolor"  : "#3a3d4d",
    "axes.labelcolor" : "#e0e0e0",
    "xtick.color"     : "#a0a0b0",
    "ytick.color"     : "#a0a0b0",
    "text.color"      : "#e0e0e0",
    "grid.color"      : "#2a2d3d",
    "grid.linestyle"  : "--",
    "grid.linewidth"  : 0.5,
})

ACCENT_POS  = "#4ade80"   # green
ACCENT_NEU  = "#facc15"   # yellow
ACCENT_NEG  = "#f87171"   # red
ACCENT_SK   = "#fb923c"   # orange (silent killers)
ACCENT_BLUE = "#60a5fa"   # blue

fig = plt.figure(figsize=(22, 18))
fig.suptitle(
    "PRODUCT RESCUE MISSION  ·  Day 1 Diagnostic\nAmazon Fire TV Stick — Sentiment vs Star-Rating Analysis",
    fontsize=16, fontweight="bold", y=0.98, color="white",
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: Star vs Sentiment scatter ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])

colors = df["sentiment_label"].map(
    {"positive": ACCENT_POS, "neutral": ACCENT_NEU, "negative": ACCENT_NEG}
)
sc = ax1.scatter(
    df["rating"] + np.random.uniform(-0.15, 0.15, len(df)),   # jitter
    df["sentiment_score"],
    c=colors, alpha=0.65, s=18, linewidths=0,
)
ax1.axhline(0.2,  color=ACCENT_SK,  linewidth=1.4, linestyle="--", label="Silent-Killer threshold (0.2)")
ax1.axhline(0.0,  color="#ffffff30", linewidth=0.8, linestyle=":")
ax1.axvline(3.5,  color="#ffffff30", linewidth=0.8, linestyle=":")

# Shade the "Silent Killer" zone
ax1.fill_betweenx([-1, 0.2], 3.5, 5.4,
                  color=ACCENT_SK, alpha=0.07, label="Silent Killer zone")

ax1.set_xlabel("Star Rating", fontsize=11)
ax1.set_ylabel("RoBERTa Sentiment Score  (pos − neg)", fontsize=11)
ax1.set_title("Star Rating  vs  Text Sentiment Score", fontsize=12, pad=8)
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_xlim(0.5, 5.8)
ax1.set_ylim(-1.05, 1.05)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True)

# Annotation
ax1.annotate(
    f"⚠  {pct_sk:.1f}% of ≥4★ reviews\nare Silent Killers",
    xy=(4.5, 0.1), fontsize=9, color=ACCENT_SK,
    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1d27", ec=ACCENT_SK, lw=1),
)


# ── Panel 2: KPI cards ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")

kpi_data = [
    ("Total Reviews",          f"{len(df)}",          ACCENT_BLUE),
    ("Avg Star Rating",        f"{df['rating'].mean():.2f} ★", ACCENT_NEU),
    ("Avg Sentiment Score",    f"{df['sentiment_score'].mean():.3f}", ACCENT_POS),
    ("Silent Killers",         f"{len(silent_killers)} ({pct_sk:.1f}%)", ACCENT_SK),
    ("Clearly Negative Text",  f"{(df['sentiment_label']=='negative').sum()}", ACCENT_NEG),
]

for i, (label, value, color) in enumerate(kpi_data):
    y = 0.88 - i * 0.18
    rect = FancyBboxPatch((0.02, y - 0.06), 0.96, 0.14,
                          boxstyle="round,pad=0.01",
                          linewidth=1.2, edgecolor=color,
                          facecolor=color + "18",
                          transform=ax2.transAxes, clip_on=False)
    ax2.add_patch(rect)
    ax2.text(0.08, y + 0.02, label,  transform=ax2.transAxes,
             fontsize=9,  color="#b0b0c0")
    ax2.text(0.92, y + 0.02, value,  transform=ax2.transAxes,
             fontsize=12, color=color, ha="right", fontweight="bold")

ax2.set_title("Key Metrics", fontsize=12, pad=8)


# ── Panel 3: Monthly avg star vs avg sentiment (timeline) ────────────────────
ax3 = fig.add_subplot(gs[1, :])

monthly = (
    df.groupby("month")
    .agg(avg_star=("rating", "mean"), avg_sent=("sentiment_score", "mean"),
         n=("rating", "count"))
    .reset_index()
)
monthly["month_dt"] = monthly["month"].dt.to_timestamp()

# Smooth
smooth_star = uniform_filter1d(monthly["avg_star"],  size=3)
smooth_sent = uniform_filter1d(monthly["avg_sent"],  size=3)
smooth_sent_scaled = smooth_sent * 2.5 + 2.5   # rescale [-1,1] → [0,5] for overlay

ax3.fill_between(monthly["month_dt"], smooth_star, smooth_sent_scaled,
                 where=(smooth_star > smooth_sent_scaled),
                 alpha=0.25, color=ACCENT_SK, label="Star > Sentiment (potential gap)")

ax3_r = ax3.twinx()
ax3.plot(monthly["month_dt"], smooth_star,       color=ACCENT_NEU, linewidth=2.2, label="Avg Star Rating (smoothed)")
ax3_r.plot(monthly["month_dt"], smooth_sent,     color=ACCENT_POS, linewidth=2.2, label="Avg Sentiment Score (smoothed)", linestyle="--")

# Mark the point where sentiment starts to dip below 0
dip_months = monthly[monthly["avg_sent"] < 0.2]["month_dt"]
if not dip_months.empty:
    ax3.axvline(dip_months.iloc[0], color=ACCENT_NEG, linewidth=1.5, linestyle=":",
                label=f"First dip below threshold ({dip_months.iloc[0].strftime('%b %Y')})")

ax3.set_xlabel("Date", fontsize=11)
ax3.set_ylabel("Avg Star Rating", fontsize=11, color=ACCENT_NEU)
ax3_r.set_ylabel("Avg Sentiment Score", fontsize=11, color=ACCENT_POS)
ax3.set_title("Timeline: When Did Sentiment Start to Sour?", fontsize=12, pad=8)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax3.set_ylim(1, 5.5)
ax3_r.set_ylim(-1.1, 1.1)
ax3.grid(True)

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_r.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower left")

ax3_r.tick_params(colors=ACCENT_POS)
ax3.tick_params(axis="y", colors=ACCENT_NEU)


# ── Panel 4: Sentiment label distribution ────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])

label_counts = df["sentiment_label"].value_counts()[["positive","neutral","negative"]]
bars = ax4.bar(
    label_counts.index,
    label_counts.values,
    color=[ACCENT_POS, ACCENT_NEU, ACCENT_NEG],
    width=0.5, edgecolor="#ffffff20",
)
for bar, val in zip(bars, label_counts.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax4.set_title("Sentiment Label Distribution", fontsize=11, pad=8)
ax4.set_ylabel("Review Count", fontsize=10)
ax4.grid(True, axis="y")


# ── Panel 5: Silent Killer breakdown ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])

sk_by_rating = df[df["is_silent_killer"]]["rating"].value_counts().sort_index()
ax5.bar(
    sk_by_rating.index.astype(str),
    sk_by_rating.values,
    color=ACCENT_SK, width=0.5, edgecolor="#ffffff20",
)
for i, val in enumerate(sk_by_rating.values):
    ax5.text(i, val + 0.3, str(val), ha="center", fontsize=10, fontweight="bold", color=ACCENT_SK)
ax5.set_title("Silent Killers by Star Rating", fontsize=11, pad=8)
ax5.set_xlabel("Star Rating", fontsize=10)
ax5.set_ylabel("Count", fontsize=10)
ax5.grid(True, axis="y")


# ── Panel 6: Sentiment score distribution ─────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])

ax6.hist(df[df["rating"] >= 4]["sentiment_score"], bins=30,
         color=ACCENT_BLUE, alpha=0.75, label="★ ≥ 4 reviews", edgecolor="#ffffff10")
ax6.hist(df[df["is_silent_killer"]]["sentiment_score"], bins=15,
         color=ACCENT_SK,  alpha=0.85, label="Silent Killers", edgecolor="#ffffff10")
ax6.axvline(0.2, color=ACCENT_SK, linewidth=1.5, linestyle="--", label="Threshold 0.2")
ax6.set_title("Sentiment Score: High-Star Reviews", fontsize=11, pad=8)
ax6.set_xlabel("Sentiment Score", fontsize=10)
ax6.set_ylabel("Count", fontsize=10)
ax6.legend(fontsize=8)
ax6.grid(True, axis="y")
plt.savefig("day1_diagnostic_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\n📊  Dashboard saved → day1_diagnostic_dashboard.png")

plt.close()


# ── 8. Print Top Silent Killers ───────────────────────────────────────────────
print("\n" + "="*70)
print("TOP 10 SILENT KILLERS  (High Star ≥ 4, Sentiment Score < 0.2)")
print("="*70)
top_sk = (
    silent_killers
    .sort_values("sentiment_score")
    [["date","rating","sentiment_score","prob_neg","clean_text"]]
    .head(10)
)
for _, row in top_sk.iterrows():
    print(f"\n  [{row['date'].date()}]  ★{row['rating']:.0f}  "
          f"Sentiment={row['sentiment_score']:+.3f}  neg_prob={row['prob_neg']:.2f}")
    snippet = row["clean_text"][:180].replace("\n", " ")
    print(f"  \"{snippet}...\"")

print("\n✅  Day 1 complete. Outputs saved in the current directory.")
