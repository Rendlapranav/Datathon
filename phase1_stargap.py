"""
Phase 1 — The Star-Gap Diagnostic
Amazon Product Review Sentiment Analysis

Requirements:
    pip install pandas vaderSentiment matplotlib

Usage:
    python phase1_stargap.py
    (place your CSV in the same folder, update CSV_PATH below)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                  # change to 'TkAgg' if you want an interactive window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH    = "datathon_target_B075X8471B.csv"   # ← update if needed
OUTPUT_PNG  = "phase1_stargap.png"
OUTPUT_CSV  = "reviews_phase1.csv"               # enriched dataset for Phase 2/3
MIN_REVIEWS = 3                                  # min reviews/month to include in chart
ROLLING_WIN = 3                                  # months for smoothing

# ── 1. LOAD & PREP ────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['date']       = pd.to_datetime(df['timestamp'], unit='ms')
df['year_month'] = df['date'].dt.to_period('M')
df['full_text']  = df['title'].fillna('') + '. ' + df['text'].fillna('')
df['norm_stars'] = (df['rating'] - 1) / 4          # normalise stars → 0–1

# ── 2. VADER SENTIMENT ────────────────────────────────────────────────────────
sia = SentimentIntensityAnalyzer()
df['vader_compound'] = df['full_text'].apply(
    lambda t: sia.polarity_scores(str(t))['compound']
)
df['sentiment_01'] = (df['vader_compound'] + 1) / 2   # compound (-1,1) → 0–1

# ── 3. GAP & REVIEW CLASSIFICATION ───────────────────────────────────────────
df['star_gap']      = df['norm_stars'] - df['sentiment_01']
df['silent_killer'] = (df['rating'] >= 4) & (df['vader_compound'] < -0.05)
df['hidden_gem']    = (df['rating'] <= 2) & (df['vader_compound'] >  0.05)

def classify(row):
    if row['silent_killer']:   return 'Silent Killer'
    if row['hidden_gem']:      return 'Hidden Gem'
    if row['star_gap'] >  0.2: return 'Overstated'
    if row['star_gap'] < -0.2: return 'Understated'
    return 'Aligned'

df['review_type'] = df.apply(classify, axis=1)
df.to_csv(OUTPUT_CSV, index=False)

# ── PRINT KEY STATS ───────────────────────────────────────────────────────────
total = len(df)
sk    = df['silent_killer'].sum()
print(f"{'─'*50}")
print(f"  Total reviews    : {total}")
print(f"  Silent Killers   : {sk} ({sk/total*100:.1f}%)")
print(f"  Avg Star Rating  : {df['rating'].mean():.2f} / 5")
print(f"  Avg Sentiment    : {df['sentiment_01'].mean()*4+1:.2f} / 5  (rescaled)")
print(f"  Mean Star-Gap    : {df['star_gap'].mean():.3f}")
print(f"\n  Review type breakdown:")
print(df['review_type'].value_counts().to_string())
print(f"{'─'*50}")

# ── 4. MONTHLY AGGREGATION ────────────────────────────────────────────────────
monthly = (
    df.groupby('year_month')
    .agg(
        avg_stars = ('rating',        'mean'),
        avg_sent  = ('sentiment_01',  'mean'),
        avg_gap   = ('star_gap',      'mean'),
        n         = ('rating',        'count'),
        n_silent  = ('silent_killer', 'sum'),
    )
    .reset_index()
)
monthly = monthly[monthly['n'] >= MIN_REVIEWS].copy()

# Smooth & rescale sentiment to 1–5 (same axis as stars)
monthly['stars_s'] = monthly['avg_stars'].rolling(ROLLING_WIN, min_periods=1).mean()
monthly['sent_s']  = (monthly['avg_sent'].rolling(ROLLING_WIN, min_periods=1).mean() * 4 + 1)
monthly['gap_s']   = monthly['avg_gap'].rolling(ROLLING_WIN, min_periods=1).mean()
monthly['xstr']    = monthly['year_month'].astype(str)

x           = list(range(len(monthly)))
xtick_step  = max(1, len(monthly) // 10)
xticks      = x[::xtick_step]
xlabels     = monthly['xstr'].iloc[::xtick_step].tolist()

# ── 5. COLOUR PALETTE ─────────────────────────────────────────────────────────
BG       = '#0f1117'
CARD     = '#1a1d27'
STAR_C   = '#f5c518'
SENT_C   = '#00c9a7'
GAP_C    = '#ff6b6b'
SILENT_C = '#ff9f43'
GRID_C   = '#2a2d3a'
TEXT_C   = '#e8eaf0'
SUB_C    = '#9da3b4'
ACCENT   = '#7c83fd'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor':  CARD,
    'text.color':       TEXT_C, 'axes.labelcolor': TEXT_C,
    'xtick.color':      SUB_C,  'ytick.color':     SUB_C,
    'axes.edgecolor':   GRID_C, 'grid.color':      GRID_C,
    'grid.alpha': 0.5,  'font.family': 'DejaVu Sans',
    'axes.titlecolor':  TEXT_C,
})

# ── 6. FIGURE LAYOUT ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 11), facecolor=BG)
fig.suptitle(
    'PHASE 1 — THE STAR-GAP DIAGNOSTIC\n'
    'Amazon Fire Stick (B075X8471B) · 500 Reviews · 2016–2021',
    fontsize=15, fontweight='bold', color=TEXT_C, y=0.98, linespacing=1.5
)

gs = gridspec.GridSpec(
    3, 1, figure=fig, hspace=0.55,
    top=0.91, bottom=0.08, left=0.07, right=0.97,
    height_ratios=[2.2, 1.2, 0.9]
)

# ── ROW 1: Stars vs Sentiment ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(CARD)
ax1.grid(True, linestyle='--', linewidth=0.5)

ax1.plot(x, monthly['stars_s'], color=STAR_C, lw=2.5, marker='o', ms=4,
         label='⭐ Avg Star Rating (3-mo avg)')
ax1.plot(x, monthly['sent_s'],  color=SENT_C, lw=2.5, marker='s', ms=4,
         linestyle='--', label='💬 Sentiment Score (1–5 scale)')

ax1.fill_between(x, monthly['sent_s'], monthly['stars_s'],
                 where=monthly['stars_s'] >= monthly['sent_s'],
                 alpha=0.18, color=STAR_C, label='Star Illusion Zone')
ax1.fill_between(x, monthly['sent_s'], monthly['stars_s'],
                 where=monthly['stars_s'] < monthly['sent_s'],
                 alpha=0.15, color=SENT_C, label='Sentiment > Stars')

# Annotate peak illusion month
peak_idx   = monthly['gap_s'].idxmax()
peak_loc   = monthly.index.get_loc(peak_idx)
peak_x     = x[peak_loc]
peak_stars = monthly['stars_s'].iloc[peak_loc]
peak_sent  = monthly['sent_s'].iloc[peak_loc]
ax1.annotate(
    f"  Peak illusion\n  ★{peak_stars:.2f} vs 💬{peak_sent:.2f}",
    xy=(peak_x, peak_stars), xytext=(peak_x + 1.5, peak_stars + 0.25),
    arrowprops=dict(arrowstyle='->', color=GAP_C, lw=1.5),
    fontsize=8.5, color=GAP_C, fontweight='bold'
)

avg_star_str = f"{df['rating'].mean():.2f}"
avg_sent_str = f"{df['sentiment_01'].mean()*4+1:.2f}"
ax1.text(0.99, 0.97,
         f"Avg ★ {avg_star_str}  |  Avg Sentiment {avg_sent_str}  |  Gap +{df['star_gap'].mean():.3f}",
         transform=ax1.transAxes, fontsize=8, color=SUB_C, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#252837',
                   edgecolor=GRID_C, alpha=0.9))

ax1.set_title('⭐ Star Rating vs. 💬 Text Sentiment  (3-month rolling avg, 1–5 scale)',
              fontsize=11, pad=8)
ax1.set_ylabel('Score (1–5)', fontsize=9)
ax1.set_ylim(2.5, 5.8)
ax1.set_xticks(xticks);  ax1.set_xticklabels(xlabels, rotation=40, ha='right', fontsize=8)
ax1.legend(loc='lower left', fontsize=8, framealpha=0.5,
           facecolor='#1e2130', edgecolor=GRID_C)

# ── ROW 2: Gap bar chart ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(CARD)
ax2.grid(True, axis='y', linestyle='--', linewidth=0.5)

bar_colors = [GAP_C if g > 0.10 else (ACCENT if g < 0 else '#555878')
              for g in monthly['gap_s']]
ax2.bar(x, monthly['gap_s'], color=bar_colors, width=0.75, alpha=0.9)
ax2.axhline(0.10, color=GAP_C,  linestyle='--', lw=1.3)
ax2.axhline(0.00, color=GRID_C, linestyle='-',  lw=0.8)
ax2.text(len(x) - 0.5, 0.105, ' danger', color=GAP_C, fontsize=7.5, va='bottom')

legend_patches = [
    mpatches.Patch(color=GAP_C,    label='High gap >0.10 — star inflation'),
    mpatches.Patch(color='#555878',label='Normal gap'),
    mpatches.Patch(color=ACCENT,   label='Negative gap — sentiment > stars'),
]
ax2.set_title('📊 Star–Sentiment Gap  (positive = stars inflate reality)',
              fontsize=11, pad=8)
ax2.set_ylabel('Gap (0–1 scale)', fontsize=9)
ax2.set_xticks(xticks); ax2.set_xticklabels(xlabels, rotation=40, ha='right', fontsize=8)
ax2.legend(handles=legend_patches, fontsize=7.5, framealpha=0.4,
           facecolor='#1e2130', edgecolor=GRID_C, loc='upper right')

# ── ROW 3: Silent Killers ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(CARD)
ax3.grid(True, axis='y', linestyle='--', linewidth=0.5)

ax3.bar(x, monthly['n_silent'], color=SILENT_C, width=0.75, alpha=0.92,
        label='Silent Killers (#)')

ax3_r = ax3.twinx()
ax3_r.plot(x, monthly['n'], color='#555878', lw=1.2, linestyle=':', alpha=0.6,
           label='Total reviews/month')
ax3_r.set_ylabel('Total reviews', fontsize=8, color='#555878')
ax3_r.tick_params(colors='#555878')
ax3_r.set_facecolor(CARD)

ax3.set_title('🔴 Silent Killers per Month  (4–5★ with negative sentiment text)',
              fontsize=11, pad=8)
ax3.set_ylabel('Silent Killers (#)', fontsize=9)
ax3.set_xticks(xticks); ax3.set_xticklabels(xlabels, rotation=40, ha='right', fontsize=8)

h1, l1 = ax3.get_legend_handles_labels()
h2, l2 = ax3_r.get_legend_handles_labels()
ax3.legend(h1 + h2, l1 + l2, fontsize=7.5, framealpha=0.4,
           facecolor='#1e2130', edgecolor=GRID_C)

# ── BOTTOM METRICS BAR ───────────────────────────────────────────────────────
ov  = (df['review_type'] == 'Overstated').sum()
metrics_txt = (
    f"  🔑 KEY FINDINGS:   "
    f"{sk} Silent Killers ({sk/total*100:.1f}% of reviews)   |   "
    f"{ov} Overstated reviews ({ov/total*100:.1f}%)   |   "
    f"Mean star-gap = +{df['star_gap'].mean():.3f}   |   "
    f"Peak illusion: {monthly['xstr'].iloc[peak_loc]} (gap {monthly['gap_s'].iloc[peak_loc]:.3f})"
)
fig.text(0.01, 0.012, metrics_txt, fontsize=7.8, color=SUB_C,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#141720',
                   edgecolor=GRID_C, alpha=0.95))

# ── SAVE ─────────────────────────────────────────────────────────────────────
plt.savefig(OUTPUT_PNG, dpi=160, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close()
print(f"\n✓ Chart saved → {OUTPUT_PNG}")
print(f"✓ Enriched CSV → {OUTPUT_CSV}  (use this as input for Phase 2)")
