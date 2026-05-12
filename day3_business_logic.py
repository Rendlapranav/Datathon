import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, warnings
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

INPUT_CSV  = "data/clean/day2_aspects.csv"
FIG_DIR    = "figures"
JSON_OUT   = "data/clean/ceo_summary.json"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load & Validate ──────────────────────────────────────────────
print(f"Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# Safety check: ensure sentiment column exists
if 'roberta_score' not in df.columns:
    if 'sentiment_score' in df.columns:
        df['roberta_score'] = df['sentiment_score']
    else:
        raise KeyError("Could not find 'roberta_score' or 'sentiment_score' in CSV.")

# Safely parse dates and drop NaTs
df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
df = df.dropna(subset=['review_date'])
df = df[df['topic_id'] != -1].copy()   # drop outliers
print(f"Loaded {len(df):,} reviews | {df['topic_label'].nunique()} aspects")

# ═══════════════════════════════════════════════════════════════════
# MODULE 1 — STRATEGIC PRIORITIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODULE 1: STRATEGIC PRIORITIZATION ENGINE")
print("="*60)

total_reviews = len(df)

aspect_stats = (
    df.groupby('topic_label')
    .agg(
        review_count     = ('topic_label',  'count'),
        mean_anger       = ('emo_anger',    'mean'),
        mean_sadness     = ('emo_sadness',  'mean'),
        mean_joy         = ('emo_joy',      'mean'),
        mean_roberta     = ('roberta_score','mean'),
    )
    .reset_index()
)

# Metric 1 — Frequency (% of total reviews)
aspect_stats['frequency_pct'] = (aspect_stats['review_count'] / total_reviews) * 100

# Metric 2 — Emotional Intensity (anger + sadness)
aspect_stats['emotional_intensity'] = aspect_stats['mean_anger'] + aspect_stats['mean_sadness']

# Normalize both to [0, 1] for the 2×2 grid
scaler = MinMaxScaler()
aspect_stats[['freq_norm', 'intensity_norm']] = scaler.fit_transform(
    aspect_stats[['frequency_pct', 'emotional_intensity']]
)

# Quadrant assignment
def assign_quadrant(row):
    hi_freq      = row['freq_norm'] >= 0.5
    hi_intensity = row['intensity_norm'] >= 0.5
    if hi_freq and hi_intensity:
        return "🔴 Urgent Redesign"
    elif not hi_freq and hi_intensity:
        return "🟠 Quality Control Edge Case"
    elif hi_freq and not hi_intensity:
        return "🟡 Marketing / Expectation Issue"
    else:
        return "🟢 Monitor Only"

aspect_stats['quadrant'] = aspect_stats.apply(assign_quadrant, axis=1)

print("\nAspect Priority Table:")
print(aspect_stats[['topic_label','frequency_pct','emotional_intensity',
                     'freq_norm','intensity_norm','quadrant']]
      .sort_values('intensity_norm', ascending=False)
      .to_string(index=False))

# ── Priority Matrix Plot ──────────────────────────────────────────
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(11, 8))

# Quadrant background shading
ax.axhspan(0.5, 1.05, xmin=0.5, xmax=1.05, color='#7f1d1d', alpha=0.15)
ax.axhspan(0.5, 1.05, xmin=0,   xmax=0.5,  color='#78350f', alpha=0.15)
ax.axhspan(0,   0.5,  xmin=0.5, xmax=1.05, color='#713f12', alpha=0.10)
ax.axhspan(0,   0.5,  xmin=0,   xmax=0.5,  color='#14532d', alpha=0.10)

ax.axhline(0.5, color='white', lw=0.8, ls='--', alpha=0.4)
ax.axvline(0.5, color='white', lw=0.8, ls='--', alpha=0.4)

ax.text(0.75, 0.97, "URGENT REDESIGN", transform=ax.transAxes, ha='center', va='top', fontsize=9, color='#fca5a5', alpha=0.7)
ax.text(0.25, 0.97, "EDGE CASE", transform=ax.transAxes, ha='center', va='top', fontsize=9, color='#fdba74', alpha=0.7)
ax.text(0.75, 0.03, "MARKETING FIX", transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='#fde68a', alpha=0.7)
ax.text(0.25, 0.03, "MONITOR ONLY", transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='#86efac', alpha=0.7)

size_scale = aspect_stats['review_count'] / aspect_stats['review_count'].max() * 1200 + 200

scatter = ax.scatter(
    aspect_stats['freq_norm'],
    aspect_stats['intensity_norm'],
    s=size_scale,
    c=aspect_stats['emotional_intensity'],
    cmap='RdYlGn_r',
    alpha=0.85,
    edgecolors='white',
    linewidths=0.8,
    zorder=5
)

for _, row in aspect_stats.iterrows():
    ax.annotate(
        row['topic_label'],
        (row['freq_norm'], row['intensity_norm']),
        textcoords="offset points", xytext=(10, 6),
        fontsize=9, color='white',
        bbox=dict(boxstyle='round,pad=0.2', fc='#1e293b', alpha=0.7)
    )

plt.colorbar(scatter, ax=ax, label='Emotional Intensity (anger+sadness)')
ax.set_xlabel('Complaint Frequency (normalised)', fontsize=12)
ax.set_ylabel('Emotional Intensity (normalised)', fontsize=12)
ax.set_title('CEO Priority Matrix\nBubble size = review volume | Color = emotional intensity', fontsize=13, pad=15)
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(-0.05, 1.1)
ax.grid(True, alpha=0.1)

plt.tight_layout()
matrix_path = os.path.join(FIG_DIR, 'day3_priority_matrix.png')
plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
print(f"\nPriority matrix saved → {matrix_path}")

# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — ROI SIMULATION ("WHAT-IF" ANALYSIS)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODULE 2: ROI SIMULATION")
print("="*60)

baseline_sentiment = df['roberta_score'].mean()
print(f"Baseline average sentiment score : {baseline_sentiment:+.4f}")

target_row   = aspect_stats.sort_values('mean_anger', ascending=False).iloc[0]
target_label = target_row['topic_label']
print(f"Most anger-heavy aspect          : '{target_label}'")

df_sim = df.copy()
target_mask = df_sim['topic_label'] == target_label

# Prevent Division by Zero
divisor = abs(baseline_sentiment) if abs(baseline_sentiment) > 0.001 else 1.0

# Scenario A — Neutral fix
df_sim_neutral = df_sim.copy()
df_sim_neutral.loc[target_mask, 'roberta_score'] = 0.0
sentiment_after_neutral = df_sim_neutral['roberta_score'].mean()
delta_neutral = ((sentiment_after_neutral - baseline_sentiment) / divisor) * 100

# Scenario B — Positive fix
df_sim_positive = df_sim.copy()
df_sim_positive.loc[target_mask, 'roberta_score'] = 0.30
sentiment_after_positive = df_sim_positive['roberta_score'].mean()
delta_positive = ((sentiment_after_positive - baseline_sentiment) / divisor) * 100

print(f"\n[ROI SIMULATION — Fixing '{target_label}']")
print(f"  ✦ SLIDE 106 PITCH:")
print(f"    \"Fixing {target_label} alone could improve overall brand sentiment by {delta_neutral:.0f}–{delta_positive:.0f}%.\"")

# ═══════════════════════════════════════════════════════════════════
# MODULE 3 — STRATEGIC ROADMAP LOGIC
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODULE 3: STRATEGIC ROADMAP")
print("="*60)

# Adjusted to common aspects, defaults to generic if not found
TECHNICAL_ASPECTS     = ["Battery & Charging", "Connectivity", "Sound Quality", "Build Quality", "Microphone & Calls"]
INSTRUCTIONAL_ASPECTS = ["Value for Money", "Comfort & Fit"]

roadmap = []
for _, row in aspect_stats.iterrows():
    label     = row['topic_label']
    intensity = row['intensity_norm']

    if intensity >= 0.6:
        tier   = "Short-Term (0–3 months)"
        action = "Firmware / Engineering Fix" if label in TECHNICAL_ASPECTS else "Urgent Quality Review"
    elif intensity < 0.4:
        tier   = "Mid-Term (3–6 months)"
        action = "Marketing Campaign / Tutorial Content" if label in INSTRUCTIONAL_ASPECTS else "UX / Design Improvement"
    else:
        tier   = "Mid-Term (3–6 months)"
        action = "Engineering Review"

    roadmap.append({
        "aspect":    label,
        "quadrant":  row['quadrant'],
        "intensity": round(float(intensity), 3),
        "tier":      tier,
        "action":    action
    })

roadmap.append({
    "aspect":    "All Aspects",
    "quadrant":  "Global",
    "intensity": None,
    "tier":      "Long-Term (6–12 months)",
    "action":    "Integrate Real-Time Sentiment Dashboard into Product QA Pipeline"
})

roadmap_df = pd.DataFrame(roadmap).sort_values('tier')
print("\nStrategic Roadmap:")
print(roadmap_df[['tier','aspect','action']].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# MODULE 4 — EARLY WARNING DASHBOARD (KPI EXPORT)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODULE 4: KPI DASHBOARD EXPORT")
print("="*60)

# Safe Datetime extraction
df['year_month'] = df['review_date'].dt.to_period('M')
monthly = df.groupby('year_month').agg(
    total = ('topic_label', 'count'),
    silent_killers = ('is_silent_killer', 'sum') if 'is_silent_killer' in df.columns else ('topic_label', lambda x: 0)
).reset_index().sort_values('year_month')

monthly['sk_rate'] = monthly['silent_killers'] / monthly['total']

# Mismatch monitor
if len(monthly) >= 2:
    curr_month   = monthly.iloc[-1]
    prev_month   = monthly.iloc[-2]
    sk_delta     = float(curr_month['sk_rate'] - prev_month['sk_rate'])
    sk_direction = "↑ WORSENING" if sk_delta > 0 else "↓ IMPROVING"
else:
    sk_delta     = 0.0
    curr_month   = monthly.iloc[-1] if not monthly.empty else pd.Series({'year_month': 'N/A', 'sk_rate': 0})
    prev_month   = curr_month
    sk_direction = "N/A"

def detect_anger_spikes(df, threshold=0.15):
    spikes = []
    periods = sorted(df['year_month'].unique())
    if len(periods) < 2: return spikes
    curr_p, prev_p = periods[-1], periods[-2]
    curr_anger = df[df['year_month'] == curr_p].groupby('topic_label')['emo_anger'].mean()
    prev_anger = df[df['year_month'] == prev_p].groupby('topic_label')['emo_anger'].mean()
    for label in curr_anger.index:
        if label in prev_anger.index:
            delta = curr_anger[label] - prev_anger[label]
            if delta > threshold:
                spikes.append({
                    "aspect": label, "anger_delta": round(float(delta), 3),
                    "alert": f"⚠ Anger spike +{delta*100:.1f}pp in last 30 days"
                })
    return spikes

anger_spikes = detect_anger_spikes(df)

# Output JSON
ceo_summary = {
    "product_id": str(df['product_id'].iloc[0]) if 'product_id' in df.columns else "Unknown",
    "total_reviews": int(total_reviews),
    "roi_simulation": {
        "target_aspect": target_label,
        "slide_106_pitch": f"Fixing {target_label} alone could improve brand sentiment by {delta_neutral:.0f}–{delta_positive:.0f}%."
    },
    "priority_matrix": aspect_stats[['topic_label','frequency_pct','quadrant']].to_dict(orient='records'),
    "strategic_roadmap": roadmap
}

with open(JSON_OUT, 'w') as f:
    json.dump(ceo_summary, f, indent=2)

print(f"\n✓ CEO summary saved → {JSON_OUT}")
print("\nDAY 3 COMPLETE — DELIVERABLES READY FOR PPT")
