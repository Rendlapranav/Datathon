import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, warnings

warnings.filterwarnings("ignore")

INPUT_CSV  = "data/clean/day2_aspects.csv"
FIG_DIR    = "figures"
JSON_OUT   = "data/clean/ceo_summary.json"
os.makedirs(FIG_DIR, exist_ok=True)

# Mirrors day2's ACTIVE_EMOTIONS and DISPLAY_LABELS exactly.
# If day2 changes these, change them here too.
ACTIVE_EMOTIONS = ["anger", "sadness", "joy"]
DISPLAY_LABELS  = {
    "anger"  : "Anger",
    "sadness": "Disappointment",
    "joy"    : "Joy",
}

# -- Load & Validate -----------------------------------------------
print(f"Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

if 'roberta_score' not in df.columns:
    if 'sentiment_score' in df.columns:
        df['roberta_score'] = df['sentiment_score']
    else:
        raise KeyError("Could not find 'roberta_score' or 'sentiment_score' in CSV.")

# Validate that emotion columns from day2 are present
missing_emo_cols = [f"emo_{e}" for e in ACTIVE_EMOTIONS if f"emo_{e}" not in df.columns]
if missing_emo_cols:
    raise KeyError(
        f"Expected emotion columns not found: {missing_emo_cols}\n"
        f"Ensure day2_aspects.py ran successfully with ACTIVE_EMOTIONS={ACTIVE_EMOTIONS}"
    )

df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
df = df.dropna(subset=['review_date'])
df = df[df['topic_id'] != -1].copy()
print(f"Loaded {len(df):,} reviews | {df['topic_label'].nunique()} aspects")

# =================================================================
# MODULE 1 -- STRATEGIC PRIORITIZATION ENGINE
# =================================================================
print("\n" + "="*60)
print("MODULE 1: STRATEGIC PRIORITIZATION ENGINE")
print("="*60)

FREQ_THRESHOLD      = 5.0    # % of total reviews
INTENSITY_THRESHOLD = 0.40   # mean(anger) + mean(disappointment)

total_reviews = len(df)

aspect_stats = (
    df.groupby('topic_label')
    .agg(
        review_count = ('topic_label',   'count'),
        mean_anger   = ('emo_anger',     'mean'),
        mean_disapp  = ('emo_sadness',   'mean'),   # stored as sadness, displayed as disappointment
        mean_joy     = ('emo_joy',       'mean'),
        mean_roberta = ('roberta_score', 'mean'),
    )
    .reset_index()
)

aspect_stats['frequency_pct']      = (aspect_stats['review_count'] / total_reviews) * 100
# Intensity = anger + disappointment. Consistent with day2 heatmap logic.
aspect_stats['emotional_intensity'] = aspect_stats['mean_anger'] + aspect_stats['mean_disapp']

def assign_quadrant(row):
    hi_freq      = row['frequency_pct']       >= FREQ_THRESHOLD
    hi_intensity = row['emotional_intensity'] >= INTENSITY_THRESHOLD
    if hi_freq and hi_intensity:
        return "Urgent Redesign"
    elif not hi_freq and hi_intensity:
        return "Quality Control Edge Case"
    elif hi_freq and not hi_intensity:
        return "Marketing / Expectation Issue"
    else:
        return "Monitor Only"

aspect_stats['quadrant'] = aspect_stats.apply(assign_quadrant, axis=1)

print(f"\nThresholds applied:")
print(f"  High Frequency    : >= {FREQ_THRESHOLD}% of reviews")
print(f"  High Intensity    : >= {INTENSITY_THRESHOLD} (mean anger + disappointment)")

print("\nAspect Priority Table:")
print(
    aspect_stats[['topic_label', 'frequency_pct', 'emotional_intensity', 'quadrant']]
    .sort_values('emotional_intensity', ascending=False)
    .to_string(index=False)
)

# -- Priority Matrix Plot ------------------------------------------
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(11, 8))

max_freq = max(aspect_stats['frequency_pct'].max() * 1.1, FREQ_THRESHOLD * 2)
max_int  = max(aspect_stats['emotional_intensity'].max() * 1.1, INTENSITY_THRESHOLD * 1.5)

ax.set_xlim(0, max_freq)
ax.set_ylim(0, max_int)

ax.axhline(INTENSITY_THRESHOLD, color='white', lw=0.8, ls='--', alpha=0.4)
ax.axvline(FREQ_THRESHOLD,      color='white', lw=0.8, ls='--', alpha=0.4)

ax.fill_between([FREQ_THRESHOLD, max_freq], INTENSITY_THRESHOLD, max_int, color='#7f1d1d', alpha=0.15)
ax.fill_between([0, FREQ_THRESHOLD],        INTENSITY_THRESHOLD, max_int, color='#78350f', alpha=0.15)
ax.fill_between([FREQ_THRESHOLD, max_freq], 0, INTENSITY_THRESHOLD,       color='#713f12', alpha=0.10)
ax.fill_between([0, FREQ_THRESHOLD],        0, INTENSITY_THRESHOLD,       color='#14532d', alpha=0.10)

ax.text(FREQ_THRESHOLD + (max_freq - FREQ_THRESHOLD)/2, max_int*0.97, "URGENT REDESIGN",      ha='center', va='top',    fontsize=9, color='#fca5a5', alpha=0.7)
ax.text(FREQ_THRESHOLD/2,                               max_int*0.97, "EDGE CASE",             ha='center', va='top',    fontsize=9, color='#fdba74', alpha=0.7)
ax.text(FREQ_THRESHOLD + (max_freq - FREQ_THRESHOLD)/2, max_int*0.03, "MARKETING FIX",         ha='center', va='bottom', fontsize=9, color='#fde68a', alpha=0.7)
ax.text(FREQ_THRESHOLD/2,                               max_int*0.03, "MONITOR ONLY",          ha='center', va='bottom', fontsize=9, color='#86efac', alpha=0.7)

size_scale = aspect_stats['review_count'] / aspect_stats['review_count'].max() * 1200 + 200

scatter = ax.scatter(
    aspect_stats['frequency_pct'],
    aspect_stats['emotional_intensity'],
    s          = size_scale,
    c          = aspect_stats['emotional_intensity'],
    cmap       = 'RdYlGn_r',
    alpha      = 0.85,
    edgecolors = 'white',
    linewidths = 0.8,
    zorder     = 5,
)

for _, row in aspect_stats.iterrows():
    ax.annotate(
        row['topic_label'],
        (row['frequency_pct'], row['emotional_intensity']),
        textcoords = "offset points",
        xytext     = (10, 6),
        fontsize   = 9,
        color      = 'white',
        bbox       = dict(boxstyle='round,pad=0.2', fc='#1e293b', alpha=0.7),
    )

plt.colorbar(scatter, ax=ax, label='Emotional Intensity (anger + disappointment)')
ax.set_xlabel('Complaint Frequency (% of total reviews)', fontsize=12)
ax.set_ylabel('Emotional Intensity (anger + disappointment probability)', fontsize=12)
ax.set_title(
    'CEO Priority Matrix\nBubble size = review volume  |  Color = emotional intensity',
    fontsize=13, pad=15,
)
ax.grid(True, alpha=0.1)
plt.tight_layout()

matrix_path = os.path.join(FIG_DIR, 'day3_priority_matrix.png')
plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
print(f"\nPriority matrix saved -> {matrix_path}")

# =================================================================
# MODULE 2 -- ROI SIMULATION & MONETIZATION
# =================================================================
print("\n" + "="*60)
print("MODULE 2: ROI SIMULATION & MONETIZATION")
print("="*60)

# AOV: Amazon Electronics category average.
# CHURN_MULTIPLIER: Bain & Company lower bound (acquiring a new customer
# costs 5-25x more than retaining one). We use 5x — conservative.
AOV              = 150.00
CHURN_MULTIPLIER = 5
ROI_TARGET_SENTIMENT = 0.50

baseline_sentiment = df['roberta_score'].mean()
target_row   = aspect_stats.sort_values('mean_anger', ascending=False).iloc[0]
target_label = target_row['topic_label']
target_mean  = target_row['mean_roberta']
improvement_delta = ROI_TARGET_SENTIMENT - target_mean
target_mask  = df['topic_label'] == target_label

df_sim_neutral  = df.copy()
df_sim_positive = df.copy()
df_sim_neutral.loc[target_mask,  'roberta_score'] = np.clip(
    df_sim_neutral.loc[target_mask, 'roberta_score'] + (0.0 - target_mean), -1.0, 1.0
)
df_sim_positive.loc[target_mask, 'roberta_score'] = np.clip(
    df_sim_positive.loc[target_mask, 'roberta_score'] + improvement_delta, -1.0, 1.0
)

sentiment_after_neutral  = df_sim_neutral['roberta_score'].mean()
sentiment_after_positive = df_sim_positive['roberta_score'].mean()

divisor = abs(baseline_sentiment) if abs(baseline_sentiment) > 0.001 else 1.0
delta_neutral_pct  = ((sentiment_after_neutral  - baseline_sentiment) / divisor) * 100
delta_positive_pct = ((sentiment_after_positive - baseline_sentiment) / divisor) * 100

# Revenue at Risk: prefer the sk_weight-adjusted column written by day1
# (which accounts for helpful_votes and verified_purchase).
# Fall back to naive calculation only if that column is absent.
if 'revenue_at_risk' in df.columns:
    revenue_at_risk      = df['revenue_at_risk'].sum()
    silent_killers_count = int(df['is_silent_killer'].sum()) if 'is_silent_killer' in df.columns else 0
    rar_source = "day1 weighted (helpful_votes + verified_purchase)"
else:
    silent_killers_count = int(df['is_silent_killer'].sum()) if 'is_silent_killer' in df.columns else 0
    revenue_at_risk      = silent_killers_count * CHURN_MULTIPLIER * AOV
    rar_source = "naive fallback (sk_weight column missing)"

print(f"\n[BUSINESS IMPACT -- The Cost of Inaction]")
print(f"  Revenue at Risk source  : {rar_source}")
print(f"  Detected Silent Killers : {silent_killers_count:,}")
print(f"  Est. Invisible Churn    : {silent_killers_count * CHURN_MULTIPLIER:,} customers")
print(f"  AOV assumption          : ${AOV:.2f} (Electronics category average)")
print(f"  Churn multiplier        : {CHURN_MULTIPLIER}x (Bain & Company lower bound)")
print(f"  Revenue at Risk         : ${revenue_at_risk:,.2f}")

print(f"\n[ROI SIMULATION -- Fixing '{target_label}']")
print(f"  Scenario A (neutralise sentiment) : {delta_neutral_pct:+.1f}% overall sentiment shift")
print(f"  Scenario B (drive to positive)    : {delta_positive_pct:+.1f}% overall sentiment shift")

# =================================================================
# MODULE 3 -- STRATEGIC ROADMAP
# =================================================================
print("\n" + "="*60)
print("MODULE 3: STRATEGIC ROADMAP")
print("="*60)

TECHNICAL_ASPECTS     = ["Battery & Charging", "Connectivity", "Sound Quality",
                          "Build Quality", "Microphone & Calls"]
INSTRUCTIONAL_ASPECTS = ["Value for Money", "Comfort & Fit"]

roadmap = []
for _, row in aspect_stats.iterrows():
    label     = row['topic_label']
    intensity = row['emotional_intensity']

    if intensity >= INTENSITY_THRESHOLD * 1.5:
        tier   = "Short-Term (0-3 months)"
        action = "Firmware / Engineering Fix" if label in TECHNICAL_ASPECTS else "Urgent Quality Review"
    elif intensity < INTENSITY_THRESHOLD * 0.67:
        tier   = "Mid-Term (3-6 months)"
        action = "Marketing Campaign / Tutorial Content" if label in INSTRUCTIONAL_ASPECTS else "UX / Design Improvement"
    else:
        tier   = "Mid-Term (3-6 months)"
        action = "Engineering Review"

    roadmap.append({
        "aspect"   : label,
        "quadrant" : row['quadrant'],
        "intensity": round(float(intensity), 3),
        "tier"     : tier,
        "action"   : action,
    })

roadmap_df = pd.DataFrame(roadmap).sort_values('tier')
print("\nStrategic Roadmap:")
print(roadmap_df[['tier', 'aspect', 'action']].to_string(index=False))

# =================================================================
# MODULE 4 -- CEO JSON EXPORT
# =================================================================
df['year_month'] = df['review_date'].dt.to_period('M').astype(str)

ceo_summary = {
    "product_id"  : str(df['product_id'].iloc[0]) if 'product_id' in df.columns else "Unknown",
    "total_reviews": int(total_reviews),
    "emotion_model": {
        "active_signals"  : ACTIVE_EMOTIONS,
        "display_labels"  : DISPLAY_LABELS,
        "discarded_signals": ["disgust", "fear", "surprise", "neutral"],
        "intensity_formula": "mean(emo_anger) + mean(emo_sadness)",
    },
    "financials": {
        "silent_killers"     : silent_killers_count,
        "revenue_at_risk_usd": round(float(revenue_at_risk), 2),
        "aov_usd"            : AOV,
        "churn_multiplier"   : CHURN_MULTIPLIER,
        "churn_multiplier_source": "Bain & Company — conservative lower bound of 5-25x range",
        "rar_source"         : rar_source,
    },
    "roi_simulation": {
        "target_aspect"       : target_label,
        "improvement_pct_low" : round(float(delta_neutral_pct),  1),
        "improvement_pct_high": round(float(delta_positive_pct), 1),
        "slide_pitch"         : (
            f"Fixing {target_label} alone rescues ${revenue_at_risk:,.0f} in ARR "
            f"and boosts overall brand sentiment by {delta_positive_pct:.0f}%."
        ),
    },
    "priority_matrix": aspect_stats[[
        'topic_label', 'frequency_pct', 'emotional_intensity', 'quadrant'
    ]].to_dict(orient='records'),
    "strategic_roadmap": roadmap,
}

with open(JSON_OUT, 'w') as f:
    json.dump(ceo_summary, f, indent=2)

print(f"\nCEO summary saved -> {JSON_OUT}")
print("\nDAY 3 COMPLETE -- DELIVERABLES READY")
