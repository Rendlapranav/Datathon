import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

INPUT_CSV  = "data/clean/day2_aspects.csv"
FIG_DIR    = "figures"
JSON_OUT   = "data/clean/ceo_summary.json"
os.makedirs(FIG_DIR, exist_ok=True)

# -- Load & Validate -----------------------------------------------
print(f"Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# FIX: strip non-breaking spaces (\xa0) that survive CSV round-trips
# in string columns (topic_label, product_id, dominant_emotion, etc.)
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda col: col.str.replace('\xa0', ' ', regex=False))

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

# =================================================================
# MODULE 1 -- STRATEGIC PRIORITIZATION ENGINE
# =================================================================
print("\n" + "="*60)
print("MODULE 1: STRATEGIC PRIORITIZATION ENGINE")
print("="*60)

# Absolute thresholds (no normalization)
FREQ_THRESHOLD      = 5.0   # % of total reviews
INTENSITY_THRESHOLD = 0.40  # mean(anger) + mean(sadness)

total_reviews = len(df)

aspect_stats = (
    df.groupby('topic_label')
    .agg(
        review_count = ('topic_label',   'count'),
        mean_anger   = ('emo_anger',     'mean'),
        mean_sadness = ('emo_sadness',   'mean'),
        mean_joy     = ('emo_joy',       'mean'),
        mean_roberta = ('roberta_score', 'mean'),
    )
    .reset_index()
)

# Metric 1 -- Frequency (% of total reviews)
aspect_stats['frequency_pct'] = (aspect_stats['review_count'] / total_reviews) * 100

# Metric 2 -- Emotional Intensity (anger + sadness), raw probabilities
aspect_stats['emotional_intensity'] = aspect_stats['mean_anger'] + aspect_stats['mean_sadness']

# Quadrant assignment via absolute thresholds
def assign_quadrant(row):
    hi_freq      = row['frequency_pct']      >= FREQ_THRESHOLD
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
print(f"  High Frequency  : >= {FREQ_THRESHOLD}% of reviews")
print(f"  High Intensity  : >= {INTENSITY_THRESHOLD} (mean anger + sadness)")

print("\nAspect Priority Table:")
print(aspect_stats[['topic_label', 'frequency_pct', 'emotional_intensity', 'quadrant']]
      .sort_values('emotional_intensity', ascending=False)
      .to_string(index=False))

# [Keep Matplotlib priority matrix plot exactly as before]

# =================================================================
# MODULE 2 -- ROI SIMULATION ("WHAT-IF" ANALYSIS)
# =================================================================
print("\n" + "="*60)
print("MODULE 2: ROI SIMULATION")
print("="*60)

ROI_TARGET_SENTIMENT = 0.50   # aspirational mean score after the fix

baseline_sentiment = df['roberta_score'].mean()
print(f"Baseline average sentiment score : {baseline_sentiment:+.4f}")

target_row   = aspect_stats.sort_values('mean_anger', ascending=False).iloc[0]
target_label = target_row['topic_label']
target_mean  = target_row['mean_roberta']
print(f"Most anger-heavy aspect          : '{target_label}'")
print(f"Current mean roberta_score       : {target_mean:+.4f}")

# improvement_delta = gap between the aspirational target and current aspect mean.
# Shifting all scores in the aspect by this delta preserves the shape of the
# distribution (variance, relative ordering) while moving the mean to the target.
improvement_delta = ROI_TARGET_SENTIMENT - target_mean
print(f"Improvement delta (shift)        : {improvement_delta:+.4f}  "
      f"(target {ROI_TARGET_SENTIMENT:.2f} - current {target_mean:.4f})")

target_mask = df['topic_label'] == target_label

# Scenario A -- Neutral fix: shift scores to mean = 0.0
delta_neutral = 0.0 - target_mean
df_sim_neutral = df.copy()
df_sim_neutral.loc[target_mask, 'roberta_score'] = np.clip(
    df_sim_neutral.loc[target_mask, 'roberta_score'] + delta_neutral,
    -1.0, 1.0
)
sentiment_after_neutral = df_sim_neutral['roberta_score'].mean()

# Scenario B -- Positive fix: shift scores to mean = ROI_TARGET_SENTIMENT
df_sim_positive = df.copy()
df_sim_positive.loc[target_mask, 'roberta_score'] = np.clip(
    df_sim_positive.loc[target_mask, 'roberta_score'] + improvement_delta,
    -1.0, 1.0
)
sentiment_after_positive = df_sim_positive['roberta_score'].mean()

# Prevent division by zero for percentage delta calculation
divisor = abs(baseline_sentiment) if abs(baseline_sentiment) > 0.001 else 1.0
delta_neutral_pct  = ((sentiment_after_neutral  - baseline_sentiment) / divisor) * 100
delta_positive_pct = ((sentiment_after_positive - baseline_sentiment) / divisor) * 100

print(f"\n[ROI SIMULATION -- Fixing '{target_label}']")
print(f"  Scenario A (neutral fix, mean -> 0.0):")
print(f"    Overall sentiment: {baseline_sentiment:+.4f} -> {sentiment_after_neutral:+.4f} "
      f"({delta_neutral_pct:+.1f}%)")
print(f"  Scenario B (positive fix, mean -> {ROI_TARGET_SENTIMENT:.2f}):")
print(f"    Overall sentiment: {baseline_sentiment:+.4f} -> {sentiment_after_positive:+.4f} "
      f"({delta_positive_pct:+.1f}%)")
print(f"\n  SLIDE 106 PITCH:")
print(f"    \"Fixing {target_label} alone could improve overall brand sentiment "
      f"by {delta_neutral_pct:.0f}-{delta_positive_pct:.0f}%.\"")

# =================================================================
# MODULE 3 -- STRATEGIC ROADMAP LOGIC
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
    intensity = row['emotional_intensity']   # raw value, not normalized

    if intensity >= INTENSITY_THRESHOLD * 1.5:   # e.g., >= 0.60 -- high urgency band
        tier   = "Short-Term (0-3 months)"
        action = ("Firmware / Engineering Fix"
                  if label in TECHNICAL_ASPECTS else "Urgent Quality Review")
    elif intensity < INTENSITY_THRESHOLD * 0.67:  # e.g., < ~0.27 -- low urgency band
        tier   = "Mid-Term (3-6 months)"
        action = ("Marketing Campaign / Tutorial Content"
                  if label in INSTRUCTIONAL_ASPECTS else "UX / Design Improvement")
    else:
        tier   = "Mid-Term (3-6 months)"
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
    "tier":      "Long-Term (6-12 months)",
    "action":    "Integrate Real-Time Sentiment Dashboard into Product QA Pipeline"
})

roadmap_df = pd.DataFrame(roadmap).sort_values('tier')
print("\nStrategic Roadmap:")
print(roadmap_df[['tier', 'aspect', 'action']].to_string(index=False))

# =================================================================
# MODULE 4 -- EARLY WARNING DASHBOARD (KPI EXPORT)
# =================================================================
print("\n" + "="*60)
print("MODULE 4: KPI DASHBOARD EXPORT")
print("="*60)

df['year_month'] = df['review_date'].dt.to_period('M')
monthly = df.groupby('year_month').agg(
    total          = ('topic_label', 'count'),
    silent_killers = ('is_silent_killer', 'sum')
    if 'is_silent_killer' in df.columns
    else ('topic_label', lambda x: 0)
).reset_index().sort_values('year_month')

monthly['sk_rate'] = monthly['silent_killers'] / monthly['total']

if len(monthly) >= 2:
    curr_month   = monthly.iloc[-1]
    prev_month   = monthly.iloc[-2]
    sk_delta     = float(curr_month['sk_rate'] - prev_month['sk_rate'])
    sk_direction = "WORSENING" if sk_delta > 0 else "IMPROVING"
else:
    sk_delta     = 0.0
    curr_month   = monthly.iloc[-1] if not monthly.empty else pd.Series({'year_month': 'N/A', 'sk_rate': 0})
    prev_month   = curr_month
    sk_direction = "N/A"

def detect_anger_spikes(df, threshold=0.15):
    spikes  = []
    periods = sorted(df['year_month'].unique())
    if len(periods) < 2:
        return spikes
    curr_p, prev_p = periods[-1], periods[-2]
    curr_anger = df[df['year_month'] == curr_p].groupby('topic_label')['emo_anger'].mean()
    prev_anger = df[df['year_month'] == prev_p].groupby('topic_label')['emo_anger'].mean()
    for label in curr_anger.index:
        if label in prev_anger.index:
            delta = curr_anger[label] - prev_anger[label]
            if delta > threshold:
                spikes.append({
                    "aspect":      label,
                    "anger_delta": round(float(delta), 3),
                    "alert":       f"Anger spike +{delta*100:.1f}pp in last 30 days"
                })
    return spikes

anger_spikes = detect_anger_spikes(df)

# Output JSON
ceo_summary = {
    "product_id": str(df['product_id'].iloc[0]) if 'product_id' in df.columns else "Unknown",
    "total_reviews": int(total_reviews),
    "thresholds": {
        "freq_pct_threshold":      FREQ_THRESHOLD,
        "intensity_threshold":     INTENSITY_THRESHOLD,
        "roi_target_sentiment":    ROI_TARGET_SENTIMENT,
    },
    "roi_simulation": {
        "target_aspect":      target_label,
        "improvement_delta":  round(float(improvement_delta), 4),
        "slide_106_pitch":    (f"Fixing {target_label} alone could improve brand sentiment "
                               f"by {delta_neutral_pct:.0f}-{delta_positive_pct:.0f}%."),
    },
    "priority_matrix": aspect_stats[['topic_label', 'frequency_pct',
                                      'emotional_intensity', 'quadrant']]
                        .to_dict(orient='records'),
    "strategic_roadmap": roadmap,
}

with open(JSON_OUT, 'w') as f:
    json.dump(ceo_summary, f, indent=2)

print(f"\nCEO summary saved -> {JSON_OUT}")
print("\nDAY 3 COMPLETE -- DELIVERABLES READY FOR PPT")
