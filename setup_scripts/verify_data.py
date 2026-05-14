# verify_data.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE : Sanity-check the clean CSV before any NLP work
# RUN     : python verify_data.py
# EXPECTED: All checks print ✓  — if any print ✗, fix before Day 1
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
CSV_PATH = "data/clean/subcategory_sample.csv"
REQUIRED_COLS_OPTIONS = [
    # 2023 HuggingFace version column names
    ["text", "rating", "timestamp"],
    # Older McAuley version column names
    ["reviewText", "overall", "unixReviewTime"],
    # Already-renamed standardised names
    ["review_text", "star_rating", "unix_time"],
    # Kaggle CSV version sometimes uses these
    ["review_text", "star_rating", "review_date"],
]

PASS = "  ✓"
FAIL = "  ✗"

print("=" * 55)
print("PHASE 0 — DATA VERIFICATION")
print("=" * 55)

# ── Check 1: File exists ──────────────────────────────────────────
print("\n[1] File existence")
path = Path(CSV_PATH)
if path.exists():
    size_mb = path.stat().st_size / 1_000_000
    print(f"{PASS} File found: {CSV_PATH}")
    print(f"{PASS} File size : {size_mb:.2f} MB")
else:
    print(f"{FAIL} FILE NOT FOUND at: {CSV_PATH}")
    print(f"     Current working directory: {os.getcwd()}")
    print(f"     Files in data/clean/ :")
    clean_dir = Path("data/clean")
    if clean_dir.exists():
        for f in clean_dir.iterdir():
            print(f"       - {f.name}")
    else:
        print("       data/clean/ directory does not exist")
    sys.exit(1)

# ── Load ──────────────────────────────────────────────────────────
print("\n[2] Loading CSV")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"{PASS} Loaded successfully")
    print(f"{PASS} Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
except Exception as e:
    print(f"{FAIL} Failed to load: {e}")
    sys.exit(1)

# ── Check 2: Columns ──────────────────────────────────────────────
print("\n[3] Column names found in your file:")
for col in df.columns.tolist():
    print(f"     - {col}")

# Try to detect which version of the dataset this is
detected_version = None
for version_cols in REQUIRED_COLS_OPTIONS:
    if all(c in df.columns for c in version_cols):
        detected_version = version_cols
        break

if detected_version:
    print(f"{PASS} Recognised column format: {detected_version}")
else:
    print(f"{FAIL} Could not auto-detect column format.")
    print(f"     Share this output — I'll update the column map for you.")

# ── Check 3: Null counts ──────────────────────────────────────────
print("\n[4] Null value counts per column:")
null_counts = df.isnull().sum()
for col, n in null_counts.items():
    pct = n / len(df) * 100
    flag = PASS if pct < 5 else FAIL
    print(f"{flag} {col:<30} {n:>6,} nulls ({pct:.1f}%)")

# ── Check 4: Star/rating column ───────────────────────────────────
print("\n[5] Rating column check")
rating_col = next(
    (c for c in ["rating", "overall", "star_rating"] if c in df.columns), None
)
if rating_col:
    unique_vals = sorted(df[rating_col].dropna().unique())
    print(f"{PASS} Rating column : '{rating_col}'")
    print(f"{PASS} Unique values : {unique_vals}")
    expected = [1.0, 2.0, 3.0, 4.0, 5.0]
    if all(v in expected for v in unique_vals):
        print(f"{PASS} Values are valid 1–5 stars")
    else:
        print(f"{FAIL} Unexpected values — check for data issues")

    print(f"\n     Distribution:")
    vc = df[rating_col].value_counts().sort_index()
    for stars, count in vc.items():
        bar = "█" * int(count / len(df) * 40)
        print(f"     {int(stars)}★  {bar}  {count:,} ({count/len(df)*100:.1f}%)")
else:
    print(f"{FAIL} No rating column found")

# ── Check 5: Text column ──────────────────────────────────────────
print("\n[6] Review text column check")
text_col = next(
    (c for c in ["text", "reviewText", "review_text"] if c in df.columns), None
)
if text_col:
    word_counts = df[text_col].dropna().str.split().str.len()
    print(f"{PASS} Text column   : '{text_col}'")
    print(f"{PASS} Avg length    : {word_counts.mean():.0f} words")
    print(f"{PASS} Median length : {word_counts.median():.0f} words")
    print(f"{PASS} Min length    : {word_counts.min()} words")
    print(f"{PASS} Max length    : {word_counts.max()} words")
    empty_text = (df[text_col].isna() | (df[text_col].str.strip() == "")).sum()
    print(f"{'✓' if empty_text == 0 else '✗'} Empty/null texts: {empty_text:,}")
else:
    print(f"{FAIL} No review text column found")

# ── Check 6: Timestamp ────────────────────────────────────────────
print("\n[7] Timestamp column check")
time_col = next(
    (c for c in ["timestamp", "unixReviewTime", "unix_time", "review_date"]
     if c in df.columns), None
)
if time_col:
    sample_val = df[time_col].dropna().iloc[0]
    print(f"{PASS} Timestamp column : '{time_col}'")
    print(f"     Sample raw value  : {sample_val} (type: {type(sample_val).__name__})")

    # Auto-detect unit
    try:
        val = float(sample_val)
        if val > 1e12:
            unit = "ms"
            parsed = pd.to_datetime(df[time_col], unit="ms", errors="coerce")
        elif val > 1e9:
            unit = "s"
            parsed = pd.to_datetime(df[time_col], unit="s", errors="coerce")
        else:
            unit = "unknown"
            parsed = pd.to_datetime(df[time_col], errors="coerce")
        print(f"{PASS} Detected unit     : {unit}")
        print(f"{PASS} Parsed date range : {parsed.min().date()} → {parsed.max().date()}")
        years = parsed.dt.year.value_counts().sort_index()
        print(f"     Years covered     : {years.index.min()} – {years.index.max()}")
    except Exception as e:
        # Already a datetime string column
        parsed = pd.to_datetime(df[time_col], errors="coerce")
        print(f"{PASS} Parsed date range : {parsed.min()} → {parsed.max()}")
else:
    print(f"{FAIL} No timestamp column found")

# ── Check 7: ASIN / product_id ────────────────────────────────────
print("\n[8] Product ID check")
asin_col = next(
    (c for c in ["asin", "parent_asin", "product_id"] if c in df.columns), None
)
if asin_col:
    unique_products = df[asin_col].nunique()
    print(f"{PASS} Product ID column : '{asin_col}'")
    print(f"{PASS} Unique products   : {unique_products:,}")
    print(f"     Most reviewed     : {df[asin_col].value_counts().index[0]} "
          f"({df[asin_col].value_counts().iloc[0]:,} reviews)")
else:
    print(f"  ⚠  No product ID column — not critical for Day 1")

# ── Check 8: Minimum sample size ─────────────────────────────────
print("\n[9] Statistical adequacy")
n = len(df)
if n >= 10_000:
    print(f"{PASS} {n:,} rows — excellent, well above 5k floor")
elif n >= 5_000:
    print(f"{PASS} {n:,} rows — meets 5k minimum (±5% MoE at 95% CI)")
elif n >= 1_000:
    print(f"  ⚠  {n:,} rows — below ideal but workable for a demo")
else:
    print(f"{FAIL} {n:,} rows — too small, results won't be reliable")

# ── Final summary ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("VERIFICATION COMPLETE")
print("=" * 55)
print(f"  Rows          : {len(df):,}")
print(f"  Columns       : {df.shape[1]}")
print(f"  Rating col    : {rating_col or 'NOT FOUND'}")
print(f"  Text col      : {text_col or 'NOT FOUND'}")
print(f"  Timestamp col : {time_col or 'NOT FOUND'}")
print(f"  ASIN col      : {asin_col or 'NOT FOUND'}")
print()
if rating_col and text_col and time_col:
    print("  ✓ All critical columns present.")
    print("  → You are clear to run day1_sentiment.py")
else:
    print("  ✗ Missing critical columns — paste this output")
    print("    and I will fix the column map before Day 1.")
print("=" * 55)
