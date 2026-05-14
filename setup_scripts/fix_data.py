# fix_data.py
# Run this ONCE from your project root: python fix_data.py
# This replaces your 326-row file with a proper subcategory sample

from datasets import load_dataset
import pandas as pd
import numpy as np
import math
from pathlib import Path

CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# ── Step 1: Load full Electronics reviews from HuggingFace ────────
print("Loading Electronics reviews from HuggingFace...")
print("(This will take 5-10 min depending on your connection)")

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Electronics",
    split="full",
    trust_remote_code=True
)
df_raw = dataset.to_pandas()
print(f"Full dataset loaded: {len(df_raw):,} rows")
print(f"Columns: {df_raw.columns.tolist()}")

# ── Step 2: Load metadata to find subcategories ───────────────────
print("\nLoading Electronics metadata...")
meta_ds = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Electronics",
    split="full",
    trust_remote_code=True
)
meta = meta_ds.to_pandas()
print(f"Metadata loaded: {len(meta):,} products")

# ── Step 3: Extract main_category from metadata ───────────────────
# 2023 version uses 'main_category' and 'categories' fields
print("\nExtracting subcategories...")

if "main_category" in meta.columns:
    cat_col = "main_category"
elif "categories" in meta.columns:
    # Older format — nested list
    meta["main_category"] = meta["categories"].apply(
        lambda x: x[0][-1] if isinstance(x, list) and len(x) > 0
                  and isinstance(x[0], list) and len(x[0]) > 1
                  else (x[0] if isinstance(x, list) and len(x) > 0 else None)
    )
    cat_col = "main_category"

subcat_counts = meta[cat_col].value_counts()
print("\nTop 10 subcategories by product count:")
print(subcat_counts.head(10).to_string())

# ── Step 4: Pick top subcategory and get its ASINs ────────────────
# Skip the top-level "Electronics" label if it appears
skip_labels = {"electronics", "amazon devices", "all electronics"}
for subcat, count in subcat_counts.items():
    if str(subcat).lower().strip() not in skip_labels:
        TOP_SUBCAT = subcat
        break

print(f"\nAuto-selected subcategory: '{TOP_SUBCAT}'")

asin_col = "parent_asin" if "parent_asin" in meta.columns else "asin"
target_asins = set(
    meta[meta[cat_col] == TOP_SUBCAT][asin_col].dropna().unique()
)
print(f"ASINs in subcategory: {len(target_asins):,}")

# ── Step 5: Filter reviews to this subcategory ────────────────────
review_asin_col = "parent_asin" if "parent_asin" in df_raw.columns else "asin"
df_subcat = df_raw[df_raw[review_asin_col].isin(target_asins)].copy()
population = len(df_subcat)
print(f"Reviews in subcategory: {population:,}")

# ── Step 6: Calculate statistically valid n ───────────────────────
def cochran_n(N, z=1.96, e=0.05, p=0.5):
    n0 = (z**2 * p * (1 - p)) / (e**2)
    return math.ceil(n0 / (1 + (n0 - 1) / N))

n_required = cochran_n(population)
SAMPLE_N = max(n_required, 5_000)
SAMPLE_N = min(SAMPLE_N, population)

print(f"\nCochran n (95% CI, ±5% MoE) : {n_required:,}")
print(f"User floor                   : 5,000")
print(f"Final sample size            : {SAMPLE_N:,}")

# ── Step 7: Stratified sample by star rating ──────────────────────
df_sample = (
    df_subcat
    .groupby("rating", group_keys=False)
    .apply(lambda x: x.sample(
        n=max(1, int(len(x) / population * SAMPLE_N)),
        random_state=SEED
    ))
    .reset_index(drop=True)
)

# ── Step 8: Rename columns to standard names ──────────────────────
COL_MAP = {
    "text"        : "review_text",
    "rating"      : "star_rating",
    "timestamp"   : "unix_time",
    "user_id"     : "reviewer_id",
    "parent_asin" : "product_id",
    "asin"        : "asin",
    "title"       : "review_summary",
    "helpful_vote": "helpful_vote",
    "verified_purchase": "verified_purchase",
}
cols = {k: v for k, v in COL_MAP.items() if k in df_sample.columns}
df_sample = df_sample[list(cols.keys())].rename(columns=cols)

# ── Step 9: Parse timestamps ──────────────────────────────────────
df_sample["review_date"] = pd.to_datetime(
    df_sample["unix_time"], unit="ms", errors="coerce"
)
df_sample["year"]       = df_sample["review_date"].dt.year
df_sample["year_month"] = df_sample["review_date"].dt.to_period("M")
df_sample["word_count"] = df_sample["review_text"].str.split().str.len()
df_sample["subcategory"] = TOP_SUBCAT

# ── Step 10: Save ─────────────────────────────────────────────────
safe_name = TOP_SUBCAT.replace(" ", "_").replace("/", "_").lower()
out_path = CLEAN_DIR / f"subcategory_sample_{safe_name}.csv"
df_sample.to_csv(out_path, index=False)

print("\n" + "=" * 55)
print("FIXED DATASET READY")
print("=" * 55)
print(f"  Subcategory  : {TOP_SUBCAT}")
print(f"  Sample size  : {len(df_sample):,}")
print(f"  Saved to     : {out_path}")
print(f"  Star dist    :")
for s, g in df_sample.groupby("star_rating"):
    bar = "█" * int(len(g)/len(df_sample)*40)
    print(f"    {int(s)}★  {bar}  {len(g):,} ({len(g)/len(df_sample)*100:.1f}%)")
print("\n→ Update CSV_PATH in day1_sentiment.py to this file path")
print("=" * 55)
