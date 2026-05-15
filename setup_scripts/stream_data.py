"""
stream_data.py — Stream 50,000 Amazon Electronics reviews from HuggingFace.
Creates: data/clean/subcategory_sample.csv
Run: python setup_scripts/stream_data.py
"""
import os
import sys

os.makedirs("data/clean", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

OUTPUT_CSV = "data/clean/subcategory_sample.csv"
N_REVIEWS  = 50_000

print("=" * 60)
print("Streaming Amazon Electronics Reviews from HuggingFace")
print(f"   Target: {N_REVIEWS:,} reviews -> {OUTPUT_CSV}")
print("=" * 60)

try:
    from datasets import load_dataset
    import pandas as pd
except ImportError:
    print("ERROR: Missing packages. Run: pip install datasets pandas")
    sys.exit(1)

print("\n[1/3] Connecting to HuggingFace hub...")
dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Electronics",
    split="full",
    streaming=True,
    trust_remote_code=True,
)

print(f"[2/3] Streaming {N_REVIEWS:,} rows (this takes ~2–5 min on broadband)...")
reviews = []
for i, row in enumerate(dataset):
    reviews.append(row)
    if i % 5_000 == 0 and i > 0:
        print(f"   {i:,} / {N_REVIEWS:,} ({i/N_REVIEWS*100:.0f}%)")
    if i >= N_REVIEWS - 1:
        break

print(f"\n[3/3] Building DataFrame and saving...")
import pandas as pd
df = pd.DataFrame(reviews)

# Standardize key column names expected by the pipeline
rename_map = {}
for candidate in [("text", "review_text"), ("rating", "star_rating"), ("timestamp", "review_date")]:
    if candidate[0] in df.columns and candidate[1] not in df.columns:
        rename_map[candidate[0]] = candidate[1]

# McAuley-Lab 2023 field names
if "text" in df.columns:       rename_map["text"]       = "review_text"
if "rating" in df.columns:     rename_map["rating"]     = "star_rating"
if "timestamp" in df.columns:  rename_map["timestamp"]  = "review_date"
if "asin" in df.columns:       rename_map["asin"]       = "product_id"

df = df.rename(columns=rename_map)

# Convert timestamp (epoch ms → datetime string)
if "review_date" in df.columns:
    import pandas as pd
    df["review_date"] = pd.to_datetime(df["review_date"], unit="ms", errors="coerce")
    df["review_date"] = df["review_date"].dt.strftime("%Y-%m-%d")

# Ensure star_rating is numeric
if "star_rating" in df.columns:
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")

# Drop rows with no review text
if "review_text" in df.columns:
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.strip() != ""]

df = df.reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {len(df):,} reviews to {OUTPUT_CSV}")
print(f"   Columns: {list(df.columns)}")
print(f"   Date range: {df.get('review_date', pd.Series()).min()} -> {df.get('review_date', pd.Series()).max()}")
print(f"\nNext step: python day1_roberta_sentiment.py")
