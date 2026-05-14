from datasets import load_dataset
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os

# --- Configuration ---
TARGET_COUNT = 5000
OUTPUT_PATH = "data/clean/subcategory_sample.csv"
# We'll scan up to 3 million rows to find a product with 5k reviews
MAX_SCAN = 3000000 

os.makedirs("data/clean", exist_ok=True)

print(f"--- INITIATING FLAGSHIP EXTRACTION (Target: {TARGET_COUNT} Reviews) ---")

# Stream the 2023 Electronics reviews
dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Electronics",
    split="full",
    streaming=True,
    trust_remote_code=True
)

asin_buckets = defaultdict(list)
flagship_asin = None

# Scan the stream
pbar = tqdm(total=MAX_SCAN, desc="Searching for Flagship Product")
for i, review in enumerate(dataset):
    asin = review['parent_asin']
    asin_buckets[asin].append(review)
    
    # Check if we hit the 5,000 review goal for this product
    if len(asin_buckets[asin]) >= TARGET_COUNT:
        flagship_asin = asin
        pbar.close()
        print(f"\n[TARGET ACQUIRED] ASIN: {flagship_asin} has reached {TARGET_COUNT} reviews.")
        break
    
    pbar.update(1)
    if i >= MAX_SCAN:
        pbar.close()
        # Fallback: pick the product with the most reviews found so far
        flagship_asin = max(asin_buckets, key=lambda k: len(asin_buckets[k]))
        print(f"\n[SCAN LIMIT] Picking best found: {flagship_asin} with {len(asin_buckets[flagship_asin])} reviews.")
        break

# --- Data Formatting ---
# Take exactly TARGET_COUNT reviews
df = pd.DataFrame(asin_buckets[flagship_asin][:TARGET_COUNT])

# Map Hugging Face 2023 columns to your PPT standard
COL_MAP = {
    "text": "review_text",
    "title": "review_summary",
    "rating": "star_rating",
    "timestamp": "unix_time",
    "parent_asin": "product_id"
}
df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})

# Convert timestamp (2023 data uses ms)
df['review_date'] = pd.to_datetime(df['unix_time'], unit='ms', errors='coerce')

# Final Cleanup
df = df.dropna(subset=['review_text', 'review_date']).copy()
df = df.sort_values(by="review_date").reset_index(drop=True)

# Save
df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "="*50)
print(f"SUCCESS: {len(df)} reviews saved to {OUTPUT_PATH}")
print(f"Product ID: {flagship_asin}")
print(f"Avg Rating: {df['star_rating'].mean():.2f}")
print("="*50)
print("→ Next Step: Run day1_roberta_sentiment.py to find the Star-Gap.")
