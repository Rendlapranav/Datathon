"""
fetch_metadata_targeted.py — one-off targeted metadata lookup for the 4
ASINs already selected by stream_data.py (flagship + 3 competitors).

Earlier attempts to metadata-enrich the full top-50 candidate list timed
out badly (metadata stream order has no relation to review popularity, so
finding 50 specific ASINs meant scanning a large unpredictable fraction of
~1.6M records — one run stalled 80+ minutes and still hadn't finished).
Needing only 4 ASINs instead of 50 changes the odds a lot: the pass exits
the instant all 4 are found, rather than needing every one of 50 scattered
matches. Still bounded/streamed — no full dataset download.

RUN:    python setup_scripts/fetch_metadata_targeted.py
OUTPUT: data/clean/competitor_products.csv (brand/title/category columns filled in)
"""
import os
import sys
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

DATASET_REPO = "McAuley-Lab/Amazon-Reviews-2023"
META_CONFIG  = "raw_meta_Electronics"
PRODUCTS_CSV = "data/clean/competitor_products.csv"
SCAN_CAP     = int(os.environ.get("PRM_META_SCAN", 1_500_000))

products_df = pd.read_csv(PRODUCTS_CSV)
wanted = set(products_df["product_id"].tolist())
print(f"Looking up metadata for {len(wanted)} ASINs: {sorted(wanted)}")

meta_ds = load_dataset(DATASET_REPO, META_CONFIG, split="full", streaming=True, trust_remote_code=True)
found = {}
for i, rec in enumerate(tqdm(meta_ds, total=SCAN_CAP, desc="Targeted metadata lookup")):
    asin = rec.get("parent_asin")
    if asin in wanted and asin not in found:
        categories = rec.get("categories") or []
        leaf_category = categories[-1] if categories else rec.get("main_category")
        found[asin] = {
            "title": rec.get("title", ""),
            "brand": rec.get("store", "") or "Unknown",
            "category": leaf_category or rec.get("main_category", "Electronics"),
        }
        print(f"  FOUND {asin}: brand='{found[asin]['brand']}' title=\"{found[asin]['title'][:70]}\"")
    if len(found) >= len(wanted) or i + 1 >= SCAN_CAP:
        break

print(f"\nMatched {len(found)}/{len(wanted)}.")

for col in ["title", "brand", "category"]:
    products_df[col] = products_df.apply(
        lambda row: found.get(row["product_id"], {}).get(col, row.get(col, "Unknown")), axis=1
    )
products_df.to_csv(PRODUCTS_CSV, index=False)
print(f"Saved -> {PRODUCTS_CSV}")
