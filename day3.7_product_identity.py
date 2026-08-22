"""
day3.7_product_identity.py
============================
Resolves "what is this product actually called, and who makes it?" for
the flagship + competitors, for the CEO-facing dashboard.

Prefers official metadata (brand/title from setup_scripts/
fetch_metadata_targeted.py, if it completed) and falls back to a
frequency-based brand guess straight from review text when metadata is
still "Unknown" — the Amazon Reviews 2023 metadata stream has no
ordering relationship to review popularity, so a targeted lookup for a
handful of specific ASINs can legitimately still be running or may not
finish on a slow connection. The fallback: count capitalized,
non-common-English words across a product's own reviews (excluding a
stoplist of frequently-name-dropped OTHER brands used for comparison,
e.g. "better than my Bose") — a product's own brand name shows up far
more often in its own reviews than any single competitor comparison
does. This is clearly labeled in the output as a best-effort guess.

Also records whether any additional products from the flagship's own
company were found among the discovery candidates (none were, in this
run's data — that itself is worth telling the CEO honestly rather than
inventing a catalog).

RUN:    python day3.7_product_identity.py
INPUT:  data/clean/competitor_products.csv
        data/clean/subcategory_sample.csv
        data/clean/competitor_reviews.csv
OUTPUT: data/clean/product_identity.json
"""
import json
import os
import re
from collections import Counter

import pandas as pd

DATA_DIR = "data/clean"
OUT_JSON = os.path.join(DATA_DIR, "product_identity.json")

# Words that show up capitalized in reviews constantly but are never the
# product's own brand: common English words most people don't bother
# lowercasing consistently, plus other real brands people compare against.
STOPWORDS = set(w.lower() for w in [
    "I", "The", "This", "These", "They", "It", "But", "And", "So", "We",
    "My", "If", "When", "After", "Before", "Once", "Now", "Overall",
    "Also", "However", "Then", "There", "That", "For", "In", "On", "At",
    "As", "Not", "No", "Yes", "Ok", "Okay", "Great", "Good", "Bad",
    "Update", "Amazon", "Bluetooth", "USB", "LED", "ANC", "USA", "TV",
    "PC", "Mac", "Windows", "Skype", "YouTube", "Google", "Wi", "Fi",
    "Apple", "iPhone", "iPad", "iPod", "Android", "Samsung", "Sony",
    "Bose", "Beats", "JBL", "Skullcandy", "Anker", "Soundcore", "Jabra",
    "Christmas", "First", "Second", "Five", "Star", "Stars", "One", "Two",
])

# Capitalized word that is NOT immediately preceded by sentence-ending
# punctuation (or start-of-string) — i.e. capitalized mid-sentence, which
# is a real signal of a proper noun (brand name) rather than just being
# the first word of a sentence ("Sound quality is great" capitalizes
# "Sound" for a reason that has nothing to do with it being a brand).
MID_SENTENCE_RE = re.compile(r"(?<![.!?\n]\s)(?<!^)\b([A-Z][a-z]{2,15})\b")


def guess_brand(texts, min_share=0.02):
    """Most frequent mid-sentence-capitalized non-stopword token, if it
    appears in at least `min_share` of reviews (own-brand mentions
    cluster; a single competitor comparison does not)."""
    n = len(texts)
    if n == 0:
        return None
    counts = Counter()
    for t in texts:
        t = str(t)
        seen_in_this_review = set()
        for m in MID_SENTENCE_RE.finditer(t):
            lw = m.group(1).lower()
            if lw in STOPWORDS:
                continue
            seen_in_this_review.add(lw)
        counts.update(seen_in_this_review)
    if not counts:
        return None
    word, count = counts.most_common(1)[0]
    if count / n < min_share:
        return None
    return word.capitalize()


# Same idea as setup_scripts/stream_data.py's product-type detection, but
# that result was never persisted anywhere — recomputing it cheaply here
# (no network, just string search over already-local review text) lets us
# build a readable fallback title like "Senso Headphones" instead of the
# bland "Senso product" when official metadata never arrives.
PRODUCT_TYPE_KEYWORDS = [
    "earbuds", "earphones", "headphones", "headset", "speaker", "soundbar",
    "webcam", "keyboard", "mouse", "monitor", "charger", "cable", "hub",
    "router", "smartwatch", "tablet", "laptop", "camera", "microphone",
    "projector", "power bank", "adapter",
]


def guess_product_type(texts):
    counts = Counter()
    for t in texts:
        low = str(t).lower()
        counts.update(kw for kw in PRODUCT_TYPE_KEYWORDS if kw in low)
    return counts.most_common(1)[0][0].title() if counts else None


def resolve_identity(product_id, official_brand, official_title, review_texts):
    # pandas round-trips empty CSV strings as NaN (float), not "" — normalize.
    # Some rows also literally stored the string "Unknown" as a placeholder
    # instead of leaving it blank — treat both the same way.
    official_brand = "" if pd.isna(official_brand) else str(official_brand)
    official_title = "" if pd.isna(official_title) or str(official_title) == "Unknown" else str(official_title)
    if official_brand and official_brand != "Unknown":
        return {"brand": official_brand, "title": official_title, "brand_source": "metadata"}
    guessed = guess_brand(review_texts)
    if guessed:
        title = official_title or guess_product_type(review_texts) or ""
        return {"brand": guessed, "title": title, "brand_source": "review-text (best-effort)"}
    return {"brand": "Unknown", "title": official_title or "", "brand_source": "unresolved"}


print("=" * 60)
print("DAY 3.7: PRODUCT IDENTITY RESOLUTION")
print("=" * 60)

products_df = pd.read_csv(os.path.join(DATA_DIR, "competitor_products.csv"))
flagship_row = products_df[products_df["is_flagship"] == True].iloc[0]
comp_rows = products_df[products_df["is_flagship"] == False]

flagship_reviews = pd.read_csv(os.path.join(DATA_DIR, "subcategory_sample.csv"))["review_text"].tolist()
comp_reviews_df = pd.read_csv(os.path.join(DATA_DIR, "competitor_reviews.csv"))

flagship_identity = resolve_identity(
    flagship_row["product_id"], flagship_row.get("brand"), flagship_row.get("title"), flagship_reviews
)
flagship_identity["product_id"] = flagship_row["product_id"]
flagship_identity["category"] = flagship_row.get("category", "Electronics")
print(f"Flagship: {flagship_identity['brand']} ({flagship_identity['brand_source']}) "
      f"— \"{flagship_identity['title'][:70]}\"")

competitors = []
for _, row in comp_rows.iterrows():
    rev_texts = comp_reviews_df.loc[comp_reviews_df["product_id"] == row["product_id"], "review_text"].tolist()
    ident = resolve_identity(row["product_id"], row.get("brand"), row.get("title"), rev_texts)
    ident["product_id"] = row["product_id"]
    competitors.append(ident)
    print(f"  Competitor: {ident['brand']} ({ident['brand_source']}) — \"{ident['title'][:60]}\"")

# Sibling products: none found in this run's discovery scan (top ~50
# candidates were already checked against a different-brand filter during
# competitor selection — nothing shared the flagship's brand). Recorded
# honestly rather than invented.
result = {
    "flagship": flagship_identity,
    "competitors": competitors,
    "sibling_products": [],
    "sibling_products_note": (
        "No additional products from this company were found among the "
        "discovery scan's top candidate list. This does not rule out "
        "other products existing outside the scan window — it means "
        "none were popular enough (by review volume) to surface in it."
    ),
}
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved -> {OUT_JSON}")

# -- Patch resolved brand/title back into competitor_products.csv --------
# day3.5_competitor_benchmark.py (and the dashboard's Competitor Benchmark
# tab) read brand names straight from this CSV, not from product_identity
# .json — without this, they'd keep showing "Unknown" even after we just
# resolved real brand names above. Run this BEFORE day3.5 in the pipeline
# so the benchmark picks up the resolved names.
by_id = {flagship_identity["product_id"]: flagship_identity}
by_id.update({c["product_id"]: c for c in competitors})
for idx, row in products_df.iterrows():
    ident = by_id.get(row["product_id"])
    if ident:
        products_df.at[idx, "brand"] = ident["brand"]
        if ident.get("title"):
            products_df.at[idx, "title"] = ident["title"]
products_df.to_csv(os.path.join(DATA_DIR, "competitor_products.csv"), index=False)
print(f"Patched resolved brand/title into competitor_products.csv")
