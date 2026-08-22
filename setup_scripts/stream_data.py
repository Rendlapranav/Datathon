"""
stream_data.py — Flagship product discovery + competitor benchmark set.

Per the problem statement ("Focus on a specific category... Electronics"),
this finds the single Electronics product with the LARGEST number of
reviews (for statistical significance) and a set of direct competitor
products in the same category, so the rest of the pipeline can diagnose
the flagship AND benchmark it against rivals.

Why this replaces the old approach:
  - The previous stream_data.py just grabbed the first 50,000 reviews off
    the stream, spanning thousands of unrelated products — there was no
    single "flagship product in trouble" at all, despite that being the
    premise of the whole mission.
  - The previous setup_scripts/load_data.py picked whichever ASIN happened
    to be the FIRST to accumulate TARGET_COUNT reviews in streaming order —
    an artifact of stream order, not the product with the most reviews.

Design notes (learned by actually running this against a flaky network):
  - ONE streaming pass over reviews does everything: counts reviews per
    ASIN, detects product type from review text (see PRODUCT_TYPE_KEYWORDS
    below), AND buffers the row so it's already on hand once we know who
    the flagship/competitors are. A separate "go back and collect rows"
    pass was tried first and cut — it re-opens a fresh HTTP stream against
    the same remote file, which is exactly where this connection stalled
    hardest in testing (a supposedly-2-minute pass once took over an hour
    on a bad network day). One pass halves that exposure.
  - Total memory is bounded by MAX_SCAN regardless of how buffered rows
    are distributed across ASINs (buffering unevenly doesn't cost more
    than buffering evenly — the total can never exceed rows scanned), so
    there's no need for a per-ASIN cap during collection; the cap is
    simply how many scanned rows we keep, not how much RAM we use.
  - Metadata (official brand/category) lives in a SEPARATE HuggingFace
    config with no ordering relationship to review popularity, so finding
    a specific handful of ASINs in it means scanning a large, unrelated
    fraction of ~1.6M records — and in testing this was the slowest,
    least reliable part of the whole script (observed under 1,200 rows/
    sec with frequent read timeouts, vs ~6,000 rows/sec on the review
    stream). It's cosmetic only (brand/category display) — nothing
    downstream needs it — so it's off by default. Set PRM_FETCH_META=1 to
    enable it.

RUN:      python setup_scripts/stream_data.py
OUTPUTS:  data/clean/subcategory_sample.csv     (flagship reviews — feeds Day 1-4 unchanged)
          data/clean/competitor_products.csv    (flagship + competitor metadata)
          data/clean/competitor_reviews.csv     (competitor reviews — feeds competitor benchmark)
"""
import math
import os
import sys
from collections import Counter, defaultdict

os.makedirs("data/clean", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

# -- Config --------------------------------------------------------------
DATASET_REPO      = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORY_CONFIG   = "raw_review_Electronics"
META_CONFIG       = "raw_meta_Electronics"

# How many rows to scan. `streaming=True` below means HuggingFace never
# downloads or caches the full ~40M-row dataset to disk — rows are pulled
# one at a time over the network and (aside from the ones we keep, below)
# discarded after use. MAX_SCAN bounds how long that streaming pass runs.
# Cochran's minimum sample for a 95% CI / +/-5% margin of error converges
# to ~385 regardless of population size, so even a modest scan comfortably
# clears statistical significance for a genuinely popular product.
# Override with env var PRM_MAX_SCAN for a deeper (slower) scan.
MAX_SCAN          = int(os.environ.get("PRM_MAX_SCAN", 500_000))

FLAGSHIP_CAP      = 5_000     # reviews kept for the flagship (matches prior TARGET_COUNT)
N_COMPETITORS     = 4         # direct rivals to benchmark against
COMPETITOR_CAP    = 1_500     # reviews kept per competitor (well above the Cochran floor above)
TOP_CANDIDATES    = 50        # how many high-volume ASINs are eligible as competitors

FETCH_META        = os.environ.get("PRM_FETCH_META", "0") == "1"
META_SCAN_CAP     = int(os.environ.get("PRM_META_SCAN", 150_000))

OUT_FLAGSHIP      = "data/clean/subcategory_sample.csv"
OUT_COMPETITORS   = "data/clean/competitor_reviews.csv"
OUT_PRODUCTS      = "data/clean/competitor_products.csv"


def cochran_n(N, z=1.96, e=0.05, p=0.5):
    """Minimum sample size for a proportion estimate at 95% CI, +/-5% MoE."""
    n0 = (z ** 2 * p * (1 - p)) / (e ** 2)
    return math.ceil(n0 / (1 + (n0 - 1) / N))


# `main_category` in the metadata is often a broad umbrella (e.g. "All
# Electronics") that alone would group unrelated products together — a
# phone case and a speaker can share it. Product type, detected straight
# from review text (title + first part of body — reviewers do mention
# what they're reviewing), is a cheaper and more specific signal that
# needs no separate metadata lookup at all.
PRODUCT_TYPE_KEYWORDS = [
    "earbuds", "earphone", "headphone", "headset", "speaker", "soundbar",
    "webcam", "keyboard", "mouse", "monitor", "charger", "cable", "hub",
    "router", "smartwatch", "tablet", "laptop", "camera", "microphone",
    "projector", "power bank", "adapter",
]


def find_product_types(text):
    """All keywords present in text (there can be more than one per review)."""
    t = (text or "").lower()
    return [kw for kw in PRODUCT_TYPE_KEYWORDS if kw in t]


def extract_product_type(text):
    hits = find_product_types(text)
    return hits[0] if hits else None


def majority_product_type(reviews):
    """Vote across every review for one ASIN, not just the first — a single
    incidental mention (e.g. "didn't include a charger" in a headphones
    review) shouldn't decide the type. Checking title + a text snippet
    from every review and taking the most common hit is far more robust."""
    votes = Counter()
    for r in reviews:
        snippet = f"{r.get('title', '') or ''} {(r.get('text', '') or '')[:300]}"
        votes.update(find_product_types(snippet))
    return votes.most_common(1)[0][0] if votes else None


try:
    from datasets import load_dataset
    import pandas as pd
    from tqdm import tqdm
except ImportError:
    print("ERROR: Missing packages. Run: pip install datasets pandas tqdm")
    sys.exit(1)

print("=" * 70)
print("FLAGSHIP + COMPETITOR DISCOVERY — Amazon Electronics Reviews")
print("=" * 70)
print("Streaming mode: rows are pulled one at a time; nothing is cached or")
print("downloaded in full. A single pass counts, type-tags, AND buffers")
print(f"rows at once (network-thrifty). Scan bound: {MAX_SCAN:,} rows")
print("(set PRM_MAX_SCAN to change).")

# ── Single pass: count + buffer rows (product type is voted afterward, ──
# only for the handful of candidates that matter — see below) ───────────
print(f"\n[1/3] Scanning up to {MAX_SCAN:,} reviews...")
dataset = load_dataset(DATASET_REPO, CATEGORY_CONFIG, split="full", streaming=True, trust_remote_code=True)

asin_counts   = Counter()
rows_by_asin  = defaultdict(list)
for i, review in enumerate(tqdm(dataset, total=MAX_SCAN, desc="Counting + buffering")):
    asin = review["parent_asin"]
    asin_counts[asin] += 1
    rows_by_asin[asin].append(review)
    if i + 1 >= MAX_SCAN:
        break

if not asin_counts:
    print("ERROR: No reviews found in stream.")
    sys.exit(1)

ranked = asin_counts.most_common(TOP_CANDIDATES)
flagship_asin, flagship_count = ranked[0]

# Product type: majority vote across every buffered review for each of the
# (at most 50) candidate ASINs — cheap now, since we're only doing this for
# candidates that already matter, not all 500K scanned rows.
asin_type_hint = {
    asin: majority_product_type(rows_by_asin[asin])
    for asin, _ in ranked
}
asin_type_hint = {k: v for k, v in asin_type_hint.items() if v}

n_required = cochran_n(sum(asin_counts.values()))
print(f"\n[TARGET ACQUIRED] Flagship ASIN: {flagship_asin} — {flagship_count:,} reviews "
      f"(largest in the {len(asin_counts):,} products scanned)")
print(f"  Cochran minimum for 95% CI / +/-5% MoE: {n_required:,} reviews")
if flagship_count < n_required:
    print(f"  [WARN] Flagship review count is below the Cochran minimum — "
          f"consider raising PRM_MAX_SCAN for a deeper scan.")
else:
    print(f"  Flagship clears the statistical-significance floor.")

# ── Optional: metadata lookup — best-effort brand/category enrichment ───
meta_by_asin = {}
if FETCH_META:
    print(f"\n[2/3] Looking up metadata for the top {len(ranked)} candidate products "
          f"(best-effort — brand/category display only)...")
    meta_ds = load_dataset(DATASET_REPO, META_CONFIG, split="full", streaming=True, trust_remote_code=True)
    wanted_meta_asins = {asin for asin, _ in ranked}
    for i, rec in enumerate(tqdm(meta_ds, total=META_SCAN_CAP, desc="Matching metadata (best-effort)")):
        asin = rec.get("parent_asin")
        if asin in wanted_meta_asins and asin not in meta_by_asin:
            categories = rec.get("categories") or []
            leaf_category = categories[-1] if categories else rec.get("main_category")
            meta_by_asin[asin] = {
                "product_id": asin,
                "title": rec.get("title", ""),
                "brand": rec.get("store", "") or "Unknown",
                "category": leaf_category or rec.get("main_category", "Electronics"),
                "avg_rating_meta": rec.get("average_rating"),
            }
        if len(meta_by_asin) >= len(wanted_meta_asins) or i + 1 >= META_SCAN_CAP:
            break
    print(f"  Matched metadata for {len(meta_by_asin)}/{len(wanted_meta_asins)} candidate ASINs.")
else:
    print(f"\n[2/3] Skipping metadata lookup (cosmetic only, and the slowest/least reliable "
          f"part of this pipeline on a constrained connection). Set PRM_FETCH_META=1 to enable it.")

flagship_meta = meta_by_asin.get(flagship_asin, {
    "product_id": flagship_asin, "title": "", "brand": "Unknown",
    "category": "Electronics", "avg_rating_meta": None,
})
flagship_title_display = flagship_meta["title"] or asin_type_hint.get(flagship_asin, "Unknown")
print(f"  Flagship: \"{flagship_title_display[:70]}\" | Brand: {flagship_meta['brand']} "
      f"| Category: {flagship_meta['category']}")

# ── Select competitors: matching product type (from reviews), different brand ─
# Product type comes from asin_type_hint (review-derived — always
# available). Brand comes from metadata when we have it; if metadata is
# missing for either side we don't exclude on brand (better to risk
# including a same-brand variant than to systematically drop real
# competitors because metadata coverage is incomplete/disabled).
flagship_type = asin_type_hint.get(flagship_asin) or extract_product_type(flagship_meta["title"])
print(f"\n[3/3] Selecting up to {N_COMPETITORS} direct competitors "
      f"(matching product type, different brand where known, ranked by review volume)"
      + (f" — product type: '{flagship_type}'" if flagship_type else " — no product type detected, using volume only")
      + "...")
competitor_asins = []
for asin, count in ranked[1:]:
    cand_type = asin_type_hint.get(asin) or extract_product_type(meta_by_asin.get(asin, {}).get("title", ""))
    if flagship_type and cand_type != flagship_type:
        continue
    cand_brand = meta_by_asin.get(asin, {}).get("brand", "Unknown")
    if cand_brand != "Unknown" and flagship_meta["brand"] != "Unknown" and cand_brand == flagship_meta["brand"]:
        continue  # same brand = not a competitor, likely a variant/accessory
    competitor_asins.append(asin)
    if len(competitor_asins) >= N_COMPETITORS:
        break

if not competitor_asins:
    print("  [WARN] No matching-type competitors found among top candidates — "
          "falling back to next highest-volume products regardless of type.")
    competitor_asins = [a for a, _ in ranked[1:N_COMPETITORS + 1]]

for asin in competitor_asins:
    meta = meta_by_asin.get(asin, {"title": "Unknown", "brand": "Unknown"})
    print(f"  - {asin} | {meta['brand']:20s} | {meta['title'][:55] or asin_type_hint.get(asin, 'Unknown'):55} "
          f"| {asin_counts[asin]:,} reviews")

# ── Build output CSVs from rows already buffered in Pass 1 ──────────────
print(f"\nBuilding output files from already-buffered rows (no further network calls)...")
COL_MAP = {
    "text": "review_text", "title": "review_summary", "rating": "star_rating",
    "timestamp": "unix_time", "parent_asin": "product_id",
}


def build_df(asin_caps):
    """asin_caps: dict of {asin: max_rows_to_keep}"""
    frames = []
    for asin, cap in asin_caps.items():
        rows = rows_by_asin[asin][:cap]
        if not rows:
            continue
        d = pd.DataFrame(rows)
        d = d.rename(columns={k: v for k, v in COL_MAP.items() if k in d.columns})
        d["review_date"] = pd.to_datetime(d["unix_time"], unit="ms", errors="coerce")
        d["brand"] = meta_by_asin.get(asin, {}).get("brand", "Unknown")
        d["product_title"] = meta_by_asin.get(asin, {}).get("title", "")
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["review_text", "review_date"]).copy()
    return out.sort_values("review_date").reset_index(drop=True)


flagship_df   = build_df({flagship_asin: FLAGSHIP_CAP})
competitor_df = build_df({a: COMPETITOR_CAP for a in competitor_asins})

flagship_df.to_csv(OUT_FLAGSHIP, index=False)
competitor_df.to_csv(OUT_COMPETITORS, index=False)

products_df = pd.DataFrame([
    {**meta_by_asin.get(flagship_asin, flagship_meta),
     "scanned_review_count": flagship_count, "sampled_review_count": len(flagship_df),
     "is_flagship": True},
    *[
        {**meta_by_asin.get(a, {"product_id": a, "title": "Unknown", "brand": "Unknown", "category": "Electronics"}),
         "scanned_review_count": asin_counts[a], "sampled_review_count": len(rows_by_asin[a][:COMPETITOR_CAP]),
         "is_flagship": False}
        for a in competitor_asins
    ],
])
products_df.to_csv(OUT_PRODUCTS, index=False)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)
print(f"  Flagship   : {flagship_asin} ({flagship_meta['brand']}) -> {len(flagship_df):,} reviews -> {OUT_FLAGSHIP}")
print(f"  Competitors: {len(competitor_asins)} products -> {len(competitor_df):,} reviews -> {OUT_COMPETITORS}")
print(f"  Product metadata -> {OUT_PRODUCTS}")
print(f"\nNext step: python day1_roberta_sentiment.py")
