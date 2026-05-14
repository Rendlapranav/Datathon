import pandas as pd
import re

CSV_PATH = "data/clean/subcategory_sample.csv"
print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# 1. Map columns to our standard names
COL_MAP = {
    "content": "review_text",
    "title": "review_summary",
    "rating": "star_rating",
    "author": "reviewer_id",
    # product_id is already correct
}
df = df.rename(columns=COL_MAP)

# 2. Extract Date from messy Amazon string
# Target pattern: "June 3, 2019" inside "Reviewed in the United States on June 3, 2019"
def extract_amazon_date(text):
    if not isinstance(text, str): return None
    match = re.search(r'([A-Z][a-z]+ \d{1,2}, \d{4})', text)
    return match.group(1) if match else None

print("Fixing timestamps...")
df['clean_date_str'] = df['timestamp'].apply(extract_amazon_date)
df['review_date'] = pd.to_datetime(df['clean_date_str'], errors='coerce')

# 3. Drop rows with null text (we can't do NLP on empty reviews)
df = df.dropna(subset=['review_text'])

# 4. Save and overwrite
df.to_csv(CSV_PATH, index=False)
print(f"\n[SUCCESS] Dataset patched! Final shape: {df.shape}")
print("Expected Columns: review_text, review_summary, star_rating, product_id, review_date")
print(f"Actual Columns: {[c for c in df.columns if c in ['review_text', 'review_summary', 'star_rating', 'product_id', 'review_date']]}")
