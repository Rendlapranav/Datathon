from datasets import load_dataset
import pandas as pd
from collections import defaultdict

# 1. Load the streaming dataset (using your verified setup)
dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023", 
    "raw_review_Electronics", 
    trust_remote_code=True,
    streaming=True,
    split="full"
)

target_review_count = 500
asin_reviews = defaultdict(list)
target_asin = None

print(f"Scanning stream for a product with {target_review_count} reviews. This may take a minute...")

# 2. Iterate through the stream dynamically
for i, review in enumerate(dataset):
    # The 2023 dataset uses 'parent_asin' to group product variations
    asin = review['parent_asin'] 
    asin_reviews[asin].append(review)
    
    # Check if we hit our target
    if len(asin_reviews[asin]) == target_review_count:
        target_asin = asin
        print(f"\n[SUCCESS] Found Target Product ASIN: {target_asin}")
        print(f"Scanned {i+1} total rows to find it.")
        break
        
    if i % 50000 == 0 and i > 0:
        print(f"Still searching... Scanned {i} rows.")

# 3. Convert strictly those 500 reviews into a Pandas DataFrame
df = pd.DataFrame(asin_reviews[target_asin])

# 4. Save to CSV so you NEVER have to stream this step again
file_name = f"datathon_target_{target_asin}.csv"
df.to_csv(file_name, index=False)

print(f"\nData saved locally to {file_name}")
print("Your Day 1 subset is ready. Here is a preview:")
print(df[['rating', 'title', 'text', 'timestamp']].head())
