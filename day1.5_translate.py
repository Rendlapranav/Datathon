import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
from langdetect import detect, LangDetectException
from transformers import pipeline

# -- Config --------------------------------------------------------
INPUT_CSV  = "data/clean/day1_reviews_roberta.csv"
OUTPUT_CSV = "data/clean/day1.5_translated.csv"
MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"
BATCH_SIZE = 32     # larger than RoBERTa batches: translation model is lighter

# -- Load ----------------------------------------------------------
print(f"Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
df["review_text"] = df["review_text"].astype(str)
print(f"Loaded {len(df):,} rows")

# -- Language detection --------------------------------------------
# langdetect is non-deterministic by design; results are stable enough
# for filtering purposes. Default to 'en' on any failure so bad/short
# text never crashes the pipeline.
def detect_lang(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"

print("Detecting languages...")
df["lang"] = df["review_text"].apply(detect_lang)

lang_counts = df["lang"].value_counts()
n_foreign   = (df["lang"] != "en").sum()
print(f"Language breakdown:\n{lang_counts.to_string()}")
print(f"\nRows to translate: {n_foreign:,} / {len(df):,}")

# -- Early exit if nothing to translate ---------------------------
if n_foreign == 0:
    print("All reviews already in English. Saving unchanged.")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    raise SystemExit(0)

# -- Build translation pipeline ------------------------------------
# opus-mt-mul-en is a lightweight many-to-English model (~300MB).
# device=-1 forces CPU; this model is fast enough that CPU is fine
# and avoids MPS/CUDA memory issues on student laptops.
print(f"\nLoading translation model: {MODEL_NAME}")
translator = pipeline(
    task       = "translation",
    model      = MODEL_NAME,
    framework  = "pt",
    device     = -1,
    max_length = 512,
)

# -- Translate in batches ------------------------------------------
foreign_mask  = df["lang"] != "en"
foreign_texts = df.loc[foreign_mask, "review_text"].tolist()

print(f"Translating {len(foreign_texts):,} reviews (batch_size={BATCH_SIZE})...")

translated_texts = []
for i in range(0, len(foreign_texts), BATCH_SIZE):
    batch   = foreign_texts[i : i + BATCH_SIZE]
    results = translator(batch, truncation=True, max_length=512)
    translated_texts.extend([r["translation_text"] for r in results])

    # Minimal progress indicator without tqdm dependency
    done = min(i + BATCH_SIZE, len(foreign_texts))
    print(f"  {done:,} / {len(foreign_texts):,}", end="\r")

print()     # newline after \r progress

# -- Merge back into original dataframe ---------------------------
# Write translated text only to the rows that needed it.
# All English rows are untouched.
df.loc[foreign_mask, "review_text"] = translated_texts

# Mark which rows were translated so Day 2 can log/audit if needed
df["was_translated"] = foreign_mask.astype(int)

# -- Save ----------------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nRows translated : {n_foreign:,}")
print(f"Rows unchanged  : {len(df) - n_foreign:,}")
print(f"Saved           : {OUTPUT_CSV}")
print("\nDay 1.5 complete. Pass day1.5_translated.csv to day2_aspects.py")
