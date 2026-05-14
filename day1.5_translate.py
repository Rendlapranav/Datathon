import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
from langdetect import detect, LangDetectException, DetectorFactory
from transformers import pipeline

DetectorFactory.seed = 0   # make langdetect deterministic across runs

# -- Config --------------------------------------------------------
INPUT_CSV  = "data/clean/day1_reviews_roberta.csv"
OUTPUT_CSV = "data/clean/day1.5_translated.csv"
MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"
BATCH_SIZE = 32
MAX_CHARS  = 1500   # ~512 tokens; pre-truncate before hitting tokenizer

# -- Load ----------------------------------------------------------
print(f"Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
df["review_text"] = df["review_text"].astype(str)
print(f"Loaded {len(df):,} rows")

# -- Language detection --------------------------------------------
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

# -- Early exit if nothing to translate ----------------------------
if n_foreign == 0:
    print("All reviews already in English. Saving unchanged.")
    df["was_translated"] = False
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    raise SystemExit(0)

# -- Build translation pipeline ------------------------------------
print(f"\nLoading translation model: {MODEL_NAME}")
translator = pipeline(
    task       = "translation",
    model      = MODEL_NAME,
    framework  = "pt",
    device     = -1,
    max_length = 512,
)

# -- Pre-truncate long reviews -------------------------------------
foreign_mask  = df["lang"] != "en"
raw_texts     = df.loc[foreign_mask, "review_text"].tolist()

truncated_count = sum(1 for t in raw_texts if len(t) > MAX_CHARS)
if truncated_count:
    print(f"[WARN] {truncated_count} reviews exceed {MAX_CHARS} chars "
          f"and will be pre-truncated before translation.")

foreign_texts = [
    t[:MAX_CHARS] if len(t) > MAX_CHARS else t
    for t in raw_texts
]

# -- Translate in batches ------------------------------------------
print(f"Translating {len(foreign_texts):,} reviews (batch_size={BATCH_SIZE})...")
translated_texts = []

for i in range(0, len(foreign_texts), BATCH_SIZE):
    batch   = foreign_texts[i : i + BATCH_SIZE]
    results = translator(
        batch,
        truncation                   = True,
        max_length                   = 512,
        clean_up_tokenization_spaces = True,
    )
    translated_texts.extend([r["translation_text"] for r in results])
    done = min(i + BATCH_SIZE, len(foreign_texts))
    print(f"  {done:,} / {len(foreign_texts):,}", end="\r", flush=True)

print()

# -- Merge back ----------------------------------------------------
df.loc[foreign_mask, "review_text"] = translated_texts
df["was_translated"] = foreign_mask.astype(bool)

# -- Save ----------------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nRows translated : {n_foreign:,}")
print(f"Rows unchanged  : {len(df) - n_foreign:,}")
if truncated_count:
    print(f"Rows truncated  : {truncated_count:,} (pre-translation, >{MAX_CHARS} chars)")
print(f"Saved           : {OUTPUT_CSV}")
print("\nDay 1.5 complete. Pass day1.5_translated.csv to day2_aspects.py")
