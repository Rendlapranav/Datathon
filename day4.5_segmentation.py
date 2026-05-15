
import pandas as pd
import numpy as np
import plotly.express as px
import os

DATA_DIR = "data/clean"
INPUT_CSV = os.path.join(DATA_DIR, "day2_aspects.csv")

def generate_segmentation():
    if not os.path.exists(INPUT_CSV):
        print(f"File not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Define Segments
    # 1. Silent Killers: Stars >= 4, Sentiment < 0
    # 2. Loyalists: Stars >= 4, Sentiment >= 0.5
    # 3. Vocal Detractors: Stars <= 2, Sentiment < 0
    # 4. At Risk: Stars 3, Sentiment < 0
    
    conditions = [
        (df['star_rating'] >= 4) & (df['roberta_score'] < 0),
        (df['star_rating'] >= 4) & (df['roberta_score'] >= 0.5),
        (df['star_rating'] <= 2) & (df['roberta_score'] < 0),
        (df['star_rating'] == 3) & (df['roberta_score'] < 0)
    ]
    choices = ['Silent Killers', 'Loyalists', 'Vocal Detractors', 'At Risk']
    df['segment'] = np.select(conditions, choices, default='Other')
    
    # Save the updated df
    df.to_csv(INPUT_CSV, index=False)
    print("Updated day2_aspects.csv with 'segment' column.")
    
    # Generate summary for dashboard
    seg_summary = df['segment'].value_counts().reset_index()
    seg_summary.columns = ['segment', 'count']
    seg_summary.to_csv(os.path.join(DATA_DIR, "customer_segments.csv"), index=False)
    print("Saved customer_segments.csv")

if __name__ == "__main__":
    generate_segmentation()
