import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/clean/day2_aspects.csv")

def segment(row):
    if row["roberta_score"] > 0.2:
        return "Loyal Promoters"
    elif row["is_silent_killer"]:
        return "Silent Quitters"
    else:
        return "Vocal Critics"

df["segment"] = df.apply(segment, axis=1)
counts = df["segment"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%",
        colors=["#67e8f9","#94a3b8","#1e40af"])
plt.title("Customer Segmentation")
plt.savefig("figures/customer_segmentation.png", dpi=300)
df["review_date"] = pd.to_datetime(df["review_date"])
median_date = df["review_date"].median()

early  = df[df["review_date"] <= median_date]
recent = df[df["review_date"] >  median_date]

early_avg  = early.groupby("topic_label")["roberta_score"].mean()
recent_avg = recent.groupby("topic_label")["roberta_score"].mean()

import numpy as np
aspects = early_avg.index
x = np.arange(len(aspects))
w = 0.35

plt.figure(figsize=(10,5))
plt.bar(x - w/2, early_avg,  w, label="Early Reviews",  color="#93c5fd")
plt.bar(x + w/2, recent_avg, w, label="Recent Reviews", color="#f87171")
plt.xticks(x, aspects, rotation=30, ha="right")
plt.ylabel("Mean Sentiment Score")
plt.title("Early vs Recent Sentiment Per Feature")
plt.legend()
plt.tight_layout()
plt.savefig("figures/early_vs_recent.png", dpi=300)
