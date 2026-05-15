"""
day4_ml_models.py
=================
MODEL 1: XGBoost Churn Risk Classifier
  - Predicts is_silent_killer from 12 behavioural features
  - Exports: AUROC, F1, precision, recall, confusion matrix, feature importance

MODEL 2: Isolation Forest Anomaly Detection
  - Flags months with unusual review patterns (review bombing, quality cliff)
  - Exports: monthly_anomalies.csv

MODEL 3: Sentiment Trend Forecasting (polynomial regression + CI)
  - Forecasts next 6 months of sentiment
  - Exports: sentiment_forecast.csv

MODEL 4: Semantic Search Embeddings
  - Encodes all reviews with sentence-transformers
  - Exports: review_embeddings.npy + review_index.csv
  - Dashboard uses cosine similarity at query time (no model reloading)

RUN:  python day4_ml_models.py
INPUT:  data/clean/day2_aspects.csv
OUTPUT: data/clean/ml_metrics.json
        data/clean/feature_importance.csv
        data/clean/churn_predictions.csv
        data/clean/monthly_anomalies.csv
        data/clean/sentiment_forecast.csv
        data/clean/review_embeddings.npy
        data/clean/review_index.csv
        models/churn_classifier.pkl
"""

import os, json, warnings, sys
os.environ["USE_TF"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DATA_DIR   = "data/clean"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

INPUT_CSV = os.path.join(DATA_DIR, "day2_aspects.csv")

print("=" * 60)
print("DAY 4: ML MODEL SUITE")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)
df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
df["review_length"] = df["review_text"].astype(str).apply(len)
print(f"Loaded {len(df):,} reviews")


# ===================================================================
# MODEL 1: XGBoost Churn Risk Classifier
# ===================================================================
print("\n" + "=" * 60)
print("MODEL 1: XGBoost Churn Risk Classifier")
print("=" * 60)

FEATURES = [
    "star_rating", "helpful_votes", "verified_purchase",
    "prob_negative", "prob_neutral", "prob_positive",
    "emo_anger", "emo_sadness", "emo_joy",
    "review_length", "star_normalized", "star_gap",
]

TARGET = "is_silent_killer"

feat_cols = [f for f in FEATURES if f in df.columns]
clf_df = df[feat_cols + [TARGET]].dropna().copy()
clf_df["verified_purchase"] = clf_df["verified_purchase"].astype(int)
clf_df[TARGET] = clf_df[TARGET].astype(int)

print(f"Training data: {len(clf_df):,} rows | Features: {feat_cols}")
print(f"Class balance: {clf_df[TARGET].mean()*100:.1f}% Silent Killers")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

X = clf_df[feat_cols].values
y = clf_df[TARGET].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Try XGBoost first, fall back to RandomForest
try:
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # handle imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model_name = "XGBoost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=42
    )
    model_name = "GradientBoosting"

print(f"Training {model_name}...")
clf.fit(X_train, y_train)

y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

auroc    = roc_auc_score(y_test, y_proba)
f1       = f1_score(y_test, y_pred)
prec     = precision_score(y_test, y_pred, zero_division=0)
rec      = recall_score(y_test, y_pred, zero_division=0)
cm       = confusion_matrix(y_test, y_pred)

print(f"\n  AUROC     : {auroc:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  Confusion Matrix:\n{cm}")

# Cross-validation for robust estimate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for tr, va in cv.split(X, y):
    clf_cv = type(clf)(**clf.get_params())
    clf_cv.fit(X[tr], y[tr])
    cv_scores.append(roc_auc_score(y[va], clf_cv.predict_proba(X[va])[:, 1]))
cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)
print(f"  5-Fold CV AUROC: {cv_mean:.4f} +/- {cv_std:.4f}")

# Feature importance
if hasattr(clf, "feature_importances_"):
    fi = pd.DataFrame({
        "feature": feat_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
else:
    fi = pd.DataFrame({"feature": feat_cols, "importance": [0.0] * len(feat_cols)})

fi.to_csv(os.path.join(DATA_DIR, "feature_importance.csv"), index=False)
print("\n  Top 5 features:")
for _, r in fi.head(5).iterrows():
    bar = "#" * int(r["importance"] * 40)
    print(f"    {r['feature']:25s} {r['importance']:.4f}  {bar}")

# Predictions on full dataset
X_full = clf_df[feat_cols].values
clf_df["churn_risk_score"] = clf.predict_proba(X_full)[:, 1]
clf_df["churn_risk_label"] = clf.predict(X_full)

# Save predictions joined back to full df
pred_df = df.copy()
pred_df = pred_df.merge(
    clf_df[["churn_risk_score", "churn_risk_label"]],
    left_index=True, right_index=True, how="left"
)
pred_df["churn_risk_score"] = pred_df["churn_risk_score"].fillna(0)
pred_df["churn_risk_label"] = pred_df["churn_risk_label"].fillna(0)
pred_df.to_csv(os.path.join(DATA_DIR, "churn_predictions.csv"), index=False)
print(f"\n  Churn predictions saved ({len(pred_df):,} rows)")

# Save model
import pickle
with open(os.path.join(MODELS_DIR, "churn_classifier.pkl"), "wb") as f:
    pickle.dump({"model": clf, "features": feat_cols, "model_name": model_name}, f)
print(f"  Model saved -> models/churn_classifier.pkl")


# ===================================================================
# MODEL 2: Isolation Forest Anomaly Detection
# ===================================================================
print("\n" + "=" * 60)
print("MODEL 2: Isolation Forest Anomaly Detection")
print("=" * 60)

from sklearn.ensemble import IsolationForest

# Build monthly features for anomaly detection
_agg = {
    "mean_sentiment": ("roberta_score", "mean"),
    "review_count":   ("roberta_score", "count"),
    "sk_count":       ("is_silent_killer", "sum"),
    "mean_anger":     ("emo_anger", "mean"),
    "mean_sadness":   ("emo_sadness", "mean"),
}
try:
    monthly = (
        df.set_index("review_date")
        .resample("ME")
        .agg(**_agg)
        .reset_index()
    )
except Exception:
    monthly = (
        df.set_index("review_date")
        .resample("M")
        .agg(**_agg)
        .reset_index()
    )

monthly = monthly[monthly["review_count"] >= 3].copy()
monthly["sk_rate"] = (monthly["sk_count"] / monthly["review_count"] * 100).clip(0, 100)
monthly["urgency"] = monthly["mean_anger"] + monthly["mean_sadness"]

iso_features = ["mean_sentiment", "sk_rate", "urgency", "review_count"]
iso_X = monthly[iso_features].fillna(0).values

iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
monthly["anomaly_score"] = iso.fit_predict(iso_X)  # -1 = anomaly, 1 = normal
monthly["anomaly_prob"]  = -iso.score_samples(iso_X)  # higher = more anomalous
monthly["is_anomaly"]    = monthly["anomaly_score"] == -1

n_anomalies = monthly["is_anomaly"].sum()
print(f"  Detected {n_anomalies} anomalous months out of {len(monthly)}")
print(f"  Top 3 anomalies:")
for _, r in monthly[monthly["is_anomaly"]].sort_values("anomaly_prob", ascending=False).head(3).iterrows():
    print(f"    {r['review_date'].strftime('%Y-%m')}: "
          f"sent={r['mean_sentiment']:.3f}, sk_rate={r['sk_rate']:.1f}%, "
          f"urgency={r['urgency']:.3f}, anomaly_score={r['anomaly_prob']:.3f}")

monthly.to_csv(os.path.join(DATA_DIR, "monthly_anomalies.csv"), index=False)
print(f"  Saved -> data/clean/monthly_anomalies.csv")


# ===================================================================
# MODEL 3: Sentiment Trend Forecasting
# ===================================================================
print("\n" + "=" * 60)
print("MODEL 3: Sentiment Trend Forecasting")
print("=" * 60)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# Use smoothed monthly data
monthly_sorted = monthly.sort_values("review_date").copy()
monthly_sorted = monthly_sorted[monthly_sorted["review_count"] >= 3].copy()

if len(monthly_sorted) >= 6:
    # Convert dates to numeric
    monthly_sorted["t"] = (monthly_sorted["review_date"] - monthly_sorted["review_date"].min()).dt.days

    t_vals = monthly_sorted["t"].values.reshape(-1, 1)
    y_vals = monthly_sorted["mean_sentiment"].values

    # Polynomial degree 3 + Ridge regularisation
    model_forecast = Pipeline([
        ("poly", PolynomialFeatures(degree=3)),
        ("ridge", Ridge(alpha=10.0)),
    ])
    model_forecast.fit(t_vals, y_vals)

    # Extend 6 months forward
    last_date = monthly_sorted["review_date"].max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=6, freq="ME")
    t_last = monthly_sorted["t"].max()
    future_t = np.array([
        t_last + 30 * (i + 1) for i in range(len(future_dates))
    ]).reshape(-1, 1)

    future_pred = model_forecast.predict(future_t)

    # Bootstrap confidence intervals (500 iterations)
    n_boot = 500
    boot_preds = np.zeros((n_boot, len(future_t)))
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        idx = rng.integers(0, len(t_vals), size=len(t_vals))
        m = Pipeline([("poly", PolynomialFeatures(degree=3)), ("ridge", Ridge(alpha=10.0))])
        m.fit(t_vals[idx], y_vals[idx])
        boot_preds[i] = m.predict(future_t)

    ci_lo = np.percentile(boot_preds, 5, axis=0)
    ci_hi = np.percentile(boot_preds, 95, axis=0)

    # Historical fitted values
    hist_fitted = model_forecast.predict(t_vals)
    hist_df = pd.DataFrame({
        "date": monthly_sorted["review_date"].values,
        "actual": y_vals,
        "fitted": hist_fitted,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "is_forecast": False,
    })
    fut_df = pd.DataFrame({
        "date": future_dates,
        "actual": np.nan,
        "fitted": future_pred,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "is_forecast": True,
    })
    forecast_df = pd.concat([hist_df, fut_df], ignore_index=True)
    forecast_df.to_csv(os.path.join(DATA_DIR, "sentiment_forecast.csv"), index=False)

    direction = "improving" if future_pred[-1] > future_pred[0] else "declining"
    print(f"  Forecasting 6 months forward...")
    print(f"  Current sentiment: {y_vals[-1]:.3f}")
    print(f"  6-month forecast: {future_pred[-1]:.3f}  ({direction})")
    print(f"  90% CI: [{ci_lo[-1]:.3f}, {ci_hi[-1]:.3f}]")
    print(f"  Saved -> data/clean/sentiment_forecast.csv")
else:
    print(f"  Not enough monthly data ({len(monthly_sorted)} months). Skipping forecast.")
    forecast_df = pd.DataFrame()


# ===================================================================
# MODEL 4: Semantic Search Embeddings
# ===================================================================
print("\n" + "=" * 60)
print("MODEL 4: Semantic Review Search (sentence-transformers)")
print("=" * 60)

try:
    from sentence_transformers import SentenceTransformer

    # Use a small, fast model
    EMB_MODEL = "all-MiniLM-L6-v2"
    print(f"  Loading {EMB_MODEL}...")
    emb_model = SentenceTransformer(EMB_MODEL)

    # Use a representative subset (up to 2000 reviews) to keep files small
    emb_df = df[["review_text", "topic_label", "star_rating",
                  "roberta_score", "is_silent_killer",
                  "revenue_at_risk"]].copy()
    emb_df = emb_df.dropna(subset=["review_text"]).head(2000).reset_index(drop=True)

    texts = emb_df["review_text"].astype(str).tolist()
    print(f"  Encoding {len(texts):,} reviews...")
    embeddings = emb_model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity = dot product
    )

    np.save(os.path.join(DATA_DIR, "review_embeddings.npy"), embeddings)
    emb_df.to_csv(os.path.join(DATA_DIR, "review_index.csv"), index=False)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Saved -> data/clean/review_embeddings.npy")
    print(f"  Saved -> data/clean/review_index.csv")
    sem_search_ok = True

except Exception as e:
    print(f"  Skipping semantic search: {e}")
    sem_search_ok = False


# ===================================================================
# EXPORT: Master Metrics JSON
# ===================================================================
print("\n" + "=" * 60)
print("EXPORTING MASTER METRICS")
print("=" * 60)

ml_metrics = {
    "churn_classifier": {
        "model_name"  : model_name,
        "features"    : feat_cols,
        "n_train"     : int(len(X_train)),
        "n_test"      : int(len(X_test)),
        "auroc"       : round(auroc,  4),
        "f1"          : round(f1,     4),
        "precision"   : round(prec,   4),
        "recall"      : round(rec,    4),
        "cv_auroc_mean": round(cv_mean, 4),
        "cv_auroc_std" : round(cv_std,  4),
        "confusion_matrix": cm.tolist(),
        "top_features": fi.head(5).to_dict(orient="records"),
    },
    "anomaly_detection": {
        "model"           : "IsolationForest",
        "n_months_analyzed": int(len(monthly)),
        "n_anomalies"     : int(n_anomalies),
        "contamination"   : 0.10,
    },
    "sentiment_forecast": {
        "model"            : "Polynomial Ridge Regression (degree=3)",
        "horizon_months"   : 6,
        "available"        : len(forecast_df) > 0,
        "forecast_direction": direction if len(monthly_sorted) >= 6 else "N/A",
    },
    "semantic_search": {
        "model"     : "all-MiniLM-L6-v2",
        "n_indexed" : int(len(texts)) if sem_search_ok else 0,
        "available" : sem_search_ok,
    },
}

with open(os.path.join(DATA_DIR, "ml_metrics.json"), "w") as f:
    json.dump(ml_metrics, f, indent=2)

print(f"\nML Metrics saved -> data/clean/ml_metrics.json")
print("\n" + "=" * 60)
print("DAY 4 COMPLETE")
print("=" * 60)
print(f"  {model_name} AUROC  : {auroc:.4f}")
print(f"  5-Fold CV AUROC     : {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"  Anomalies detected  : {n_anomalies}")
print(f"  Semantic search     : {'OK' if sem_search_ok else 'SKIP'}")
print(f"\nNext: python -m streamlit run gui/app.py")
