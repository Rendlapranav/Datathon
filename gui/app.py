import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -- Page Configuration ------------------------------------------------
st.set_page_config(
    page_title="Product Rescue Mission | CEO Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;700;800;900&display=swap');
    
    /* Global Base */
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    .stApp {
        background-color: #050505;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(76, 158, 255, 0.03), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(0, 255, 179, 0.03), transparent 25%);
    }
    
    /* Hide top padding and header */
    header {visibility: hidden;}
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
    
    /* Premium Glassmorphism KPI Cards */
    .premium-kpi-container {
        display: flex;
        gap: 24px;
        margin-top: 20px;
        margin-bottom: 40px;
    }
    .premium-card {
        flex: 1;
        background: rgba(15, 20, 30, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 28px 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .premium-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* Card Glow Tops */
    .card-red::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background: linear-gradient(90deg, transparent, #FF4C4C, transparent); opacity: 0.8; }
    .card-green::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background: linear-gradient(90deg, transparent, #00FFB3, transparent); opacity: 0.8; }
    .card-blue::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background: linear-gradient(90deg, transparent, #4C9EFF, transparent); opacity: 0.8; }
    .card-gold::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background: linear-gradient(90deg, transparent, #FFD700, transparent); opacity: 0.8; }

    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 6px;
        letter-spacing: -1px;
    }
    
    /* Text Gradients */
    .val-red { background: linear-gradient(135deg, #FF4C4C 20%, #FF8F8F 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .val-green { background: linear-gradient(135deg, #00FFB3 20%, #4C9EFF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .val-white { color: #f8fafc; }

    .kpi-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 400;
    }
    
    /* SO WHAT Callouts */
    .so-what {
        background: linear-gradient(90deg, rgba(0, 255, 179, 0.05), transparent);
        border-left: 4px solid #00FFB3; 
        border-radius: 0 16px 16px 0;
        padding: 20px 24px; 
        margin: 20px 0;
    }
    .so-what-title { color: #00FFB3; font-weight: 800; font-family: 'Outfit', sans-serif; font-size: 1rem; margin-bottom: 6px; letter-spacing: 1px; }
    
    /* Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background: rgba(15, 23, 42, 0.4);
        border-radius: 12px 12px 0 0;
        border: 1px solid rgba(255,255,255,0.05);
        border-bottom: none;
        color: #94a3b8;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.05) !important;
        color: #f8fafc !important;
        border-top: 2px solid #4C9EFF !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.03);
    }

    /* Titles */
    .dash-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem; font-weight: 900;
        background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -1px;
    }
    .dash-subtitle {
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 20px;
    }

    /* ===== HERO SECTION ===== */
    .hero-stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 28px;
        margin-bottom: 36px;
    }
    .hero-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.07);
        padding: 44px 36px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .hero-card:hover { transform: translateY(-6px); }
    .hero-card-red  { box-shadow: 0 0 60px rgba(255,76,76,0.20); border-color: rgba(255,76,76,0.30); }
    .hero-card-gold { box-shadow: 0 0 60px rgba(255,215,0,0.18); border-color: rgba(255,215,0,0.28); }
    .hero-card-green{ box-shadow: 0 0 60px rgba(0,255,179,0.18); border-color: rgba(0,255,179,0.28); }
    .hero-badge {
        display: inline-block;
        font-size: 0.62rem; font-weight: 800;
        letter-spacing: 2.5px; text-transform: uppercase;
        padding: 4px 14px; border-radius: 20px;
        background: rgba(255,255,255,0.05);
        color: #64748b; margin-bottom: 20px;
    }
    .hero-number {
        font-family: 'Outfit', sans-serif;
        font-size: 4.8rem; font-weight: 900;
        line-height: 1; letter-spacing: -2px;
        margin-bottom: 14px;
    }
    .hero-sub {
        font-size: 1rem; color: #94a3b8;
        font-weight: 400; line-height: 1.5;
    }
    .hero-tagline {
        text-align: center;
        font-family: 'Outfit', sans-serif;
        font-style: italic;
        font-size: 1.55rem; font-weight: 500;
        color: #e2e8f0;
        margin: 8px 0 36px;
        opacity: 0.9;
        letter-spacing: -0.3px;
    }
    .hero-timeline {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: nowrap;
        padding: 18px 28px;
        background: rgba(255,255,255,0.02);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 36px;
        gap: 0;
    }
    .timeline-step {
        font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.5px;
        padding: 8px 16px; border-radius: 8px;
        white-space: nowrap; color: #00FFB3;
        background: rgba(0,255,179,0.07);
        border: 1px solid rgba(0,255,179,0.18);
        text-shadow: 0 0 14px rgba(0,255,179,0.5);
    }
    .timeline-arrow {
        color: #334155; font-size: 1.1rem;
        padding: 0 8px; flex-shrink: 0;
    }
    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin: 28px 0;
    }
    </style>
""", unsafe_allow_html=True)

# -- Constants (must mirror day2 / day3) --------------------------------
ACTIVE_EMOTIONS     = ["anger", "sadness", "joy"]
DISPLAY_LABELS      = {"anger": "Anger", "sadness": "Disappointment", "joy": "Joy"}
FREQ_THRESHOLD      = 5.0
INTENSITY_THRESHOLD = 0.40

DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="Inter"),
    margin=dict(l=40, r=40, t=50, b=40),
)

DATA_DIR = "data/clean"

# -- Load Assets --------------------------------------------------------
@st.cache_data
def load_data():
    try:
        with open(os.path.join(DATA_DIR, "ceo_summary.json"), "r") as f:
            summary = json.load(f)
        quotes_df  = pd.read_csv(os.path.join(DATA_DIR, "day2_representative_quotes.csv"))
        aspects_df = pd.read_csv(os.path.join(DATA_DIR, "day2_aspects.csv"))
        return summary, quotes_df, aspects_df
    except FileNotFoundError as e:
        st.error(f"❌ Data files not found: {e}\n\nRun the pipeline: `python run.sh` or `./run.ps1`")
        st.stop()

summary, quotes_df, aspects_df = load_data()

@st.cache_data
def load_ml_data():
    ml_metrics = {}
    if os.path.exists(os.path.join(DATA_DIR, "ml_metrics.json")):
        with open(os.path.join(DATA_DIR, "ml_metrics.json"), "r") as f:
            ml_metrics = json.load(f)
    feat_imp = pd.read_csv(os.path.join(DATA_DIR, "feature_importance.csv")) if os.path.exists(os.path.join(DATA_DIR, "feature_importance.csv")) else pd.DataFrame()
    anomalies = pd.read_csv(os.path.join(DATA_DIR, "monthly_anomalies.csv")) if os.path.exists(os.path.join(DATA_DIR, "monthly_anomalies.csv")) else pd.DataFrame()
    forecast = pd.read_csv(os.path.join(DATA_DIR, "sentiment_forecast.csv")) if os.path.exists(os.path.join(DATA_DIR, "sentiment_forecast.csv")) else pd.DataFrame()
    return ml_metrics, feat_imp, anomalies, forecast

ml_metrics, feat_imp, anomalies, forecast = load_ml_data()

@st.cache_data
def load_competitor_data():
    comp_csv  = os.path.join(DATA_DIR, "competitor_comparison.csv")
    comp_json = os.path.join(DATA_DIR, "competitor_summary.json")
    comparison = pd.read_csv(comp_csv) if os.path.exists(comp_csv) else pd.DataFrame()
    summary_j  = {}
    if os.path.exists(comp_json):
        with open(comp_json, "r") as f:
            summary_j = json.load(f)
    return comparison, summary_j

competitor_comparison, competitor_summary = load_competitor_data()

@st.cache_data
def load_identity_and_ai_summary():
    identity_path = os.path.join(DATA_DIR, "product_identity.json")
    ai_summary_path = os.path.join(DATA_DIR, "ai_complaint_summary.json")
    identity = {}
    ai_summary = {}
    if os.path.exists(identity_path):
        with open(identity_path, "r") as f:
            identity = json.load(f)
    if os.path.exists(ai_summary_path):
        with open(ai_summary_path, "r") as f:
            ai_summary = json.load(f)
    return identity, ai_summary

identity, ai_summary = load_identity_and_ai_summary()
flagship_identity = identity.get("flagship", {})
flagship_brand = flagship_identity.get("brand", "Unknown")
flagship_title = flagship_identity.get("title") or ""
if flagship_brand not in ("Unknown", "") and flagship_title:
    product_display_name = f"{flagship_brand} {flagship_title}"
elif flagship_brand not in ("Unknown", ""):
    product_display_name = f"{flagship_brand} product"
else:
    product_display_name = "the flagship product"

df_valid = aspects_df[aspects_df["topic_id"] != -1].copy()
df_valid["review_date"] = pd.to_datetime(df_valid["review_date"], errors="coerce")
df_valid = df_valid.dropna(subset=["review_date"])

missing = [f"emo_{e}" for e in ACTIVE_EMOTIONS if f"emo_{e}" not in df_valid.columns]
if missing:
    st.error(f"Expected emotion columns not found: {missing}. Re-run day2_aspects.py.")
    st.stop()

financials = summary.get("financials", {})
sim        = summary.get("roi_simulation", {})
total_rev  = summary.get("total_reviews", 0)
sk_count   = financials.get("silent_killers", 0)
rar_total  = financials.get("revenue_at_risk_usd", 0.0)

# -- Sidebar ------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 4px;">
      <div style="font-family:'Outfit',sans-serif; font-size:1.3rem; font-weight:900;
                  background:linear-gradient(135deg,#00FFB3,#4C9EFF);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  letter-spacing:-0.5px;">🚨 Mission Control</div>
      <div style="font-size:0.72rem; color:#475569; margin-top:2px; letter-spacing:1px;">PRODUCT RESCUE MISSION</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**Product:** {product_display_name}")
    st.markdown(f"**Category:** `Consumer Electronics`")
    st.markdown(f"**Reviews Analyzed:** `{summary.get('total_reviews', 0):,}`")
    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(255,76,76,0.08); border:1px solid rgba(255,76,76,0.2);
                border-radius:12px; padding:14px 16px; margin-bottom:12px;">
      <div style="font-size:0.7rem; color:#64748b; font-weight:700; text-transform:uppercase;
                  letter-spacing:1.5px; margin-bottom:4px;">Revenue at Risk</div>
      <div style="font-family:'Outfit',sans-serif; font-size:2rem; font-weight:900; color:#FF4C4C;">
        ${rar_total:,.0f}</div>
    </div>
    <div style="background:rgba(255,76,76,0.05); border:1px solid rgba(255,76,76,0.1);
                border-radius:12px; padding:14px 16px; margin-bottom:16px;">
      <div style="font-size:0.7rem; color:#64748b; font-weight:700; text-transform:uppercase;
                  letter-spacing:1.5px; margin-bottom:4px;">Silent Killers Detected</div>
      <div style="font-family:'Outfit',sans-serif; font-size:2rem; font-weight:900; color:#FF8F8F;">
        {sk_count:,}</div>
    </div>
    """.format(rar_total=rar_total, sk_count=sk_count), unsafe_allow_html=True)
    st.info(
        "**Silent Killer** — A 4-5★ reviewer whose text signals hidden anger. "
        "They churn silently, without filing a support ticket."
    )
    with st.expander("⚙️ Technical stack (for the data team)"):
        st.caption("• XLM-RoBERTa → Sentiment scoring")
        st.caption("• Helsinki-NLP → Multilingual translation")
        st.caption("• BERTopic → Aspect mining")
        st.caption("• mDeBERTa → Emotion detection")
        st.caption("• XGBoost → Churn classification")
        st.caption("• Isolation Forest → Anomaly detection")
        st.caption("• Sentence-Transformers → Semantic search")
        st.caption("• DistilBART → AI complaint summarization")
        src = financials.get("churn_multiplier_source", "")
        if src:
            st.caption(f"RAR basis: {src}")

# -- Hero Screen --------------------------------------------------------
# Dynamic hero numbers
_high_star = aspects_df[aspects_df["star_rating"] >= 4] if "star_rating" in aspects_df.columns else aspects_df
_sk_high   = _high_star[_high_star["is_silent_killer"] == True] if "is_silent_killer" in _high_star.columns else pd.DataFrame()
hero_sk_pct = round(len(_sk_high) / max(len(_high_star), 1) * 100, 1)

_pm_hero = pd.DataFrame(summary.get("priority_matrix", []))
if not _pm_hero.empty and "frequency_pct" in _pm_hero.columns:
    _freq_total  = max(_pm_hero["frequency_pct"].sum(), 1)
    _top3_weight = _pm_hero.nlargest(3, "emotional_intensity")["frequency_pct"].sum()
    hero_recovery = int(rar_total * (_top3_weight / _freq_total))
else:
    hero_recovery = int(rar_total * 0.85)

st.markdown('<div class="dash-title">Product Rescue Mission</div>', unsafe_allow_html=True)
st.markdown(f'<div class="dash-subtitle">{product_display_name} · Electronics · AI-Powered Business Intelligence</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="hero-stat-grid">
  <div class="hero-card hero-card-red">
    <div class="hero-badge">The Villain</div>
    <div class="hero-number" style="color:#FF4C4C;">{hero_sk_pct}%</div>
    <div class="hero-sub">of 4&#8211;5&#9733; reviewers are<br>secretly frustrated</div>
  </div>
  <div class="hero-card hero-card-gold">
    <div class="hero-badge">The Stakes</div>
    <div class="hero-number" style="color:#FFD700;">${rar_total:,.0f}</div>
    <div class="hero-sub">in annual revenue<br>currently at risk</div>
  </div>
  <div class="hero-card hero-card-green">
    <div class="hero-badge">The Recovery</div>
    <div class="hero-number" style="color:#00FFB3;">${hero_recovery:,.0f}</div>
    <div class="hero-sub">recoverable with<br>3 targeted fixes</div>
  </div>
</div>

<div class="hero-tagline">
  &ldquo;Star ratings said everything was fine. Our AI found the truth.&rdquo;
</div>
""", unsafe_allow_html=True)

with st.expander("See how this was built"):
    st.markdown("""
<div class="hero-timeline">
  <div class="timeline-step">Data Ingestion</div>
  <div class="timeline-arrow">&rarr;</div>
  <div class="timeline-step">Sentiment Analysis</div>
  <div class="timeline-arrow">&rarr;</div>
  <div class="timeline-step">Silent Killer Detection</div>
  <div class="timeline-arrow">&rarr;</div>
  <div class="timeline-step">Root Cause Mining</div>
  <div class="timeline-arrow">&rarr;</div>
  <div class="timeline-step">Revenue Quantification</div>
  <div class="timeline-arrow">&rarr;</div>
  <div class="timeline-step">Strategic Roadmap</div>
</div>
""", unsafe_allow_html=True)

sk_rate   = (sk_count / total_rev * 100) if total_rev > 0 else 0
sim_boost = sim.get('improvement_pct_high', 0)

# -- Tabs ---------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Executive Summary",
    "🎯 Priority Matrix & Roadmap",
    "🧠 Emotion Deep Dive",
    "🗣️ Voice of Customer",
    "📈 Temporal Trend",
    "🔍 Silent Killer Drill-Down",
    "🤖 Advanced ML Analytics",
    "📊 Competitor Benchmark",
])

# ── Tab 0: Executive Summary ──────────────────────────────────────────
with tab0:
    st.subheader(f"🏠 {product_display_name}")
    st.markdown(
        "**What we did:** read every review of this product (and its direct competitors), "
        "scored the real sentiment behind the words — not just the star rating — and found "
        "customers who post 4-5★ reviews while their words signal quiet frustration. "
        "**Why it matters for you:** stars alone would have told you this product is fine. "
        "They're not telling you the whole story."
    )

    st.markdown("---")
    id_col1, id_col2, id_col3 = st.columns(3)
    with id_col1:
        st.markdown("##### 📦 This Product")
        st.markdown(f"**{product_display_name}**")
        src = flagship_identity.get("brand_source", "")
        if src and src != "metadata":
            st.caption(f"Brand identified from review text ({src})")
    with id_col2:
        st.markdown("##### 🏢 Other Products From This Company")
        siblings = identity.get("sibling_products", [])
        if siblings:
            for s in siblings:
                st.markdown(f"- {s}")
        else:
            st.caption(identity.get("sibling_products_note",
                       "None found in this scan — this appears to be a single-line seller."))
    with id_col3:
        st.markdown("##### ⚔️ Direct Competitors Benchmarked")
        comps = identity.get("competitors", [])
        if comps:
            for c in comps:
                st.markdown(f"- **{c.get('brand', 'Unknown')}**")
        else:
            st.caption("No competitor data available — run the pipeline's discovery step.")

    st.markdown("---")
    st.markdown("### The Numbers That Matter")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Revenue at Risk", f"${rar_total:,.0f}")
    m2.metric("Silent Killer Rate", f"{sk_rate:.1f}%", help="Share of analyzed reviews flagged as hidden frustration")
    comp_sent_gap = None
    if competitor_summary.get("competitor_avg_sentiment") is not None:
        comp_sent_gap = competitor_summary["flagship_avg_sentiment"] - competitor_summary["competitor_avg_sentiment"]
        m3.metric("Sentiment vs. Competitors", f"{comp_sent_gap:+.3f}",
                   delta=f"{comp_sent_gap:+.3f}", delta_color="inverse")
    else:
        m3.metric("Sentiment vs. Competitors", "N/A")
    m4.metric("Top Fix ROI", f"+{sim.get('improvement_pct_high', 0):.0f}%",
               help=f"Sentiment lift if '{sim.get('target_aspect', 'top issue')}' is fixed")

    if competitor_summary.get("insight"):
        st.markdown(
            f'<div class="so-what"><div class="so-what-title">SO WHAT?</div>'
            f'{competitor_summary["insight"]}</div>', unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🤖 AI Summary of Customer Complaints")
    st.caption(
        f"Generated from {ai_summary.get('n_complaints_summarized', 0):,} negative/frustrated reviews — "
        "read this instead of scrolling raw quotes."
    )
    overall_ai = ai_summary.get("overall", "")
    if overall_ai:
        st.markdown(
            f'<div class="so-what"><div class="so-what-title">OVERALL</div>{overall_ai}</div>',
            unsafe_allow_html=True,
        )
    by_aspect_ai = ai_summary.get("by_aspect", {})
    if by_aspect_ai:
        acols = st.columns(len(by_aspect_ai))
        for col, (aspect, text) in zip(acols, by_aspect_ai.items()):
            with col:
                st.markdown(f"""
                <div style="background:#0f172a;padding:18px;border-radius:12px;border:1px solid #1e293b;
                            border-top:3px solid #FF8F8F;height:100%;">
                    <div style="font-weight:700;color:#FF8F8F;margin-bottom:8px;font-size:0.9rem;">{aspect}</div>
                    <div style="font-size:0.85rem;color:#cbd5e1;line-height:1.5;">{text}</div>
                </div>""", unsafe_allow_html=True)
    elif not overall_ai:
        st.info("Run `day3.6_ai_complaint_summary.py` to generate this section.")

    st.markdown("---")
    st.success(f"**Bottom line:** {sim.get('slide_pitch', 'Run the pipeline to generate a recommendation.')}")
    st.caption(
        "Explore the tabs above for the supporting detail behind every number here — "
        "Priority Matrix, Competitor Benchmark, and the Silent Killer Drill-Down."
    )

# ── Tab 1: Priority Matrix ─────────────────────────────────────────────
with tab1:
    st.subheader("Actionable Priorities")
    col_matrix, col_roadmap = st.columns([3, 2])

    with col_matrix:
        pm_data = pd.DataFrame(summary.get("priority_matrix", []))
        if not pm_data.empty:
            max_x = max(pm_data["frequency_pct"].max() * 1.2,      FREQ_THRESHOLD * 2)
            max_y = max(pm_data["emotional_intensity"].max() * 1.2, INTENSITY_THRESHOLD * 1.5)

            fig_matrix = px.scatter(
                pm_data,
                x="frequency_pct", y="emotional_intensity",
                size="frequency_pct", color="emotional_intensity",
                color_continuous_scale="RdYlGn_r",
                hover_name="topic_label",
                hover_data={"topic_label": False, "quadrant": True,
                            "frequency_pct": ":.1f", "emotional_intensity": ":.2f"},
                text="topic_label",
                title="CEO Priority Matrix",
            )
            fig_matrix.update_traces(
                textposition="top center",
                marker=dict(line=dict(width=1.5, color="#00FFB3")),
            )
            fig_matrix.add_hline(y=INTENSITY_THRESHOLD, line_dash="dash", line_color="#FF4C4C", opacity=0.6,
                                  annotation_text="Intensity threshold", annotation_font_color="#FF4C4C")
            fig_matrix.add_vline(x=FREQ_THRESHOLD, line_dash="dash", line_color="#FF4C4C", opacity=0.6,
                                  annotation_text="Freq threshold", annotation_font_color="#FF4C4C")
            fig_matrix.update_layout(**DARK,
                xaxis_title="Complaint Frequency (% of reviews)",
                yaxis_title="Emotional Intensity (Anger + Disappointment)",
                coloraxis_colorbar=dict(title="Pain Level"),
            )
            fig_matrix.update_xaxes(range=[0, max_x], gridcolor="#1e293b")
            fig_matrix.update_yaxes(range=[0, max_y], gridcolor="#1e293b")
            st.plotly_chart(fig_matrix, use_container_width=True)

    with col_roadmap:
        st.markdown("### 🗺️ Strategic Roadmap")
        roadmap_df = pd.DataFrame(summary.get("strategic_roadmap", []))
        if not roadmap_df.empty:
            def _highlight(val):
                return "background-color:#7f1d1d;color:#fca5a5" if val == "Urgent Redesign" else ""
            st.dataframe(
                roadmap_df[["tier", "aspect", "action", "quadrant"]].style.map(_highlight, subset=["quadrant"]),
                hide_index=True, height=300,
            )
        st.success(f"**Bottom Line:** {sim.get('slide_pitch', '')}")

    # ROI Scenario Calculator
    st.markdown("---")
    st.markdown("### 🎛️ Interactive ROI Scenario Calculator")
    st.caption("Simulate: *If we improve [aspect] sentiment by X%, how much revenue do we recover?*")
    pm_full = pd.DataFrame(summary.get("priority_matrix", []))
    if not pm_full.empty:
        top3 = pm_full.sort_values("emotional_intensity", ascending=False).head(3).reset_index(drop=True)
        freq_sum = pm_full["frequency_pct"].sum()
        rcols = st.columns(len(top3))
        for i, row in top3.iterrows():
            with rcols[i]:
                pct = st.slider(
                    f"Fix **{row['topic_label']}** by",
                    min_value=0, max_value=100, value=30, step=5,
                    key=f"roi_{i}", format="%d%%",
                )
                weight    = row["frequency_pct"] / freq_sum if freq_sum > 0 else 1 / len(top3)
                recovered = rar_total * (pct / 100) * weight
                st.markdown(f"""
                <div class="roi-card">
                  <div class="roi-amount">${recovered:,.0f}</div>
                  <div class="roi-label">ARR rescued<br>{row['topic_label']}</div>
                </div>""", unsafe_allow_html=True)

# ── Tab 2: Emotion Deep Dive ──────────────────────────────────────────
with tab2:
    st.subheader("Emotion Signal Analysis")
    st.caption("Which features trigger anger vs. disappointment vs. joy.")
    with st.expander("Methodology"):
        st.caption(
            f"Active signals: **{', '.join(DISPLAY_LABELS.values())}**  |  "
            "Disgust, fear, surprise, neutral excluded — no actionable hardware signal."
        )

    col_full, col_urgency = st.columns(2)

    with col_full:
        if "dominant_emotion_display" in df_valid.columns:
            heat_data = (df_valid.groupby(["topic_label", "dominant_emotion_display"])
                         .size().unstack(fill_value=0))
            heat_pct = heat_data.div(heat_data.sum(axis=1), axis=0) * 100
            display_order = [DISPLAY_LABELS[e] for e in ACTIVE_EMOTIONS if DISPLAY_LABELS[e] in heat_pct.columns]
            heat_pct = heat_pct[display_order].round(1)
        else:
            heat_rows = {}
            for label, grp in df_valid.groupby("topic_label"):
                heat_rows[label] = {DISPLAY_LABELS[e]: grp[f"emo_{e}"].mean() * 100 for e in ACTIVE_EMOTIONS}
            heat_pct = pd.DataFrame(heat_rows).T.round(1)

        fig_heat = px.imshow(heat_pct, color_continuous_scale="RdYlGn_r",
                             text_auto=True, aspect="auto",
                             title="Product Aspect × Dominant Emotion (%)")
        fig_heat.update_layout(**DARK)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_urgency:
        urgency_df = (df_valid.groupby("topic_label")[["emo_anger", "emo_sadness"]]
                      .mean() * 100).round(1)
        urgency_df.columns = ["Anger", "Disappointment"]
        fig_urg = px.imshow(urgency_df, color_continuous_scale="YlOrRd",
                            text_auto=True, aspect="auto",
                            title="🚨 Urgency: Mean Anger & Disappointment (%)")
        fig_urg.update_layout(**DARK)
        st.plotly_chart(fig_urg, use_container_width=True)

    # Stacked bar
    ae = df_valid.groupby("topic_label").agg(
        mean_anger=("emo_anger", "mean"), mean_sadness=("emo_sadness", "mean")
    ).reset_index().sort_values("mean_anger")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(y=ae["topic_label"], x=ae["mean_anger"],  name="Anger",
                             orientation="h", marker_color="#FF4C4C"))
    fig_bar.add_trace(go.Bar(y=ae["topic_label"], x=ae["mean_sadness"], name="Disappointment",
                             orientation="h", marker_color="#FFD700"))
    fig_bar.update_layout(**DARK, barmode="stack", title="Emotional Pain by Aspect",
                          xaxis_title="Mean Probability")
    fig_bar.update_xaxes(gridcolor="#1e293b")
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Tab 3: Voice of Customer ──────────────────────────────────────────
with tab3:
    st.subheader("🗣️ Voice of the Customer")
    st.markdown(
        "Auto-extracted quotes from the mathematical center of each topic cluster. "
        "These ground AI findings in actual human language."
    )
    aspects         = quotes_df["topic_label"].unique()
    selected_aspect = st.selectbox("Filter by Aspect:", aspects, label_visibility="collapsed")
    filtered_quotes = quotes_df[quotes_df["topic_label"] == selected_aspect].head(3)

    cols = st.columns(3)
    for col, (_, row) in zip(cols, filtered_quotes.iterrows()):
        qt = str(row["quote_text"]).strip()
        if len(qt) > 300:
            qt = qt[:297] + "..."
        with col:
            st.markdown(f"""
            <div style="background:#0f172a;padding:24px;border-radius:12px;border:1px solid #1e293b;
                        border-top:4px solid #4C9EFF;min-height:240px;display:flex;
                        flex-direction:column;justify-content:space-between;">
                <div>
                    <div style="font-size:30px;color:#4C9EFF;line-height:1;margin-bottom:10px;">&#10077;</div>
                    <p style="font-style:italic;font-size:14px;color:#cbd5e1;line-height:1.6;">{qt}</p>
                </div>
                <div style="text-align:right;border-top:1px solid #1e293b;margin-top:12px;padding-top:10px;">
                    <span style="font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;
                                 letter-spacing:1px;">Verified Analysis</span>
                </div>
            </div>""", unsafe_allow_html=True)

# ── Tab 4: Temporal Trend ─────────────────────────────────────────────
with tab4:
    st.subheader("Sentiment Drift Over Time")
    st.markdown("*When did the product start failing customers?*")

    if "roberta_score" not in df_valid.columns:
        st.warning("Column `roberta_score` not found. Re-run the pipeline from Day 1.")
    elif df_valid["review_date"].isna().all():
        st.warning("No valid review dates found.")
    else:
        # ── Build monthly aggregates ──────────────────────────────────
        _agg = {
            "mean_sentiment": ("roberta_score", "mean"),
            "review_count":   ("roberta_score", "count"),
        }
        if "is_silent_killer" in df_valid.columns:
            _agg["sk_count"] = ("is_silent_killer", "sum")
        else:
            _agg["sk_count"] = ("roberta_score", "count")

        try:
            monthly = (df_valid.set_index("review_date").resample("ME").agg(**_agg).reset_index())
        except ValueError:
            monthly = (df_valid.set_index("review_date").resample("M").agg(**_agg).reset_index())

        # Filter to months with enough reviews to avoid noise spikes
        MIN_MONTHLY = 3
        monthly = monthly[monthly["review_count"] >= MIN_MONTHLY].copy()
        monthly["sk_rate"] = (monthly["sk_count"] / monthly["review_count"] * 100).clip(0, 100)

        # Rolling average (window=5 for smoother trend)
        monthly["smooth_sentiment"] = (
            monthly["mean_sentiment"]
            .rolling(window=5, center=True, min_periods=2)
            .mean()
        )
        monthly["smooth_sk_rate"] = (
            monthly["sk_rate"]
            .rolling(window=5, center=True, min_periods=2)
            .mean()
        )

        # Changepoint: biggest MoM drop in the SMOOTHED signal (avoids noise artifacts)
        monthly["sentiment_delta"] = monthly["smooth_sentiment"].diff()
        # Only consider months with enough data for a meaningful changepoint
        stable = monthly[monthly["review_count"] >= 5]
        if not stable.empty:
            cp_idx  = stable["sentiment_delta"].idxmin()
            cp_date = monthly.loc[cp_idx, "review_date"]
            cp_drop = abs(monthly.loc[cp_idx, "sentiment_delta"])
        else:
            cp_idx = monthly["sentiment_delta"].idxmin()
            cp_date = monthly.loc[cp_idx, "review_date"] if not monthly.empty else None
            cp_drop = abs(monthly.loc[cp_idx, "sentiment_delta"]) if cp_date is not None else 0

        # ── CHART 1: Sentiment Trend (filled area) ───────────────────
        fig_trend = go.Figure()

        # Zero baseline fill
        fig_trend.add_hrect(
            y0=-1, y1=0,
            fillcolor="rgba(255,76,76,0.04)", line_width=0,
        )
        fig_trend.add_hrect(
            y0=0, y1=1,
            fillcolor="rgba(0,255,179,0.04)", line_width=0,
        )

        # Raw monthly sentiment — subtle dots only
        fig_trend.add_trace(go.Scatter(
            x=monthly["review_date"], y=monthly["mean_sentiment"],
            mode="markers",
            name="Monthly Avg",
            marker=dict(color="#00FFB3", size=4, opacity=0.35),
            hovertemplate="<b>%{x|%b %Y}</b><br>Raw: %{y:.3f}<extra></extra>",
        ))

        # Smoothed sentiment — filled area + line (the hero trace)
        fig_trend.add_trace(go.Scatter(
            x=monthly["review_date"], y=monthly["smooth_sentiment"],
            mode="lines",
            name="Smoothed Trend",
            line=dict(color="#00FFB3", width=3),
            fill="tozeroy",
            fillcolor="rgba(0,255,179,0.08)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Smoothed: %{y:.3f}<extra></extra>",
        ))

        # Changepoint annotation
        if cp_date is not None:
            fig_trend.add_vline(
                x=cp_date.timestamp() * 1000,
                line_dash="dash", line_color="rgba(255,215,0,0.7)", line_width=1.5,
                annotation_text="Inflection Point",
                annotation_font_color="#FFD700",
                annotation_font_size=11,
                annotation_position="top right",
            )

        # Zero sentiment reference
        fig_trend.add_hline(y=0, line_color="rgba(255,255,255,0.12)", line_width=1)

        fig_trend.update_layout(
            **DARK,
            title=dict(text="Sentiment Score Over Time (5-Month Smoothed)", font=dict(size=15)),
            height=340,
            xaxis=dict(title="", gridcolor="#1e293b", showgrid=True),
            yaxis=dict(
                title="Sentiment Score",
                gridcolor="#1e293b",
                range=[-1.05, 1.05],
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["−1 (Negative)", "−0.5", "Neutral", "+0.5", "+1 (Positive)"],
                zeroline=True, zerolinecolor="rgba(255,255,255,0.15)", zerolinewidth=1,
            ),
            legend=dict(
                bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155",
                borderwidth=1, orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            ),
            showlegend=True,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── SO WHAT callout ─────────────────────────────────────────
        if cp_date is not None:
            _cp_month = cp_date.strftime('%B %Y')
            _cp_drop  = f"{cp_drop:.3f}"
            st.markdown(
                f'<div class="so-what">'
                f'<div class="so-what-title">SO WHAT?</div>'
                f'<b>The inflection point was {_cp_month}.</b> '
                f'Smoothed sentiment dropped {_cp_drop} points in a single month. '
                f'This is the forensic anchor date - cross-reference with firmware releases, '
                f'supplier changes, or marketing campaigns.'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── CHART 2: Silent Killer Rate + Review Volume (side-by-side) ─
        col_sk, col_vol = st.columns(2)

        with col_sk:
            if monthly["sk_rate"].sum() > 0:
                fig_sk = go.Figure()

                # Smoothed area
                fig_sk.add_trace(go.Scatter(
                    x=monthly["review_date"], y=monthly["smooth_sk_rate"],
                    mode="lines",
                    name="Smoothed SK Rate",
                    line=dict(color="#FF4C4C", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(255,76,76,0.12)",
                    hovertemplate="<b>%{x|%b %Y}</b><br>SK Rate: %{y:.1f}%<extra></extra>",
                ))
                # Raw dots
                fig_sk.add_trace(go.Scatter(
                    x=monthly["review_date"], y=monthly["sk_rate"],
                    mode="markers",
                    name="Monthly SK%",
                    marker=dict(color="#FF4C4C", size=4, opacity=0.4),
                    showlegend=False,
                ))
                if cp_date is not None:
                    fig_sk.add_vline(
                        x=cp_date.timestamp() * 1000,
                        line_dash="dash", line_color="rgba(255,215,0,0.5)", line_width=1,
                    )
                fig_sk.update_layout(
                    **DARK,
                    title=dict(text="Silent Killer Rate % (Monthly)", font=dict(size=14)),
                    height=260,
                    xaxis=dict(gridcolor="#1e293b", showgrid=True),
                    yaxis=dict(
                        title="SK Rate (%)", gridcolor="#1e293b",
                        range=[0, min(100, monthly["sk_rate"].max() * 1.3 + 1)],
                        ticksuffix="%",
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig_sk, use_container_width=True)
            else:
                st.info("No Silent Killers in the current dataset window.")

        with col_vol:
            # Gradient bar coloring: more reviews = more opaque
            max_count = monthly["review_count"].max()
            bar_colors = [
                f"rgba(76,158,255,{0.3 + 0.7 * (v / max_count):.2f})"
                for v in monthly["review_count"]
            ]
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=monthly["review_date"], y=monthly["review_count"],
                marker_color=bar_colors,
                name="Reviews/Month",
                hovertemplate="<b>%{x|%b %Y}</b><br>Reviews: %{y}<extra></extra>",
            ))
            if cp_date is not None:
                fig_vol.add_vline(
                    x=cp_date.timestamp() * 1000,
                    line_dash="dash", line_color="rgba(255,215,0,0.5)", line_width=1,
                )
            fig_vol.update_layout(
                **DARK,
                title=dict(text="Review Volume (Monthly)", font=dict(size=14)),
                height=260,
                xaxis=dict(gridcolor="#1e293b", showgrid=True),
                yaxis=dict(title="Review Count", gridcolor="#1e293b"),
                showlegend=False,
            )
            st.plotly_chart(fig_vol, use_container_width=True)

# ── Tab 5: Silent Killer Drill-Down — NEW ────────────────────────────
with tab5:
    st.subheader("🔍 Silent Killer Review Drill-Down")
    st.markdown("All reviews flagged as Silent Killers — customers who gave 4-5★ but signal deep frustration.")

    if "is_silent_killer" not in df_valid.columns:
        st.warning("`is_silent_killer` column not found. Re-run Day 1 pipeline to generate this flag.")
    else:
        sk_df = df_valid[df_valid["is_silent_killer"] == True].copy()
        sk_df = sk_df.sort_values("star_gap" if "star_gap" in sk_df.columns else "roberta_score",
                                  ascending=True)

        if "revenue_at_risk" in sk_df.columns:
            _sk_header = f"**{len(sk_df):,} Silent Killers** | Total Revenue at Risk: **${sk_df['revenue_at_risk'].sum():,.0f}**"
        else:
            _sk_header = f"**{len(sk_df):,} Silent Killers detected**"
        st.markdown(_sk_header)

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            if "star_rating" in sk_df.columns:
                star_filter = st.multiselect("Star Rating", options=sorted(sk_df["star_rating"].dropna().unique()),
                                              default=sorted(sk_df["star_rating"].dropna().unique()))
                if star_filter:
                    sk_df = sk_df[sk_df["star_rating"].isin(star_filter)]
        with fc2:
            if "topic_label" in sk_df.columns:
                aspect_filter = st.multiselect("Aspect", options=sorted(sk_df["topic_label"].dropna().unique()),
                                                default=sorted(sk_df["topic_label"].dropna().unique()))
                if aspect_filter:
                    sk_df = sk_df[sk_df["topic_label"].isin(aspect_filter)]
        with fc3:
            if not sk_df.empty and "review_date" in sk_df.columns:
                min_d = sk_df["review_date"].min().date()
                max_d = sk_df["review_date"].max().date()
                date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
                if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                    sk_df = sk_df[
                        (sk_df["review_date"].dt.date >= date_range[0]) &
                        (sk_df["review_date"].dt.date <= date_range[1])
                    ]

        # Display table
        display_cols = ["review_date", "star_rating", "roberta_score", "star_gap",
                        "topic_label", "review_text", "revenue_at_risk"]
        display_cols = [c for c in display_cols if c in sk_df.columns]

        sk_display = sk_df[display_cols].copy()
        if "review_text" in sk_display.columns:
            sk_display["review_text"] = sk_display["review_text"].str[:120] + "..."
        if "review_date" in sk_display.columns:
            sk_display["review_date"] = sk_display["review_date"].dt.strftime("%Y-%m-%d")
        if "roberta_score" in sk_display.columns:
            sk_display["roberta_score"] = sk_display["roberta_score"].round(3)
        if "star_gap" in sk_display.columns:
            sk_display["star_gap"] = sk_display["star_gap"].round(3)
        if "revenue_at_risk" in sk_display.columns:
            sk_display["revenue_at_risk"] = sk_display["revenue_at_risk"].apply(lambda x: f"${x:,.0f}")

        st.dataframe(
            sk_display.rename(columns={
                "review_date": "Date", "star_rating": "Stars", "roberta_score": "Sentiment",
                "star_gap": "Star Gap ↑", "topic_label": "Aspect",
                "review_text": "Review Snippet", "revenue_at_risk": "Revenue at Risk",
            }),
            use_container_width=True,
            height=450,
            hide_index=True,
        )

        st.markdown("---")
        # Revenue distribution by aspect
        if "revenue_at_risk" in sk_df.columns and "topic_label" in sk_df.columns:
            rar_by_aspect = (sk_df.groupby("topic_label")["revenue_at_risk"]
                             .sum().reset_index()
                             .sort_values("revenue_at_risk", ascending=True))
            rar_by_aspect["revenue_at_risk_num"] = rar_by_aspect["revenue_at_risk"]

            fig_rar = go.Figure(go.Bar(
                y=rar_by_aspect["topic_label"],
                x=rar_by_aspect["revenue_at_risk_num"],
                orientation="h",
                marker=dict(
                    color=rar_by_aspect["revenue_at_risk_num"],
                    colorscale="RdYlGn_r",
                    showscale=True,
                ),
                text=[f"${v:,.0f}" for v in rar_by_aspect["revenue_at_risk_num"]],
                textposition="outside",
            ))
            fig_rar.update_layout(**DARK, title="Revenue at Risk by Aspect (Silent Killers Only)",
                                  xaxis_title="Total Revenue at Risk ($)", xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(fig_rar, use_container_width=True)

# ── Tab 6: Advanced ML Analytics ─────────────────────────────────────────────
with tab6:
    st.subheader("🤖 Advanced ML Analytics")
    st.markdown("*Behind-the-scenes models driving the intelligence suite.*")
    st.caption("This tab is for the technical team. For the business read, see 🏠 Executive Summary.")

    ml_col1, ml_col2 = st.columns(2)

    with ml_col1:
        st.markdown("#### 1. XGBoost Churn Risk Classifier")
        if ml_metrics.get("churn_classifier"):
            ch = ml_metrics["churn_classifier"]
            st.caption(f"Trained on {ch['n_train']:,} reviews, tested on {ch['n_test']:,}")
            
            c_metrics = st.columns(4)
            c_metrics[0].metric("AUROC", f"{ch['auroc']:.4f}")
            c_metrics[1].metric("F1 Score", f"{ch['f1']:.4f}")
            c_metrics[2].metric("Precision", f"{ch['precision']:.4f}")
            c_metrics[3].metric("Recall", f"{ch['recall']:.4f}")
            
            st.markdown("**Top Features by Importance:**")
            if not feat_imp.empty:
                fig_feat = px.bar(
                    feat_imp.head(7).sort_values("importance", ascending=True),
                    x="importance", y="feature", orientation="h",
                    color_discrete_sequence=["#4C9EFF"]
                )
                fig_feat.update_layout(**DARK, height=250,
                                       xaxis=dict(showgrid=False), yaxis_title="")
                st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.warning("Churn model metrics not found.")

        st.markdown("---")
        st.markdown("#### 3. Query Copilot — Semantic Search + Insights")
        st.caption("Ask a question in plain English. We find the most similar reviews by "
                   "meaning (not keyword match) and summarize what they say as a group.")
        if ml_metrics.get("semantic_search", {}).get("available"):

            @st.cache_resource
            def _load_embedder():
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer("all-MiniLM-L6-v2")

            @st.cache_data
            def _load_semantic_index():
                embs = np.load(os.path.join(DATA_DIR, "review_embeddings.npy"))
                idx = pd.read_csv(os.path.join(DATA_DIR, "review_index.csv"))
                return embs, idx

            search_query = st.text_input("Query", placeholder="e.g. 'the battery dies too fast'")
            top_k = st.slider("Results", min_value=3, max_value=15, value=6, key="query_top_k")

            if search_query:
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    with st.spinner("Searching via embeddings..."):
                        emb_model = _load_embedder()
                        review_embs, review_index = _load_semantic_index()
                        query_emb = emb_model.encode([search_query])

                        sims = cosine_similarity(query_emb, review_embs)[0]
                        top_idx = np.argsort(sims)[::-1][:top_k]
                        matches = review_index.iloc[top_idx].copy()
                        matches["similarity"] = sims[top_idx]

                        # -- Synthesized insight over the matched reviews --------
                        avg_sent   = matches["roberta_score"].mean() if "roberta_score" in matches.columns else None
                        avg_star   = matches["star_rating"].mean() if "star_rating" in matches.columns else None
                        sk_rate    = (matches["is_silent_killer"].mean() * 100
                                      if "is_silent_killer" in matches.columns else None)
                        top_aspect = (matches["topic_label"].mode().iloc[0]
                                      if "topic_label" in matches.columns and not matches["topic_label"].mode().empty
                                      else None)

                        insight_bits = []
                        if avg_sent is not None:
                            tone = "negative" if avg_sent < -0.1 else ("positive" if avg_sent > 0.1 else "mixed/neutral")
                            insight_bits.append(f"Average sentiment among these matches is **{avg_sent:+.2f}** ({tone})")
                        if avg_star is not None:
                            insight_bits.append(f"average star rating **{avg_star:.1f}★**")
                        if top_aspect is not None:
                            insight_bits.append(f"most commonly tied to **{top_aspect}**")
                        if sk_rate is not None and sk_rate > 0:
                            insight_bits.append(f"**{sk_rate:.0f}%** are Silent Killers (high stars, hidden frustration)")

                        if insight_bits:
                            st.markdown(
                                f'<div class="so-what">'
                                f'<div class="so-what-title">INSIGHT</div>'
                                f'{"; ".join(insight_bits)}.'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        for _, row in matches.iterrows():
                            aspect_bit = f" | Aspect: {row['topic_label']}" if "topic_label" in row else ""
                            star_bit = f" | Stars: {row['star_rating']}" if "star_rating" in row else ""
                            st.markdown(f"""
                            <div style="background:#0f172a; padding:12px; border-radius:8px; margin-bottom:8px; border-left:3px solid #00FFB3;">
                                <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:4px;">
                                    <strong>Similarity: {row['similarity']:.2f}</strong>{aspect_bit}{star_bit}
                                </div>
                                <div style="font-size:0.9rem;">{row['review_text']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Search error: {e}")
        else:
            st.warning("Semantic search not available. Run day4_ml_models.py to build the review index.")

    with ml_col2:
        st.markdown("#### 2. Isolation Forest Anomaly Detection")
        if not anomalies.empty:
            anom_plot = anomalies.copy()
            anom_plot["review_date"] = pd.to_datetime(anom_plot["review_date"])
            fig_anom = go.Figure()
            
            normal = anom_plot[~anom_plot["is_anomaly"]]
            fig_anom.add_trace(go.Scatter(
                x=normal["review_date"], y=normal["anomaly_prob"],
                mode="markers", name="Normal", marker=dict(color="#4C9EFF", opacity=0.5, size=6),
                hovertemplate="%{x|%b %Y}<br>Score: %{y:.3f}<extra></extra>"
            ))
            
            anoms = anom_plot[anom_plot["is_anomaly"]]
            fig_anom.add_trace(go.Scatter(
                x=anoms["review_date"], y=anoms["anomaly_prob"],
                mode="markers", name="Anomaly", marker=dict(color="#FF4C4C", size=10, symbol="x"),
                hovertemplate="<b>%{x|%b %Y}</b><br>Score: %{y:.3f}<extra></extra>"
            ))
            fig_anom.update_layout(**DARK, height=280, title="Monthly Anomaly Scores (Higher = Anomalous)", 
                                   legend=dict(orientation="h", y=1.1, x=0))
            st.plotly_chart(fig_anom, use_container_width=True)
            st.caption(f"Detected {len(anoms)} unusual monthly clusters based on sentiment, volume, and anger/sadness.")
        else:
            st.warning("Anomaly data not found.")

        st.markdown("---")
        st.markdown("#### 4. Polynomial Ridge Sentiment Forecast")
        if not forecast.empty:
            forecast["date"] = pd.to_datetime(forecast["date"])
            fig_fc = go.Figure()
            
            hist = forecast[~forecast["is_forecast"]]
            fig_fc.add_trace(go.Scatter(
                x=hist["date"], y=hist["actual"], mode="lines", name="Historical", 
                line=dict(color="#00FFB3", width=2), hovertemplate="%{x|%b %Y}<br>Actual: %{y:.3f}<extra></extra>"
            ))
            
            fut = forecast[forecast["is_forecast"]]
            if not fut.empty:
                fig_fc.add_trace(go.Scatter(
                    x=fut["date"], y=fut["fitted"], mode="lines", name="Forecast", 
                    line=dict(color="#FFD700", width=2, dash="dash"), hovertemplate="%{x|%b %Y}<br>Forecast: %{y:.3f}<extra></extra>"
                ))
                fig_fc.add_trace(go.Scatter(
                    x=pd.concat([fut["date"], fut["date"][::-1]]), 
                    y=pd.concat([fut["ci_hi"], fut["ci_lo"][::-1]]), 
                    fill="toself", fillcolor="rgba(255,215,0,0.15)", line=dict(color="rgba(255,255,255,0)"), 
                    name="90% CI", showlegend=False, hoverinfo="skip"
                ))
            fig_fc.update_layout(**DARK, height=280, title="6-Month Sentiment Forecast", 
                                 legend=dict(orientation="h", y=1.1, x=0))
            st.plotly_chart(fig_fc, use_container_width=True)
            
            direction = ml_metrics.get("sentiment_forecast", {}).get("forecast_direction", "N/A")
            st.caption(f"Projected trend: **{direction}**")
        else:
            st.warning("Forecast data not found.")

# ── Tab 7: Competitor Benchmark ───────────────────────────────────────
with tab7:
    st.subheader("📊 Competitor Benchmark")
    st.markdown(
        "The flagship vs. its direct competitors — same category, different brand, "
        "auto-selected by review volume during data ingestion. Compares **real "
        "sentiment**, not just the star rating management already sees."
    )

    if competitor_comparison.empty:
        st.warning(
            "No competitor data found. Run `python setup_scripts/stream_data.py` "
            "(discovers competitors) then `python day3.5_competitor_benchmark.py` "
            "(scores + compares them)."
        )
    else:
        insight_text = competitor_summary.get("insight")
        if insight_text:
            st.markdown(
                f'<div class="so-what">'
                f'<div class="so-what-title">SO WHAT?</div>'
                f'{insight_text}'
                f'</div>',
                unsafe_allow_html=True,
            )

        cdf = competitor_comparison.copy()
        cdf["label"] = cdf["brand"].fillna("Unknown") + np.where(cdf["is_flagship"], " (Flagship)", "")
        cdf = cdf.sort_values("is_flagship", ascending=False)

        col_star, col_sent = st.columns(2)
        bar_colors = ["#FF4C4C" if f else "#4C9EFF" for f in cdf["is_flagship"]]

        with col_star:
            fig_star = go.Figure(go.Bar(
                y=cdf["label"], x=cdf["avg_star"], orientation="h",
                marker_color=bar_colors, text=cdf["avg_star"].round(2), textposition="outside",
            ))
            fig_star.update_layout(**DARK, title="Average Star Rating (what management sees)",
                                   xaxis=dict(range=[0, 5], gridcolor="#1e293b"), height=320)
            st.plotly_chart(fig_star, use_container_width=True)

        with col_sent:
            fig_sent = go.Figure(go.Bar(
                y=cdf["label"], x=cdf["avg_sentiment"], orientation="h",
                marker_color=bar_colors, text=cdf["avg_sentiment"].round(3), textposition="outside",
            ))
            fig_sent.update_layout(**DARK, title="Average Sentiment Score (what customers feel)",
                                   xaxis=dict(range=[-1, 1], gridcolor="#1e293b"), height=320)
            fig_sent.add_vline(x=0, line_color="rgba(255,255,255,0.2)")
            st.plotly_chart(fig_sent, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Full Comparison Table")
        display_cdf = cdf[["label", "n_reviews", "avg_star", "avg_sentiment", "silent_killer_rate_pct"]].copy()
        display_cdf["avg_star"] = display_cdf["avg_star"].round(2)
        display_cdf["avg_sentiment"] = display_cdf["avg_sentiment"].round(3)
        display_cdf["silent_killer_rate_pct"] = display_cdf["silent_killer_rate_pct"].round(1)
        st.dataframe(
            display_cdf.rename(columns={
                "label": "Product", "n_reviews": "Reviews",
                "avg_star": "Avg Star", "avg_sentiment": "Avg Sentiment",
                "silent_killer_rate_pct": "Silent Killer Rate (%)",
            }),
            use_container_width=True, hide_index=True,
        )
        sk_thresh = competitor_summary.get("sk_threshold_shared")
        if sk_thresh is not None:
            st.caption(
                f"Silent Killer rates use a single shared threshold "
                f"(prob_negative ≥ {sk_thresh:.3f}) computed once across the flagship "
                f"and all competitors, so every product is held to the same bar."
            )
