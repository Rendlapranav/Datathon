import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# -- Page Configuration --------------------------------------------
st.set_page_config(
    page_title="CEO Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 0rem;}
    </style>
""", unsafe_allow_html=True)

# -- Constants: must mirror day2 ACTIVE_EMOTIONS / DISPLAY_LABELS --
# If day2 changes these, change them here too.
ACTIVE_EMOTIONS = ["anger", "sadness", "joy"]
DISPLAY_LABELS  = {
    "anger"  : "Anger",
    "sadness": "Disappointment",
    "joy"    : "Joy",
}

# Thresholds: must mirror day3 FREQ_THRESHOLD / INTENSITY_THRESHOLD.
# day3's ceo_summary.json does not export these, so they are hardcoded
# here with a comment. If you change them in day3, change them here too.
FREQ_THRESHOLD      = 5.0
INTENSITY_THRESHOLD = 0.40

DATA_DIR = "data/clean"

# -- Load Assets ---------------------------------------------------
@st.cache_data
def load_data():
    try:
        with open(os.path.join(DATA_DIR, "ceo_summary.json"), "r") as f:
            summary = json.load(f)
        quotes_df  = pd.read_csv(os.path.join(DATA_DIR, "day2_representative_quotes.csv"))
        aspects_df = pd.read_csv(os.path.join(DATA_DIR, "day2_aspects.csv"))
        return summary, quotes_df, aspects_df
    except FileNotFoundError:
        st.error("Data files not found. Please run the Day 1-3 pipeline first.")
        st.stop()

summary, quotes_df, aspects_df = load_data()
df_valid = aspects_df[aspects_df["topic_id"] != -1].copy()

# Validate that emotion columns exist before any plot attempts
missing = [f"emo_{e}" for e in ACTIVE_EMOTIONS if f"emo_{e}" not in df_valid.columns]
if missing:
    st.error(f"Expected emotion columns not found in day2_aspects.csv: {missing}. "
             f"Re-run day2_aspects.py.")
    st.stop()

# -- Sidebar -------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Mission Control")
    st.markdown(f"**Target Product:** `{summary.get('product_id', 'All')}`")
    st.markdown(f"**Total Reviews Processed:** `{summary.get('total_reviews', 0):,}`")

    emotion_meta = summary.get("emotion_model", {})
    if emotion_meta:
        st.markdown("---")
        st.markdown("### 🧠 Emotion Model")
        st.markdown(f"**Active signals:** {', '.join(emotion_meta.get('active_signals', ACTIVE_EMOTIONS))}")
        st.markdown(f"**Discarded:** {', '.join(emotion_meta.get('discarded_signals', []))}")
        st.caption("Discarded emotions (fear, surprise, disgust, neutral) "
                   "carry no actionable signal for hardware product analysis.")

    st.markdown("---")
    st.markdown("### The Silent Killer Metric")
    st.info(
        "Reviews with 4-5 stars where the NLP model detects high anger / disappointment. "
        "These customers will churn without filing a support ticket."
    )

    financials = summary.get("financials", {})
    if financials.get("churn_multiplier_source"):
        st.caption(f"Churn multiplier source: {financials['churn_multiplier_source']}")

# -- Header --------------------------------------------------------
st.title("📊 Executive Sentiment & ROI Dashboard")
st.markdown("Translating deep-learning text analytics into actionable business priorities.")

# -- KPI Row -------------------------------------------------------
col1, col2, col3 = st.columns(3)
financials = summary.get("financials", {})
sim        = summary.get("roi_simulation", {})

col1.metric(
    "🚨 Silent Killers Detected",
    f"{financials.get('silent_killers', 0):,}",
)
col2.metric(
    "💸 Revenue at Risk (Est.)",
    f"${financials.get('revenue_at_risk_usd', 0):,.0f}",
    f"- AOV ${financials.get('aov_usd', 150):.0f} × "
    f"{financials.get('churn_multiplier', 5)}x churn multiplier",
)
col3.metric(
    f"📈 ROI: Fix {sim.get('target_aspect', 'Top Issue')}",
    f"+{sim.get('improvement_pct_high', 0)}%",
    "Brand Sentiment Boost",
)

st.markdown("---")

# -- Tabs ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Priority Matrix & Roadmap",
    "🧠 Emotion Deep Dive",
    "🗣️ Voice of Customer",
])

# ── Tab 1: Priority Matrix ────────────────────────────────────────
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
                x              = "frequency_pct",
                y              = "emotional_intensity",
                size           = "frequency_pct",
                color          = "emotional_intensity",
                color_continuous_scale = "RdYlGn_r",
                hover_name     = "topic_label",
                hover_data     = {
                    "topic_label"       : False,
                    "quadrant"          : True,
                    "frequency_pct"     : ":.1f",
                    "emotional_intensity": ":.2f",
                },
                text  = "topic_label",
                title = "Interactive CEO Priority Matrix",
            )
            fig_matrix.update_traces(
                textposition = "top center",
                marker       = dict(line=dict(width=1, color="DarkSlateGrey")),
            )
            fig_matrix.add_hline(
                y=INTENSITY_THRESHOLD, line_dash="dash", line_color="gray", opacity=0.5,
            )
            fig_matrix.add_vline(
                x=FREQ_THRESHOLD, line_dash="dash", line_color="gray", opacity=0.5,
            )
            fig_matrix.add_vrect(
                x0=FREQ_THRESHOLD, x1=max_x,
                fillcolor="red", opacity=0.05, layer="below", line_width=0,
            )
            fig_matrix.update_layout(
                xaxis_title = "Complaint Frequency (% of total reviews)",
                yaxis_title = "Emotional Intensity (Anger + Disappointment)",
                coloraxis_colorbar = dict(title="Pain Level"),
            )
            fig_matrix.update_xaxes(range=[0, max_x])
            fig_matrix.update_yaxes(range=[0, max_y])

            st.plotly_chart(fig_matrix, use_container_width=True)

    with col_roadmap:
        st.markdown("### 🗺️ Strategic Roadmap")
        roadmap = pd.DataFrame(summary.get("strategic_roadmap", []))
        if not roadmap.empty:
            def highlight_urgent(val):
                return "background-color: #7f1d1d" if val == "Urgent Redesign" else ""

            st.dataframe(
                roadmap[["tier", "aspect", "action", "quadrant"]]
                .style.map(highlight_urgent, subset=["quadrant"]),
                hide_index = True,
                height     = 400,
            )
        st.success(f"**Bottom Line:** {sim.get('slide_pitch', '')}")

# ── Tab 2: Emotion Deep Dive ──────────────────────────────────────
with tab2:
    st.subheader("Emotion Heatmaps")
    st.caption(
        f"Active signals: **{', '.join(DISPLAY_LABELS.values())}**. "
        f"Disgust, fear, surprise, and neutral were excluded — "
        f"they carry no actionable signal on hardware product reviews."
    )

    col_full, col_urgency = st.columns(2)

    with col_full:
        # Dominant emotion % per aspect — only ACTIVE_EMOTIONS columns.
        # Uses dominant_emotion_display written by day2 (already renamed).
        if "dominant_emotion_display" in df_valid.columns:
            heat_data = (
                df_valid
                .groupby(["topic_label", "dominant_emotion_display"])
                .size()
                .unstack(fill_value=0)
            )
            heat_pct = heat_data.div(heat_data.sum(axis=1), axis=0) * 100

            # Order columns by display label order for consistency
            display_order = [DISPLAY_LABELS[e] for e in ACTIVE_EMOTIONS
                             if DISPLAY_LABELS[e] in heat_pct.columns]
            heat_pct = heat_pct[display_order].round(1)
        else:
            # Fallback: build from raw emo_ columns using display rename
            heat_rows = {}
            for label, grp in df_valid.groupby("topic_label"):
                heat_rows[label] = {
                    DISPLAY_LABELS[e]: grp[f"emo_{e}"].mean() * 100
                    for e in ACTIVE_EMOTIONS
                }
            heat_pct = pd.DataFrame(heat_rows).T.round(1)

        fig_heat = px.imshow(
            heat_pct,
            color_continuous_scale = "RdYlGn_r",
            text_auto              = True,
            aspect                 = "auto",
            title                  = "Product Aspect × Dominant Emotion (%)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_urgency:
        # Urgency heatmap: Anger + Disappointment mean probabilities.
        # Column is emo_sadness in the CSV; label shown as Disappointment.
        urgency_df = (
            df_valid
            .groupby("topic_label")[["emo_anger", "emo_sadness"]]
            .mean() * 100
        ).round(1)
        urgency_df.columns = ["Anger", "Disappointment"]   # display rename

        fig_urgency = px.imshow(
            urgency_df,
            color_continuous_scale = "YlOrRd",
            text_auto              = True,
            aspect                 = "auto",
            title                  = "Urgency Signal: Mean Anger & Disappointment (%)",
        )
        st.plotly_chart(fig_urgency, use_container_width=True)

# ── Tab 3: Voice of Customer ──────────────────────────────────────
with tab3:
    st.subheader("🗣️ Voice of the Customer")
    st.markdown(
        "Auto-extracted quotes from the mathematical center of each topic cluster. "
        "These ground the AI's findings in actual human language."
    )

    aspects          = quotes_df["topic_label"].unique()
    selected_aspect  = st.selectbox("Filter by Aspect:", aspects, label_visibility="collapsed")
    filtered_quotes  = quotes_df[quotes_df["topic_label"] == selected_aspect].head(3)

    cols = st.columns(3)
    for col, (_, row) in zip(cols, filtered_quotes.iterrows()):
        quote_text = row["quote_text"].strip()
        if len(quote_text) > 300:
            quote_text = quote_text[:297] + "..."

        with col:
            st.markdown(f"""
            <div style="
                background-color: #0f172a;
                padding: 25px;
                border-radius: 12px;
                border: 1px solid #1e293b;
                border-top: 4px solid #3b82f6;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
                min-height: 250px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div>
                    <div style="font-size:32px;color:#3b82f6;line-height:1;margin-bottom:10px;">❝</div>
                    <p style="font-style:italic;font-size:15px;color:#cbd5e1;line-height:1.6;">
                        {quote_text}
                    </p>
                </div>
                <div style="text-align:right;border-top:1px solid #1e293b;margin-top:15px;padding-top:10px;">
                    <span style="font-size:12px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:1px;">
                        Verified Analysis
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
