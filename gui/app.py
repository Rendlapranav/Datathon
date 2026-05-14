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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1400px; }

    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155; border-radius: 16px; padding: 20px 16px; text-align: center;
        margin-bottom: 8px;
    }
    .metric-val-green {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #00FFB3, #4C9EFF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-val-red {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #FF4C4C, #FFD700);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-label { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; font-weight: 500; }
    .so-what {
        background: linear-gradient(135deg, #0f2027, #1a3a2a);
        border-left: 4px solid #00FFB3; border-radius: 0 12px 12px 0;
        padding: 14px 18px; margin: 12px 0;
    }
    .so-what-title { color: #00FFB3; font-weight: 700; font-size: 0.85rem; margin-bottom: 4px; }
    .roi-card {
        background: #0f172a; border: 1px solid #00FFB3; border-radius: 12px;
        padding: 18px; text-align: center;
    }
    .roi-amount { font-size: 1.7rem; font-weight: 800; color: #00FFB3; }
    .roi-label  { font-size: 0.75rem; color: #64748b; margin-top: 4px; }
    .dash-title {
        font-size: 1.9rem; font-weight: 800;
        background: linear-gradient(135deg, #00FFB3, #4C9EFF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# -- Constants (must mirror day2 / day3) --------------------------------
ACTIVE_EMOTIONS     = ["anger", "sadness", "joy"]
DISPLAY_LABELS      = {"anger": "Anger", "sadness": "Disappointment", "joy": "Joy"}
FREQ_THRESHOLD      = 5.0
INTENSITY_THRESHOLD = 0.40

DARK = dict(
    paper_bgcolor="#0D0D0D", plot_bgcolor="#0D0D0D",
    font=dict(color="#e2e8f0", family="Inter"),
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
    st.markdown("## 🚨 Mission Control")
    st.markdown(f"**Product:** `{summary.get('product_id', 'Electronics')}`")
    st.markdown(f"**Reviews Analyzed:** `{total_rev:,}`")
    st.markdown("---")
    st.markdown(f"<div class='metric-val-red'>${rar_total:,.0f}</div><div class='metric-label'>Revenue at Risk</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-val-red'>{sk_count:,}</div><div class='metric-label'>Silent Killers Detected</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧠 Model Stack")
    st.caption("• **RoBERTa** (Cardiff XLM) → Sentiment")
    st.caption("• **Helsinki-NLP** opus-mt → Translation")
    st.caption("• **BERTopic** → Aspect Mining")
    st.caption("• **mDeBERTa** XNLI → Emotion")
    st.markdown("---")
    st.info(
        "**Silent Killer** — A 4-5★ reviewer whose text signals hidden anger. "
        "They will churn without filing a support ticket."
    )
    src = financials.get("churn_multiplier_source", "")
    if src:
        st.caption(f"Churn multiplier: {src}")

# -- Header + KPIs ------------------------------------------------------
st.markdown('<div class="dash-title">📊 Product Rescue Mission</div>', unsafe_allow_html=True)
st.markdown("**Executive Intelligence Dashboard** — Translating customer language into revenue decisions.")
st.markdown("---")

k1, k2, k3, k4 = st.columns(4)
sk_rate = (sk_count / total_rev * 100) if total_rev > 0 else 0

with k1:
    st.metric("🚨 Silent Killers",     f"{sk_count:,}")
with k2:
    st.metric("💸 Revenue at Risk",    f"${rar_total:,.0f}")
with k3:
    st.metric("📈 Sentiment Boost",    f"+{sim.get('improvement_pct_high', 0)}%",  "if top issue fixed")
with k4:
    st.metric("📊 SK Rate",            f"{sk_rate:.1f}%", "of all reviews")

st.markdown("---")

# -- Tabs ---------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Priority Matrix & Roadmap",
    "🧠 Emotion Deep Dive",
    "🗣️ Voice of Customer",
    "📈 Temporal Trend",
    "🔍 Silent Killer Drill-Down",
])

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
    st.caption(
        f"Active: **{', '.join(DISPLAY_LABELS.values())}**  |  "
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
                    <div style="font-size:30px;color:#4C9EFF;line-height:1;margin-bottom:10px;">❝</div>
                    <p style="font-style:italic;font-size:14px;color:#cbd5e1;line-height:1.6;">{qt}</p>
                </div>
                <div style="text-align:right;border-top:1px solid #1e293b;margin-top:12px;padding-top:10px;">
                    <span style="font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;
                                 letter-spacing:1px;">Verified Analysis</span>
                </div>
            </div>""", unsafe_allow_html=True)

# ── Tab 4: Temporal Trend — NEW ───────────────────────────────────────
with tab4:
    st.subheader("📈 Sentiment Drift Over Time")
    st.markdown("*When did the product start failing customers?*")

    if "roberta_score" not in df_valid.columns:
        st.warning("Column `roberta_score` not found. Re-run the pipeline from Day 1.")
    elif df_valid["review_date"].isna().all():
        st.warning("No valid review dates found.")
    else:
        monthly = (
            df_valid
            .set_index("review_date")
            .resample("ME")
            .agg(
                mean_sentiment=("roberta_score", "mean"),
                sk_count=("is_silent_killer", "sum") if "is_silent_killer" in df_valid.columns else ("roberta_score", "count"),
                review_count=("roberta_score", "count"),
            )
            .reset_index()
        )
        monthly = monthly[monthly["review_count"] >= 10].copy()

        if "is_silent_killer" in df_valid.columns:
            monthly["sk_rate"] = monthly["sk_count"] / monthly["review_count"] * 100
        else:
            monthly["sk_rate"] = 0

        # Detect changepoint: month with the biggest MoM drop in sentiment
        monthly["sentiment_delta"] = monthly["mean_sentiment"].diff()
        changepoint_idx = monthly["sentiment_delta"].idxmin()
        changepoint_date = monthly.loc[changepoint_idx, "review_date"] if not monthly.empty else None

        fig_trend = go.Figure()

        # Sentiment line
        fig_trend.add_trace(go.Scatter(
            x=monthly["review_date"], y=monthly["mean_sentiment"],
            mode="lines+markers", name="Sentiment Score",
            line=dict(color="#00FFB3", width=2.5),
            marker=dict(size=6),
            hovertemplate="<b>%{x|%b %Y}</b><br>Sentiment: %{y:.3f}<extra></extra>",
        ))

        # 3-month rolling average
        monthly["rolling_sentiment"] = monthly["mean_sentiment"].rolling(3, center=True).mean()
        fig_trend.add_trace(go.Scatter(
            x=monthly["review_date"], y=monthly["rolling_sentiment"],
            mode="lines", name="3-Month Rolling Avg",
            line=dict(color="#4C9EFF", width=2, dash="dot"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Rolling Avg: %{y:.3f}<extra></extra>",
        ))

        # SK rate bars
        if monthly["sk_rate"].sum() > 0:
            fig_trend.add_trace(go.Bar(
                x=monthly["review_date"], y=monthly["sk_rate"],
                name="Silent Killer %", yaxis="y2",
                marker_color="#FF4C4C", opacity=0.45,
                hovertemplate="<b>%{x|%b %Y}</b><br>SK Rate: %{y:.1f}%<extra></extra>",
            ))

        # Changepoint vertical line
        if changepoint_date is not None:
            fig_trend.add_vline(
                x=changepoint_date.timestamp() * 1000,
                line_dash="dash", line_color="#FFD700", line_width=2,
                annotation_text="⚠️ The Day It Broke",
                annotation_font_color="#FFD700",
                annotation_position="top left",
            )

        fig_trend.update_layout(
            **DARK,
            title="Monthly Sentiment Score vs Silent Killer Rate",
            xaxis=dict(title="Month", gridcolor="#1e293b"),
            yaxis=dict(title="Mean Sentiment Score", gridcolor="#1e293b", range=[-1, 1]),
            yaxis2=dict(title="Silent Killer Rate (%)", overlaying="y", side="right",
                        showgrid=False, range=[0, monthly["sk_rate"].max() * 3 + 1]),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155", borderwidth=1),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        if changepoint_date is not None:
            st.markdown(f"""
            <div class="so-what">
              <div class="so-what-title">💡 SO WHAT?</div>
              <b>The inflection point was {changepoint_date.strftime('%B %Y')}.</b>
              Sentiment dropped {abs(monthly.loc[changepoint_idx,'sentiment_delta']):.3f} points month-over-month.
              This is the date to reference in engineering post-mortems and board presentations.
            </div>""", unsafe_allow_html=True)

        # Monthly review volume
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=monthly["review_date"], y=monthly["review_count"],
            marker_color="#4C9EFF", opacity=0.7, name="Reviews/Month",
        ))
        fig_vol.update_layout(**DARK, title="Monthly Review Volume", height=250,
                              xaxis_title="Month", yaxis_title="Reviews",
                              xaxis=dict(gridcolor="#1e293b"),
                              yaxis=dict(gridcolor="#1e293b"))
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

        st.markdown(f"**{len(sk_df):,} Silent Killers** | Total Revenue at Risk: "
                    f"**${sk_df['revenue_at_risk'].sum():,.0f}**" if "revenue_at_risk" in sk_df.columns
                    else f"**{len(sk_df):,} Silent Killers detected**")

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
