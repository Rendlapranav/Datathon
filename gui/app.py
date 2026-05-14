import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# -- Page Configuration --
st.set_page_config(page_title="CEO Sentiment Dashboard", layout="wide", initial_sidebar_state="expanded")
# -- UI Polish (Hide Streamlit branding) --
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {padding-top: 2rem; padding-bottom: 0rem;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
# -- Load Assets --
DATA_DIR = "data/clean"

@st.cache_data
def load_data():
    try:
        with open(os.path.join(DATA_DIR, "ceo_summary.json"), 'r') as f:
            summary = json.load(f)
        quotes_df = pd.read_csv(os.path.join(DATA_DIR, "day2_representative_quotes.csv"))
        aspects_df = pd.read_csv(os.path.join(DATA_DIR, "day2_aspects.csv"))
        return summary, quotes_df, aspects_df
    except FileNotFoundError:
        st.error("Data files not found. Please run the Day 1-3 pipeline first.")
        st.stop()

summary, quotes_df, aspects_df = load_data()
df_valid_aspects = aspects_df[aspects_df['topic_id'] != -1] # Exclude outliers for plots

# -- Sidebar: Filters & Context --
with st.sidebar:
    st.title("⚙️ Mission Control")
    st.markdown(f"**Target Product:** `{summary.get('product_id', 'All')}`")
    st.markdown(f"**Total Reviews Processed:** `{summary.get('total_reviews', 0):,}`")
    st.markdown("---")
    st.markdown("### The Silent Killer Metric")
    st.info("Reviews with 4-5 stars where the NLP model detects high anger/sadness. These are customers who will churn without complaining.")

# -- Header --
st.title("📊 Executive Sentiment & ROI Dashboard")
st.markdown("Translating deep-learning text analytics into actionable business priorities.")

# -- Top KPI Row --
col1, col2, col3 = st.columns(3)
financials = summary.get('financials', {})
sim = summary.get('roi_simulation', {})

col1.metric("🚨 Silent Killers Detected", f"{financials.get('silent_killers', 0):,}")
col2.metric("💸 Revenue at Risk (Est.)", f"${financials.get('revenue_at_risk_usd', 0):,.0f}", "- Immediate Churn Threat")
col3.metric(f"📈 ROI: Fix {sim.get('target_aspect', 'Top Issue')}", f"+{sim.get('improvement_pct_high', 0)}%", "Brand Sentiment Boost")

st.markdown("---")

# -- Main Content Tabs --
tab1, tab2, tab3 = st.tabs(["🎯 Priority Matrix & Roadmap", "🧠 AI Emotion Deep Dive", "🗣️ Voice of Customer"])

with tab1:
    st.subheader("Actionable Priorities")
    row1_col1, row1_col2 = st.columns([3, 2])
    
    with row1_col1:
        # PLOTLY MATRIX: Interactive Bubble Chart
        pm_data = pd.DataFrame(summary.get('priority_matrix', []))
        if not pm_data.empty:
            thresh_int = summary.get('thresholds', {}).get('intensity_threshold', 0.4)
            thresh_freq = summary.get('thresholds', {}).get('freq_pct_threshold', 5.0)

            fig_matrix = px.scatter(
                pm_data, 
                x='frequency_pct', 
                y='emotional_intensity',
                size='frequency_pct', 
                color='emotional_intensity',
                color_continuous_scale='RdYlGn_r',
                hover_name='topic_label',
                hover_data={'topic_label': False, 'quadrant': True, 'frequency_pct': ':.1f', 'emotional_intensity': ':.2f'},
                text='topic_label',
                title="Interactive CEO Priority Matrix"
            )
            
            # Format text position and axes
            fig_matrix.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            fig_matrix.add_hline(y=thresh_int, line_dash="dash", line_color="gray", opacity=0.5)
            fig_matrix.add_vline(x=thresh_freq, line_dash="dash", line_color="gray", opacity=0.5)
            fig_matrix.update_layout(
                xaxis_title="Complaint Frequency (% of total reviews)",
                yaxis_title="Emotional Intensity (Anger + Sadness)",
                coloraxis_colorbar=dict(title="Pain Level")
            )
            
            # Shaded Quadrants
            max_x = max(pm_data['frequency_pct'].max() * 1.2, thresh_freq * 2)
            max_y = max(pm_data['emotional_intensity'].max() * 1.2, thresh_int * 1.5)
            fig_matrix.add_vrect(x0=thresh_freq, x1=max_x, fillcolor="red", opacity=0.05, layer="below", line_width=0)
            fig_matrix.update_xaxes(range=[0, max_x])
            fig_matrix.update_yaxes(range=[0, max_y])
            
            st.plotly_chart(fig_matrix, use_container_width=True)
            
    with row1_col2:
        st.markdown("### 🗺️ Strategic Roadmap")
        roadmap = pd.DataFrame(summary.get('strategic_roadmap', []))
        if not roadmap.empty:
            def highlight_urgent(val):
                color = '#7f1d1d' if val == 'Urgent Redesign' else ''
                return f'background-color: {color}'
            st.dataframe(
                roadmap[['tier', 'aspect', 'action', 'quadrant']].style.map(highlight_urgent, subset=['quadrant']),
                hide_index=True,
                height=400
            )
        st.success(f"**Bottom Line:** {sim.get('slide_pitch', '')}")

with tab2:
    st.subheader("Multilingual Emotion Heatmaps")
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # PLOTLY HEATMAP: Dominant Emotion
        emotions_to_plot = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        heat_data = df_valid_aspects.groupby(['topic_label', 'dominant_emotion']).size().unstack(fill_value=0)
        heat_pct = heat_data.div(heat_data.sum(axis=1), axis=0) * 100
        cols_present = [e for e in emotions_to_plot if e in heat_pct.columns]
        heat_pct = heat_pct[cols_present].round(1)

        fig_heat = px.imshow(
            heat_pct, 
            color_continuous_scale='RdYlGn_r',
            text_auto=True,
            aspect="auto",
            title="Product Aspect × Dominant Emotion (%)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
            
    with row2_col2:
        # PLOTLY HEATMAP: Urgency (Anger + Sadness)
        urgency_cols = ['emo_anger', 'emo_sadness']
        cols_present_u = [e for e in urgency_cols if e in df_valid_aspects.columns]
        heatmap_urgency = (df_valid_aspects.groupby('topic_label')[cols_present_u].mean() * 100).round(1)
        heatmap_urgency.columns = [e.replace('emo_', '').capitalize() for e in cols_present_u]

        fig_urgency = px.imshow(
            heatmap_urgency, 
            color_continuous_scale='YlOrRd',
            text_auto=True,
            aspect="auto",
            title="Urgency Lever: Mean Anger & Sadness Probability (%)"
        )
        st.plotly_chart(fig_urgency, use_container_width=True)
with tab3:
    st.subheader("🗣️ Voice of the Customer")
    st.markdown("Direct, auto-extracted quotes representing the mathematical center of each topic cluster. Use these to ground abstract AI metrics in human reality.")
    
    aspects = quotes_df['topic_label'].unique()
    # Collapsed label looks cleaner in UI
    selected_aspect = st.selectbox("Filter by Aspect:", aspects, label_visibility="collapsed")
    
    # We requested exactly 3 representative docs in day2_aspects.py
    filtered_quotes = quotes_df[quotes_df['topic_label'] == selected_aspect].head(3)
    
    # Create a clean 3-column grid
    cols = st.columns(3)
    
    for idx, (col, row_data) in enumerate(zip(cols, filtered_quotes.iterrows())):
        _, row = row_data
        # Cap length to keep the cards relatively uniform in height
        quote_text = row["quote_text"].strip()
        if len(quote_text) > 300:
            quote_text = quote_text[:297] + "..."
            
        with col:
            # HTML/CSS Injection for SaaS-style Testimonial Cards
            st.markdown(f"""
            <div style="
                background-color: #0f172a;
                padding: 25px;
                border-radius: 12px;
                border: 1px solid #1e293b;
                border-top: 4px solid #3b82f6;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                min-height: 250px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div>
                    <div style="font-size: 32px; color: #3b82f6; line-height: 1; margin-bottom: 10px;">❝</div>
                    <p style="font-style: italic; font-size: 15px; color: #cbd5e1; line-height: 1.6;">
                        {quote_text}
                    </p>
                </div>
                <div style="text-align: right; border-top: 1px solid #1e293b; margin-top: 15px; padding-top: 10px;">
                    <span style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                        Verified Analysis
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
