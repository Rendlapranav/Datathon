import os
import json
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# -- Constants --
ACTIVE_EMOTIONS = ["anger", "sadness", "joy"]
DISPLAY_LABELS  = {"anger": "Anger", "sadness": "Disappointment", "joy": "Joy"}
FREQ_THRESHOLD = 5.0
INTENSITY_THRESHOLD = 0.40
DATA_DIR = "data/clean"

# -- Load Data --
def load_data():
    if not os.path.exists(os.path.join(DATA_DIR, "ceo_summary.json")):
        raise FileNotFoundError("Data files not found. Run Day 1-3 pipeline.")
    with open(os.path.join(DATA_DIR, "ceo_summary.json"), "r") as f:
        summary = json.load(f)
    quotes_df  = pd.read_csv(os.path.join(DATA_DIR, "day2_representative_quotes.csv"))
    aspects_df = pd.read_csv(os.path.join(DATA_DIR, "day2_aspects.csv"))
    return summary, quotes_df, aspects_df

summary, quotes_df, aspects_df = load_data()
df_valid = aspects_df[aspects_df["topic_id"] != -1].copy()
financials = summary.get("financials", {})
sim = summary.get("roi_simulation", {})

# -- Initialize App (Using Dark Theme) --
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Product Risk & ROI Triage"

# -- Pre-compute Static Figures --
# 1. Priority Matrix
pm_data = pd.DataFrame(summary.get("priority_matrix", []))
max_x = max(pm_data["frequency_pct"].max() * 1.2, FREQ_THRESHOLD * 2) if not pm_data.empty else 10
max_y = max(pm_data["emotional_intensity"].max() * 1.2, INTENSITY_THRESHOLD * 1.5) if not pm_data.empty else 1

fig_matrix = px.scatter(
    pm_data, x="frequency_pct", y="emotional_intensity", size="frequency_pct",
    color="emotional_intensity", color_continuous_scale="RdYlGn_r",
    hover_name="topic_label", text="topic_label", template="plotly_dark",
    title="Defect Prioritization vs. Emotional Impact"
)
fig_matrix.add_hline(y=INTENSITY_THRESHOLD, line_dash="dash", line_color="gray", opacity=0.5)
fig_matrix.add_vline(x=FREQ_THRESHOLD, line_dash="dash", line_color="gray", opacity=0.5)
fig_matrix.add_vrect(x0=FREQ_THRESHOLD, x1=max_x, fillcolor="red", opacity=0.05, layer="below", line_width=0)
fig_matrix.update_traces(textposition="top center")
fig_matrix.update_layout(xaxis_title="Defect Frequency (%)", yaxis_title="Severity (Anger + Disappointment)")

# 2. Emotion Heatmap
heat_rows = {label: {DISPLAY_LABELS[e]: grp[f"emo_{e}"].mean() * 100 for e in ACTIVE_EMOTIONS} 
             for label, grp in df_valid.groupby("topic_label")}
heat_pct = pd.DataFrame(heat_rows).T.round(1)
fig_heat = px.imshow(heat_pct, color_continuous_scale="RdYlGn_r", text_auto=True, 
                     aspect="auto", template="plotly_dark", title="Hardware Feature × Dominant Emotion (%)")

# 3. Urgency Heatmap
urgency_df = (df_valid.groupby("topic_label")[["emo_anger", "emo_sadness"]].mean() * 100).round(1)
urgency_df.columns = ["Anger", "Disappointment"]
fig_urgency = px.imshow(urgency_df, color_continuous_scale="YlOrRd", text_auto=True, 
                        aspect="auto", template="plotly_dark", title="Urgency Signal: Mean Probabilities")

# -- App Layout --
app.layout = dbc.Container([
    dbc.Row([
        # SIDEBAR
        dbc.Col([
            html.H3("⚙️ System Parameters", className="mt-4"),
            html.Hr(),
            html.P(f"Analyzed SKU: {summary.get('product_id', 'All')}"),
            html.P(f"Sample Size (N): {summary.get('total_reviews', 0):,}"),
            html.Hr(),
            html.H5("The 'Silent Killer' Heuristic"),
            html.P("Rating ≥ 4 Stars AND (Anger + Disappointment > 0.55). Represents immediate churn risk disguised by inflated star ratings.", className="text-muted text-sm")
        ], width=3, className="bg-dark text-white p-4 vh-100"),
        
        # MAIN CONTENT
        dbc.Col([
            html.H1("📊 Product Risk & ROI Triage Dashboard", className="mt-4"),
            html.P("Quantifying text-based sentiment to prioritize engineering resources.", className="lead"),
            html.Hr(),
            
            # KPI Cards
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🚨 Immediate Churn Risk", className="card-title text-danger"),
                    html.H2(f"{financials.get('silent_killers', 0):,}")
                ]), color="dark", inverse=True)),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("💸 Revenue at Risk (Est.)", className="card-title text-warning"),
                    html.H2(f"${financials.get('revenue_at_risk_usd', 0):,.0f}")
                ]), color="dark", inverse=True)),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5(f"📈 ROI: Fix {sim.get('target_aspect', 'Top Issue')}", className="card-title text-success"),
                    html.H2(f"+{sim.get('improvement_pct_high', 0)}%")
                ]), color="dark", inverse=True)),
            ], className="mb-4"),
            
            # Tabs
            dbc.Tabs([
                # TAB 1: Matrix
                dbc.Tab(label="🎯 Churn Risk Matrix", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig_matrix), width=12)
                    ], className="mt-4")
                ]),
                
                # TAB 2: Heatmaps
                dbc.Tab(label="🧠 Sentiment Distribution", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig_heat), width=6),
                        dbc.Col(dcc.Graph(figure=fig_urgency), width=6)
                    ], className="mt-4")
                ]),
                
                # TAB 3: Voice of Customer (Interactive Callback)
                dbc.Tab(label="🗣️ Root Cause Evidence", children=[
                    html.Div([
                        html.Label("Filter by Hardware Feature:", className="mt-4 fw-bold"),
                        dcc.Dropdown(
                            id='aspect-dropdown',
                            options=[{'label': i, 'value': i} for i in quotes_df["topic_label"].unique()],
                            value=quotes_df["topic_label"].unique()[0],
                            clearable=False,
                            className="text-dark mb-4"
                        ),
                        dbc.Row(id='quotes-container') # Populated by callback
                    ])
                ]),
            ])
        ], width=9, className="p-4")
    ])
], fluid=True)

# -- Callbacks (The magic of Dash) --
@app.callback(
    Output('quotes-container', 'children'),
    Input('aspect-dropdown', 'value')
)
def update_quotes(selected_aspect):
    filtered_quotes = quotes_df[quotes_df["topic_label"] == selected_aspect].head(3)
    cards = []
    for _, row in filtered_quotes.iterrows():
        text = row["quote_text"].strip()
        if len(text) > 300: text = text[:297] + "..."
        cards.append(dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H1("❝", className="text-danger"),
                    html.P(text, className="font-italic text-light"),
                    html.Hr(),
                    html.Small("Verified Defect Report", className="text-muted text-uppercase fw-bold")
                ]), color="dark", outline=True, style={"borderTop": "4px solid #ef4444"}
            ), width=4
        ))
    return cards

# -- Run Server --
if __name__ == '__main__':
	app.run(debug=True, port=8050)
