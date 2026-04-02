import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import os

# =========================
# LOAD DATA
# =========================
df8 = pd.read_csv("testfile.csv", index_col=0)

# =========================
# MEDIA COLUMNS
# =========================
media_cols = [
    'FBS (%)', 'P/S(%)', 'Cell Boost 5 (%)', 'Cell Boost 6 (%)',
    'Yeast Extract (%)', 'Insulin (X)', 'Transferrin (X)',
    'Sodium Selenite (X)', 'Sodium Bicarbonate(%)', 'ITS',
    'bFGF (ng/ml)', 'Ascorbic acid (X)', 'Lipid mix (X)',
    'YE 903 (mg/mL)', 'Trace Element Mix', 'Vit B12',
    'Nutek YE FP100 (µL/mL)',
    'Kerry mix (1510/7504/4601N) (µL/mL)',
    'Chlorella (µL/mL)', 'Red Hardtii (µg/mL)',
    'Green Hardtii (µg/mL)', 'Nanochloropsis (µg/mL)',
    'Lglutathione reduced (ug/ml)', 'Superoxide Dismutase (X)',
    'Seratonine Creatinine Sulfate Monohydrate (ng/mL)',
    'NRG1 (ng/mL)', 'Replicate',
    'Oleic Acid  µM', 'Linoleic A µM', 'Lipoic Acid µM',
    'Ethanolamine(ug/ml)', 'Vit E µM', 'Seed Oil',
    'Phytosterols µg/m', 'Algal Oil(%)', 'Vit D µM',
    'Lysine µM', 'GAA µM', 'Resveratol µM', 'Quercetin µM',
    'Taurine', 'SOY', 'B Alanine', 'BSA(%)',
    'Protein µg/ml', 'ug protein/104 cells',
    'Lipid µg/ml', 'µg lipid/104 cell',
    'Total Cell number'
]

# Output columns — zeros are meaningless (not measured), replace with NaN
output_cols = [
    'Total Cell number', 'Protein µg/ml', 'ug protein/104 cells',
    'Lipid µg/ml', 'µg lipid/104 cell'
]

available_cols = [c for c in media_cols if c in df8.columns]
df8 = df8[available_cols].copy()

# FIX 1: Replace zeros in output columns with NaN
for col in output_cols:
    if col in df8.columns:
        df8[col] = df8[col].replace(0, np.nan)

# Drop rows missing target
df8 = df8.dropna(subset=["Total Cell number"])

# Convert to numeric
for col in df8.columns:
    df8[col] = pd.to_numeric(df8[col], errors='coerce')

# Fill input NaNs with 0 (inputs not added = 0)
input_cols = [c for c in df8.columns if c not in output_cols]
df8[input_cols] = df8[input_cols].fillna(0)

# Model copy — fill remaining output NaNs with 0 for training only
df8_model = df8.copy().fillna(0)

# =========================
# MODEL
# =========================
X = df8_model.drop(columns=["Total Cell number"])
y = df8_model["Total Cell number"]

model = CatBoostRegressor(verbose=0)
model.fit(X, y)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.get_feature_importance()
}).sort_values(by="Importance", ascending=False)

# =========================
# PCA — 3 components
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_ * 100
cumulative_variance = np.cumsum(explained_variance)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "PC3": X_pca[:, 2],
    "TCC": y.values
})

pc_options = [
    {"label": f"PC1 ({explained_variance[0]:.1f}% var)", "value": "PC1"},
    {"label": f"PC2 ({explained_variance[1]:.1f}% var)", "value": "PC2"},
    {"label": f"PC3 ({explained_variance[2]:.1f}% var)", "value": "PC3"},
]

# =========================
# DASH APP
# =========================
app = Dash(__name__)
server = app.server

dist_options = [c for c in ["Total Cell number", "Protein µg/ml", "Lipid µg/ml"] if c in df8.columns]
pdp_features = [c for c in X.columns if c != "Replicate"]

DARK = "#0b0f14"
CARD = "#131920"
ACCENT = "#00e5ff"
dropdown_style = {"color": "black", "backgroundColor": "white"}

app.layout = html.Div(style={"backgroundColor": DARK, "color": "white", "padding": "24px", "fontFamily": "sans-serif"}, children=[

    html.H1("🧪 Media Screening Dashboard", style={"color": ACCENT}),

    # ── DISTRIBUTION ──────────────────────────────────────────────
    html.H3("Distribution Analysis"),
    dcc.Dropdown(id="dist-metric", options=[{"label": c, "value": c} for c in dist_options],
                 value=dist_options[0], style=dropdown_style),
    dcc.Graph(id="dist-plot"),

    # ── FEATURE IMPORTANCE ────────────────────────────────────────
    html.H3("Sensitivity Analysis (Top 15)"),
    dcc.Graph(
        figure=px.bar(
            importance_df.head(15), x="Importance", y="Feature",
            orientation="h", template="plotly_dark",
            color="Importance", color_continuous_scale="teal"
        ).update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD)
    ),

    # ── SENSITIVITY CURVE — FIX 2 ─────────────────────────────────
    html.H3("Sensitivity Curve"),
    dcc.Dropdown(
        id="pdp-feature",
        options=[{"label": c, "value": c} for c in pdp_features],
        value=pdp_features[0],
        clearable=False,
        style=dropdown_style
    ),
    dcc.Graph(id="pdp-plot"),

    # ── PCA 3D — FIX 3 + 4 ───────────────────────────────────────
    html.H3("Formulation Space (3D PCA)"),

    # Variance explained chart
    dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(name="Per PC", x=["PC1", "PC2", "PC3"],
                       y=list(explained_variance), marker_color=ACCENT),
                go.Scatter(name="Cumulative %", x=["PC1", "PC2", "PC3"],
                           y=list(cumulative_variance), mode="lines+markers",
                           line=dict(color="orange", width=2), yaxis="y2")
            ],
            layout=go.Layout(
                title="Variance Explained by PC",
                template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
                yaxis=dict(title="% Variance per PC"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
                legend=dict(x=0.01, y=0.99)
            )
        )
    ),

    # Axis selectors
    html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "12px"}, children=[
        html.Div([html.Label("X Axis", style={"color": "white"}),
                  dcc.Dropdown(id="pca-x", options=pc_options, value="PC1", clearable=False, style=dropdown_style)],
                 style={"flex": 1}),
        html.Div([html.Label("Y Axis", style={"color": "white"}),
                  dcc.Dropdown(id="pca-y", options=pc_options, value="PC2", clearable=False, style=dropdown_style)],
                 style={"flex": 1}),
        html.Div([html.Label("Z Axis", style={"color": "white"}),
                  dcc.Dropdown(id="pca-z", options=pc_options, value="PC3", clearable=False, style=dropdown_style)],
                 style={"flex": 1}),
    ]),
    dcc.Graph(id="pca-plot"),

    # ── CLUSTERING — FIX 5 ────────────────────────────────────────
    html.H3("Clustering Analysis"),
    html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "12px"}, children=[
        html.Div([
            html.Label("Algorithm", style={"color": "white"}),
            dcc.Dropdown(
                id="cluster-algo",
                options=[
                    {"label": "KMeans", "value": "kmeans"},
                    {"label": "DBSCAN", "value": "dbscan"},
                    {"label": "Agglomerative", "value": "agglomerative"},
                ],
                value="kmeans", clearable=False, style=dropdown_style
            )
        ], style={"flex": 1}),
        html.Div([
            html.Label("Number of Clusters (KMeans / Agglomerative)", style={"color": "white"}),
            dcc.Slider(id="n-clusters", min=2, max=8, step=1, value=3,
                       marks={i: {"label": str(i), "style": {"color": "white"}} for i in range(2, 9)})
        ], style={"flex": 2}),
    ]),
    dcc.Graph(id="cluster-plot"),
])

# =========================
# CALLBACKS
# =========================

@app.callback(Output("dist-plot", "figure"), Input("dist-metric", "value"))
def update_dist(metric):
    fig = px.histogram(df8, x=metric, nbins=40, marginal="box",
                       template="plotly_dark", color_discrete_sequence=[ACCENT])
    fig.update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD)
    return fig


# FIX 2: Proper sensitivity curve — iterates over feature grid correctly
@app.callback(Output("pdp-plot", "figure"), Input("pdp-feature", "value"))
def update_pdp(feature):
    col_min = float(df8_model[feature].min())
    col_max = float(df8_model[feature].max())
    if col_min == col_max:
        col_max = col_min + 1.0
    grid = np.linspace(col_min, col_max, 50)
    preds = []
    for val in grid:
        temp = X.copy()
        temp[feature] = val
        preds.append(float(model.predict(temp).mean()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid, y=preds, mode="lines",
        line=dict(color=ACCENT, width=3),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.08)"
    ))
    fig.update_layout(
        title=f"{feature} vs Predicted TCC",
        xaxis_title=feature, yaxis_title="Predicted Total Cell Number",
        template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD
    )
    return fig


# FIX 3: 3D PCA with selectable axes
@app.callback(
    Output("pca-plot", "figure"),
    Input("pca-x", "value"), Input("pca-y", "value"), Input("pca-z", "value")
)
def update_pca(x_ax, y_ax, z_ax):
    fig = px.scatter_3d(
        pca_df, x=x_ax, y=y_ax, z=z_ax, color="TCC",
        color_continuous_scale="viridis", template="plotly_dark",
        title=f"3D PCA: {x_ax} vs {y_ax} vs {z_ax}"
    )
    fig.update_layout(paper_bgcolor=CARD)
    return fig


# FIX 5: Multiple clustering algorithms
@app.callback(
    Output("cluster-plot", "figure"),
    Input("cluster-algo", "value"), Input("n-clusters", "value"),
    Input("pca-x", "value"), Input("pca-y", "value"), Input("pca-z", "value")
)
def update_cluster(algo, n_clusters, x_ax, y_ax, z_ax):
    if algo == "kmeans":
        labels = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(X_scaled)
    elif algo == "dbscan":
        labels = DBSCAN(eps=0.5, min_samples=3).fit_predict(X_scaled)
    else:
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_scaled)

    plot_df = pca_df.copy()
    plot_df["Cluster"] = labels.astype(str)

    fig = px.scatter_3d(
        plot_df, x=x_ax, y=y_ax, z=z_ax, color="Cluster",
        template="plotly_dark",
        title=f"{algo.capitalize()} Clustering — {x_ax} / {y_ax} / {z_ax}"
    )
    fig.update_layout(paper_bgcolor=CARD)
    return fig


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
