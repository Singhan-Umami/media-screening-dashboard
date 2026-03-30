import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA  (reads testfile.csv as df8)
# =========================
df8 = pd.read_csv("testfile.csv", index_col=0)

# =========================
# FINAL MEDIA VARIABLES
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

# Keep only available columns
available_cols = [c for c in media_cols if c in df8.columns]
df8 = df8[available_cols].copy()

# Drop missing target
df8 = df8.dropna(subset=["Total Cell number"])

# Convert all to numeric
for col in df8.columns:
    df8[col] = pd.to_numeric(df8[col], errors='coerce')

df8 = df8.fillna(0)

# =========================
# MODEL
# =========================
X = df8.drop(columns=["Total Cell number"])
y = df8["Total Cell number"]

model = CatBoostRegressor(verbose=0)
model.fit(X, y)

# Feature Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.get_feature_importance()
}).sort_values(by="Importance", ascending=False)

# =========================
# PCA
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "TCC": y
})

# =========================
# DASH APP
# =========================
app = Dash(__name__)
server = app.server  # required for deployment (Render / Ploomber)

dist_options = [c for c in ["Total Cell number", "Protein µg/ml", "Lipid µg/ml"] if c in df8.columns]

app.layout = html.Div(style={
    "backgroundColor": "#0b0f14",
    "color": "white",
    "padding": "20px",
    "fontFamily": "sans-serif"
}, children=[

    html.H1("🧪 Media Screening Dashboard"),

    # -------------------------
    # DISTRIBUTION
    # -------------------------
    html.H3("Distribution Analysis"),
    dcc.Dropdown(
        id="dist-metric",
        options=[{"label": c, "value": c} for c in dist_options],
        value=dist_options[0] if dist_options else None,
        style={"color": "black"}
    ),
    dcc.Graph(id="dist-plot"),

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------
    html.H3("Sensitivity Analysis (Top 15 Features)"),
    dcc.Graph(
        figure=px.bar(
            importance_df.head(15),
            x="Importance",
            y="Feature",
            orientation="h",
            template="plotly_dark"
        )
    ),

    # -------------------------
    # PARTIAL DEPENDENCE
    # -------------------------
    html.H3("Sensitivity Curve"),
    dcc.Dropdown(
        id="pdp-feature",
        options=[{"label": c, "value": c} for c in X.columns if c != "Replicate"],
        value=[c for c in X.columns if c != "Replicate"][0],
        style={"color": "black"}
    ),
    dcc.Graph(id="pdp-plot"),

    # -------------------------
    # PCA PLOT
    # -------------------------
    html.H3("Formulation Space (PCA vs TCC)"),
    dcc.Graph(id="pca-plot"),
])

# =========================
# CALLBACKS
# =========================

@app.callback(
    Output("dist-plot", "figure"),
    Input("dist-metric", "value")
)
def update_dist(metric):
    fig = px.histogram(
        df8,
        x=metric,
        nbins=40,
        marginal="box",
        template="plotly_dark"
    )
    return fig


@app.callback(
    Output("pdp-plot", "figure"),
    Input("pdp-feature", "value")
)
def update_pdp(feature):
    grid = np.linspace(df8[feature].min(), df8[feature].max(), 50)
    preds = []
    for val in grid:
        temp = X.copy()
        temp[feature] = val
        preds.append(model.predict(temp).mean())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=preds, mode="lines"))
    fig.update_layout(
        title=f"{feature} vs Predicted TCC",
        xaxis_title=feature,
        yaxis_title="Predicted Total Cell Number",
        template="plotly_dark"
    )
    return fig


@app.callback(
    Output("pca-plot", "figure"),
    Input("dist-metric", "value")
)
def update_pca(_):
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="TCC",
        color_continuous_scale="viridis",
        template="plotly_dark",
        title="PCA Projection of Media Space (Colored by TCC)"
    )
    return fig


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
