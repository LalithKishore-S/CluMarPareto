"""
CluMarPareto Benchmark Dashboard
---------------------------------
Place this file in the SAME folder as benchmarking.ipynb.
The notebook saves CSVs to Results_csv/ — this dashboard reads them directly.

Run:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Paths (same folder structure the notebook uses) ───────────────────────────
BASE_DIR     = Path(__file__).parent
RESULTS_CSV  = BASE_DIR / "Results_csv" / "benchmark_results_CluMarPareto_DBSCAN_IAMB_modified_NSGA2_GRA_weighted_optimised_run3.csv"
SUMMARY_CSV  = BASE_DIR / "Results_csv" / "benchmark_summary_CluMarPareto_DBSCAN_IAMB_modified_NSGA2_GRA_weighted_optimised_run3.csv"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CluMarPareto Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b0f1a; color: #e4e8f0; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

[data-testid="stSidebar"] {
    background: #0e1220 !important;
    border-right: 1px solid #1e2640;
}

.metric-card {
    background: linear-gradient(135deg, #141929 0%, #1a2035 100%);
    border: 1px solid #2a3050;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa);
}
.metric-label { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #6b7fa3; margin-bottom: 6px; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 28px; font-weight: 700; color: #e4e8f0; }
.metric-sub   { font-size: 12px; color: #4f8ef7; margin-top: 4px; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4f8ef7;
    border-bottom: 1px solid #2a3050;
    padding-bottom: 8px;
    margin-bottom: 20px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "CluMarPareto":     "#4f8ef7",
    "NSGA2 Standalone": "#a78bfa",
    "DBSCAN + IAMB":    "#34d399",
    "IAMB Only":        "#fb923c",
    "RF Importance":    "#facc15",
    "LASSO":            "#f472b6",
    "IAMB + NSGA2":     "#22d3ee",
}

# Fallback palette for any method names not in the dict above
FALLBACK_PALETTE = ["#e879f9","#4ade80","#f97316","#38bdf8","#fb7185","#a3e635","#fbbf24"]

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#111827",
    font=dict(color="#c9d1e0", family="DM Sans"),
    xaxis=dict(gridcolor="#1e2640", linecolor="#2a3050", zerolinecolor="#1e2640"),
    yaxis=dict(gridcolor="#1e2640", linecolor="#2a3050", zerolinecolor="#1e2640"),
    legend=dict(bgcolor="rgba(20,25,41,0.8)", bordercolor="#2a3050", borderwidth=1),
    margin=dict(l=10, r=10, t=40, b=10),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not RESULTS_CSV.exists():
        return None, None

    df = pd.read_csv(RESULTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    for col in ["runtime", "n_selected", "reduction_pct", "test_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = None
    if SUMMARY_CSV.exists():
        # summary is saved with index=True (method names are the index)
        summary = pd.read_csv(SUMMARY_CSV, index_col=0)
        summary.index.name = "method"
        summary = summary.reset_index()
        summary.columns = [c.strip() for c in summary.columns]

    return df, summary

df_full, summary_df = load_data()

# ── Guard: CSV not found ──────────────────────────────────────────────────────
if df_full is None:
    st.markdown("""
    <div style="max-width:560px;margin:100px auto;text-align:center;">
      <div style="font-family:'Space Mono',monospace;font-size:2rem;
        background:linear-gradient(90deg,#4f8ef7,#a78bfa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:16px;">
        ⚡ CluMarPareto
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.error(
        f"CSV not found at:\n\n`{RESULTS_CSV}`\n\n"
        "Make sure you have run all cells in **benchmarking.ipynb** first "
        "and that `dashboard.py` is in the same folder as the notebook.",
        icon="📂",
    )
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ CluMarPareto")
    st.markdown("*Benchmark Dashboard*")
    st.markdown("---")

    all_methods  = sorted(df_full["method"].unique().tolist())
    all_datasets = sorted(df_full["dataset"].unique().tolist())

    sel_methods  = st.multiselect("Methods",  all_methods,  default=all_methods)
    sel_datasets = st.multiselect("Datasets", all_datasets, default=all_datasets)

df = df_full.copy()
if sel_methods:
    df = df[df["method"].isin(sel_methods)]
if sel_datasets:
    df = df[df["dataset"].isin(sel_datasets)]

# Build color map from actual method names in the data — no hardcoded assumptions
all_actual_methods = sorted(df["method"].unique().tolist())
color_map = {}
fallback_idx = 0
for m in all_actual_methods:
    if m in METHOD_COLORS:
        color_map[m] = METHOD_COLORS[m]
    else:
        color_map[m] = FALLBACK_PALETTE[fallback_idx % len(FALLBACK_PALETTE)]
        fallback_idx += 1


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-family:'Space Mono',monospace;font-size:2rem;margin-bottom:0;
background:linear-gradient(90deg,#4f8ef7,#a78bfa);-webkit-background-clip:text;
-webkit-text-fill-color:transparent;">
CluMarPareto — Benchmark Results
</h1>
<p style="color:#6b7fa3;font-size:14px;margin-top:4px;">
Feature Selection Method Comparison · Runtime · Accuracy · Feature Reduction
</p>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Overview</p>', unsafe_allow_html=True)

best_acc = df.loc[df["test_acc"].idxmax()]
best_red = df.loc[df["reduction_pct"].idxmax()]
fastest  = df.loc[df["runtime"].idxmin()]

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Best Accuracy</div>
      <div class="metric-value">{best_acc['test_acc']:.1%}</div>
      <div class="metric-sub">{best_acc['method']} · {best_acc['dataset']}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Best Reduction</div>
      <div class="metric-value">{best_red['reduction_pct']:.1f}%</div>
      <div class="metric-sub">{best_red['method']} · {best_red['dataset']}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Fastest Method</div>
      <div class="metric-value">{fastest['runtime']:.1f}s</div>
      <div class="metric-sub">{fastest['method']} · {fastest['dataset']}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Avg Accuracy</div>
      <div class="metric-value">{df['test_acc'].mean():.1%}</div>
      <div class="metric-sub">Across all methods & datasets</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Datasets</div>
      <div class="metric-value">{df['dataset'].nunique()}</div>
      <div class="metric-sub">{df['method'].nunique()} methods compared</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Accuracy", "⏱️ Runtime", "✂️ Feature Reduction", "🔀 Trade-off", "🏆 Rankings"
])

# ── TAB 1: ACCURACY ───────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Test Accuracy by Method & Dataset</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(df, x="dataset", y="test_acc", color="method",
                     barmode="group", color_discrete_map=color_map,
                     title="Test Accuracy — Grouped by Dataset",
                     labels={"test_acc": "Test Accuracy", "dataset": "Dataset"})
        fig.update_yaxes(tickformat=".0%")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(apply_theme(fig), use_container_width=True)
    with c2:
        avg_m = df.groupby("method")["test_acc"].mean().reset_index().sort_values("test_acc")
        fig2 = px.bar(avg_m, x="test_acc", y="method", orientation="h",
                      color="method", color_discrete_map=color_map,
                      title="Avg Accuracy (all datasets)",
                      labels={"test_acc": "Mean Accuracy", "method": ""})
        fig2.update_xaxes(tickformat=".0%")
        fig2.update_traces(marker_line_width=0)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(apply_theme(fig2), use_container_width=True)

    st.markdown('<p class="section-header">Accuracy Heatmap</p>', unsafe_allow_html=True)
    pivot = df.pivot_table(index="method", columns="dataset", values="test_acc")
    fig3 = px.imshow(pivot, color_continuous_scale="Blues", text_auto=".2f",
                     aspect="auto", title="Test Accuracy Heatmap (method × dataset)")
    fig3.update_coloraxes(colorbar_tickformat=".0%")
    st.plotly_chart(apply_theme(fig3), use_container_width=True)

# ── TAB 2: RUNTIME ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Runtime Comparison</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(df, x="dataset", y="runtime", color="method",
                     barmode="group", color_discrete_map=color_map,
                     log_y=True, title="Runtime (seconds, log scale)",
                     labels={"runtime": "Runtime (s)", "dataset": "Dataset"})
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(apply_theme(fig), use_container_width=True)
    with c2:
        avg_rt = df.groupby("method")["runtime"].mean().reset_index().sort_values("runtime")
        fig2 = px.bar(avg_rt, x="runtime", y="method", orientation="h",
                      color="method", color_discrete_map=color_map,
                      log_x=True, title="Avg Runtime (log scale)",
                      labels={"runtime": "Seconds (log)", "method": ""})
        fig2.update_traces(marker_line_width=0)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(apply_theme(fig2), use_container_width=True)

    st.markdown('<p class="section-header">Runtime Distribution</p>', unsafe_allow_html=True)
    fig3 = px.box(df, x="method", y="runtime", color="method",
                  color_discrete_map=color_map, log_y=True, points="all",
                  title="Runtime Distribution per Method",
                  labels={"runtime": "Seconds (log)", "method": ""})
    fig3.update_layout(showlegend=False)
    st.plotly_chart(apply_theme(fig3), use_container_width=True)

# ── TAB 3: FEATURE REDUCTION ─────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Feature Reduction & Count</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df, x="dataset", y="reduction_pct", color="method",
                     barmode="group", color_discrete_map=color_map,
                     title="Feature Reduction %",
                     labels={"reduction_pct": "Reduction (%)", "dataset": "Dataset"})
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(apply_theme(fig), use_container_width=True)
    with c2:
        fig2 = px.bar(df, x="dataset", y="n_selected", color="method",
                      barmode="group", color_discrete_map=color_map,
                      title="Features Selected (count)",
                      labels={"n_selected": "# Features Selected", "dataset": "Dataset"})
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(apply_theme(fig2), use_container_width=True)

    st.markdown('<p class="section-header">Multi-Metric Method Profile (Radar)</p>', unsafe_allow_html=True)
    agg = df.groupby("method").agg(
        test_acc=("test_acc", "mean"),
        reduction_pct=("reduction_pct", "mean"),
        runtime=("runtime", "mean"),
    ).reset_index()
    rt_range = agg["runtime"].max() - agg["runtime"].min()
    agg["speed_score"] = 1 - (agg["runtime"] - agg["runtime"].min()) / (rt_range + 1e-9)

    fig_radar = go.Figure()
    for _, row in agg.iterrows():
        vals = [row["test_acc"], row["reduction_pct"] / 100, row["speed_score"]]
        cats = ["Accuracy", "Reduction %", "Speed Score"]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", name=row["method"],
            line_color=color_map.get(row["method"], "#94a3b8"),
            fillcolor=color_map.get(row["method"], "#94a3b8"),
            opacity=0.3,
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a3050", linecolor="#2a3050"),
            angularaxis=dict(gridcolor="#2a3050", linecolor="#2a3050"),
            bgcolor="#111827",
        ),
        title="Normalised Method Profile (avg across datasets)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1e0", family="DM Sans"),
        legend=dict(bgcolor="rgba(20,25,41,0.8)", bordercolor="#2a3050", borderwidth=1),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ── TAB 4: TRADE-OFF ──────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">Accuracy vs Runtime</p>', unsafe_allow_html=True)

    # Only plot rows where runtime and test_acc exist (all successful runs)
    df_scatter = df.dropna(subset=["runtime", "test_acc"]).copy()
    # Fill missing reduction_pct with a small default so bubble still renders
    df_scatter["reduction_pct"] = df_scatter["reduction_pct"].fillna(1.0).clip(lower=1.0)

    fig = px.scatter(df_scatter, x="runtime", y="test_acc", color="method",
                     symbol="dataset", size="reduction_pct",
                     color_discrete_map=color_map, log_x=True,
                     title="Accuracy vs Runtime (bubble size = feature reduction %)",
                     labels={"runtime": "Runtime (s, log)", "test_acc": "Test Accuracy"},
                     hover_data=["dataset", "n_selected", "reduction_pct"])
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    st.plotly_chart(apply_theme(fig), use_container_width=True)

    st.markdown('<p class="section-header">Accuracy vs Features Selected</p>', unsafe_allow_html=True)
    df_scatter2 = df.dropna(subset=["n_selected", "test_acc"]).copy()
    fig2 = px.scatter(df_scatter2, x="n_selected", y="test_acc", color="method",
                      symbol="dataset", color_discrete_map=color_map,
                      title="Accuracy vs Number of Selected Features",
                      labels={"n_selected": "Features Selected", "test_acc": "Test Accuracy"},
                      hover_data=["dataset", "runtime"])
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(apply_theme(fig2), use_container_width=True)

# ── TAB 5: RANKINGS ───────────────────────────────────────────────────────────
with tab5:
    st.markdown('<p class="section-header">Average Rank Summary (lower = better)</p>', unsafe_allow_html=True)

    if summary_df is not None:
        rank_table = summary_df.copy()
    else:
        r_acc  = df.pivot_table(index="dataset", columns="method", values="test_acc")
        r_rt   = df.pivot_table(index="dataset", columns="method", values="runtime")
        r_nsel = df.pivot_table(index="dataset", columns="method", values="n_selected")
        rank_table = pd.DataFrame({
            "Avg Rank (Accuracy ↑)": r_acc.rank(axis=1, ascending=False).mean(),
            "Avg Rank (Runtime ↓)":  r_rt.rank(axis=1, ascending=True).mean(),
            "Avg Rank (Features ↓)": r_nsel.rank(axis=1, ascending=True).mean(),
        }).dropna(how="all")
        rank_table["Overall Avg Rank"] = rank_table.mean(axis=1)
        rank_table = rank_table.sort_values("Overall Avg Rank").reset_index()

    num_cols = rank_table.select_dtypes("number").columns.tolist()
    st.dataframe(
        rank_table.style
            .format({c: "{:.2f}" for c in num_cols})
            .highlight_min(axis=0, subset=num_cols, color="#1a3a2a"),
        use_container_width=True,
    )

    if "Overall Avg Rank" in rank_table.columns and "method" in rank_table.columns:
        fig = px.bar(rank_table.sort_values("Overall Avg Rank"),
                     x="method", y="Overall Avg Rank",
                     color="method", color_discrete_map=color_map,
                     title="Overall Average Rank (lower = better)",
                     labels={"method": "", "Overall Avg Rank": "Avg Rank"})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    st.markdown('<p class="section-header">Raw Results Table</p>', unsafe_allow_html=True)
    display = df.copy()
    if "test_acc"      in display.columns: display["test_acc"]      = display["test_acc"].map("{:.1%}".format)
    if "reduction_pct" in display.columns: display["reduction_pct"] = display["reduction_pct"].map("{:.1f}%".format)
    if "runtime"       in display.columns: display["runtime"]       = display["runtime"].map("{:.2f}s".format)
    st.dataframe(display, use_container_width=True, height=400)

    st.download_button(
        "⬇️ Download Filtered Results CSV",
        data=df.to_csv(index=False).encode(),
        file_name="benchmark_filtered.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#2a3050;margin-top:40px;">
<p style="color:#3a4a6a;font-size:12px;text-align:center;font-family:'Space Mono',monospace;">
CluMarPareto · DBSCAN + IAMB + NSGA2 Feature Selection Pipeline
</p>
""", unsafe_allow_html=True)