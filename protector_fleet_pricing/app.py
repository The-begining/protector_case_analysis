"""
Protector Fleet Pricing Engine — Streamlit Dashboard
Single-file app with deep client analysis, portfolio benchmarking, and interactive pricing.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline import run_pipeline, VALID_UW_CATEGORIES
from pricing import (
    client_risk_metrics, category_risk_metrics, portfolio_averages,
    calculate_price, PricingParams, sensitivity_analysis, build_category_risk_factors,
)

# ── Page Config ──
st.set_page_config(
    page_title="Fleet Pricing Engine", page_icon="🚗",
    layout="wide", initial_sidebar_state="expanded",
)


# Data Loading (cached)
@st.cache_data(show_spinner="Loading and processing data...")
def get_data():
    return run_pipeline()


@st.cache_data(show_spinner="Calculating metrics...")
def get_metrics(_fleet, _claims):
    metrics = client_risk_metrics(_fleet, _claims)
    avgs = portfolio_averages(metrics)
    return metrics, avgs


# Sidebar — Client Selector + Pricing Sliders
def render_sidebar(clients: list[str]) -> tuple[str, PricingParams]:
    with st.sidebar:
        st.header("⚙️ Controls")

        default_idx = clients.index("Client - 18") if "Client - 18" in clients else 0
        client = st.selectbox("Select Client", clients, index=default_idx)

        st.divider()
        st.subheader("📐 Pricing Assumptions")
        st.caption("Adjust these to see instant impact on the price proposal")

        # Frequency & severity
        freq_adj = st.slider("Frequency Adjustment", 0.5, 2.0, 1.0, 0.05, format="%.2f×",
                              help="Multiplier on observed claim frequency")
        severity_adj = st.slider("Severity Adjustment", 0.5, 2.0, 1.0, 0.05, format="%.2f×",
                                  help="Multiplier on observed average claim cost")

        cost_mode = st.radio("Average Claim Cost", ["Use observed (adjusted)", "Override manually"], horizontal=True)
        cost_override = None
        if cost_mode == "Override manually":
            cost_override = st.number_input("Override (SEK)", 0, 500000, 50000, 5000)

        st.divider()
        st.subheader("📊 Risk Model")

        use_risk_factors = st.toggle("Category risk factors", value=True,
                                      help="Apply different risk weights per UW category")
        fleet_age_adj = st.toggle("Fleet age adjustment", value=True,
                                   help="Older-than-median fleets get +10% risk loading")
        large_loss_cap = st.number_input("Large loss cap (SEK)", 0, 1000000, 0, 50000,
                                          help="Cap individual claims at this amount (0 = no cap)")

        st.divider()
        st.subheader("💰 Loadings")

        margin_pct = st.slider("Profit Margin (%)", 0, 30, 10, 1)
        expense_pct = st.slider("Expense Loading (%)", 5, 25, 15, 1)
        reinsurance_pct = st.slider("Reinsurance Loading (%)", 0, 15, 5, 1)

        margin = margin_pct / 100
        expense = expense_pct / 100
        reinsurance = reinsurance_pct / 100

        params = PricingParams(
            frequency_adj=freq_adj,
            severity_adj=severity_adj,
            avg_cost_override=cost_override,
            use_risk_factors=use_risk_factors,
            profit_margin=margin,
            expense_loading=expense,
            reinsurance_loading=reinsurance,
            fleet_age_adj=fleet_age_adj,
            large_loss_cap=large_loss_cap,
        )

        st.divider()
        total_load = params.total_loading
        st.metric("Total Loading", f"{total_load:.1%}")

        if st.button("🔄 Reset to Defaults"):
            st.rerun()

    return client, params


# Tab 1: Client Deep-Dive
def tab_deep_dive(client: str, fleet: pd.DataFrame, claims: pd.DataFrame,
                  metrics: pd.DataFrame, avgs: dict, params: PricingParams):
    cf = fleet[fleet["Client"] == client]
    cc = claims[claims["Client"] == client]
    cm = metrics[metrics["Client"] == client]

    if cm.empty:
        st.error(f"No data found for {client}")
        return

    cm_row = cm.iloc[0]

    # ── KPI Row: Client vs Portfolio ──
    st.subheader("Key Metrics vs Portfolio")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Fleet Size", f"{int(cm_row['vehicle_count']):,}",
              delta=f"{cm_row['vehicle_count'] - avgs['avg_fleet_size']:+.0f} vs avg",
              delta_color="off")
    k2.metric("Total Claims", f"{int(cm_row['claim_count']):,}")
    k3.metric("Claim Frequency", f"{cm_row['claim_frequency']:.3f}",
              delta=f"{cm_row['claim_frequency'] - avgs['avg_claim_frequency']:+.4f} vs avg",
              delta_color="inverse")
    k4.metric("Avg Claim Cost", f"{cm_row['avg_claim_cost']:,.0f} SEK",
              delta=f"{cm_row['avg_claim_cost'] - avgs['avg_claim_cost']:+,.0f} vs avg",
              delta_color="inverse")
    k5.metric("Loss / Vehicle", f"{cm_row['loss_per_vehicle']:,.0f} SEK",
              delta=f"{cm_row['loss_per_vehicle'] - avgs['avg_loss_per_vehicle']:+,.0f} vs avg",
              delta_color="inverse")

    st.divider()

    # ── Price Proposal ──
    price = calculate_price(fleet, claims, client, params)

    st.subheader("💰 Price Proposal")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Total Premium", f"{price['total_price']:,.0f} SEK")
    p2.metric("Pure Premium", f"{price['pure_premium']:,.0f} SEK")
    p3.metric("Per Vehicle", f"{price['price_per_vehicle']:,.0f} SEK")
    p4.metric("Loading Factor", f"{price['loading_factor']:.1%}")
    p5.metric("Exposure Years", f"{price['exposure_years']}")

    # Assumptions transparency panel
    with st.expander("📋 Active Assumptions (underwriter review)", expanded=True):
        assumptions = price.get("assumptions", [])
        a_col1, a_col2 = st.columns(2)
        mid = len(assumptions) // 2
        with a_col1:
            for name, val in assumptions[:mid]:
                st.markdown(f"**{name}:** {val}")
        with a_col2:
            for name, val in assumptions[mid:]:
                st.markdown(f"**{name}:** {val}")

    with st.expander("📋 Price Breakdown by UW Category", expanded=True):
        breakdown = price["per_category"].copy()
        display_cols = ["UW_Kategori", "vehicle_count", "risk_factor", "adj_frequency",
                        "expected_claims", "expected_cost", "pure_premium", "gross_price", "price_per_vehicle"]
        display_cols = [c for c in display_cols if c in breakdown.columns]
        breakdown = breakdown[display_cols]
        col_names = {
            "UW_Kategori": "UW Category", "vehicle_count": "Vehicles",
            "risk_factor": "Risk Factor", "adj_frequency": "Adj. Frequency",
            "expected_claims": "Exp. Claims", "expected_cost": "Exp. Cost (SEK)",
            "pure_premium": "Pure Premium", "gross_price": "Gross Price (SEK)",
            "price_per_vehicle": "Per Vehicle (SEK)",
        }
        breakdown = breakdown.rename(columns=col_names)
        st.dataframe(
            breakdown.style.format({
                "Risk Factor": "{:.2f}",
                "Adj. Frequency": "{:.4f}",
                "Exp. Claims": "{:.1f}",
                "Exp. Cost (SEK)": "{:,.0f}",
                "Pure Premium": "{:,.0f}",
                "Gross Price (SEK)": "{:,.0f}",
                "Per Vehicle (SEK)": "{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # ── Fleet Composition Deep Analysis ──
    st.subheader("🚗 Fleet Composition")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        cat_counts = cf["UW_Kategori"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.bar(cat_counts.sort_values("Count"), x="Count", y="Category",
                     orientation="h", color="Count", color_continuous_scale="Blues",
                     title="Vehicles by UW Category")
        fig.update_layout(showlegend=False, height=400, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Classification method breakdown
        method_counts = cf["classification_method"].value_counts().reset_index()
        method_counts.columns = ["Method", "Count"]
        fig2 = px.pie(method_counts, values="Count", names="Method",
                      title="Classification Methods", hole=0.4)
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Fleet details table
    col_c, col_d = st.columns([1, 1])
    with col_c:
        # Brand distribution
        brand_counts = cf["Marke"].value_counts().head(10).reset_index()
        brand_counts.columns = ["Brand", "Count"]
        fig3 = px.bar(brand_counts, x="Brand", y="Count", title="Top 10 Vehicle Brands",
                      color="Count", color_continuous_scale="Viridis")
        fig3.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        # Model year distribution
        if cf["Arsmodell"].notna().any():
            year_counts = cf["Arsmodell"].value_counts().sort_index().reset_index()
            year_counts.columns = ["Year", "Count"]
            fig4 = px.bar(year_counts, x="Year", y="Count", title="Fleet Age Distribution",
                          color="Count", color_continuous_scale="RdYlGn")
            fig4.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig4, use_container_width=True)

    # Body type (Karosserikod) breakdown
    kaross_counts = cf["Karosserikod"].value_counts().head(10).reset_index()
    kaross_counts.columns = ["Body Type", "Count"]
    fig_k = px.bar(kaross_counts, x="Body Type", y="Count",
                   title="Top 10 Body Types (Karosserikod)", color="Count",
                   color_continuous_scale="Tealgrn")
    fig_k.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_k, use_container_width=True)

    st.divider()

    # ── Claims Deep Analysis ──
    st.subheader("📊 Claims Analysis")

    if cc.empty:
        st.info("No claims data for this client.")
        return

    # Claims over time — dual axis
    yearly = cc.groupby("CLAIM_YEAR").agg(
        claim_count=("Client", "size"),
        total_incurred=("Incurred idx", "sum"),
        avg_cost=("Incurred idx", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        non_zero=("Ex. 0", "sum"),
        minor=("minor", "sum"),
        major=("major", "sum"),
    ).reset_index()

    col_e, col_f = st.columns([3, 2])
    with col_e:
        fig5 = make_subplots(specs=[[{"secondary_y": True}]])
        fig5.add_trace(go.Bar(x=yearly["CLAIM_YEAR"], y=yearly["claim_count"],
                              name="Claims Count", marker_color="#636EFA"), secondary_y=False)
        fig5.add_trace(go.Scatter(x=yearly["CLAIM_YEAR"], y=yearly["total_incurred"],
                                  name="Total Incurred (idx)", mode="lines+markers",
                                  marker_color="#EF553B"), secondary_y=True)
        fig5.update_layout(title="Claims Over Time", height=380)
        fig5.update_yaxes(title_text="Count", secondary_y=False)
        fig5.update_yaxes(title_text="Incurred (idx)", secondary_y=True)
        st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        # Claim type breakdown
        if "CLAIM_TYPE" in cc.columns:
            type_counts = cc["CLAIM_TYPE"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            fig6 = px.pie(type_counts, values="Count", names="Type",
                          title="Claim Types", hole=0.4)
            fig6.update_layout(height=380)
            st.plotly_chart(fig6, use_container_width=True)

    # Claim cause analysis
    col_g, col_h = st.columns([1, 1])
    with col_g:
        if "CLAIM_CAUSE" in cc.columns:
            cause_counts = cc["CLAIM_CAUSE"].value_counts().head(8).reset_index()
            cause_counts.columns = ["Cause", "Count"]
            fig7 = px.bar(cause_counts, x="Count", y="Cause", orientation="h",
                          title="Top Claim Causes", color="Count",
                          color_continuous_scale="Reds")
            fig7.update_layout(showlegend=False, height=350, yaxis_title="")
            st.plotly_chart(fig7, use_container_width=True)

    with col_h:
        # Minor vs Major claims over time
        fig8 = go.Figure()
        fig8.add_trace(go.Bar(x=yearly["CLAIM_YEAR"], y=yearly["minor"],
                              name="Minor", marker_color="#00CC96"))
        fig8.add_trace(go.Bar(x=yearly["CLAIM_YEAR"], y=yearly["major"],
                              name="Major", marker_color="#EF553B"))
        fig8.update_layout(title="Minor vs Major Claims", barmode="stack", height=350)
        st.plotly_chart(fig8, use_container_width=True)

    # Severity distribution
    non_zero_claims = cc[cc["Incurred idx"] > 0]
    if not non_zero_claims.empty:
        fig9 = px.histogram(non_zero_claims, x="Incurred idx", nbins=30,
                            title="Claim Severity Distribution (non-zero)",
                            labels={"Incurred idx": "Incurred Amount (idx)"},
                            color_discrete_sequence=["#636EFA"])
        fig9.update_layout(height=300)
        st.plotly_chart(fig9, use_container_width=True)

    # Claims summary stats
    st.markdown("#### Claims Summary Statistics")
    stats_data = {
        "Metric": ["Total Claims", "Non-Zero Claims", "Zero Claims",
                    "Minor Claims", "Major Claims",
                    "Total Incurred (idx)", "Avg Cost (non-zero)",
                    "Median Cost (non-zero)", "Max Single Claim",
                    "Claim Years Span"],
        "Value": [
            f"{len(cc):,}",
            f"{int(cc['Ex. 0'].sum()):,}" if "Ex. 0" in cc.columns else "N/A",
            f"{len(cc) - int(cc['Ex. 0'].sum()):,}" if "Ex. 0" in cc.columns else "N/A",
            f"{int(cc['minor'].sum()):,}" if "minor" in cc.columns else "N/A",
            f"{int(cc['major'].sum()):,}" if "major" in cc.columns else "N/A",
            f"{cc['Incurred idx'].sum():,.0f}",
            f"{non_zero_claims['Incurred idx'].mean():,.0f}" if not non_zero_claims.empty else "N/A",
            f"{non_zero_claims['Incurred idx'].median():,.0f}" if not non_zero_claims.empty else "N/A",
            f"{cc['Incurred idx'].max():,.0f}",
            f"{int(cc['CLAIM_YEAR'].max() - cc['CLAIM_YEAR'].min() + 1)}" if cc["CLAIM_YEAR"].notna().any() else "N/A",
        ],
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # Full vehicle table
    with st.expander("📋 Full Vehicle List"):
        display_cols = ["regnr", "Marke", "Karosserikod", "Fordonsslag",
                        "Arsmodell", "TotalVikt", "UW_Kategori", "classification_method"]
        avail = [c for c in display_cols if c in cf.columns]
        st.dataframe(cf[avail].sort_values("UW_Kategori"), use_container_width=True, hide_index=True)


# Tab 2: Portfolio Benchmark
def tab_benchmark(client: str, fleet: pd.DataFrame, claims: pd.DataFrame,
                  metrics: pd.DataFrame, avgs: dict):

    st.subheader("Portfolio Overview")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Clients", len(metrics))
    o2.metric("Total Vehicles", f"{avgs['total_vehicles']:,}")
    o3.metric("Total Claims", f"{avgs['total_claims']:,}")
    o4.metric("Avg Frequency", f"{avgs['avg_claim_frequency']:.3f}")

    st.divider()

    # ── Fleet Size Comparison ──
    col1, col2 = st.columns(2)
    with col1:
        df_sorted = metrics.sort_values("vehicle_count", ascending=True)
        colors = ["#EF553B" if c == client else "#636EFA" for c in df_sorted["Client"]]
        fig = go.Figure(go.Bar(x=df_sorted["vehicle_count"], y=df_sorted["Client"],
                               orientation="h", marker_color=colors))
        fig.update_layout(title=f"Fleet Size (▪ {client})", height=500, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk scatter
        df = metrics.copy()
        df["highlight"] = df["Client"] == client
        fig2 = px.scatter(df, x="vehicle_count", y="claim_frequency",
                          size="total_incurred", color="highlight",
                          color_discrete_map={True: "#EF553B", False: "#636EFA"},
                          hover_data=["Client"], title="Fleet Size vs Claim Frequency",
                          labels={"vehicle_count": "Fleet Size",
                                  "claim_frequency": "Claim Frequency"})
        fig2.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Loss per Vehicle Comparison ──
    col3, col4 = st.columns(2)
    with col3:
        df_sorted2 = metrics.sort_values("loss_per_vehicle", ascending=True)
        colors2 = ["#EF553B" if c == client else "#00CC96" for c in df_sorted2["Client"]]
        fig3 = go.Figure(go.Bar(x=df_sorted2["loss_per_vehicle"], y=df_sorted2["Client"],
                                orientation="h", marker_color=colors2))
        fig3.update_layout(title=f"Loss per Vehicle (▪ {client})", height=500, yaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Avg claim cost comparison
        df_sorted3 = metrics.sort_values("avg_claim_cost", ascending=True)
        colors3 = ["#EF553B" if c == client else "#AB63FA" for c in df_sorted3["Client"]]
        fig4 = go.Figure(go.Bar(x=df_sorted3["avg_claim_cost"], y=df_sorted3["Client"],
                                orientation="h", marker_color=colors3))
        fig4.update_layout(title=f"Avg Claim Cost (▪ {client})", height=500, yaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Radar Chart: Client vs Portfolio ──
    st.subheader(f"Radar: {client} vs Portfolio Average")
    client_row = metrics[metrics["Client"] == client].iloc[0]

    radar_metrics = ["vehicle_count", "claim_frequency", "avg_claim_cost", "loss_per_vehicle"]
    radar_labels = ["Fleet Size", "Claim Frequency", "Avg Claim Cost", "Loss/Vehicle"]

    # Normalize to 0-1 scale for radar
    client_vals, avg_vals = [], []
    for m in radar_metrics:
        max_val = metrics[m].max() if metrics[m].max() > 0 else 1
        client_vals.append(client_row[m] / max_val)
        avg_vals.append(avgs.get(f"avg_{m}", metrics[m].mean()) / max_val)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatterpolar(r=client_vals + [client_vals[0]],
                                    theta=radar_labels + [radar_labels[0]],
                                    name=client, fill="toself", fillcolor="rgba(239,85,59,0.2)",
                                    line_color="#EF553B"))
    fig5.add_trace(go.Scatterpolar(r=avg_vals + [avg_vals[0]],
                                    theta=radar_labels + [radar_labels[0]],
                                    name="Portfolio Avg", fill="toself",
                                    fillcolor="rgba(99,110,250,0.2)", line_color="#636EFA"))
    fig5.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=450)
    st.plotly_chart(fig5, use_container_width=True)

    # ── Ranking Table ──
    st.subheader("Client Ranking Table")
    display = metrics[["Client", "vehicle_count", "claim_count", "claim_frequency",
                       "avg_claim_cost", "total_incurred", "loss_per_vehicle",
                       "minor_claims", "major_claims"]].copy()
    display = display.sort_values("vehicle_count", ascending=False)

    def highlight_client(row):
        return ["background-color: #fff3cd" if row["Client"] == client else "" for _ in row]

    st.dataframe(
        display.style.apply(highlight_client, axis=1).format({
            "claim_frequency": "{:.4f}",
            "avg_claim_cost": "{:,.0f}",
            "total_incurred": "{:,.0f}",
            "loss_per_vehicle": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # ── UW Category Composition Comparison ──
    st.subheader("Fleet Composition Comparison")
    cat_comp = fleet.groupby(["Client", "UW_Kategori"]).size().reset_index(name="count")
    cat_comp_pct = cat_comp.copy()
    totals = cat_comp_pct.groupby("Client")["count"].transform("sum")
    cat_comp_pct["pct"] = cat_comp_pct["count"] / totals * 100

    fig6 = px.bar(cat_comp_pct, x="Client", y="pct", color="UW_Kategori",
                  title="Fleet Composition by Client (%)",
                  labels={"pct": "% of Fleet"})
    fig6.update_layout(height=450, barmode="stack")
    st.plotly_chart(fig6, use_container_width=True)


# Tab 3: Classification Review
def tab_classification(fleet: pd.DataFrame):
    st.subheader("Classification Coverage")

    # Apply any stored overrides from session state
    overrides = st.session_state.get("classification_overrides", {})
    if overrides:
        for regnr, new_cat in overrides.items():
            mask = fleet["regnr"] == regnr
            fleet.loc[mask, "UW_Kategori"] = new_cat
            fleet.loc[mask, "classification_method"] = "manual"

    total = len(fleet)
    method_counts = fleet["classification_method"].value_counts()

    cols = st.columns(len(method_counts) + 1)
    cols[0].metric("Total Vehicles", f"{total:,}")
    for i, (method, count) in enumerate(method_counts.items()):
        pct = count / total * 100
        cols[i + 1].metric(method, f"{count:,} ({pct:.1f}%)")

    st.divider()

    # Vehicles classified by heuristic (need review)
    heuristic = fleet[fleet["classification_method"] == "heuristic"].copy()
    if not heuristic.empty:
        st.subheader(f"Heuristic Classifications ({len(heuristic)} vehicles)")
        st.markdown("These vehicles were not found in API data and classified by Fordonsslag rules. "
                    "Consider verifying with the underwriter.")
        display_cols = ["Client", "regnr", "Marke", "Karosserikod", "Fordonsslag",
                        "TotalVikt", "UW_Kategori"]
        avail = [c for c in display_cols if c in heuristic.columns]
        st.dataframe(heuristic[avail].sort_values(["Client", "UW_Kategori"]),
                     use_container_width=True, hide_index=True)

    st.divider()

    # Manual override — human-in-the-loop for heuristic vehicles
    non_rule = fleet[fleet["classification_method"] == "heuristic"].copy()
    if not non_rule.empty:
        st.subheader("✏️ Manual Override")
        st.markdown("Select a heuristic-classified vehicle to override its category.")
        regnr_opts = non_rule["regnr"].dropna().unique().tolist()
        if regnr_opts:
            sel = st.selectbox("Vehicle (regnr)", regnr_opts)
            sel_row = non_rule[non_rule["regnr"] == sel].iloc[0]
            st.caption(f"Current: **{sel_row['UW_Kategori']}** | "
                       f"Brand: {sel_row.get('Marke','?')} | "
                       f"Body: {sel_row.get('Karosserikod','?')} | "
                       f"Type: {sel_row.get('Fordonsslag','?')}")
            new_cat = st.selectbox("Assign UW Category", VALID_UW_CATEGORIES)
            if st.button("✅ Apply Override"):
                if "classification_overrides" not in st.session_state:
                    st.session_state["classification_overrides"] = {}
                st.session_state["classification_overrides"][sel] = new_cat
                st.success(f"Override saved: {sel} → {new_cat}")
                st.rerun()
    else:
        st.success("All vehicles classified via rule-based mapping — no manual review needed!")


# Tab 3: Pricing Model & Assumptions (Underwriter UI)
def tab_pricing_model(client: str, fleet: pd.DataFrame, claims: pd.DataFrame,
                      params: PricingParams):
    """Dedicated pricing tab — exposes all assumptions for underwriter review."""

    st.subheader("📐 Pricing Model Methodology")
    st.markdown("""
    **Model:** Risk-Differentiated Burning Cost with Category-Level Risk Factors

    ```
    Per Category:
        Pure Premium = Vehicles × (Base Frequency × Risk Factor × Age Adj × Freq Adj) × (Avg Severity × Sev Adj)
        Gross Premium = Pure Premium × (1 + Profit Margin + Expense Loading + Reinsurance Loading)
    ```

    This model goes beyond flat-rate burning cost by applying **16 category-specific risk factors**
    derived from vehicle characteristics (type, weight, usage pattern). The underwriter can toggle
    each assumption using the sidebar controls and see instant impact below.
    """)

    st.divider()

    # ── Current assumptions summary ──
    st.subheader("⚙️ Active Assumptions")
    price = calculate_price(fleet, claims, client, params)

    if "error" in price:
        st.error(price["error"])
        return

    assumptions = price.get("assumptions", [])
    cols = st.columns(3)
    per_col = len(assumptions) // 3 + 1
    for i, col in enumerate(cols):
        with col:
            for name, val in assumptions[i * per_col : (i + 1) * per_col]:
                st.markdown(f"• **{name}:** `{val}`")

    st.divider()

    # ── Category Risk Factors Table ──
    st.subheader("📊 Category Risk Factors (Data-Driven)")
    st.markdown("""
    Risk factors are **computed from actual claims data** using `FLEET_TYPE`:
    - `Loss Rate = (Claims / Vehicles / Years) × Avg Severity`
    - `Risk Factor = Loss Rate(category) / Loss Rate(Personbil)`
    - **Personbil = 1.00** (baseline). Categories without claims data are interpolated.
    """)

    cat_risk = build_category_risk_factors(fleet, claims)
    display_cols = ["UW_Kategori", "vehicle_count", "risk_factor"]
    col_names = {"UW_Kategori": "UW Category", "vehicle_count": "Portfolio Vehicles",
                 "risk_factor": "Risk Factor"}
    if "factor_source" in cat_risk.columns:
        display_cols.append("factor_source")
        col_names["factor_source"] = "Source"
    cat_display = cat_risk[display_cols].copy()
    cat_display = cat_display.sort_values("risk_factor", ascending=False)
    cat_display = cat_display.rename(columns=col_names)

    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.dataframe(
            cat_display.style.format({"Risk Factor": "{:.2f}"}).background_gradient(
                subset=["Risk Factor"], cmap="RdYlGn_r"
            ),
            use_container_width=True, hide_index=True,
        )

    with col_b:
        fig_rf = px.bar(
            cat_display.sort_values("Risk Factor"),
            x="Risk Factor", y="UW Category",
            orientation="h", color="Risk Factor",
            color_continuous_scale="RdYlGn_r",
            title="Risk Factor by UW Category",
        )
        fig_rf.add_vline(x=1.0, line_dash="dash", line_color="black",
                         annotation_text="Baseline (1.0)")
        fig_rf.update_layout(height=500, showlegend=False, yaxis_title="")
        st.plotly_chart(fig_rf, use_container_width=True)

    st.divider()

    # ── Price Breakdown with Risk Factors ──
    st.subheader(f"💰 {client} — Price Breakdown")
    breakdown = price["per_category"].copy()
    display_cols = ["UW_Kategori", "vehicle_count", "risk_factor", "age_adj",
                    "adj_frequency", "expected_claims", "pure_premium", "gross_price",
                    "price_per_vehicle"]
    display_cols = [c for c in display_cols if c in breakdown.columns]
    bk = breakdown[display_cols].copy()
    col_names = {
        "UW_Kategori": "UW Category", "vehicle_count": "Vehicles",
        "risk_factor": "Risk Factor", "age_adj": "Age Adj",
        "adj_frequency": "Adj. Frequency", "expected_claims": "Exp. Claims",
        "pure_premium": "Pure Premium (SEK)", "gross_price": "Gross Premium (SEK)",
        "price_per_vehicle": "Per Vehicle (SEK)",
    }
    bk = bk.rename(columns=col_names)

    st.dataframe(
        bk.style.format({
            "Risk Factor": "{:.2f}", "Age Adj": "{:.2f}",
            "Adj. Frequency": "{:.4f}", "Exp. Claims": "{:.1f}",
            "Pure Premium (SEK)": "{:,.0f}", "Gross Premium (SEK)": "{:,.0f}",
            "Per Vehicle (SEK)": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # Summary metrics
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Pure Premium", f"{price['pure_premium']:,.0f} SEK")
    s2.metric("Gross Premium", f"{price['total_price']:,.0f} SEK")
    s3.metric("Per Vehicle", f"{price['price_per_vehicle']:,.0f} SEK")
    s4.metric("Loading", f"{price['loading_factor']:.1%}")

    st.divider()

    # ── Sensitivity Analysis ──
    st.subheader("📈 Sensitivity Analysis")
    st.markdown("How does the price change when we vary each assumption?")

    sens = sensitivity_analysis(fleet, claims, client, params)

    for assumption in sens["Assumption"].unique():
        sub = sens[sens["Assumption"] == assumption].copy()
        col_l, col_r = st.columns([2, 3])

        with col_l:
            st.markdown(f"**{assumption}**")
            st.dataframe(
                sub[["Scenario", "Total Price (SEK)", "Change %"]].style.format({
                    "Total Price (SEK)": "{:,.0f}",
                    "Change %": "{:+.1f}%",
                }),
                use_container_width=True, hide_index=True,
            )

        with col_r:
            fig_s = px.bar(
                sub, x="Scenario", y="Total Price (SEK)",
                color="Change %", color_continuous_scale="RdYlGn_r",
                title=f"Impact of {assumption} on Total Premium",
            )
            fig_s.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_s, use_container_width=True)

    st.divider()
    st.subheader("ℹ️ Model Documentation")
    with st.expander("How does this pricing model work?", expanded=False):
        st.markdown("""
        ### Burning Cost with Risk Differentiation

        **1. Base Metrics (from historical data)**
        - **Claim Frequency** = Total claims ÷ Total vehicles ÷ Exposure years
        - **Average Severity** = Mean of non-zero claims (indexed)

        **2. Category Risk Factors (Data-Driven)**
        - Computed from `FLEET_TYPE` in claims data: `Loss Rate = Frequency × Severity`
        - Risk Factor = `Loss Rate(category) / Loss Rate(Personbil)`
        - Example: Tung lastbil = 3.12× (low freq but very high severity ~57K SEK)
        - Släp = 0.02× (almost never claims)
        - Categories without claims data are interpolated from related types

        **3. Fleet Age Adjustment**
        - Categories where the average model year is below the portfolio median
          receive a +10% risk loading (older vehicles = more claims)

        **4. Large Loss Cap**
        - Optional: cap individual claim amounts to reduce impact of outliers
        - When set, recalculates average severity from capped claims

        **5. Loadings**
        - Profit margin: target return for the insurer
        - Expense loading: administration, acquisition, claims handling costs
        - Reinsurance loading: cost of reinsurance protection

        **6. Formula**
        ```
        Pure Premium = Σ (Vehicles_cat × Base_Freq × Risk_Factor × Age_Adj × Freq_Adj × Severity × Sev_Adj)
        Gross Premium = Pure Premium × (1 + Profit + Expenses + Reinsurance)
        ```

        **Limitations:**
        - Claims data lacks vehicle-level linkage (no regnr) → frequency is client-level
        - Risk factors use FLEET_TYPE as a proxy for UW category (close but not exact)
        - Categories with very few claims (e.g., Buss=10, Terrängfordon=3) have volatile factors
        - A production model would use GLM/gradient boosting fitted on claim-level data
        """)


# Tab 5: AI & ML Insights
def tab_ml_insights(client: str, fleet: pd.DataFrame, claims: pd.DataFrame, params: PricingParams):
    """ML models, credibility weighting, uncertainty, anomaly detection, clustering."""
    from models import (
        engineer_client_features, fit_frequency_model, credibility_analysis,
        bootstrap_premium, detect_claim_anomalies,
    )

    features = engineer_client_features(fleet, claims)

    # ── 1. Predictive Model ──────────────────────────────────────────────────
    st.subheader("📈 Predictive Model: Poisson GLM for Claim Frequency")
    st.markdown(
        "A **Poisson GLM** predicts claim frequency from fleet characteristics. "
        "Poisson regression is the standard actuarial approach for count data. "
        "With only 18 clients, we use **Leave-One-Out cross-validation** to honestly assess generalization."
    )

    model, scaler, pred_df, importance, metrics = fit_frequency_model(features)

    # ── Model performance summary ──
    improvement = (1 - metrics["loo_mae"] / metrics["baseline_mae"]) * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Error (LOO-CV)", f"{metrics['loo_mae']:.4f}",
              help="Mean Absolute Error when each client is predicted using a model trained on the other 17")
    m2.metric("Baseline Error (Mean)", f"{metrics['baseline_mae']:.4f}",
              help="Error if we just predicted the portfolio average for everyone")
    m3.metric("Improvement", f"{improvement:+.1f}%",
              delta=f"{improvement:+.1f}%",
              delta_color="normal" if improvement > 0 else "inverse",
              help="Negative = model is worse than just using the average")

    # ── Actual vs Predicted bar chart — one bar per client, easy to read ──
    bar_df = pred_df[["Client", "claim_frequency", "loo_predicted_freq"]].copy()
    bar_df = bar_df.sort_values("claim_frequency", ascending=True)
    bar_df = bar_df.rename(columns={
        "claim_frequency": "Actual Frequency",
        "loo_predicted_freq": "Model Prediction",
    })
    bar_melt = bar_df.melt(id_vars="Client", var_name="Type", value_name="Claim Frequency")

    fig = px.bar(
        bar_melt, y="Client", x="Claim Frequency", color="Type", barmode="group",
        color_discrete_map={"Actual Frequency": "#636EFA", "Model Prediction": "#FFA15A"},
        title="Actual vs Model-Predicted Claim Frequency per Client",
    )
    fig.update_layout(height=500, yaxis_title="", legend_title="",
                      legend=dict(orientation="h", y=1.08))
    # Highlight selected client
    fig.add_annotation(
        x=bar_df[bar_df["Client"] == client]["Actual Frequency"].values[0] if client in bar_df["Client"].values else 0,
        y=client, text="◄ Selected", showarrow=False, xanchor="left", xshift=5,
        font=dict(color="#EF553B", size=12, family="Arial Black"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance — normalized to 100% so values are meaningful ──
    imp = importance.copy()
    total = imp["Abs_Coef"].sum()
    imp["Contribution %"] = (imp["Abs_Coef"] / total * 100).round(1)
    imp["Direction"] = imp["Coefficient"].apply(lambda c: "↑ More claims" if c > 0 else "↓ Fewer claims")
    # Friendly names
    name_map = {
        "pct_personbil": "% Passenger cars", "fleet_age": "Fleet age",
        "log_fleet_size": "Fleet size", "pct_brand": "% Fire vehicles",
        "pct_heavy": "% Heavy vehicles", "fleet_diversity": "Vehicle type diversity",
    }
    imp["Feature"] = imp["Feature"].map(name_map).fillna(imp["Feature"])

    fig2 = px.bar(
        imp, x="Contribution %", y="Feature", orientation="h",
        color="Direction",
        color_discrete_map={"↑ More claims": "#EF553B", "↓ Fewer claims": "#00CC96"},
        title="What Drives Claim Frequency? (Feature Importance)",
        text="Contribution %",
    )
    fig2.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig2.update_layout(height=350, yaxis_title="", xaxis_title="Relative Importance (%)",
                       legend_title="", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Client-specific insight ──
    cr = pred_df[pred_df["Client"] == client]
    if not cr.empty:
        r = cr.iloc[0]
        diff = r["loo_predicted_freq"] - r["claim_frequency"]
        if abs(diff) < 0.02:
            verdict = "The model prediction is **close to observed** — this client behaves as expected given their fleet."
        elif diff > 0:
            verdict = f"The model predicts **higher** frequency than observed ({r['loo_predicted_freq']:.3f} vs {r['claim_frequency']:.3f}). This client may be **better than expected** — a positive sign for renewal."
        else:
            verdict = f"The model predicts **lower** frequency than observed ({r['loo_predicted_freq']:.3f} vs {r['claim_frequency']:.3f}). This client may carry **higher risk than their fleet profile suggests**."
        st.info(f"**{client}:** {verdict}")

    st.divider()

    # ── 2. Credibility Weighting ─────────────────────────────────────────────
    st.subheader("⚖️ Bühlmann Credibility Weighting")
    st.markdown(
        "**Problem:** A client with 30 claims might look terrible — but is that real or just bad luck? "
        "Credibility weighting answers this by blending each client's own experience with the portfolio average. "
        "Clients with **more data get more trust** (higher Z)."
    )

    portfolio_freq = features["claim_frequency"].mean()
    cred_df = credibility_analysis(features, portfolio_freq)

    # ── Table with highlighted selected client ──
    def _hl(row):
        return ["background-color: #fff3cd" if row["Client"] == client else "" for _ in row]

    st.dataframe(
        cred_df.style.apply(_hl, axis=1).format({
            "Observed Freq": "{:.4f}", "Portfolio Freq": "{:.4f}",
            "Z-Factor": "{:.3f}", "Credibility Freq": "{:.4f}", "Change %": "{:+.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )

    # ── Client-specific plain-English insight ──
    cc = cred_df[cred_df["Client"] == client]
    if not cc.empty:
        z = cc.iloc[0]
        trust_pct = int(z["Z-Factor"] * 100)
        if z["Change %"] > 5:
            meaning = (
                f"Their observed frequency ({z['Observed Freq']:.3f}) is **below** the portfolio average "
                f"({z['Portfolio Freq']:.3f}), but with Z={z['Z-Factor']:.2f} we only trust their data "
                f"{trust_pct}%. The adjusted frequency is pulled **up** to {z['Credibility Freq']:.3f}. "
                "For pricing, this means we should charge slightly **more** than raw experience suggests."
            )
        elif z["Change %"] < -5:
            meaning = (
                f"Their observed frequency ({z['Observed Freq']:.3f}) is **above** the portfolio average, "
                f"but credibility pulls it **down** to {z['Credibility Freq']:.3f}. "
                "Some of their bad experience may just be bad luck."
            )
        else:
            meaning = (
                f"With {int(z['Claims'])} claims, we trust their data {trust_pct}% (Z={z['Z-Factor']:.2f}). "
                f"Adjustment is small ({z['Change %']:+.1f}%) — their experience is reliable enough to price on directly."
            )
        st.info(f"**{client}:** {meaning}")

    with st.expander("📐 Formula"):
        st.markdown(r"""
        $$Z = \frac{n}{n + k}, \quad \text{Adjusted Freq} = Z \cdot f_{\text{client}} + (1 - Z) \cdot f_{\text{portfolio}}$$

        - $n$ = number of claims (more claims → more trust)
        - $k$ = credibility parameter (default 50 — industry standard)
        - $Z$ close to 1 = trust the client's data; $Z$ close to 0 = use portfolio average instead
        """)

    st.divider()

    # ── 3. Bootstrap Uncertainty ─────────────────────────────────────────────
    st.subheader("📊 Premium Uncertainty (Bootstrap)")
    st.markdown(
        f"**2,000 bootstrap iterations** for {client}: draw claim count from Poisson(observed), "
        "resample severities, recalculate premium. This quantifies how much the premium "
        "estimate could vary due to randomness in claims experience."
    )

    boot = bootstrap_premium(claims, fleet, client, loading=params.total_loading)

    if "error" not in boot:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Mean Premium", f"{boot['mean']:,.0f} SEK")
        b2.metric("90% CI Lower", f"{boot['ci_5']:,.0f} SEK")
        b3.metric("90% CI Upper", f"{boot['ci_95']:,.0f} SEK")
        b4.metric("Std Dev", f"{boot['std']:,.0f} SEK")

        ci_width = boot["ci_95"] - boot["ci_5"]
        st.caption(f"90% confidence interval width: **{ci_width:,.0f} SEK** "
                   f"({ci_width / boot['mean'] * 100:.1f}% of mean)")

        fig3 = px.histogram(
            x=boot["premiums"], nbins=50,
            title=f"Bootstrap Premium Distribution — {client}",
            labels={"x": "Premium (SEK)", "count": "Frequency"},
        )
        fig3.add_vline(x=boot["ci_5"], line_dash="dash", line_color="red",
                       annotation_text="5th pctl")
        fig3.add_vline(x=boot["ci_95"], line_dash="dash", line_color="red",
                       annotation_text="95th pctl")
        fig3.add_vline(x=boot["mean"], line_dash="solid", line_color="black",
                       annotation_text="Mean")
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── 4. Anomaly Detection ─────────────────────────────────────────────────
    st.subheader("🔍 Claim Anomaly Detection")
    st.markdown(
        "IQR-based outlier detection flags unusually severe claims. "
        "**Outlier:** > Q3 + 1.5×IQR  |  **Severe Outlier:** > Q3 + 3×IQR"
    )

    anomalies = detect_claim_anomalies(claims, client)
    if not anomalies.empty:
        n_outliers = (anomalies["flag"] != "Normal").sum()
        n_severe = (anomalies["flag"] == "Severe Outlier").sum()
        a1, a2, a3 = st.columns(3)
        a1.metric("Non-Zero Claims", len(anomalies))
        a2.metric("Outliers", int(n_outliers))
        a3.metric("Severe Outliers", int(n_severe))

        col_a, col_b = st.columns([3, 2])
        with col_a:
            fig_a = px.histogram(
                anomalies, x="Incurred idx", color="flag",
                color_discrete_map={"Normal": "#636EFA", "Outlier": "#FFA15A", "Severe Outlier": "#EF553B"},
                nbins=30, title="Severity Distribution with Anomaly Flags",
                labels={"Incurred idx": "Incurred Amount (SEK)"},
            )
            fig_a.update_layout(height=350)
            st.plotly_chart(fig_a, use_container_width=True)

        with col_b:
            if n_outliers > 0:
                st.markdown("**Flagged claims:**")
                out_df = anomalies[anomalies["flag"] != "Normal"][
                    ["CLAIM_TYPE", "CLAIM_CAUSE", "CLAIM_YEAR", "Incurred idx", "flag"]
                ].sort_values("Incurred idx", ascending=False)
                st.dataframe(
                    out_df.style.format({"Incurred idx": "{:,.0f}"}),
                    use_container_width=True, hide_index=True, height=300,
                )
            else:
                st.success("No anomalous claims detected for this client.")
    else:
        st.info("No non-zero claims to analyze.")


# Main App
def main():
    st.title("🚗 Protector Fleet Pricing Engine")

    try:
        data = get_data()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info("Place the Excel file in `data/raw/`")
        return

    fleet = data["fleet"]
    claims = data["claims"]
    clients = sorted(fleet["Client"].unique().tolist())

    selected_client, params = render_sidebar(clients)
    metrics, avgs = get_metrics(fleet, claims)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"🔍 {selected_client} Deep Dive",
        "📊 Portfolio Benchmark",
        "💰 Pricing Model & Assumptions",
        "🤖 AI & ML Insights",
        "🏷️ Classification Review",
    ])

    with tab1:
        tab_deep_dive(selected_client, fleet, claims, metrics, avgs, params)

    with tab2:
        tab_benchmark(selected_client, fleet, claims, metrics, avgs)

    with tab3:
        tab_pricing_model(selected_client, fleet, claims, params)

    with tab4:
        tab_ml_insights(selected_client, fleet, claims, params)

    with tab5:
        tab_classification(fleet)


if __name__ == "__main__":
    main()
