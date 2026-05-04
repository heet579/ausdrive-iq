"""
AusAutoIQ — Streamlit Dashboard
Australian Vehicle Registration & Insurance Risk Intelligence Platform

Run: streamlit run dashboard/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.data_generator import (
    generate_vehicle_registrations,
    generate_insurance_records,
    generate_ev_adoption_timeseries,
    generate_road_trauma_costs,
)
import importlib
import src.feature_engineering
importlib.reload(src.feature_engineering)

from src.feature_engineering import (
    engineer_registration_features,
    engineer_insurance_features,
    engineer_ev_features,
)
from src.models.compliance_model import train_compliance_model, predict_compliance_risk
from src.models.insurance_risk_model import train_insurance_risk_model, score_premium_adequacy
from src.models.ev_forecast_model import train_ev_forecast_model, forecast_to_2030
from src.models.road_trauma_model import (
    build_cost_features, train_cost_model,
    compute_cost_breakdown, get_state_cost_summary,
)


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AusAutoIQ | Vehicle Intelligence Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #38bdf8; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #f1f5f9;
        border-bottom: 2px solid #38bdf8; padding-bottom: 8px; margin-bottom: 20px;
    }
    .badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .badge-blue  { background: #1e40af; color: #bfdbfe; }
    .badge-green { background: #14532d; color: #bbf7d0; }
    .badge-red   { background: #7f1d1d; color: #fecaca; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e293b; border-radius: 8px; color: #94a3b8;
        padding: 8px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background: #0ea5e9 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

PALETTE = px.colors.qualitative.Set2
STATE_COLOURS = {
    "NSW": "#38bdf8", "VIC": "#818cf8", "QLD": "#fb923c",
    "WA": "#4ade80",  "SA": "#facc15", "TAS": "#f472b6",
    "ACT": "#a78bfa", "NT": "#fb7185",
}


# ── Data Loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating synthetic Australian vehicle dataset…")
def load_data():
    vehicles = generate_vehicle_registrations(50_000)
    insurance = generate_insurance_records(vehicles)
    ev = generate_ev_adoption_timeseries()
    trauma = generate_road_trauma_costs()
    return vehicles, insurance, ev, trauma


@st.cache_resource(show_spinner="Training ML models (XGBoost + Optuna)…")
def load_models(vehicles, insurance):
    X_comp = engineer_registration_features(vehicles)
    y_comp = vehicles["compliance_risk"]
    comp_model, comp_explainer, comp_shap, comp_metrics = train_compliance_model(
        X_comp, y_comp, n_trials=20, n_folds=3
    )

    X_ins, y_ins = engineer_insurance_features(vehicles, insurance)
    ins_model, ins_explainer, ins_shap, ins_sample, ins_metrics = train_insurance_risk_model(
        X_ins, y_ins, n_trials=20, n_folds=3
    )

    ev_feat = engineer_ev_features(generate_ev_adoption_timeseries())
    ev_model, ev_metrics = train_ev_forecast_model(ev_feat)

    return (
        comp_model, comp_explainer, comp_shap, comp_metrics, X_comp, y_comp,
        ins_model, ins_explainer, ins_shap, ins_sample, ins_metrics, X_ins, y_ins,
        ev_model, ev_metrics, ev_feat,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 AusAutoIQ")
    st.markdown(
        "<span style='color:#94a3b8;font-size:0.8rem'>Australian Vehicle Registration "
        "& Insurance Risk Intelligence Platform</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Filter Dataset**")
    selected_states = st.multiselect(
        "States", options=["NSW","VIC","QLD","WA","SA","TAS","ACT","NT"],
        default=["NSW","VIC","QLD","WA","SA"],
    )
    vehicle_types_filter = st.multiselect(
        "Vehicle Types",
        options=["Sedan","SUV","Ute","Van","Hatchback","Wagon","Motorcycle","Electric Vehicle","Hybrid"],
        default=["Sedan","SUV","Ute","Electric Vehicle","Hybrid"],
    )
    seifa_filter = st.multiselect(
        "SEIFA Band",
        options=["Very Low","Low","Medium","High","Very High"],
        default=["Very Low","Low","Medium","High","Very High"],
    )

    st.divider()
    st.markdown(
        "<span style='color:#64748b;font-size:0.7rem'>"
        "⚠️ All data is synthetic — generated for portfolio/research purposes only."
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='color:#64748b;font-size:0.7rem'>"
        "Data mirrors distributions from ABS, BITRE, and data.gov.au"
        "</span>",
        unsafe_allow_html=True,
    )


# ── Load Data ─────────────────────────────────────────────────────────────────
vehicles_raw, insurance_raw, ev_raw, trauma_raw = load_data()

# Apply sidebar filters
vehicles = vehicles_raw[
    vehicles_raw["state"].isin(selected_states) &
    vehicles_raw["vehicle_type"].isin(vehicle_types_filter) &
    vehicles_raw["seifa_band"].isin(seifa_filter)
].copy()
insurance = insurance_raw[insurance_raw["vehicle_id"].isin(vehicles["vehicle_id"])].copy()

(
    comp_model, comp_explainer, comp_shap, comp_metrics, X_comp_full, y_comp_full,
    ins_model, ins_explainer, ins_shap, ins_sample, ins_metrics, X_ins_full, y_ins_full,
    ev_model, ev_metrics, ev_feat,
) = load_models(vehicles_raw, insurance_raw)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#f1f5f9;font-weight:800;font-size:2.2rem'>"
    "🚗 AusAutoIQ — Vehicle Intelligence Platform"
    "</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#94a3b8'>Real-time ML-powered registration compliance, insurance risk scoring, "
    "EV adoption forecasting & road trauma cost intelligence for modern Australia.</p>",
    unsafe_allow_html=True,
)

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
non_compliant = vehicles["compliance_risk"].sum()
claim_rate = insurance["made_claim"].mean()
avg_premium = insurance["annual_premium_aud"].mean()
ev_count = vehicles[vehicles["fuel_type"] == "Electric"]["vehicle_id"].count()

k1.metric("Records Analysed",    f"{len(vehicles):,}")
k2.metric("Non-Compliant",       f"{non_compliant:,}", delta=f"{non_compliant/len(vehicles)*100:.1f}% of fleet")
k3.metric("Claim Rate",          f"{claim_rate*100:.1f}%",  delta="industry avg ~7%")
k4.metric("Avg Annual Premium",  f"${avg_premium:,.0f}")
k5.metric("EV Registrations",    f"{ev_count:,}", delta=f"{ev_count/len(vehicles)*100:.1f}% of fleet")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Fleet Overview",
    "🔴 Compliance Risk",
    "💰 Insurance Risk",
    "⚡ EV Adoption",
    "🏥 Road Trauma Costs",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Fleet Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Australian Vehicle Fleet Overview</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        vtype_counts = vehicles["vehicle_type"].value_counts().reset_index()
        vtype_counts.columns = ["Vehicle Type", "Count"]
        fig = px.bar(
            vtype_counts, x="Count", y="Vehicle Type", orientation="h",
            color="Vehicle Type", color_discrete_sequence=PALETTE,
            title="Registrations by Vehicle Type",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                          font_color="#f1f5f9", yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        state_counts = vehicles["state"].value_counts().reset_index()
        state_counts.columns = ["State", "Count"]
        state_counts["colour"] = state_counts["State"].map(STATE_COLOURS)
        fig2 = px.bar(
            state_counts, x="State", y="Count",
            color="State", color_discrete_map=STATE_COLOURS,
            title="Registrations by State",
        )
        fig2.update_layout(showlegend=False, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                           font_color="#f1f5f9")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig3 = px.histogram(
            vehicles, x="vehicle_age_years", nbins=30,
            color="state", color_discrete_map=STATE_COLOURS,
            title="Vehicle Age Distribution by State",
            labels={"vehicle_age_years": "Vehicle Age (years)"},
        )
        fig3.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        seifa_risk = vehicles.groupby("seifa_band")["compliance_risk"].mean().reset_index()
        seifa_risk.columns = ["SEIFA Band", "Non-Compliance Rate"]
        seifa_order = ["Very Low", "Low", "Medium", "High", "Very High"]
        seifa_risk["SEIFA Band"] = pd.Categorical(seifa_risk["SEIFA Band"], categories=seifa_order, ordered=True)
        seifa_risk = seifa_risk.sort_values("SEIFA Band")
        fig4 = px.bar(
            seifa_risk, x="SEIFA Band", y="Non-Compliance Rate",
            color="Non-Compliance Rate", color_continuous_scale="Reds",
            title="Non-Compliance Rate by SEIFA (Socio-Economic Index)",
        )
        fig4.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(fig4, use_container_width=True)

    # LGA leaderboard
    st.markdown("**Top 10 LGAs by Non-Compliance Rate**")
    lga_risk = (
        vehicles.groupby(["state", "lga"])
        .agg(count=("vehicle_id", "count"), non_compliant=("compliance_risk", "sum"))
        .reset_index()
    )
    lga_risk["non_compliance_rate"] = (lga_risk["non_compliant"] / lga_risk["count"] * 100).round(1)
    lga_risk = lga_risk[lga_risk["count"] >= 30].sort_values("non_compliance_rate", ascending=False).head(10)
    st.dataframe(
        lga_risk[["state","lga","count","non_compliant","non_compliance_rate"]]
        .rename(columns={"count":"Registrations","non_compliant":"Non-Compliant","non_compliance_rate":"Rate (%)"}),
        use_container_width=True, hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Compliance Risk
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Registration Compliance Risk Model</p>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Model", "XGBoost + Optuna")
    m2.metric("CV PR-AUC", f"{comp_metrics['mean_pr_auc']:.4f}")
    m3.metric("CV ROC-AUC", f"{comp_metrics['mean_roc_auc']:.4f}")

    X_filtered = engineer_registration_features(vehicles)
    risk_results = predict_compliance_risk(comp_model, X_filtered)
    vehicles_scored = pd.concat([vehicles.reset_index(drop=True), risk_results], axis=1)

    c1, c2 = st.columns(2)

    with c1:
        risk_dist = risk_results["risk_band"].value_counts().reset_index()
        risk_dist.columns = ["Risk Band", "Count"]
        colour_map = {"Very Low":"#22c55e","Low":"#84cc16","Medium":"#f59e0b","High":"#f97316","Critical":"#ef4444"}
        fig = px.pie(
            risk_dist, values="Count", names="Risk Band",
            color="Risk Band", color_discrete_map=colour_map,
            title="Compliance Risk Distribution",
            hole=0.4,
        )
        fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        state_risk = (
            vehicles_scored.groupby("state")["compliance_risk_score"]
            .mean().reset_index()
            .sort_values("compliance_risk_score", ascending=False)
        )
        fig2 = px.bar(
            state_risk, x="state", y="compliance_risk_score",
            color="state", color_discrete_map=STATE_COLOURS,
            title="Avg Compliance Risk Score by State",
            labels={"compliance_risk_score": "Risk Score"},
        )
        fig2.update_layout(showlegend=False, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(fig2, use_container_width=True)

    # SHAP Feature Importance
    st.markdown("**SHAP Feature Importance — What Drives Registration Lapse?**")
    shap_df = pd.DataFrame(
        np.abs(comp_shap).mean(axis=0),
        index=X_comp_full.columns,
        columns=["mean_abs_shap"],
    ).sort_values("mean_abs_shap", ascending=True).tail(15)

    fig_shap = go.Figure(go.Bar(
        x=shap_df["mean_abs_shap"],
        y=shap_df.index,
        orientation="h",
        marker_color="#38bdf8",
    ))
    fig_shap.update_layout(
        title="Top 15 Features by Mean |SHAP| Value",
        plot_bgcolor="#0f172a", paper_bgcolor="#1e293b", font_color="#f1f5f9",
        height=450,
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # What-if simulator
    st.markdown("---")
    st.markdown("### 🔬 What-If Risk Simulator")
    st.markdown("Adjust vehicle & owner characteristics to see the predicted compliance risk.")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_age      = st.slider("Vehicle Age (years)", 1, 25, 8)
        sim_km       = st.slider("Annual KM", 5000, 50000, 15000, step=1000)
        sim_infring  = st.slider("Prior Infringements", 0, 8, 0)
    with sim_c2:
        sim_seifa    = st.selectbox("SEIFA Band", ["Very Low","Low","Medium","High","Very High"], index=2)
        sim_urban    = st.radio("Location", ["Urban", "Regional"]) == "Urban"
        sim_state    = st.selectbox("State", ["NSW","VIC","QLD","WA","SA","TAS","ACT","NT"])
    with sim_c3:
        sim_vtype    = st.selectbox("Vehicle Type", ["Sedan","SUV","Ute","Van","Hatchback","Wagon","Motorcycle","Electric Vehicle","Hybrid"])
        sim_fuel     = st.selectbox("Fuel Type", ["Petrol","Diesel","Electric","Hybrid"])
        sim_cost     = st.slider("Annual Rego Cost (AUD)", 200, 800, 380)

    sim_row = pd.DataFrame([{
        "vehicle_age_years": sim_age, "vehicle_age_sq": sim_age**2,
        "is_old_vehicle": int(sim_age > 10),
        "annual_km": sim_km, "km_per_year_log": np.log1p(sim_km),
        "engine_cc": 2000 if sim_fuel != "Electric" else 0,
        "prior_infringements": sim_infring,
        "has_prior_infringement": int(sim_infring > 0),
        "repeat_offender": int(sim_infring > 2),
        "is_urban": int(sim_urban),
        "seifa_ordinal": {"Very Low":0,"Low":1,"Medium":2,"High":3,"Very High":4}[sim_seifa],
        "is_low_seifa": int(sim_seifa in ["Very Low","Low"]),
        "annual_rego_cost_aud": sim_cost, "rego_cost_log": np.log1p(sim_cost),
        "state_density": {"ACT":5.0,"VIC":4.8,"NSW":4.2,"QLD":3.5,"SA":3.0,"WA":2.8,"TAS":2.4,"NT":1.5}[sim_state],
        "is_ev_or_hybrid": int(sim_fuel in ["Electric","Hybrid"]),
        "is_motorcycle": int(sim_vtype == "Motorcycle"),
        "is_commercial": int(sim_vtype in ["Van","Ute"]),
        "age_x_seifa": sim_age * {"Very Low":0,"Low":1,"Medium":2,"High":3,"Very High":4}[sim_seifa],
        "infringement_x_km": sim_infring * np.log1p(sim_km),
        "urban_x_seifa": int(sim_urban) * {"Very Low":0,"Low":1,"Medium":2,"High":3,"Very High":4}[sim_seifa],
    }])

    for s in ["NSW","VIC","QLD","WA","SA","TAS","ACT"]:
        col = f"state_{s}"
        sim_row[col] = float(sim_state == s)

    # Align to training columns
    for col in X_comp_full.columns:
        if col not in sim_row.columns:
            sim_row[col] = 0.0
    sim_row = sim_row[X_comp_full.columns]

    sim_risk = float(comp_model.predict_proba(sim_row)[0, 1])
    risk_colour = "#ef4444" if sim_risk > 0.5 else "#f59e0b" if sim_risk > 0.3 else "#22c55e"
    risk_label  = "HIGH RISK" if sim_risk > 0.5 else "MEDIUM RISK" if sim_risk > 0.3 else "LOW RISK"

    st.markdown(
        f"<div style='text-align:center;padding:24px;background:#1e293b;border-radius:12px;"
        f"border:2px solid {risk_colour};margin-top:16px'>"
        f"<div style='font-size:3rem;font-weight:900;color:{risk_colour}'>{sim_risk:.1%}</div>"
        f"<div style='color:{risk_colour};font-weight:700;font-size:1.1rem'>{risk_label}</div>"
        f"<div style='color:#94a3b8;margin-top:8px;font-size:0.85rem'>"
        f"Predicted compliance risk score for this vehicle profile</div></div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Insurance Risk
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Insurance Risk Scoring (InsurTech)</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", "XGBoost + Optuna")
    m2.metric("CV PR-AUC", f"{ins_metrics['mean_pr_auc']:.4f}")
    m3.metric("CV ROC-AUC", f"{ins_metrics['mean_roc_auc']:.4f}")
    m4.metric("Avg Claim Amount", f"${insurance[insurance['made_claim']==1]['claim_amount_aud'].mean():,.0f}")

    c1, c2 = st.columns(2)

    with c1:
        merged_ins = vehicles.merge(insurance, on="vehicle_id")
        age_claim = merged_ins.groupby("driver_age")["made_claim"].mean().reset_index()
        fig = px.scatter(
            age_claim, x="driver_age", y="made_claim",
            trendline="lowess", title="Claim Rate by Driver Age",
            labels={"driver_age":"Driver Age","made_claim":"Claim Rate"},
            color_discrete_sequence=["#38bdf8"],
        )
        fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        premium_dist = insurance.copy()
        premium_dist["risk_group"] = np.where(insurance["high_insurance_risk"], "High Risk", "Standard")
        fig2 = px.box(
            premium_dist, x="risk_group", y="annual_premium_aud",
            color="risk_group", color_discrete_map={"High Risk":"#ef4444","Standard":"#38bdf8"},
            title="Premium Distribution by Risk Group",
            labels={"annual_premium_aud":"Annual Premium (AUD)"},
        )
        fig2.update_layout(showlegend=False, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(fig2, use_container_width=True)

    # Premium adequacy
    st.markdown("**Premium Adequacy Analysis**")
    ins_risk_scores = ins_model.predict_proba(X_ins_full)[:, 1]
    adequacy = score_premium_adequacy(ins_risk_scores, insurance_raw["annual_premium_aud"].values)
    adeq_counts = adequacy["adequacy_band"].value_counts().reset_index()
    adeq_counts.columns = ["Adequacy Band", "Count"]

    adequacy_colours = {
        "Severely Under-priced": "#ef4444",
        "Under-priced":          "#f97316",
        "Adequate":              "#22c55e",
        "Over-priced":           "#60a5fa",
        "Severely Over-priced":  "#818cf8",
    }
    fig3 = px.bar(
        adeq_counts, x="Adequacy Band", y="Count",
        color="Adequacy Band", color_discrete_map=adequacy_colours,
        title="Premium Adequacy Classification (InsurTech Risk Signal)",
    )
    fig3.update_layout(showlegend=False, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9")
    st.plotly_chart(fig3, use_container_width=True)

    # SHAP importance
    st.markdown("**SHAP Feature Importance — Insurance Risk Drivers**")
    shap_ins_df = pd.DataFrame(
        np.abs(ins_shap).mean(axis=0),
        index=ins_sample.columns,
        columns=["mean_abs_shap"],
    ).sort_values("mean_abs_shap", ascending=True).tail(12)

    fig_shap2 = go.Figure(go.Bar(
        x=shap_ins_df["mean_abs_shap"],
        y=shap_ins_df.index,
        orientation="h",
        marker_color="#818cf8",
    ))
    fig_shap2.update_layout(
        title="Top 12 Insurance Risk Features by Mean |SHAP|",
        plot_bgcolor="#0f172a", paper_bgcolor="#1e293b", font_color="#f1f5f9", height=400,
    )
    st.plotly_chart(fig_shap2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EV Adoption
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">EV Adoption Forecast (2019 → 2030)</p>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Model", "XGBoost Regression")
    m2.metric("Walk-Fwd MAE", f"{ev_metrics['mean_mae']:,.0f} registrations")
    m3.metric("Walk-Fwd R²", f"{ev_metrics['mean_r2']:.4f}")

    # Historical
    ev_hist = generate_ev_adoption_timeseries()
    ev_hist["period_dt"] = pd.PeriodIndex(ev_hist["quarter"], freq="Q").to_timestamp()

    # Forecast
    forecast_df = forecast_to_2030(ev_model, engineer_ev_features(ev_hist))
    forecast_df["period_dt"] = pd.PeriodIndex(forecast_df["quarter"], freq="Q").to_timestamp()

    selected_ev_states = st.multiselect(
        "Select States", options=list(ev_hist["state"].unique()),
        default=["NSW","VIC","QLD","ACT"],
        key="ev_states",
    )

    fig_ev = go.Figure()
    for state in selected_ev_states:
        colour = STATE_COLOURS.get(state, "#94a3b8")
        hist = ev_hist[ev_hist["state"] == state]
        fore = forecast_df[forecast_df["state"] == state]

        fig_ev.add_trace(go.Scatter(
            x=hist["period_dt"], y=hist["ev_registrations"],
            name=f"{state} (actual)", line=dict(color=colour, width=2),
        ))
        fig_ev.add_trace(go.Scatter(
            x=fore["period_dt"], y=fore["ev_registrations_forecast"],
            name=f"{state} (forecast)", line=dict(color=colour, width=2, dash="dash"),
        ))

    fig_ev.add_vline(x=pd.Timestamp("2025-07-01").timestamp() * 1000, line_dash="dot",
                     line_color="#f59e0b", annotation_text="Forecast →")
    fig_ev.update_layout(
        title="EV Registrations — Historical + XGBoost Forecast to 2030",
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9",
        xaxis_title="Quarter", yaxis_title="EV Registrations",
        legend=dict(bgcolor="#1e293b"),
        height=500,
    )
    st.plotly_chart(fig_ev, use_container_width=True)

    # EV share by state
    ev_share = ev_hist.groupby("state")["ev_share_pct"].last().reset_index().sort_values("ev_share_pct", ascending=False)
    fig_share = px.bar(
        ev_share, x="state", y="ev_share_pct",
        color="state", color_discrete_map=STATE_COLOURS,
        title="Current EV Market Share by State (%)",
        labels={"ev_share_pct": "EV Share (%)"},
    )
    fig_share.update_layout(showlegend=False, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9")
    st.plotly_chart(fig_share, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Road Trauma Costs
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">Road Trauma Healthcare Cost Intelligence</p>', unsafe_allow_html=True)

    breakdown = compute_cost_breakdown(trauma_raw)
    summary = get_state_cost_summary(trauma_raw)

    m1, m2, m3 = st.columns(3)
    m1.metric("Est. National Annual Cost", "~$27B AUD")
    m2.metric("10-Year Total (Modelled)", f"${breakdown['computed_total_m'].sum()/1000:.1f}B AUD")
    m3.metric("Cost per Fatality", "~$8.6M AUD (BITRE)")

    c1, c2 = st.columns(2)

    with c1:
        annual = breakdown.groupby("year")["computed_total_m"].sum().reset_index()
        fig = px.area(
            annual, x="year", y="computed_total_m",
            title="National Road Trauma Cost Trend (AUD Millions)",
            labels={"computed_total_m": "Total Cost (AUD M)", "year": "Year"},
            color_discrete_sequence=["#ef4444"],
        )
        fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.bar(
            summary, x="state", y="total_cost_aud_millions",
            color="state", color_discrete_map=STATE_COLOURS,
            title="10-Year Cumulative Cost by State (AUD M)",
            labels={"total_cost_aud_millions": "Total Cost (AUD M)"},
        )
        fig2.update_layout(showlegend=False, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(fig2, use_container_width=True)

    # Injury breakdown stacked
    injury_by_year = breakdown.groupby("year")[
        ["cost_fatalities_m","cost_serious_m","cost_minor_m"]
    ].sum().reset_index()
    injury_by_year = injury_by_year.melt(id_vars="year", var_name="Category", value_name="Cost (AUD M)")
    label_map = {"cost_fatalities_m":"Fatalities","cost_serious_m":"Serious Injuries","cost_minor_m":"Minor Injuries"}
    injury_by_year["Category"] = injury_by_year["Category"].map(label_map)

    fig3 = px.bar(
        injury_by_year, x="year", y="Cost (AUD M)", color="Category",
        color_discrete_map={"Fatalities":"#ef4444","Serious Injuries":"#f97316","Minor Injuries":"#fbbf24"},
        title="Road Trauma Cost Breakdown by Severity (AUD Millions)",
        barmode="stack",
    )
    fig3.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#f1f5f9")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**State-Level Summary**")
    st.dataframe(
        summary.rename(columns={
            "state":"State","total_fatalities":"Total Fatalities",
            "total_serious_injuries":"Serious Injuries",
            "total_cost_aud_millions":"Total Cost (AUD M)",
            "avg_annual_cost_aud_millions":"Avg Annual Cost (AUD M)",
            "pct_national_cost":"% National",
        }),
        use_container_width=True, hide_index=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.75rem'>"
    "AusAutoIQ · Built by Heet Patel · Masters in Computer Engineering, University of Adelaide · "
    "All data is synthetic — generated for portfolio/research purposes only · "
    "Distributions based on ABS, BITRE, data.gov.au"
    "</div>",
    unsafe_allow_html=True,
)
