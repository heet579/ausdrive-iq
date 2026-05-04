# AusAutoIQ — Australian Vehicle Registration & Insurance Risk Intelligence Platform

> ML-powered analytics platform addressing Australia's $27B/year road trauma burden, rising insurance unaffordability, and EV transition — built for Transport, InsurTech, and HealthTech recruiters.

---

## What It Does

| Module | Technique | Business Value |
|---|---|---|
| **Registration Compliance Predictor** | XGBoost + SHAP + Optuna | Flag high-risk vehicles before renewal lapse |
| **Insurance Risk Scorer** | XGBoost + walk-forward CV | Premium adequacy analysis for InsurTech |
| **EV Adoption Forecaster** | XGBoost Regression + lag features | State-level forecast to 2030 for infrastructure planning |
| **Road Trauma Cost Model** | Ridge Regression + BITRE methodology | Healthcare burden quantification by state/year |
| **Interactive Dashboard** | Streamlit + Plotly | Real-time filtering, what-if simulator, SHAP viz |

---

## Stack

```
Python · XGBoost · SHAP · Optuna · Scikit-Learn
Streamlit · Plotly · Pandas · NumPy
Jupyter · Matplotlib · Seaborn
```

---

## Key Results

| Model | Metric | Score |
|---|---|---|
| Compliance Predictor | PR-AUC (5-fold CV) | **>0.85** |
| Insurance Risk Scorer | PR-AUC (5-fold CV) | **>0.82** |
| EV Forecast | Walk-forward R² | **>0.97** |
| Cost Model | R² | **>0.94** |

---

## Data Sources

All data is **synthetic** — generated to mirror real Australian distributions from:

- **ABS** — vehicle registration counts by state/year
- **BITRE** — crash statistics and road trauma cost methodology (~$27B/year nationally)
- **Austroads** — unit cost estimates per injury severity
- **data.gov.au** — vehicle registration open datasets
- **Synthetic insurance records** — generated with realistic Australian InsurTech risk factors

> ⚠️ All data is clearly synthetic and intended for portfolio/research purposes only.

---

## Australian Context

This project addresses three interconnected crises in modern Australia:

**1. Registration Non-Compliance (~8–12% of the fleet)**
Unregistered vehicles create road safety voids and uninsured motorist risk. Transport departments spend millions on manual compliance checks that ML could replace.

**2. Insurance Affordability Crisis**
Australian premiums have risen 14–28% in 2023–2024. InsurTech companies need accurate risk scoring to price fairly while remaining competitive.

**3. EV Transition**
Australia committed to 3.8M EVs by 2030. State governments need accurate adoption forecasts to plan charging infrastructure investment by LGA.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/heet579/ausdrive-iq
cd ausdrive-iq
pip install -r requirements.txt

# 2. Generate synthetic datasets
python src/data_generator.py

# 3. Launch dashboard
streamlit run dashboard/streamlit_app.py

# 4. Open notebook
jupyter notebook notebooks/01_eda_and_modelling.ipynb
```

---

## Project Structure

```
ausdrive-iq/
├── src/
│   ├── data_generator.py          # Synthetic Australian vehicle + insurance data
│   ├── feature_engineering.py     # ML-ready feature matrices
│   └── models/
│       ├── compliance_model.py    # Registration compliance XGBoost
│       ├── insurance_risk_model.py # Insurance risk XGBoost
│       ├── ev_forecast_model.py   # EV adoption forecasting
│       └── road_trauma_model.py   # Healthcare cost modelling
├── dashboard/
│   └── streamlit_app.py           # Interactive 5-tab Streamlit dashboard
├── notebooks/
│   └── 01_eda_and_modelling.ipynb # Full EDA → model → SHAP notebook
├── data/
│   └── synthetic/                 # Generated datasets (gitignored)
├── reports/
│   └── figures/                   # Saved charts
└── requirements.txt
```

---

## Target Roles

This project is designed to demonstrate skills relevant to:

- **Data Analyst** — EDA, SEIFA segmentation, BITRE cost methodology, Plotly dashboards
- **Business Analyst** — InsurTech premium adequacy, transport policy, ROI framing
- **ML Engineer / Data Scientist** — XGBoost, Optuna, SHAP, walk-forward CV, time-series
- **BI Developer** — Streamlit dashboard, Power BI-ready aggregations

**Relevant companies:** IAG · Suncorp · NRMA · REA Group · Finder · Compare the Market · Transport for NSW · Department of Infrastructure · BITRE · Australian Bureau of Statistics

---

*Built by Heet Patel | Masters in Computer Engineering, University of Adelaide*
