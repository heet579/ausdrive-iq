"""
Feature engineering pipeline for AusAutoIQ.
Transforms raw vehicle + insurance records into ML-ready feature matrices.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


SEIFA_ORDINAL = {"Very Low": 0, "Low": 1, "Medium": 2, "High": 3, "Very High": 4}

STATE_DENSITY = {
    "ACT": 5.0, "VIC": 4.8, "NSW": 4.2, "QLD": 3.5,
    "SA": 3.0, "WA": 2.8, "TAS": 2.4, "NT": 1.5,
}

STATE_EV_INDEX = {
    "ACT": 1.4, "VIC": 1.2, "NSW": 1.1, "QLD": 0.9,
    "SA": 0.95, "WA": 0.85, "TAS": 0.7, "NT": 0.5,
}


def engineer_registration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for the registration compliance model."""
    feat = pd.DataFrame()

    feat["vehicle_age_years"]         = df["vehicle_age_years"]
    feat["vehicle_age_sq"]            = df["vehicle_age_years"] ** 2
    feat["is_old_vehicle"]            = (df["vehicle_age_years"] > 10).astype(int)
    feat["annual_km"]                 = df["annual_km"]
    feat["km_per_year_log"]           = np.log1p(df["annual_km"])
    feat["engine_cc"]                 = df["engine_cc"]
    feat["prior_infringements"]       = df["prior_infringements"]
    feat["has_prior_infringement"]    = (df["prior_infringements"] > 0).astype(int)
    feat["repeat_offender"]           = (df["prior_infringements"] > 2).astype(int)
    feat["is_urban"]                  = df["is_urban"]
    feat["seifa_ordinal"]             = df["seifa_band"].map(SEIFA_ORDINAL)
    feat["is_low_seifa"]              = (feat["seifa_ordinal"] <= 1).astype(int)
    feat["annual_rego_cost_aud"]      = df["annual_rego_cost_aud"]
    feat["rego_cost_log"]             = np.log1p(df["annual_rego_cost_aud"])
    feat["state_density"]             = df["state"].map(STATE_DENSITY)
    feat["is_ev_or_hybrid"]           = df["fuel_type"].isin(["Electric", "Hybrid"]).astype(int)
    feat["is_motorcycle"]             = (df["vehicle_type"] == "Motorcycle").astype(int)
    feat["is_commercial"]             = df["vehicle_type"].isin(["Van", "Ute"]).astype(int)

    # Interaction terms
    feat["age_x_seifa"]               = feat["vehicle_age_years"] * feat["seifa_ordinal"]
    feat["infringement_x_km"]         = feat["prior_infringements"] * feat["km_per_year_log"]
    feat["urban_x_seifa"]             = feat["is_urban"] * feat["seifa_ordinal"]

    # One-hot: state (drop first to avoid multicollinearity)
    # Ensure all categories exist so get_dummies outputs the same columns when filtered
    all_states = ["ACT", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"]
    state_series = pd.Categorical(df["state"], categories=all_states)
    state_dummies = pd.get_dummies(state_series, prefix="state", drop_first=True)
    feat = pd.concat([feat, state_dummies], axis=1)

    return feat.astype(float)


def engineer_insurance_features(
    vehicle_df: pd.DataFrame,
    insurance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature matrix for the insurance risk scoring model."""
    merged = vehicle_df.merge(insurance_df, on="vehicle_id")
    feat = pd.DataFrame()

    feat["driver_age"]                = merged["driver_age"]
    feat["is_young_driver"]           = (merged["driver_age"] < 25).astype(int)
    feat["is_senior_driver"]          = (merged["driver_age"] > 70).astype(int)
    feat["driver_age_sq"]             = merged["driver_age"] ** 2
    feat["vehicle_age_years"]         = merged["vehicle_age_years"]
    feat["annual_km"]                 = merged["annual_km"]
    feat["km_per_year_log"]           = np.log1p(merged["annual_km"])
    feat["prior_infringements"]       = merged["prior_infringements"]
    feat["ncd_years"]                 = merged["ncd_years"]
    feat["is_urban"]                  = merged["is_urban"]
    feat["seifa_ordinal"]             = merged["seifa_band"].map(SEIFA_ORDINAL)
    feat["engine_cc"]                 = merged["engine_cc"]
    feat["is_motorcycle"]             = (merged["vehicle_type"] == "Motorcycle").astype(int)
    feat["is_ev"]                     = (merged["fuel_type"] == "Electric").astype(int)
    feat["state_density"]             = merged["state"].map(STATE_DENSITY)
    feat["compliance_risk"]           = merged["compliance_risk"]

    # Cover type ordinal (more cover = higher risk exposure flagged)
    cover_map = {"Third Party Property": 0, "Third Party Fire & Theft": 1, "Comprehensive": 2}
    feat["cover_type_ordinal"]        = merged["cover_type"].map(cover_map)

    # Interaction terms
    feat["age_x_km"]                  = feat["driver_age"] * feat["km_per_year_log"]
    feat["young_x_infringement"]      = feat["is_young_driver"] * feat["prior_infringements"]
    feat["seifa_x_km"]                = feat["seifa_ordinal"] * feat["km_per_year_log"]

    return feat.astype(float), merged["high_insurance_risk"]


def engineer_ev_features(ev_df: pd.DataFrame) -> pd.DataFrame:
    """Build time-series features for EV adoption forecasting."""
    df = ev_df.copy()
    df["quarter_dt"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp()
    df = df.sort_values(["state", "quarter_dt"])

    df["t"] = df.groupby("state").cumcount()
    df["t_sq"] = df["t"] ** 2
    df["t_cube"] = df["t"] ** 3
    df["state_ev_index"] = df["state"].map(STATE_EV_INDEX)

    # Lag features
    for lag in [1, 2, 4]:
        df[f"ev_lag_{lag}q"] = df.groupby("state")["ev_registrations"].shift(lag)

    # Rolling mean
    df["ev_rolling_4q"] = (
        df.groupby("state")["ev_registrations"]
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )

    # Quarter of year seasonality
    df["quarter_num"] = pd.PeriodIndex(df["quarter"], freq="Q").quarter
    df["is_q4"] = (df["quarter_num"] == 4).astype(int)

    df = df.dropna()
    return df
