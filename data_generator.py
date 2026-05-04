"""
Synthetic Australian vehicle registration + insurance dataset generator.
Mirrors real distributions from ABS, BITRE, and data.gov.au.
All data is synthetic — clearly labelled for portfolio/research use.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Australian geography ──────────────────────────────────────────────────────
STATES = {
    "NSW": {"pop_weight": 0.32, "urban_rate": 0.89, "ev_adoption_index": 1.1},
    "VIC": {"pop_weight": 0.26, "urban_rate": 0.91, "ev_adoption_index": 1.2},
    "QLD": {"pop_weight": 0.20, "urban_rate": 0.87, "ev_adoption_index": 0.9},
    "WA":  {"pop_weight": 0.11, "urban_rate": 0.85, "ev_adoption_index": 0.85},
    "SA":  {"pop_weight": 0.07, "urban_rate": 0.87, "ev_adoption_index": 0.95},
    "TAS": {"pop_weight": 0.02, "urban_rate": 0.59, "ev_adoption_index": 0.7},
    "ACT": {"pop_weight": 0.02, "urban_rate": 0.99, "ev_adoption_index": 1.4},
    "NT":  {"pop_weight": 0.01, "urban_rate": 0.46, "ev_adoption_index": 0.5},
}

LGAS = {
    "NSW": ["Sydney", "Parramatta", "Blacktown", "Liverpool", "Newcastle", "Wollongong", "Central Coast", "Penrith", "Campbelltown", "Ryde"],
    "VIC": ["Melbourne", "Casey", "Wyndham", "Hume", "Monash", "Geelong", "Ballarat", "Bendigo", "Whittlesea", "Knox"],
    "QLD": ["Brisbane", "Gold Coast", "Moreton Bay", "Sunshine Coast", "Townsville", "Cairns", "Logan", "Ipswich", "Redland", "Toowoomba"],
    "WA":  ["Perth", "Stirling", "Joondalup", "Wanneroo", "Swan", "Rockingham", "Mandurah", "Fremantle", "Armadale", "Belmont"],
    "SA":  ["Adelaide", "Onkaparinga", "Salisbury", "Tea Tree Gully", "Playford", "Marion", "Charles Sturt", "Port Adelaide", "Mitcham", "Holdfast Bay"],
    "TAS": ["Hobart", "Launceston", "Devonport", "Burnie", "Glenorchy", "Clarence", "Kingborough", "Brighton", "Sorell", "Central Highlands"],
    "ACT": ["Canberra", "Belconnen", "Tuggeranong", "Gungahlin", "Woden", "Weston Creek", "Molonglo", "Inner North", "Inner South", "Fyshwick"],
    "NT":  ["Darwin", "Palmerston", "Alice Springs", "Litchfield", "Katherine", "Jabiru", "Tennant Creek", "Nhulunbuy", "Humpty Doo", "Batchelor"],
}

VEHICLE_TYPES = ["Sedan", "SUV", "Ute", "Van", "Hatchback", "Wagon", "Motorcycle", "Electric Vehicle", "Hybrid"]

VEHICLE_MAKES = {
    "Sedan":           ["Toyota", "Honda", "Mazda", "Hyundai", "Kia", "Ford", "Holden", "Nissan"],
    "SUV":             ["Toyota", "Mazda", "Hyundai", "Kia", "Ford", "Mitsubishi", "Subaru", "BMW"],
    "Ute":             ["Toyota", "Ford", "Mitsubishi", "Nissan", "Isuzu", "Mazda", "HSV", "RAM"],
    "Van":             ["Toyota", "Mercedes-Benz", "Ford", "Volkswagen", "Fiat", "Renault", "Hyundai"],
    "Hatchback":       ["Toyota", "Mazda", "Volkswagen", "Honda", "Hyundai", "Ford", "Kia", "Renault"],
    "Wagon":           ["Subaru", "Volvo", "Volkswagen", "Toyota", "Mazda", "Skoda", "BMW"],
    "Motorcycle":      ["Honda", "Yamaha", "Kawasaki", "Suzuki", "Ducati", "BMW", "Harley-Davidson"],
    "Electric Vehicle":["Tesla", "BYD", "Hyundai", "Kia", "Polestar", "Volkswagen", "MG", "BMW"],
    "Hybrid":          ["Toyota", "Lexus", "Honda", "Hyundai", "Kia", "Ford", "Mitsubishi"],
}

FUEL_TYPES = {
    "Sedan":            ["Petrol", "Diesel", "Petrol"],
    "SUV":              ["Petrol", "Diesel", "Petrol", "Diesel"],
    "Ute":              ["Diesel", "Petrol", "Diesel"],
    "Van":              ["Diesel", "Petrol"],
    "Hatchback":        ["Petrol", "Petrol", "Diesel"],
    "Wagon":            ["Petrol", "Diesel"],
    "Motorcycle":       ["Petrol"],
    "Electric Vehicle": ["Electric"],
    "Hybrid":           ["Hybrid"],
}

SEIFA_BANDS = ["Very Low", "Low", "Medium", "High", "Very High"]  # Socio-Economic Index for Areas


# ── Generator ────────────────────────────────────────────────────────────────
def generate_vehicle_registrations(n: int = 50_000) -> pd.DataFrame:
    """Generate synthetic vehicle registration records."""
    state_list = list(STATES.keys())
    raw_weights = [STATES[s]["pop_weight"] for s in state_list]
    total_weight = sum(raw_weights)
    state_weights = [w / total_weight for w in raw_weights]
    state_weights[-1] = 1.0 - sum(state_weights[:-1])
    states = np.random.choice(state_list, size=n, p=state_weights)

    lgas = [np.random.choice(LGAS[s]) for s in states]
    vt_p = [0.22, 0.28, 0.14, 0.06, 0.10, 0.06, 0.05, 0.05, 0.04]
    vt_p[-1] = 1.0 - sum(vt_p[:-1])
    vehicle_types = np.random.choice(
        VEHICLE_TYPES,
        size=n,
        p=vt_p,
    )
    makes = [np.random.choice(VEHICLE_MAKES[vt]) for vt in vehicle_types]
    fuel_types = [np.random.choice(FUEL_TYPES[vt]) for vt in vehicle_types]

    reg_year = np.random.randint(2015, 2025, size=n)
    manufacture_year = reg_year - np.random.randint(0, 15, size=n)
    vehicle_age = 2025 - manufacture_year

    # SEIFA — socio-economic index, affects compliance & insurance risk
    seifa = np.random.choice(SEIFA_BANDS, size=n, p=[0.15, 0.20, 0.30, 0.20, 0.15])

    # Urban vs regional
    is_urban = np.array([
        np.random.rand() < STATES[s]["urban_rate"] for s in states
    ])

    # Annual kilometres driven — urban drives less per trip but more trips
    annual_km = np.where(
        is_urban,
        np.random.normal(14_000, 4_000, n).clip(3_000, 35_000),
        np.random.normal(22_000, 6_000, n).clip(5_000, 60_000),
    ).astype(int)

    # Engine size (cc) — not applicable for EVs
    engine_cc = np.where(
        np.isin(vehicle_types, ["Electric Vehicle"]),
        0,
        np.random.choice([1200, 1500, 1800, 2000, 2500, 3000, 3500, 4000], size=n,
                         p=[0.08, 0.22, 0.25, 0.20, 0.12, 0.07, 0.04, 0.02]),
    )

    # Number of prior traffic infringements (Poisson)
    seifa_infringement_map = {"Very Low": 1.8, "Low": 1.4, "Medium": 0.9, "High": 0.5, "Very High": 0.3}
    infringement_lam = np.array([seifa_infringement_map[s] for s in seifa])
    prior_infringements = np.random.poisson(infringement_lam)

    # Days overdue on registration (0 = compliant)
    base_lapse_prob = (
        0.05
        + 0.03 * (vehicle_age > 10).astype(float)
        + 0.04 * (seifa == "Very Low").astype(float)
        + 0.03 * (seifa == "Low").astype(float)
        - 0.02 * is_urban.astype(float)
        + 0.02 * (prior_infringements > 2).astype(float)
    ).clip(0.02, 0.35)
    is_lapsed = np.random.rand(n) < base_lapse_prob
    days_overdue = np.where(is_lapsed, np.random.exponential(45, n).clip(1, 365).astype(int), 0)

    # Registration expiry date
    base_date = datetime(2024, 1, 1)
    expiry_dates = [
        (base_date + timedelta(days=int(np.random.randint(-30, 365)))).strftime("%Y-%m-%d")
        for _ in range(n)
    ]

    # Annual registration cost (AUD) — varies by state and vehicle type
    base_rego_cost = {
        "NSW": 400, "VIC": 350, "QLD": 320, "WA": 370, "SA": 330, "TAS": 280, "ACT": 310, "NT": 290,
    }
    rego_cost = np.array([
        base_rego_cost[s] * (1 + 0.15 * (vt in ["SUV", "Ute", "Van"]) - 0.05 * (vt == "Motorcycle"))
        for s, vt in zip(states, vehicle_types)
    ]) + np.random.normal(0, 30, n)

    # Compliance label: 1 = lapsed/non-compliant, 0 = compliant
    compliance_risk = is_lapsed.astype(int)

    df = pd.DataFrame({
        "vehicle_id": [f"VEH{str(i).zfill(6)}" for i in range(n)],
        "state": states,
        "lga": lgas,
        "vehicle_type": vehicle_types,
        "make": makes,
        "fuel_type": fuel_types,
        "manufacture_year": manufacture_year,
        "registration_year": reg_year,
        "vehicle_age_years": vehicle_age,
        "engine_cc": engine_cc,
        "annual_km": annual_km,
        "is_urban": is_urban.astype(int),
        "seifa_band": seifa,
        "prior_infringements": prior_infringements,
        "days_overdue": days_overdue,
        "registration_expiry": expiry_dates,
        "annual_rego_cost_aud": rego_cost.round(2),
        "compliance_risk": compliance_risk,
    })
    return df


def generate_insurance_records(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic insurance records linked to vehicle registrations."""
    n = len(vehicle_df)
    rng = np.random.default_rng(RANDOM_SEED)

    vehicle_age = vehicle_df["vehicle_age_years"].values
    seifa = vehicle_df["seifa_band"].values
    annual_km = vehicle_df["annual_km"].values
    prior_inf = vehicle_df["prior_infringements"].values
    is_urban = vehicle_df["is_urban"].values
    vehicle_type = vehicle_df["vehicle_type"].values
    state = vehicle_df["state"].values

    # Driver age (independently drawn — different person may own the car)
    driver_age = rng.integers(18, 80, size=n)

    # Young/old driver risk
    age_risk = np.where(driver_age < 25, 1.4, np.where(driver_age > 70, 1.2, 1.0))

    # Vehicle type base premium multiplier
    type_multiplier = {
        "Sedan": 1.0, "SUV": 1.1, "Ute": 1.15, "Van": 1.2, "Hatchback": 0.95,
        "Wagon": 1.0, "Motorcycle": 1.5, "Electric Vehicle": 1.3, "Hybrid": 1.05,
    }
    type_mult = np.array([type_multiplier[vt] for vt in vehicle_type])

    seifa_premium_map = {"Very Low": 1.3, "Low": 1.15, "Medium": 1.0, "High": 0.9, "Very High": 0.85}
    seifa_mult = np.array([seifa_premium_map[s] for s in seifa])

    # Annual premium (AUD)
    base_premium = 900
    annual_premium = (
        base_premium
        * age_risk
        * type_mult
        * seifa_mult
        * (1 + 0.01 * vehicle_age)
        * (1 + 0.008 * (annual_km / 1000))
        * (1 + 0.12 * (prior_inf > 0).astype(float))
        + rng.normal(0, 80, n)
    ).clip(400, 6000).round(2)

    # Claim probability — driven by compound risk factors
    claim_prob = (
        0.06
        + 0.04 * (driver_age < 25).astype(float)
        + 0.03 * (driver_age > 70).astype(float)
        + 0.03 * (vehicle_age > 12).astype(float)
        + 0.02 * is_urban.astype(float)
        + 0.05 * (prior_inf > 1).astype(float)
        + 0.02 * (vehicle_type == "Motorcycle").astype(float)
        + rng.normal(0, 0.01, n)
    ).clip(0.02, 0.50)

    made_claim = (rng.random(n) < claim_prob).astype(int)

    # Claim amount when a claim is made
    claim_amount = np.where(
        made_claim,
        rng.exponential(4500, n).clip(200, 80_000).round(2),
        0.0,
    )

    # No-claim discount years
    ncd_years = rng.integers(0, 6, size=n)

    # Cover type
    cover_types = rng.choice(["Comprehensive", "Third Party Property", "Third Party Fire & Theft"],
                             size=n, p=[0.62, 0.28, 0.10])

    return pd.DataFrame({
        "vehicle_id": vehicle_df["vehicle_id"].values,
        "driver_age": driver_age,
        "cover_type": cover_types,
        "annual_premium_aud": annual_premium,
        "ncd_years": ncd_years,
        "made_claim": made_claim,
        "claim_amount_aud": claim_amount,
        "claim_probability": claim_prob.round(4),
        "high_insurance_risk": (claim_prob > 0.15).astype(int),
    })


def generate_ev_adoption_timeseries() -> pd.DataFrame:
    """
    State-level EV registration counts by quarter (2019 Q1 → 2025 Q2).
    Growth follows a logistic adoption curve with state-specific rates.
    """
    rows = []
    quarters = pd.period_range(start="2019Q1", end="2025Q2", freq="Q")
    t_max = len(quarters)

    for state, meta in STATES.items():
        k = meta["ev_adoption_index"]
        # Logistic growth: L / (1 + e^(-r*(t - t0)))
        L = int(50_000 * meta["pop_weight"] * k)  # saturation capacity
        r = 0.35                                    # growth rate
        t0 = t_max * 0.55                           # inflection point

        for t_idx, period in enumerate(quarters):
            base_count = L / (1 + np.exp(-r * (t_idx - t0)))
            noise = np.random.normal(0, base_count * 0.04)
            ev_count = max(0, int(base_count + noise))
            rows.append({
                "quarter": str(period),
                "state": state,
                "ev_registrations": ev_count,
                "total_registrations": int(ev_count / max(0.001, 0.002 + 0.003 * (t_idx / t_max))),
                "ev_share_pct": round(min(100, (0.002 + 0.003 * (t_idx / t_max)) * 100 * k), 3),
            })

    return pd.DataFrame(rows)


def generate_road_trauma_costs() -> pd.DataFrame:
    """
    State-level annual road trauma healthcare cost estimates (AUD millions).
    Based on BITRE / Austroads methodology: ~$27B/year nationally.
    """
    years = list(range(2015, 2026))
    rows = []
    national_base = 27_000  # $M nationally

    for state, meta in STATES.items():
        state_share = meta["pop_weight"]
        for year in years:
            trend = 1 - 0.02 * (year - 2015)  # improving road safety year-on-year
            noise = np.random.normal(1.0, 0.03)
            fatalities = int(np.random.normal(
                800 * state_share * trend * noise, 20 * state_share
            ))
            serious_injuries = fatalities * np.random.randint(8, 12)
            minor_injuries = serious_injuries * np.random.randint(4, 7)
            cost_m = round(national_base * state_share * trend * noise, 1)

            rows.append({
                "year": year,
                "state": state,
                "fatalities": max(0, fatalities),
                "serious_injuries": max(0, serious_injuries),
                "minor_injuries": max(0, minor_injuries),
                "total_cost_aud_millions": max(0, cost_m),
                "cost_per_capita_aud": round(cost_m * 1_000_000 / (25_000_000 * state_share), 2),
            })

    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating vehicle registrations (50,000 records)...")
    vehicles = generate_vehicle_registrations(50_000)
    vehicles.to_csv(os.path.join(out_dir, "vehicle_registrations.csv"), index=False)
    print(f"  ✓ {len(vehicles):,} records → vehicle_registrations.csv")

    print("Generating insurance records...")
    insurance = generate_insurance_records(vehicles)
    insurance.to_csv(os.path.join(out_dir, "insurance_records.csv"), index=False)
    print(f"  ✓ {len(insurance):,} records → insurance_records.csv")

    print("Generating EV adoption time-series...")
    ev = generate_ev_adoption_timeseries()
    ev.to_csv(os.path.join(out_dir, "ev_adoption_timeseries.csv"), index=False)
    print(f"  ✓ {len(ev):,} rows → ev_adoption_timeseries.csv")

    print("Generating road trauma cost data...")
    trauma = generate_road_trauma_costs()
    trauma.to_csv(os.path.join(out_dir, "road_trauma_costs.csv"), index=False)
    print(f"  ✓ {len(trauma):,} rows → road_trauma_costs.csv")

    print("\nAll synthetic datasets generated.")
    print("NOTE: All data is synthetic and generated for portfolio/research purposes only.")
