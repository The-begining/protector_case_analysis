"""
Pricing engine: risk-differentiated model with category-level risk factors,
sensitivity analysis, and transparent assumptions for underwriter review.

Model: frequency severity risk_factor  loading (per UW category)
"""
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


# Category Risk Factors — learned from portfolio claims data

def build_category_risk_factors(fleet: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Build risk factors per UW category from actual claims data.

    Uses FLEET_TYPE from claims to compute observed frequency and severity
    per vehicle type, then calculates a relative risk factor vs Personbil.
    Categories without claims data get a factor interpolated from vehicle
    characteristics (weight, type similarity).
    """
    exposure_years = max(1, claims["CLAIM_YEAR"].nunique()) if not claims.empty else 1

    # Vehicles per UW category
    cat_counts = fleet.groupby("UW_Kategori").agg(
        vehicle_count=("regnr", "count"),
        avg_weight=("TotalVikt", "mean"),
        avg_year=("Arsmodell", "mean"),
    ).reset_index()

    # ── Map FLEET_TYPE → UW_Kategori ──
    fleet_type_to_uw = {
        "Personbil":     "Personbil",
        "Lätt lastbil":  "Lätt lastbil",
        "Lastbil":       "Tung lastbil",
        "Brandfordon":   "Brandfordon",    # covers both Brandfordon + Lätt brandfordon
        "Buss":          "Buss",
        "Traktor":       "Traktor, minitraktor, åkgräsklippare",
        "Släp":          "Släp",            # covers Släp + Tungt släp
        "Lätt släp":     "Släp",
        "Motorredskap":  "Motorredskap lätt",
        "Terrängfordon": "Terrängfordon",
    }

    # ── Compute loss rate per FLEET_TYPE ──
    cl = claims[claims["FLEET_TYPE"].notna()].copy()
    cl["_uw"] = cl["FLEET_TYPE"].map(fleet_type_to_uw)

    # Merge similar types: Brandfordon claims → count against all fire vehicles
    uw_vehicle_counts = {}
    for uw in cat_counts["UW_Kategori"]:
        uw_vehicle_counts[uw] = cat_counts.loc[
            cat_counts["UW_Kategori"] == uw, "vehicle_count"
        ].iloc[0]
    # Brandfordon claims cover both Brandfordon + Lätt brandfordon
    uw_vehicle_counts["Brandfordon"] = (
        uw_vehicle_counts.get("Brandfordon", 0) + uw_vehicle_counts.get("Lätt brandfordon", 0)
    )
    # Släp claims cover Släp + Tungt släp + Lätt släp
    uw_vehicle_counts["Släp"] = (
        uw_vehicle_counts.get("Släp", 0) + uw_vehicle_counts.get("Tungt släp", 0)
    )

    cl_agg = cl.groupby("_uw").agg(
        claim_count=("Client", "size"),
        avg_severity=("Incurred idx", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
    ).reset_index()

    loss_rates = {}
    for _, row in cl_agg.iterrows():
        uw = row["_uw"]
        v = uw_vehicle_counts.get(uw, 0)
        if v > 0:
            freq = row["claim_count"] / v / exposure_years
            loss_rates[uw] = freq * row["avg_severity"]

    # ── Personbil baseline ──
    pb_loss = loss_rates.get("Personbil", 1)
    if pb_loss == 0:
        pb_loss = 1

    # ── Compute data-driven risk factors ──
    data_driven = {uw: round(lr / pb_loss, 2) for uw, lr in loss_rates.items()}

    # ── Fallback for categories with no claims data ──
    # These are interpolated from the closest related category
    fallback = {
        "Personbil EL":    round(data_driven.get("Personbil", 1.0) * 0.85, 2),
        "Lätt lastbil EL": round(data_driven.get("Lätt lastbil", 0.85) * 0.85, 2),
        "Lätt brandfordon": round(data_driven.get("Brandfordon", 1.82) * 0.70, 2),
        "Tungt släp":      round(data_driven.get("Släp", 0.02) * 2.5, 2),
        "Motorcykel/ATV":  1.20,   # no claims data, use domain estimate
        "Moped":           0.30,   # no claims data, low value vehicle
        "Motorredskap tungt": round(data_driven.get("Motorredskap lätt", 1.0) * 1.5, 2),
    }

    # Merge: data-driven takes priority, then fallback
    risk_map = {**fallback, **data_driven}

    cat_counts["risk_factor"] = cat_counts["UW_Kategori"].map(risk_map).fillna(1.0)
    cat_counts["factor_source"] = cat_counts["UW_Kategori"].apply(
        lambda x: "data-driven" if x in data_driven else "interpolated"
    )

    # Portfolio baseline stats
    total_veh = len(fleet)
    total_claims = len(claims)
    base_freq = total_claims / total_veh / exposure_years if total_veh > 0 else 0
    non_zero = claims[claims["Incurred idx"] > 0] if not claims.empty else pd.DataFrame()
    base_severity = non_zero["Incurred idx"].mean() if len(non_zero) > 0 else 0
    cat_counts["base_frequency"] = base_freq
    cat_counts["base_severity"] = base_severity

    return cat_counts


# Risk Metrics

def client_risk_metrics(fleet: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Calculate risk metrics per client."""
    vehicles = fleet.groupby("Client").agg(
        vehicle_count=("regnr", "count"),
        avg_model_year=("Arsmodell", "mean"),
        avg_weight=("TotalVikt", "mean"),
    ).reset_index()

    if claims.empty or "Client" not in claims.columns:
        for col in ["claim_count", "total_incurred", "avg_claim_cost",
                     "exposure_years", "claim_frequency", "non_zero_claims",
                     "minor_claims", "major_claims"]:
            vehicles[col] = 0
        return vehicles

    cl_agg = claims.groupby("Client").agg(
        claim_count=("Client", "size"),
        total_incurred=("Incurred idx", "sum"),
        avg_claim_cost=("Incurred idx", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        min_year=("CLAIM_YEAR", "min"),
        max_year=("CLAIM_YEAR", "max"),
        non_zero_claims=("Ex. 0", "sum"),
        minor_claims=("minor", "sum"),
        major_claims=("major", "sum"),
    ).reset_index()
    cl_agg["exposure_years"] = (cl_agg["max_year"] - cl_agg["min_year"] + 1).clip(lower=1)

    result = vehicles.merge(cl_agg, on="Client", how="left").fillna(0)
    result["claim_frequency"] = np.where(
        (result["vehicle_count"] > 0) & (result["exposure_years"] > 0),
        result["claim_count"] / result["vehicle_count"] / result["exposure_years"],
        0,
    )
    result["loss_per_vehicle"] = np.where(
        result["vehicle_count"] > 0,
        result["total_incurred"] / result["vehicle_count"],
        0,
    )
    return result


def category_risk_metrics(fleet: pd.DataFrame, claims: pd.DataFrame, client: str) -> pd.DataFrame:
    """Risk metrics segmented by UW category for one client."""
    cf = fleet[fleet["Client"] == client]
    cat_veh = cf.groupby("UW_Kategori").agg(
        vehicle_count=("regnr", "count"),
        avg_model_year=("Arsmodell", "mean"),
        avg_weight=("TotalVikt", "mean"),
    ).reset_index()

    cc = claims[claims["Client"] == client]
    if not cc.empty and "FLEET_TYPE" in cc.columns:
        cl_by_type = cc.groupby("FLEET_TYPE").agg(
            claim_count=("Client", "size"),
            total_incurred=("Incurred idx", "sum"),
            avg_cost=("Incurred idx", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        ).reset_index()
    else:
        cl_by_type = pd.DataFrame()

    return cat_veh, cl_by_type


def portfolio_averages(metrics: pd.DataFrame) -> dict:
    """Portfolio-wide averages for benchmarking."""
    return {
        "avg_fleet_size": round(metrics["vehicle_count"].mean(), 1),
        "avg_claim_frequency": round(metrics["claim_frequency"].mean(), 4),
        "avg_claim_cost": round(metrics["avg_claim_cost"].mean(), 2),
        "avg_loss_per_vehicle": round(metrics["loss_per_vehicle"].mean(), 2),
        "median_claim_frequency": round(metrics["claim_frequency"].median(), 4),
        "total_vehicles": int(metrics["vehicle_count"].sum()),
        "total_claims": int(metrics["claim_count"].sum()),
    }


# Pricing Model — Risk-Differentiated by Category

@dataclass
class PricingParams:
    """User-adjustable pricing assumptions exposed to the underwriter."""
    # Frequency & severity
    frequency_adj: float = 1.0              # multiplier on observed frequency
    severity_adj: float = 1.0               # multiplier on observed avg cost
    avg_cost_override: float | None = None  # hard override for avg cost (SEK)
    use_risk_factors: bool = True           # apply category-level differentiation

    # Loadings (additive on top of pure premium)
    profit_margin: float = 0.10       # 10%
    expense_loading: float = 0.15     # 15%
    reinsurance_loading: float = 0.05 # 5%

    # Fleet adjustments
    fleet_age_adj: bool = True     # adjust for fleet age (older = higher risk)
    large_loss_cap: float = 0.0    # cap individual claim at this SEK (0 = no cap)

    @property
    def total_loading(self) -> float:
        return 1 + self.profit_margin + self.expense_loading + self.reinsurance_loading


def calculate_price(
    fleet: pd.DataFrame,
    claims: pd.DataFrame,
    client: str,
    params: PricingParams | None = None,
) -> dict:
    """Risk-differentiated price: each UW category gets its own risk factor.

    Formula per category:
        pure_premium  = vehicles (base_freq risk_factor freq_adj)  (severity sev_adj)
        gross_premium = pure_premium  loading
    """
    if params is None:
        params = PricingParams()

    cf = fleet[fleet["Client"] == client]
    cc = claims[claims["Client"] == client] if "Client" in claims.columns else pd.DataFrame()

    total_vehicles = len(cf)
    if total_vehicles == 0:
        return {"error": "No vehicles found", "total_price": 0}

    # ── Observed client-level metrics ──
    exposure_years = max(1, cc["CLAIM_YEAR"].nunique()) if not cc.empty else 1
    observed_freq = len(cc) / total_vehicles / exposure_years if total_vehicles > 0 else 0
    non_zero = cc[cc["Incurred idx"] > 0] if not cc.empty else pd.DataFrame()

    # Apply large loss cap if set
    if params.large_loss_cap > 0 and not non_zero.empty:
        capped = non_zero["Incurred idx"].clip(upper=params.large_loss_cap)
        observed_avg_cost = capped.mean()
    else:
        observed_avg_cost = non_zero["Incurred idx"].mean() if len(non_zero) > 0 else 0

    # ── Category risk factors ──
    cat_risk = build_category_risk_factors(fleet, claims)
    risk_lookup = dict(zip(cat_risk["UW_Kategori"], cat_risk["risk_factor"]))

    # ── Adjusted frequency & severity ──
    adj_freq_base = observed_freq * params.frequency_adj
    adj_cost = params.avg_cost_override if params.avg_cost_override is not None else (
        observed_avg_cost * params.severity_adj
    )

    # ── Per-category pricing ──
    cats = cf.groupby("UW_Kategori").agg(
        vehicle_count=("regnr", "count"),
        avg_year=("Arsmodell", "mean"),
    ).reset_index()

    cats["risk_factor"] = cats["UW_Kategori"].map(risk_lookup).fillna(1.0)
    if not params.use_risk_factors:
        cats["risk_factor"] = 1.0

    # Fleet age adjustment: vehicles older than median get +10% risk
    if params.fleet_age_adj:
        median_year = fleet["Arsmodell"].median()
        cats["age_adj"] = np.where(
            cats["avg_year"].notna() & (cats["avg_year"] < median_year), 1.10, 1.00
        )
    else:
        cats["age_adj"] = 1.0

    cats["adj_frequency"] = adj_freq_base * cats["risk_factor"] * cats["age_adj"]
    cats["expected_claims"] = cats["vehicle_count"] * cats["adj_frequency"]
    cats["expected_cost"] = cats["expected_claims"] * adj_cost
    cats["pure_premium"] = cats["expected_cost"]
    cats["gross_price"] = cats["pure_premium"] * params.total_loading
    cats["price_per_vehicle"] = np.where(
        cats["vehicle_count"] > 0, cats["gross_price"] / cats["vehicle_count"], 0
    )
    total_price = cats["gross_price"].sum()
    pure_total = cats["pure_premium"].sum()

    # ── Assumption log for underwriter transparency ──
    assumptions = [
        ("Observed claim frequency", f"{observed_freq:.4f} per vehicle/year"),
        ("Frequency adjustment", f"{params.frequency_adj:.2f}"),
        ("Severity adjustment", f"{params.severity_adj:.2f}" if params.avg_cost_override is None else "overridden"),
        ("Avg claim cost (observed)", f"{observed_avg_cost:,.0f} SEK"),
        ("Avg claim cost (used)", f"{adj_cost:,.0f} SEK"),
        ("Category risk factors", "Applied (16 categories)" if params.use_risk_factors else "Disabled (flat rate)"),
        ("Fleet age adjustment", "Enabled (+10% for older-than-median)" if params.fleet_age_adj else "Disabled"),
        ("Large loss cap", f"{params.large_loss_cap:,.0f} SEK" if params.large_loss_cap > 0 else "None"),
        ("Profit margin", f"{params.profit_margin:.0%}"),
        ("Expense loading", f"{params.expense_loading:.0%}"),
        ("Reinsurance loading", f"{params.reinsurance_loading:.0%}"),
        ("Total loading factor", f"{params.total_loading:.1%}"),
        ("Exposure years", f"{exposure_years}"),
    ]

    return {
        "total_price": round(total_price, 2),
        "pure_premium": round(pure_total, 2),
        "price_per_vehicle": round(total_price / total_vehicles, 2),
        "per_category": cats,
        "total_vehicles": total_vehicles,
        "total_claims": len(cc),
        "exposure_years": exposure_years,
        "observed_frequency": round(observed_freq, 4),
        "adjusted_frequency": round(adj_freq_base, 4),
        "observed_avg_cost": round(observed_avg_cost, 2),
        "adjusted_avg_cost": round(adj_cost, 2),
        "loading_factor": round(params.total_loading, 3),
        "params": params,
        "assumptions": assumptions,
    }


# Sensitivity Analysis — show underwriter how assumptions affect price

def sensitivity_analysis(
    fleet: pd.DataFrame,
    claims: pd.DataFrame,
    client: str,
    base_params: PricingParams,
) -> pd.DataFrame:
    """Vary each assumption ±20% and show impact on total price."""
    base_price = calculate_price(fleet, claims, client, base_params)["total_price"]

    results = []

    # Frequency sensitivity
    for mult in [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
        p = PricingParams(
            frequency_adj=base_params.frequency_adj * mult,
            severity_adj=base_params.severity_adj,
            avg_cost_override=base_params.avg_cost_override,
            use_risk_factors=base_params.use_risk_factors,
            profit_margin=base_params.profit_margin,
            expense_loading=base_params.expense_loading,
            reinsurance_loading=base_params.reinsurance_loading,
            fleet_age_adj=base_params.fleet_age_adj,
            large_loss_cap=base_params.large_loss_cap,
        )
        price = calculate_price(fleet, claims, client, p)["total_price"]
        results.append({
            "Assumption": "Claim Frequency",
            "Scenario": f"{mult:.1f}",
            "Total Price (SEK)": price,
            "Change vs Base": price - base_price,
            "Change %": (price - base_price) / base_price * 100 if base_price else 0,
        })

    # Severity sensitivity
    for mult in [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
        p = PricingParams(
            frequency_adj=base_params.frequency_adj,
            severity_adj=base_params.severity_adj * mult,
            avg_cost_override=base_params.avg_cost_override,
            use_risk_factors=base_params.use_risk_factors,
            profit_margin=base_params.profit_margin,
            expense_loading=base_params.expense_loading,
            reinsurance_loading=base_params.reinsurance_loading,
            fleet_age_adj=base_params.fleet_age_adj,
            large_loss_cap=base_params.large_loss_cap,
        )
        price = calculate_price(fleet, claims, client, p)["total_price"]
        results.append({
            "Assumption": "Claim Severity",
            "Scenario": f"{mult:.1f}",
            "Total Price (SEK)": price,
            "Change vs Base": price - base_price,
            "Change %": (price - base_price) / base_price * 100 if base_price else 0,
        })

    # Margin sensitivity
    for margin in [0.05, 0.10, 0.15, 0.20, 0.25]:
        p = PricingParams(
            frequency_adj=base_params.frequency_adj,
            severity_adj=base_params.severity_adj,
            avg_cost_override=base_params.avg_cost_override,
            use_risk_factors=base_params.use_risk_factors,
            profit_margin=margin,
            expense_loading=base_params.expense_loading,
            reinsurance_loading=base_params.reinsurance_loading,
            fleet_age_adj=base_params.fleet_age_adj,
            large_loss_cap=base_params.large_loss_cap,
        )
        price = calculate_price(fleet, claims, client, p)["total_price"]
        results.append({
            "Assumption": "Profit Margin",
            "Scenario": f"{margin:.0%}",
            "Total Price (SEK)": price,
            "Change vs Base": price - base_price,
            "Change %": (price - base_price) / base_price * 100 if base_price else 0,
        })

    return pd.DataFrame(results)
