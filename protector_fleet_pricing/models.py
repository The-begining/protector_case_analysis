"""
ML models, credibility weighting, uncertainty quantification, and anomaly detection.
Complements the deterministic pricing model with statistical learning approaches.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error


# ── Feature Engineering ──────────────────────────────────────────────────────

FEATURE_COLS = [
    "log_fleet_size", "fleet_age", "fleet_diversity",
    "pct_personbil", "pct_heavy", "pct_brand",
]


def engineer_client_features(fleet: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """One row per client with fleet composition features + claims targets."""
    clients = fleet.groupby("Client").agg(
        vehicle_count=("regnr", "count"),
        avg_model_year=("Arsmodell", "mean"),
        fleet_diversity=("UW_Kategori", "nunique"),
    ).reset_index()

    # Category percentages
    cat_counts = fleet.groupby(["Client", "UW_Kategori"]).size().unstack(fill_value=0)
    cat_pcts = cat_counts.div(cat_counts.sum(axis=1), axis=0)
    pct = cat_pcts.to_dict()

    clients["pct_personbil"] = clients["Client"].map(pct.get("Personbil", {})).fillna(0)
    clients["pct_heavy"] = 0.0
    for col in ["Tung lastbil", "Buss"]:
        if col in pct:
            clients["pct_heavy"] += clients["Client"].map(pct[col]).fillna(0)
    clients["pct_brand"] = 0.0
    for col in ["Brandfordon", "Lätt brandfordon"]:
        if col in pct:
            clients["pct_brand"] += clients["Client"].map(pct[col]).fillna(0)

    clients["fleet_age"] = 2024 - clients["avg_model_year"]
    clients["log_fleet_size"] = np.log1p(clients["vehicle_count"])

    # Claims targets
    if not claims.empty:
        cl = claims.groupby("Client").agg(
            claim_count=("Client", "size"),
            total_incurred=("Incurred idx", "sum"),
            avg_severity=("Incurred idx", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            exposure_years=("CLAIM_YEAR", "nunique"),
        ).reset_index()
        cl["exposure_years"] = cl["exposure_years"].clip(lower=1)
        clients = clients.merge(cl, on="Client", how="left").fillna(0)

    clients["exposure"] = clients["vehicle_count"] * clients["exposure_years"]
    clients["claim_frequency"] = np.where(
        clients["exposure"] > 0,
        clients["claim_count"] / clients["exposure"],
        0,
    )
    return clients


# ── Poisson GLM ──────────────────────────────────────────────────────────────

def fit_frequency_model(features: pd.DataFrame):
    """Poisson GLM for claim frequency with LOO cross-validation."""
    df = features.copy()
    X = df[FEATURE_COLS].fillna(0).values
    exposure = df["exposure"].values
    y_rate = df["claim_count"].values.astype(float) / np.maximum(exposure, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = PoissonRegressor(alpha=1.0, max_iter=1000)
    model.fit(X_scaled, y_rate, sample_weight=exposure)
    df["predicted_frequency"] = model.predict(X_scaled)
    df["predicted_claims"] = df["predicted_frequency"] * df["exposure"]

    # Leave-One-Out CV (honest performance on 18 data points)
    loo_preds = np.zeros(len(X))
    for tr, te in LeaveOneOut().split(X):
        m = PoissonRegressor(alpha=1.0, max_iter=1000)
        m.fit(X_scaled[tr], y_rate[tr], sample_weight=exposure[tr])
        loo_preds[te] = m.predict(X_scaled[te])
    df["loo_predicted_freq"] = loo_preds

    importance = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Coefficient": model.coef_,
        "Abs_Coef": np.abs(model.coef_),
    }).sort_values("Abs_Coef", ascending=False)

    metrics = {
        "in_sample_mae": round(mean_absolute_error(y_rate, df["predicted_frequency"]), 5),
        "loo_mae": round(mean_absolute_error(y_rate, loo_preds), 5),
        "baseline_mae": round(mean_absolute_error(y_rate, np.full_like(y_rate, y_rate.mean())), 5),
        "n_clients": len(df),
    }
    return model, scaler, df, importance, metrics


# ── Bühlmann Credibility Weighting ───────────────────────────────────────────

def credibility_weight(client_freq, portfolio_freq, n_claims, k=50):
    """Z = n/(n+k). Returns (blended_freq, z_factor)."""
    z = n_claims / (n_claims + k)
    return z * client_freq + (1 - z) * portfolio_freq, z


def credibility_analysis(features: pd.DataFrame, portfolio_freq: float) -> pd.DataFrame:
    """Credibility-weighted frequencies for all clients."""
    rows = []
    for _, r in features.iterrows():
        blended, z = credibility_weight(r["claim_frequency"], portfolio_freq, r["claim_count"])
        rows.append({
            "Client": r["Client"],
            "Vehicles": int(r["vehicle_count"]),
            "Claims": int(r["claim_count"]),
            "Observed Freq": round(r["claim_frequency"], 4),
            "Portfolio Freq": round(portfolio_freq, 4),
            "Z-Factor": round(z, 3),
            "Credibility Freq": round(blended, 4),
            "Change %": round((blended / max(r["claim_frequency"], 0.0001) - 1) * 100, 1),
        })
    return pd.DataFrame(rows)


# ── Bootstrap Uncertainty on Premium ─────────────────────────────────────────

def bootstrap_premium(claims: pd.DataFrame, fleet: pd.DataFrame, client: str,
                      loading: float = 1.30, n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap CI: resample claim count (Poisson) + severities."""
    rng = np.random.RandomState(seed)
    cc = claims[claims["Client"] == client]
    n_vehicles = len(fleet[fleet["Client"] == client])
    if cc.empty or n_vehicles == 0:
        return {"error": "No data"}

    observed_n = len(cc)
    exposure = max(1, cc["CLAIM_YEAR"].nunique())
    amounts = cc["Incurred idx"].values

    premiums = []
    for _ in range(n_boot):
        n = rng.poisson(observed_n)
        if n == 0:
            premiums.append(0.0)
            continue
        boot = rng.choice(amounts, size=n, replace=True)
        premiums.append(boot.sum() / exposure * loading)

    premiums = np.array(premiums)
    return {
        "mean": float(np.mean(premiums)),
        "median": float(np.median(premiums)),
        "std": float(np.std(premiums)),
        "ci_5": float(np.percentile(premiums, 5)),
        "ci_25": float(np.percentile(premiums, 25)),
        "ci_75": float(np.percentile(premiums, 75)),
        "ci_95": float(np.percentile(premiums, 95)),
        "premiums": premiums,
    }


# ── Anomaly Detection (IQR) ─────────────────────────────────────────────────

def detect_claim_anomalies(claims: pd.DataFrame, client: str | None = None) -> pd.DataFrame:
    """Flag outlier claims using IQR on severity."""
    cc = claims[claims["Client"] == client].copy() if client else claims.copy()
    nz = cc[cc["Incurred idx"] > 0].copy()
    if nz.empty:
        return pd.DataFrame()

    q1, q3 = nz["Incurred idx"].quantile(0.25), nz["Incurred idx"].quantile(0.75)
    iqr = q3 - q1
    upper, severe = q3 + 1.5 * iqr, q3 + 3.0 * iqr

    nz["anomaly_score"] = ((nz["Incurred idx"] - q3) / max(iqr, 1)).round(2)
    nz["flag"] = "Normal"
    nz.loc[nz["Incurred idx"] > upper, "flag"] = "Outlier"
    nz.loc[nz["Incurred idx"] > severe, "flag"] = "Severe Outlier"
    nz["threshold"] = round(upper, 0)
    return nz


# ── Client Risk Clustering ──────────────────────────────────────────────────

CLUSTER_FEATURES = ["claim_frequency", "avg_severity", "vehicle_count", "fleet_age"]


def cluster_clients(features: pd.DataFrame, n_clusters: int = 3):
    """K-means risk segmentation."""
    avail = [f for f in CLUSTER_FEATURES if f in features.columns]
    X = features[avail].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n = min(n_clusters, len(features))
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    df = features.copy()
    df["cluster"] = labels
    order = df.groupby("cluster")["claim_frequency"].mean().sort_values()
    remap = {old: new for new, old in enumerate(order.index)}
    names = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    df["cluster"] = df["cluster"].map(remap)
    df["risk_segment"] = df["cluster"].map(names)
    return df, km, scaler, X_scaled
