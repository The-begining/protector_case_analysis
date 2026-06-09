"""Core tests: data validation, pipeline integrity, and model sanity checks."""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def pipeline_data():
    """Run the pipeline once, share across all tests."""
    from pipeline import run_pipeline
    return run_pipeline()


@pytest.fixture(scope="session")
def fleet(pipeline_data):
    return pipeline_data["fleet"]


@pytest.fixture(scope="session")
def claims(pipeline_data):
    return pipeline_data["claims"]


# ── Data Quality ─────────────────────────────────────────────────────────────

class TestDataQuality:

    def test_fleet_classification_coverage(self, fleet):
        assert fleet["UW_Kategori"].notna().all(), "Some vehicles lack UW_Kategori"

    def test_valid_uw_categories(self, fleet):
        from pipeline import VALID_UW_CATEGORIES
        invalid = set(fleet["UW_Kategori"].unique()) - set(VALID_UW_CATEGORIES)
        assert not invalid, f"Invalid categories: {invalid}"

    def test_incurred_mostly_non_negative(self, claims):
        neg_pct = (claims["Incurred idx"] < 0).mean()
        assert neg_pct < 0.05, f"{neg_pct:.1%} negative incurred (expect <5%)"

    def test_claim_clients_have_vehicles(self, fleet, claims):
        missing = set(claims["Client"].unique()) - set(fleet["Client"].unique())
        assert not missing, f"Clients in claims but not fleet: {missing}"


# ── Pricing ──────────────────────────────────────────────────────────────────

class TestPricing:

    def test_positive_premium(self, fleet, claims):
        from pricing import calculate_price, PricingParams
        price = calculate_price(fleet, claims, "Client - 18", PricingParams())
        assert price["total_price"] > 0
        assert price["pure_premium"] > 0
        assert price["price_per_vehicle"] > 0

    def test_higher_loading_higher_price(self, fleet, claims):
        from pricing import calculate_price, PricingParams
        low = calculate_price(fleet, claims, "Client - 18", PricingParams(profit_margin=0.05))
        high = calculate_price(fleet, claims, "Client - 18", PricingParams(profit_margin=0.25))
        assert high["total_price"] > low["total_price"]


# ── ML Models ────────────────────────────────────────────────────────────────

class TestModels:

    @pytest.fixture(scope="class")
    def features(self, fleet, claims):
        from models import engineer_client_features
        return engineer_client_features(fleet, claims)

    def test_feature_engineering_shape(self, features):
        assert len(features) == 18
        assert "claim_frequency" in features.columns
        assert (features["vehicle_count"] > 0).all()

    def test_glm_predictions_non_negative(self, features):
        from models import fit_frequency_model
        _, _, pred_df, importance, metrics = fit_frequency_model(features)
        assert (pred_df["predicted_frequency"] >= 0).all()
        assert len(importance) == 6

    def test_credibility_z_between_0_and_1(self, features):
        from models import credibility_analysis
        cred = credibility_analysis(features, features["claim_frequency"].mean())
        assert (cred["Z-Factor"] >= 0).all() and (cred["Z-Factor"] <= 1).all()

    def test_bootstrap_produces_ci(self, fleet, claims):
        from models import bootstrap_premium
        result = bootstrap_premium(claims, fleet, "Client - 18", n_boot=100)
        assert "error" not in result
        assert result["ci_5"] < result["ci_95"]

    def test_anomaly_detection_flags(self, claims):
        from models import detect_claim_anomalies
        result = detect_claim_anomalies(claims, "Client - 18")
        assert "flag" in result.columns
        assert set(result["flag"].unique()).issubset({"Normal", "Outlier", "Severe Outlier"})
