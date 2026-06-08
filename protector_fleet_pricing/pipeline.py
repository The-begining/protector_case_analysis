"""
Data pipeline: load, clean, classify all vehicles.
Single module that handles the full ingestion → classification flow.
"""
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ──
DATA_DIR = Path(__file__).parent / "data" / "raw"
DEFAULT_EXCEL = DATA_DIR / "Case - Datasett -Analytics and AI Solutions Engineer.xlsx"

# ── Egenklass → UW Kategori mapping ──
EGENKLASS_TO_UW = {
    "Personbil": "Personbil",
    "Lätt lastbil": "Lätt lastbil",
    "Lätt släp": "Släp",
    "Släp": "Släp",
    "Brandbil": "Brandfordon",
    "Brandbil lätt": "Lätt brandfordon",
    "Lastbil": "Tung lastbil",
    "Traktor": "Traktor, minitraktor, åkgräsklippare",
    "Tungt släp": "Tungt släp",
    "Buss": "Buss",
    "Moped": "Moped",
    "Alla släp": "Släp",
    "Motorcykel/ATV": "Motorcykel/ATV",
}

# Fordonsslag-based fallback for vehicles missing from API
FORDONSSLAG_DEFAULTS = {
    "PB": "Personbil", "LB": "Lätt lastbil", "SL": "Släp",
    "BU": "Buss", "MC": "Motorcykel/ATV", "MP": "Moped",
    "TR": "Traktor, minitraktor, åkgräsklippare",
    "MR": "Motorredskap lätt", "TF": "Terrängfordon",
}

VALID_UW_CATEGORIES = [
    "Brandfordon", "Buss", "Lätt brandfordon", "Lätt lastbil",
    "Lätt lastbil EL", "Moped", "Motorcykel/ATV", "Motorredskap lätt",
    "Motorredskap tungt", "Personbil", "Personbil EL", "Släp",
    "Terrängfordon", "Traktor, minitraktor, åkgräsklippare",
    "Tung lastbil", "Tungt släp",
]

HEAVY_WEIGHT_KG = 3500


# LLM Classification (Google Gemini for vehicles missing from API data)

def _build_llm_prompt(row: pd.Series) -> str:
    """Build a classification prompt from vehicle attributes."""
    return f"""You are an insurance underwriter classifying vehicles for fleet insurance in Scandinavia.

Given the following vehicle information:
- Brand (Märke): {row.get('Marke', 'Unknown')}
- Body type (Karosserikod): {row.get('Karosserikod', 'Unknown')}
- Vehicle type (Fordonsslag): {row.get('Fordonsslag', 'Unknown')}
- Total weight (kg): {row.get('TotalVikt', 'Unknown')}
- Model year: {row.get('Arsmodell', 'Unknown')}

Classify this vehicle into exactly ONE of these underwriting categories:
{json.dumps(VALID_UW_CATEGORIES, ensure_ascii=False)}

Respond with ONLY valid JSON:
{{"uw_kategori": "<category>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""


def classify_with_llm(unclassified: pd.DataFrame) -> pd.DataFrame:
    """Classify vehicles using Google Gemini 2.5 Flash.

    Uses default Google authentication (application default credentials
    or GOOGLE_CLOUD_PROJECT env var). Falls back to empty DataFrame
    if google-genai is not installed.
    """
    try:
        from google import genai
    except ImportError:
        return pd.DataFrame()

    try:
        client = genai.Client()
    except Exception:
        return pd.DataFrame()

    results = unclassified.copy()
    categories, confidences, reasonings = [], [], []

    for _, row in results.iterrows():
        prompt = _build_llm_prompt(row)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            content = response.text.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            cat = parsed.get("uw_kategori", "")
            conf = float(parsed.get("confidence", 0.0))
            reason = parsed.get("reasoning", "")
            if cat not in VALID_UW_CATEGORIES:
                cat, conf, reason = None, 0.0, f"Invalid LLM output: {parsed.get('uw_kategori')}"
        except Exception as e:
            cat, conf, reason = None, 0.0, f"LLM error: {e}"

        categories.append(cat)
        confidences.append(conf)
        reasonings.append(reason)

    results["UW_Kategori"] = categories
    results["classification_confidence"] = confidences
    results["classification_reasoning"] = reasonings
    results["classification_method"] = "llm"
    return results


# Loading

def load_data(filepath: str | Path | None = None) -> dict:
    """Load all four sheets from the Excel workbook."""
    fp = Path(filepath) if filepath else DEFAULT_EXCEL
    if not fp.exists():
        raise FileNotFoundError(f"Data file not found: {fp}")

    objects = pd.read_excel(fp, sheet_name="Objects")

    api_data = pd.read_excel(
        fp,
        sheet_name='"External data - API data"',
        header=2,  # rows 0-1 are blank
    )

    claims = pd.read_excel(fp, sheet_name="Claims", header=2)

    uw_cats = pd.read_excel(fp, sheet_name="UW categories", header=None, skiprows=3)
    uw_categories = uw_cats.iloc[:, 0].dropna().astype(str).tolist()

    return {
        "objects": objects,
        "api_data": api_data,
        "claims": claims,
        "uw_categories": uw_categories,
    }


# Cleaning

def _clean_str_col(series: pd.Series) -> pd.Series:
    """Strip whitespace, convert 'nan' strings to None."""
    s = series.astype(str).str.strip()
    return s.replace({"nan": None, "None": None, "": None})


def clean_data(data: dict) -> dict:
    """Clean and standardize all DataFrames."""
    # ── Objects ──
    obj = data["objects"].copy()
    for col in ["Client", "regnr", "Marke", "Fordonsslag", "Fordonsstatus", "Karosserikod"]:
        if col in obj.columns:
            obj[col] = _clean_str_col(obj[col])
    if "Arsmodell" in obj.columns:
        obj["Arsmodell"] = pd.to_numeric(obj["Arsmodell"], errors="coerce")
    if "TotalVikt" in obj.columns:
        obj["TotalVikt"] = pd.to_numeric(obj["TotalVikt"], errors="coerce")
    if "Uwcategory" in obj.columns:
        obj.drop(columns=["Uwcategory"], inplace=True)

    # ── API data — keep only useful columns ──
    api = data["api_data"].copy()
    keep = ["Client", "regnr", "Marke", "Fordonsslag", "Karosserikod",
            "Egenklass", "TotalVikt", "Motoreffekt", "Handelsbet", "Karossny"]
    keep = [c for c in keep if c in api.columns]
    api = api[keep]
    for col in ["Client", "regnr", "Marke", "Egenklass", "Karosserikod"]:
        if col in api.columns:
            api[col] = _clean_str_col(api[col])
    api = api.dropna(subset=["regnr"])

    # ── Claims ──
    cl = data["claims"].copy()
    if "CLIENT" in cl.columns:
        cl.rename(columns={"CLIENT": "Client"}, inplace=True)
    if "Client" in cl.columns:
        cl["Client"] = _clean_str_col(cl["Client"])
        cl = cl[cl["Client"].notna()]
    if "CLAIM_YEAR" in cl.columns:
        cl["CLAIM_YEAR"] = pd.to_numeric(cl["CLAIM_YEAR"], errors="coerce")
    if "DAMAGE_DATE" in cl.columns:
        cl["DAMAGE_DATE"] = pd.to_numeric(cl["DAMAGE_DATE"], errors="coerce")
        cl["DAMAGE_DATE"] = pd.to_datetime(cl["DAMAGE_DATE"], unit="D", origin="1899-12-30", errors="coerce")
    # Financial columns
    for col in ["LAST_MONTH_NET_PAID", "LAST_MONTH_REMAINING_RESERVES",
                "LAST_MONTH_INCURRED", "DEDUCTIBLE", "Net paid adj",
                "Reserves adj", "Incurred adj", "idx",
                "Net paid idx", "Reserves idx", "Incurred idx"]:
        if col in cl.columns:
            cl[col] = pd.to_numeric(cl[col], errors="coerce").fillna(0)
    for col in ["Incl. 0", "Ex. 0", "minor", "major"]:
        if col in cl.columns:
            cl[col] = pd.to_numeric(cl[col], errors="coerce").fillna(0).astype(int)

    return {
        "objects": obj,
        "api_data": api,
        "claims": cl,
        "uw_categories": data["uw_categories"],
    }


# Integration & Classification

def build_fleet(objects: pd.DataFrame, api_data: pd.DataFrame) -> pd.DataFrame:
    """Join Objects with API data and classify all vehicles."""
    # LEFT JOIN to preserve every vehicle
    api_cols = ["regnr", "Egenklass"]
    for c in ["Motoreffekt", "Handelsbet", "Karossny"]:
        if c in api_data.columns:
            api_cols.append(c)
    api_sub = api_data[api_cols].drop_duplicates(subset=["regnr"])
    fleet = objects.merge(api_sub, on="regnr", how="left")
    fleet["api_matched"] = fleet["Egenklass"].notna() & (fleet["Egenklass"] != "???")

    # ── Rule-based classification ──
    fleet["UW_Kategori"] = fleet["Egenklass"].map(EGENKLASS_TO_UW)
    fleet["classification_method"] = np.where(fleet["UW_Kategori"].notna(), "rule_based", None)

    # ── Heuristic fallback for unclassified ──
    unclassified = fleet["UW_Kategori"].isna()
    if unclassified.any():
        # Try LLM first (if OPENAI_API_KEY is set)
        llm_result = classify_with_llm(fleet[unclassified])
        if not llm_result.empty and llm_result["UW_Kategori"].notna().any():
            llm_classified = llm_result["UW_Kategori"].notna()
            for col in ["UW_Kategori", "classification_method", "classification_confidence", "classification_reasoning"]:
                if col in llm_result.columns:
                    fleet.loc[unclassified, col] = llm_result[col].values
            # Refresh mask — some may still be unclassified after LLM
            unclassified = fleet["UW_Kategori"].isna()

        # Heuristic fallback for anything still unclassified
        if unclassified.any():
            cats, methods = [], []
            for _, row in fleet[unclassified].iterrows():
                fs = str(row.get("Fordonsslag", "")).strip()
                weight = row.get("TotalVikt")
                cat = FORDONSSLAG_DEFAULTS.get(fs, "Personbil")
                # Heavy truck refinement
                if fs == "LB" and weight and weight > HEAVY_WEIGHT_KG:
                    cat = "Tung lastbil"
                cats.append(cat)
                methods.append("heuristic")
            fleet.loc[unclassified, "UW_Kategori"] = cats
            fleet.loc[unclassified, "classification_method"] = methods

    return fleet


def run_pipeline(filepath: str | Path | None = None) -> dict:
    """Full pipeline: load → clean → integrate → classify. Returns all data."""
    raw = load_data(filepath)
    clean = clean_data(raw)
    fleet = build_fleet(clean["objects"], clean["api_data"])
    return {
        "fleet": fleet,
        "claims": clean["claims"],
        "uw_categories": clean["uw_categories"],
    }
