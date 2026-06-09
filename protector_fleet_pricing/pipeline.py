"""
Task 1 — Ingestion & Processing Pipeline
==========================================
Loads four sheets from the Excel file, fixes headers,
cleans data, joins Objects with API data, and classifies
all vehicles into UW categories.

Classification cascade:
  1. Rule-based: Egenklass → UW_Kategori mapping (for API-matched vehicles)
  2. LLM: Google Gemini classifies unmatched vehicles from attributes
  3. ML: Logistic Regression trained on rule-based labels (fallback)
  4. Heuristic: Fordonsslag code lookup (final fallback)
"""
import json
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

# Fordonsslag-based heuristic fallback
FORDONSSLAG_DEFAULTS = {
    "PB": "Personbil", "LB": "Lätt lastbil", "SLÄP": "Släp",
    "BU": "Buss", "MC": "Motorcykel/ATV", "MOPED": "Moped",
    "TR": "Traktor, minitraktor, åkgräsklippare", "BUSS": "Buss",
    "MRED": "Motorredskap lätt", "TGSK": "Terrängfordon",
    "TGHJUL": "Terrängfordon", "TGV": "Terrängfordon", "TGSNÖ": "Terrängfordon",
}

VALID_UW_CATEGORIES = [
    "Brandfordon", "Buss", "Lätt brandfordon", "Lätt lastbil",
    "Lätt lastbil EL", "Moped", "Motorcykel/ATV", "Motorredskap lätt",
    "Motorredskap tungt", "Personbil", "Personbil EL", "Släp",
    "Terrängfordon", "Traktor, minitraktor, åkgräsklippare",
    "Tung lastbil", "Tungt släp",
]

HEAVY_WEIGHT_KG = 3500


# ═══════════════════════════════════════════════════════════════
# Step 1: Load & Clean
# ═══════════════════════════════════════════════════════════════

def load_data(filepath: str | Path | None = None) -> dict:
    """Load all four sheets from the Excel workbook."""
    fp = Path(filepath) if filepath else DEFAULT_EXCEL
    if not fp.exists():
        raise FileNotFoundError(f"Data file not found: {fp}")

    xl = pd.ExcelFile(fp)

    # Objects — header in row 0
    df_objects = xl.parse("Objects")
    df_objects.columns = [c.strip() for c in df_objects.columns]

    # API data — rows 0-1 are blank, row 2 holds column names
    df_api = xl.parse('"External data - API data"', header=None)
    df_api = df_api.iloc[2:].copy()
    df_api.columns = [str(c).strip() for c in df_api.iloc[0].values]
    df_api = df_api.iloc[1:].reset_index(drop=True)

    # Claims — rows 0-1 are blank, row 2 holds column names
    df_claims = xl.parse("Claims", header=None)
    df_claims = df_claims.iloc[2:].copy()
    df_claims.columns = [str(c).strip() for c in df_claims.iloc[0].values]
    df_claims = df_claims.iloc[1:].reset_index(drop=True)

    # UW categories — skip header text, keep only category names
    df_uw = xl.parse("UW categories", header=None)
    uw_categories = [
        str(x).strip() for x in df_uw[0].dropna().tolist()
        if "Use this" not in str(x)
    ]

    return {
        "objects": df_objects,
        "api_data": df_api,
        "claims": df_claims,
        "uw_categories": uw_categories,
    }


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


# ═══════════════════════════════════════════════════════════════
# Step 2: Classify All Vehicles
# ═══════════════════════════════════════════════════════════════

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
    Falls back to empty DataFrame if google-genai is not installed.
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
                model="gemini-2.5-flash", contents=prompt,
            )
            content = response.text.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            cat = parsed.get("uw_kategori", "")
            conf = float(parsed.get("confidence", 0.0))
            reason = parsed.get("reasoning", "")
            if cat not in VALID_UW_CATEGORIES:
                cat, conf, reason = None, 0.0, f"Invalid: {parsed.get('uw_kategori')}"
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


def classify_with_ml(fleet: pd.DataFrame, unclassified_mask: pd.Series) -> pd.DataFrame:
    """Logistic Regression trained on rule-based labels.
    Uses Fordonsslag + Karosserikod as features.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    training = fleet[fleet["classification_method"] == "rule_based"].copy()
    to_predict = fleet[unclassified_mask].copy()

    if len(training) < 20 or to_predict.empty:
        return pd.DataFrame()

    feature_cols = ["Fordonsslag", "Karosserikod"]
    encoders = {}
    for col in feature_cols:
        le = LabelEncoder()
        all_vals = pd.concat([training[col], to_predict[col]]).fillna("MISSING").astype(str)
        le.fit(all_vals)
        training[f"{col}_enc"] = le.transform(training[col].fillna("MISSING").astype(str))
        to_predict[f"{col}_enc"] = le.transform(to_predict[col].fillna("MISSING").astype(str))
        encoders[col] = le

    enc_cols = [f"{c}_enc" for c in feature_cols]
    X_train = training[enc_cols].values.astype(float)
    y_train = training["UW_Kategori"].values.copy()
    X_pred = to_predict[enc_cols].values.astype(float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_pred = scaler.transform(X_pred)

    model = LogisticRegression(max_iter=5000, solver="saga", random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_pred)
    confidences = model.predict_proba(X_pred).max(axis=1)

    to_predict["UW_Kategori"] = predictions
    to_predict["classification_confidence"] = confidences.round(3)
    to_predict["classification_method"] = "ml_model"
    to_predict["classification_reasoning"] = [
        f"LogReg: {cat} ({conf:.0%})" for cat, conf in zip(predictions, confidences)
    ]
    return to_predict


def build_fleet(objects: pd.DataFrame, api_data: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN Objects with API data, then classify all vehicles."""
    # Join on regnr
    api_cols = ["regnr", "Egenklass"]
    for c in ["Motoreffekt", "Handelsbet", "Karossny"]:
        if c in api_data.columns:
            api_cols.append(c)
    api_sub = api_data[api_cols].drop_duplicates(subset=["regnr"])
    fleet = objects.merge(api_sub, on="regnr", how="left")
    fleet["api_matched"] = fleet["Egenklass"].notna() & (fleet["Egenklass"] != "???")

    # 1. Rule-based: Egenklass → UW_Kategori
    fleet["UW_Kategori"] = fleet["Egenklass"].map(EGENKLASS_TO_UW)
    fleet["classification_method"] = np.where(fleet["UW_Kategori"].notna(), "rule_based", None)

    unclassified = fleet["UW_Kategori"].isna()
    if not unclassified.any():
        return fleet

    # 2. LLM (if google-genai available)
    llm_result = classify_with_llm(fleet[unclassified])
    if not llm_result.empty and llm_result["UW_Kategori"].notna().any():
        for col in ["UW_Kategori", "classification_method", "classification_confidence", "classification_reasoning"]:
            if col in llm_result.columns:
                fleet.loc[unclassified, col] = llm_result[col].values
        unclassified = fleet["UW_Kategori"].isna()

    # 3. ML classifier (Logistic Regression)
    if unclassified.any():
        ml_result = classify_with_ml(fleet, unclassified)
        if not ml_result.empty:
            for col in ["UW_Kategori", "classification_method", "classification_confidence", "classification_reasoning"]:
                if col in ml_result.columns:
                    fleet.loc[unclassified, col] = ml_result[col].values
            unclassified = fleet["UW_Kategori"].isna()

    # 4. Heuristic fallback
    if unclassified.any():
        cats, methods = [], []
        for _, row in fleet[unclassified].iterrows():
            fs = str(row.get("Fordonsslag", "")).strip()
            weight = row.get("TotalVikt")
            cat = FORDONSSLAG_DEFAULTS.get(fs, "Personbil")
            if fs == "LB" and weight and weight > HEAVY_WEIGHT_KG:
                cat = "Tung lastbil"
            cats.append(cat)
            methods.append("heuristic")
        fleet.loc[unclassified, "UW_Kategori"] = cats
        fleet.loc[unclassified, "classification_method"] = methods

    return fleet


# ═══════════════════════════════════════════════════════════════
# Run Full Pipeline
# ═══════════════════════════════════════════════════════════════

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


def run_pipeline(filepath: str | Path | None = None, save: bool = True) -> dict:
    """Full pipeline: load → clean → join → classify → save (CSV + optionally DB)."""
    raw = load_data(filepath)
    clean = clean_data(raw)
    fleet = build_fleet(clean["objects"], clean["api_data"])

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        fleet.to_csv(PROCESSED_DIR / "fleet_classified.csv", index=False)
        clean["claims"].to_csv(PROCESSED_DIR / "claims_cleaned.csv", index=False)

        # Upload to PostgreSQL if configured
        from db import DB_MODE
        if DB_MODE == "postgres":
            from db import init_db, upload_to_db
            init_db()
            upload_to_db(fleet, clean["claims"])

    return {
        "fleet": fleet,
        "claims": clean["claims"],
        "uw_categories": clean["uw_categories"],
    }
