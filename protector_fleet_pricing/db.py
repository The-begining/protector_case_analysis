"""
Database layer — PostgreSQL connection, table creation, and data loading.
Supports both CSV (default) and PostgreSQL modes via environment variable.

Usage:
    Set DB_MODE=postgres and configure DB_* env vars to use PostgreSQL.
    Otherwise falls back to CSV files (no database needed).
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads .env file (git-ignored, never pushed)

# ── Configuration ──
DB_MODE = os.getenv("DB_MODE", "csv")  # "csv" or "postgres"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "protector_fleet_pricing"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

# ── Column mappings: DB (lowercase) ↔ App (original case) ──

FLEET_DB_TO_APP = {
    "client": "Client", "regnr": "regnr", "marke": "Marke",
    "fordonsslag": "Fordonsslag", "fordonsstatus": "Fordonsstatus",
    "karosserikod": "Karosserikod", "arsmodell": "Arsmodell",
    "totalvikt": "TotalVikt", "egenklass": "Egenklass",
    "motoreffekt": "Motoreffekt", "handelsbet": "Handelsbet",
    "karossny": "Karossny", "api_matched": "api_matched",
    "uw_kategori": "UW_Kategori", "classification_method": "classification_method",
    "classification_confidence": "classification_confidence",
    "classification_reasoning": "classification_reasoning",
}

CLAIMS_DB_TO_APP = {
    "client": "Client", "claim_type": "CLAIM_TYPE", "claim_cause": "CLAIM_CAUSE",
    "claim_year": "CLAIM_YEAR", "damage_date": "DAMAGE_DATE",
    "last_month_net_paid": "LAST_MONTH_NET_PAID",
    "last_month_remaining_reserves": "LAST_MONTH_REMAINING_RESERVES",
    "last_month_incurred": "LAST_MONTH_INCURRED", "deductible": "DEDUCTIBLE",
    "fleet_type": "FLEET_TYPE", "fleet_group": "FLEET_GROUP",
    "net_paid_adj": "Net paid adj", "reserves_adj": "Reserves adj",
    "incurred_adj": "Incurred adj", "idx": "idx",
    "net_paid_idx": "Net paid idx", "reserves_idx": "Reserves idx",
    "incurred_idx": "Incurred idx", "incl_zero": "Incl. 0",
    "ex_zero": "Ex. 0", "minor": "minor", "major": "major",
}

FLEET_APP_TO_DB = {v: k for k, v in FLEET_DB_TO_APP.items()}
CLAIMS_APP_TO_DB = {v: k for k, v in CLAIMS_DB_TO_APP.items()}


# ── SQL Schema ──

CREATE_FLEET_TABLE = """
CREATE TABLE IF NOT EXISTS fleet_classified (
    id                        SERIAL PRIMARY KEY,
    client                    VARCHAR(50)   NOT NULL,
    regnr                     VARCHAR(20)   NOT NULL,
    marke                     VARCHAR(50)   NOT NULL,
    fordonsslag               VARCHAR(20),
    fordonsstatus             VARCHAR(30),
    karosserikod              VARCHAR(100),
    arsmodell                 SMALLINT,
    totalvikt                 NUMERIC(10,2),
    egenklass                 VARCHAR(50),
    motoreffekt               NUMERIC(10,2),
    handelsbet                VARCHAR(100),
    karossny                  VARCHAR(100),
    api_matched               BOOLEAN       NOT NULL DEFAULT FALSE,
    uw_kategori               VARCHAR(60)   NOT NULL,
    classification_method     VARCHAR(20)   NOT NULL,
    classification_confidence NUMERIC(5,3),
    classification_reasoning  TEXT,
    created_at                TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_CLAIMS_TABLE = """
CREATE TABLE IF NOT EXISTS claims_cleaned (
    id                            SERIAL PRIMARY KEY,
    client                        VARCHAR(50)   NOT NULL,
    claim_type                    TEXT          NOT NULL,
    claim_cause                   TEXT          NOT NULL,
    claim_year                    SMALLINT      NOT NULL,
    damage_date                   DATE,
    last_month_net_paid           NUMERIC(14,2) DEFAULT 0,
    last_month_remaining_reserves NUMERIC(14,2) DEFAULT 0,
    last_month_incurred           NUMERIC(14,2) DEFAULT 0,
    deductible                    NUMERIC(14,2) DEFAULT 0,
    fleet_type                    VARCHAR(50),
    fleet_group                   VARCHAR(50),
    net_paid_adj                  NUMERIC(14,2) DEFAULT 0,
    reserves_adj                  NUMERIC(14,2) DEFAULT 0,
    incurred_adj                  NUMERIC(14,2) DEFAULT 0,
    idx                           NUMERIC(12,6) DEFAULT 1.0,
    net_paid_idx                  NUMERIC(14,2) DEFAULT 0,
    reserves_idx                  NUMERIC(14,2) DEFAULT 0,
    incurred_idx                  NUMERIC(14,2) DEFAULT 0,
    incl_zero                     SMALLINT      DEFAULT 0,
    ex_zero                       SMALLINT      DEFAULT 0,
    minor                         INTEGER       DEFAULT 0,
    major                         INTEGER       DEFAULT 0,
    created_at                    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_fleet_client ON fleet_classified(client);
CREATE INDEX IF NOT EXISTS idx_fleet_regnr ON fleet_classified(regnr);
CREATE INDEX IF NOT EXISTS idx_fleet_uw_kategori ON fleet_classified(uw_kategori);
CREATE INDEX IF NOT EXISTS idx_claims_client ON claims_cleaned(client);
CREATE INDEX IF NOT EXISTS idx_claims_year ON claims_cleaned(claim_year);
CREATE INDEX IF NOT EXISTS idx_claims_fleet_type ON claims_cleaned(fleet_type);
"""


# ── Database Functions ──

def _get_connection():
    """Get a psycopg2 connection."""
    import psycopg2
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    """Create tables and indexes if they don't exist."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(CREATE_FLEET_TABLE)
    cur.execute(CREATE_CLAIMS_TABLE)
    for stmt in CREATE_INDEXES.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)
    conn.commit()
    cur.close()
    conn.close()


def upload_to_db(fleet: pd.DataFrame, claims: pd.DataFrame):
    """Upload fleet and claims DataFrames to PostgreSQL.
    Clears existing data and replaces with new data.
    """
    import psycopg2
    from io import StringIO

    conn = _get_connection()
    cur = conn.cursor()

    # Rename columns to DB format
    fleet_db = fleet.rename(columns=FLEET_APP_TO_DB)
    claims_db = claims.rename(columns=CLAIMS_APP_TO_DB)

    # Keep only DB columns (drop any extras)
    fleet_db = fleet_db[[c for c in FLEET_DB_TO_APP.keys() if c in fleet_db.columns]]
    claims_db = claims_db[[c for c in CLAIMS_DB_TO_APP.keys() if c in claims_db.columns]]

    # Fix float → int for SMALLINT columns (e.g. 2019.0 → 2019)
    int_cols_fleet = ["arsmodell"]
    for col in int_cols_fleet:
        if col in fleet_db.columns:
            fleet_db[col] = pd.to_numeric(fleet_db[col], errors="coerce").astype("Int64")
    int_cols_claims = ["claim_year", "incl_zero", "ex_zero", "minor", "major"]
    for col in int_cols_claims:
        if col in claims_db.columns:
            claims_db[col] = pd.to_numeric(claims_db[col], errors="coerce").astype("Int64")

    # Clear existing data
    cur.execute("TRUNCATE fleet_classified RESTART IDENTITY CASCADE;")
    cur.execute("TRUNCATE claims_cleaned RESTART IDENTITY CASCADE;")

    # Bulk insert using COPY (much faster than INSERT)
    for table_name, df in [("fleet_classified", fleet_db), ("claims_cleaned", claims_db)]:
        buf = StringIO()
        df.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N")
        buf.seek(0)
        cols = ", ".join(df.columns)
        cur.copy_expert(f"COPY {table_name}({cols}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\N')", buf)

    conn.commit()
    cur.close()
    conn.close()


def load_from_db() -> dict:
    """Load fleet and claims from PostgreSQL, return with app-style column names."""
    conn = _get_connection()
    fleet = pd.read_sql("SELECT * FROM fleet_classified ORDER BY id", conn)
    claims = pd.read_sql("SELECT * FROM claims_cleaned ORDER BY id", conn)
    conn.close()

    # Drop DB-only columns
    for col in ["id", "created_at"]:
        if col in fleet.columns:
            fleet.drop(columns=[col], inplace=True)
        if col in claims.columns:
            claims.drop(columns=[col], inplace=True)

    # Rename back to app column names
    fleet.rename(columns=FLEET_DB_TO_APP, inplace=True)
    claims.rename(columns=CLAIMS_DB_TO_APP, inplace=True)

    return {"fleet": fleet, "claims": claims}


def load_from_csv() -> dict:
    """Load fleet and claims from CSV files."""
    fleet = pd.read_csv(PROCESSED_DIR / "fleet_classified.csv")
    claims = pd.read_csv(PROCESSED_DIR / "claims_cleaned.csv")
    return {"fleet": fleet, "claims": claims}


def load_data() -> dict:
    """Load data from configured source (CSV or PostgreSQL)."""
    if DB_MODE == "postgres":
        return load_from_db()
    return load_from_csv()
