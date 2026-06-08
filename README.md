# 🚗 Protector Fleet Pricing Engine

Intelligent fleet risk analysis and pricing tool for insurance underwriters.
Built as a prototype for the Analytics & AI Solutions Engineer case study.

## Quick Start

```bash
cd protector_fleet_pricing
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Client 18 is selected by default.

## Project Structure

```
protector_fleet_pricing/
├── pipeline.py                  # Data ingestion, cleaning, vehicle classification
├── pricing.py                   # Risk-differentiated pricing model & sensitivity analysis
├── app.py                       # Streamlit dashboard (4 tabs)
├── initial_data_analysis.ipynb  # Jupyter notebook: full initial data exploration
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container deployment
└── data/raw/                    # Excel data file (place here)
```

## What It Does

### Step 1: Data Ingestion & Processing (`pipeline.py`)
- Loads 4 Excel sheets (Objects, API data, Claims, UW categories)
- Handles quirks: quoted sheet names, blank header rows, Excel serial dates
- Normalizes column names (`CLIENT` → `Client`), cleans strings, converts types
- LEFT JOINs Objects with API data on `regnr`

### Step 2: Vehicle Classification (`pipeline.py`)
- **Rule-based** (82%): Maps `Egenklass` → UW category using a lookup table
- **LLM** (Google Gemini 2.5 Flash): Classifies OREG/unknown vehicles using vehicle attributes
- **Heuristic fallback**: Uses `Fordonsslag` codes when LLM is unavailable
- Result: **100% of 2,376 vehicles classified** into 16 UW categories

### Step 3: Interactive Dashboard (`app.py`)

| Tab | What It Shows |
|-----|--------------|
| **🔍 Client Deep Dive** | KPIs vs portfolio, price proposal with category breakdown, fleet composition charts, claims analysis (trend, types, causes, severity distribution) |
| **📊 Portfolio Benchmark** | All 18 clients compared: fleet size, frequency, severity, loss/vehicle. Radar chart, risk scatter, ranking table, fleet composition comparison |
| **💰 Pricing Model** | Model methodology, data-driven risk factors table + chart, per-category price breakdown, sensitivity analysis (frequency, severity, margin scenarios) |
| **🏷️ Classification Review** | Heuristic classification queue, manual override with session persistence |

### Step 4: Pricing with Adjustable Assumptions (`pricing.py`)

**Model: Risk-Differentiated Burning Cost**

```
Per Category:
  Pure Premium  = Vehicles × (Frequency × Risk Factor × Age Adj) × Severity
  Gross Premium = Pure Premium × (1 + Profit + Expenses + Reinsurance)
```

**Sidebar controls (underwriter adjustable):**
- Frequency adjustment (×0.5 to ×2.0)
- Severity adjustment (×0.5 to ×2.0)
- Category risk factors toggle (data-driven, on/off)
- Fleet age adjustment toggle (+10% for older fleets)
- Large loss cap (optional)
- Profit margin, expense loading, reinsurance loading sliders

**Risk factors are data-driven** — computed from `FLEET_TYPE` in claims:
```
Risk Factor = Loss Rate(category) / Loss Rate(Personbil)

Tung lastbil:  3.12×  (high severity, expensive repairs)
Brandfordon:   1.82×  (emergency vehicle exposure)
Personbil:     1.00×  (baseline)
Lätt lastbil:  0.85×  (lower frequency than cars)
Släp:          0.02×  (almost no claims)
```

## Client 18 Results

| Metric | Value |
|--------|-------|
| Fleet size | 258 vehicles |
| Claims (5 years) | 191 |
| Claim frequency | 0.148 (35% below portfolio avg) |
| Avg claim cost | 12,870 SEK (34% below avg) |
| Recommended premium | **530,894 SEK/year** |
| Per vehicle | 2,058 SEK |
| Loss ratio | 58.7% ✅ Profitable |

## Data Requirements

Place the Excel file in `protector_fleet_pricing/data/raw/`:
```
Case - Datasett -Analytics and AI Solutions Engineer.xlsx
```

Expected sheets:
- **Objects** — 2,376 vehicles (master list)
- **"External data - API data"** — 2,067 vehicle registry records (header row 2)
- **Claims** — 2,049 claims with financial data (header row 2)
- **UW categories** — 16 target classification labels (skip first 3 rows)

## Dependencies

```
pandas, openpyxl, streamlit, plotly, google-genai (optional, for LLM classification)
```

## Docker

```bash
cd protector_fleet_pricing
docker build -t fleet-pricing .
docker run -p 8501:8501 fleet-pricing
```