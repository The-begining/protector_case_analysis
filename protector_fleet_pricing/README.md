# Protector Fleet Pricing Engine

Interactive fleet risk analysis and pricing dashboard for insurance underwriters.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

```
protector_fleet_pricing/
├── app.py             # Streamlit dashboard (3 tabs: Deep Dive, Benchmark, Classification)
├── pipeline.py        # Data loading, cleaning, integration, classification
├── pricing.py         # Risk metrics, pricing engine, benchmarking
├── requirements.txt   # Dependencies
├── Dockerfile         # Container deployment
└── data/raw/          # Place Excel file here
```

## Features

- **Client Deep Dive**: Fleet composition, claims analysis, severity distribution, price proposal
- **Portfolio Benchmark**: Cross-client comparisons, radar chart, ranking table
- **Classification Review**: Review heuristic classifications, manual override
- **Interactive Pricing**: Adjust frequency, cost, margin, expense, reinsurance via sidebar sliders
