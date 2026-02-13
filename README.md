# Basketball ML Betting

Machine learning system for finding profitable edges in college basketball (NCAA) betting markets. Targets peer-to-peer exchanges rather than sportsbooks.

## Project Status

**Phase 0: Foundation & Infrastructure** — complete

- SQLite feature store with anti-lookahead temporal queries
- Data collectors: ESPN/cbbpy box scores, Barttorvik ratings, The Odds API, SBRO historical odds
- Feature engine: possessions, pace, ORtg/DRtg, Dean Oliver's Four Factors (EWMA)
- Naive paper trading pipeline (logistic regression, 2 features)
- 56 unit/integration tests passing

## Quick Start

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install
uv venv .venv --python 3.12
uv pip install -e ".[dev]"

# Configure API key
cp .env.example .env  # Edit with your Odds API key

# Run tests
.venv/bin/python -m pytest tests/ -v

# Build historical dataset (slow — ~3-4 hrs/season)
.venv/bin/python scripts/build_historical.py 2025 2026

# Daily collection (or set up as cron)
.venv/bin/python scripts/daily_collect.py

# Paper trading
.venv/bin/python scripts/paper_trade.py --date 2026-02-13
```

## Architecture

```
ESPN API + cbbpy ──┐
Barttorvik ────────┤──> SQLite DB ──> Feature Engine ──> Model ──> Predictions
The Odds API ──────┤
SBRO (historical) ─┘
```

All data flows into a single SQLite database (`data/basketball.db`). The feature engine computes tempo-free stats using exponential weighted moving averages with strict anti-lookahead enforcement — every query uses only data available before the prediction date.

## Project Structure

```
src/
├── data/
│   ├── cbbpy_collector.py       # ESPN box scores via cbbpy
│   ├── barttorvik_collector.py   # Barttorvik team ratings scraper
│   ├── odds_api_collector.py     # The Odds API client
│   └── sbro_loader.py           # SBRO historical odds parser
├── features/
│   └── compute.py               # Four Factors, pace, EWMA features
├── utils/
│   ├── db.py                    # SQLite FeatureStore
│   └── config.py                # Paths, constants, helpers
scripts/
├── build_historical.py          # One-time historical dataset assembly
├── daily_collect.py             # Daily data pipeline (cron-ready)
└── paper_trade.py               # Naive model paper trading
tests/
├── test_db.py                   # Database and temporal query tests
├── test_features.py             # Feature computation tests
└── test_collectors.py           # Collector parsing tests
```

## Key Design Decisions

- **SQLite** — zero setup, sufficient for ~55K games, single portable file
- **Anti-lookahead by default** — `as_of_date <= ?` enforced in the data layer, not the caller
- **FTA coefficient 0.40** — standard for college basketball possession estimation
- **EWMA decay** — pace (α=0.90), offense (α=0.93), defense (α=0.95)
- **Walk-forward validation only** — no k-fold for temporal data
