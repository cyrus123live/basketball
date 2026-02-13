"""Project configuration: paths, constants, and helpers."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ODDS_DIR = DATA_DIR / "odds"
SBRO_DIR = ODDS_DIR / "sbro"
DB_PATH = DATA_DIR / "basketball.db"
LOG_DIR = PROJECT_ROOT / "logs"
CROSSWALK_PATH = DATA_DIR / "team_name_crosswalk.csv"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, SBRO_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ── Basketball Constants ───────────────────────────────────────────────
FTA_COEFF = 0.40          # College basketball FTA coefficient for possessions
REGULATION_MINUTES = 40   # College basketball regulation game length
OT_MINUTES = 5            # Overtime period length

# ── Scraping ───────────────────────────────────────────────────────────
REQUEST_DELAY = 3.0       # Seconds between scraping requests
CBBPY_CHUNK_DAYS = 7      # Days per cbbpy request chunk

# ── Feature Computation ───────────────────────────────────────────────
EWMA_ALPHA_PACE = 0.90
EWMA_ALPHA_OFFENSE = 0.93
EWMA_ALPHA_DEFENSE = 0.95

# ── Seasons ────────────────────────────────────────────────────────────
FIRST_SEASON = 2016       # Earliest season to collect (cbbpy notation = end year)
CURRENT_SEASON = 2026     # Current season


def season_to_years(season: int) -> tuple[int, int]:
    """Convert season label (e.g. 2026) to start/end calendar years.

    cbbpy uses the end year as the season label, so season 2026 means
    the 2025-26 academic year (Nov 2025 – April 2026).
    """
    return season - 1, season


def season_date_range(season: int) -> tuple[str, str]:
    """Return (start_date, end_date) strings for a season in MM/DD/YYYY format.

    Conservative date range covering early-season tournaments through
    the championship game.
    """
    start_year, end_year = season_to_years(season)
    return f"11/01/{start_year}", f"04/15/{end_year}"
