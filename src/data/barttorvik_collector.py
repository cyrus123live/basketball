"""Scrape team ratings from Barttorvik (barttorvik.com).

Barttorvik provides KenPom-style adjusted efficiency ratings for free.
Single request returns all ~363 D1 teams with AdjO, AdjD, AdjT, Four Factors.

The HTML table has multi-level headers and values like "127.2 5" where the
number after the space is the national rank. We extract just the stat value.
"""

import logging
import time
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from src.utils.config import REQUEST_DELAY

logger = logging.getLogger(__name__)

RANKINGS_URL = "https://barttorvik.com/trank.php"

# Column names we assign by position to the parsed table.
# Barttorvik's 24-column layout (as of 2025-26):
# Rk, Team, Conf, G, Rec, AdjOE, AdjDE, Barthag,
# EFG%, EFGD%, TOR, TORD, ORB, DRB, FTR, FTRD,
# 2P%, 2PD%, 3P%, 3PD%, 3PR, 3PRD, AdjT, WAB
COLUMN_NAMES = [
    "rank", "team", "conf", "games", "record",
    "adj_o", "adj_d", "barthag",
    "efg_pct", "efg_pct_d", "tov_pct", "tov_pct_d",
    "orb_pct", "drb_pct", "ftr", "ftr_d",
    "twop_pct", "twop_pct_d", "threep_pct", "threep_pct_d",
    "threep_rate", "threep_rate_d", "adj_t", "wab",
]


def _extract_value(cell) -> float | None:
    """Extract the numeric value from a Barttorvik cell.

    Cells contain values like '127.2 5' where '127.2' is the stat and '5'
    is the rank. We want just the stat value.
    """
    if pd.isna(cell):
        return None
    s = str(cell).strip()
    if not s:
        return None
    # Take first token (the value, not the rank)
    parts = s.split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except (ValueError, TypeError):
        return None


def _get_session() -> requests.Session:
    """Create a session that handles Barttorvik's JS verification."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    })
    # Pass the JS verification check
    s.post("https://barttorvik.com/", data={"js_test_submitted": "1"}, timeout=30)
    return s


def collect_current_ratings(season: int = None) -> list[dict]:
    """Scrape current Barttorvik ratings for a season.

    Args:
        season: Season end year (e.g. 2026). Defaults to current.

    Returns:
        List of rating dicts ready for FeatureStore.upsert_barttorvik_ratings().
    """
    if season is None:
        now = datetime.now()
        season = now.year if now.month <= 6 else now.year + 1

    as_of_date = datetime.now().strftime("%Y-%m-%d")

    logger.info("Scraping Barttorvik ratings for season %d", season)

    try:
        session = _get_session()
        resp = session.get(
            RANKINGS_URL,
            params={"year": season, "sort": "", "lastx": 0, "hession": "All"},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error("Failed to fetch Barttorvik: %s", e)
        return []

    time.sleep(REQUEST_DELAY)

    try:
        tables = pd.read_html(StringIO(resp.text))
    except Exception as e:
        logger.error("Failed to parse Barttorvik HTML: %s", e)
        return []

    if not tables:
        logger.error("No tables found on Barttorvik page")
        return []

    # The main rankings table is the largest one
    df = max(tables, key=len)

    # Flatten multi-level column headers
    df.columns = range(len(df.columns))

    logger.info("Parsed table with %d rows, %d columns", len(df), len(df.columns))

    # Assign our column names
    if len(df.columns) != len(COLUMN_NAMES):
        logger.warning(
            "Expected %d columns, got %d. Attempting best-effort parse.",
            len(COLUMN_NAMES), len(df.columns),
        )
        # If we have more or fewer columns, only map what we can
        names = COLUMN_NAMES[:len(df.columns)] if len(df.columns) < len(COLUMN_NAMES) else COLUMN_NAMES
        df.columns = list(range(len(df.columns)))
        col_map = {i: names[i] for i in range(len(names))}
        df = df.rename(columns=col_map)
    else:
        df.columns = COLUMN_NAMES

    # Filter out non-data rows
    df = df[df["team"].notna()].copy()
    df = df[~df["team"].astype(str).str.lower().str.contains("^team$|^nan$")]
    # Remove rows where rank isn't numeric (header duplicates)
    df = df[df["rank"].apply(lambda x: str(x).strip().isdigit())]

    ratings = []
    for _, row in df.iterrows():
        team_name = str(row["team"]).strip()
        if not team_name:
            continue

        ratings.append({
            "team_name": team_name,
            "team_id": None,  # Will be matched later
            "season": season,
            "as_of_date": as_of_date,
            "adj_o": _extract_value(row.get("adj_o")),
            "adj_d": _extract_value(row.get("adj_d")),
            "adj_t": _extract_value(row.get("adj_t")),
            "barthag": _extract_value(row.get("barthag")),
            "efg_pct": _extract_value(row.get("efg_pct")),
            "efg_pct_d": _extract_value(row.get("efg_pct_d")),
            "tov_pct": _extract_value(row.get("tov_pct")),
            "tov_pct_d": _extract_value(row.get("tov_pct_d")),
            "orb_pct": _extract_value(row.get("orb_pct")),
            "orb_pct_d": _extract_value(row.get("drb_pct")),  # opponent ORB% = our DRB%
            "ftr": _extract_value(row.get("ftr")),
            "ftr_d": _extract_value(row.get("ftr_d")),
        })

    logger.info(
        "Collected %d Barttorvik ratings for season %d", len(ratings), season
    )
    return ratings


def collect_historical_ratings(season: int) -> list[dict]:
    """Collect end-of-season ratings for a historical season.

    Same as current ratings but for a past season. Barttorvik preserves
    final ratings for past seasons at the same URL.
    """
    return collect_current_ratings(season=season)
