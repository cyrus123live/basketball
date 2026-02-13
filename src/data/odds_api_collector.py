"""Collect live odds from The Odds API.

Free tier: 500 requests/month. We use ~1/day for NCAAB odds snapshots.
Stores timestamped snapshots for building line movement history.
"""

import logging
from datetime import datetime, timezone

import requests

from src.utils.config import ODDS_API_KEY

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4/sports"
SPORT = "basketball_ncaab"
REGIONS = "us"
ODDS_FORMAT = "american"


def collect_current_odds(
    sport: str = SPORT,
    markets: str = "h2h,spreads,totals",
) -> tuple[list[dict], dict]:
    """Fetch current odds for all available NCAAB games.

    Args:
        sport: Sport key for the API.
        markets: Comma-separated market types.

    Returns:
        Tuple of (list of odds dicts for DB, quota info dict).
    """
    if not ODDS_API_KEY or ODDS_API_KEY == "your_key_here":
        logger.warning("ODDS_API_KEY not configured, skipping odds collection")
        return [], {"remaining": "N/A", "used": "N/A"}

    url = f"{BASE_URL}/{sport}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": markets,
        "oddsFormat": ODDS_FORMAT,
    }

    logger.info("Fetching odds for %s", sport)

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            logger.error("Invalid ODDS_API_KEY")
        elif resp.status_code == 422:
            logger.error("Invalid sport or market parameter")
        else:
            logger.error("Odds API HTTP error: %s", e)
        return [], {}
    except Exception as e:
        logger.error("Odds API request failed: %s", e)
        return [], {}

    # Track API quota from response headers
    quota = {
        "remaining": resp.headers.get("x-requests-remaining", "?"),
        "used": resp.headers.get("x-requests-used", "?"),
    }
    logger.info(
        "Odds API quota: %s remaining, %s used", quota["remaining"], quota["used"]
    )

    events = resp.json()
    if not isinstance(events, list):
        logger.error("Unexpected response format: %s", type(events))
        return [], quota

    snapshot_time = datetime.now(timezone.utc).isoformat()
    odds_records = []

    for event in events:
        game_id = event.get("id", "")
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        for bookmaker in event.get("bookmakers", []):
            bk_name = bookmaker.get("key", "")

            for market in bookmaker.get("markets", []):
                market_type = market.get("key", "")

                for outcome in market.get("outcomes", []):
                    odds_records.append({
                        "game_id": game_id,
                        "snapshot_time": snapshot_time,
                        "bookmaker": bk_name,
                        "market_type": market_type,
                        "outcome_name": outcome.get("name", ""),
                        "price": outcome.get("price"),
                        "point": outcome.get("point"),
                    })

    logger.info(
        "Collected %d odds records across %d events",
        len(odds_records), len(events),
    )

    return odds_records, quota


def get_available_sports() -> list[dict]:
    """List sports available on The Odds API (useful for debugging)."""
    if not ODDS_API_KEY or ODDS_API_KEY == "your_key_here":
        logger.warning("ODDS_API_KEY not configured")
        return []

    resp = requests.get(
        f"{BASE_URL}/",
        params={"apiKey": ODDS_API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def american_to_implied_prob(american_odds: float) -> float:
    """Convert American odds to implied probability.

    -110 → 0.524 (52.4%)
    +150 → 0.400 (40.0%)
    """
    if american_odds < 0:
        return (-american_odds) / (-american_odds + 100)
    else:
        return 100 / (american_odds + 100)


def extract_game_odds_summary(odds_records: list[dict]) -> list[dict]:
    """Summarize odds by game for easier consumption.

    Returns list of dicts with consensus spread, total, and ML for each game.
    """
    from collections import defaultdict

    by_game = defaultdict(list)
    for r in odds_records:
        by_game[r["game_id"]].append(r)

    summaries = []
    for game_id, records in by_game.items():
        summary = {"game_id": game_id}

        # Find spreads (use first bookmaker's spread as reference)
        spreads = [r for r in records if r["market_type"] == "spreads"]
        if spreads:
            summary["spread_home"] = spreads[0].get("point")
            summary["spread_price"] = spreads[0].get("price")

        # Find totals
        totals = [
            r for r in records
            if r["market_type"] == "totals" and r["outcome_name"] == "Over"
        ]
        if totals:
            summary["total"] = totals[0].get("point")

        # Find ML
        h2h = [r for r in records if r["market_type"] == "h2h"]
        if h2h:
            for r in h2h:
                if r.get("price"):
                    summary.setdefault("ml_prices", []).append(r["price"])

        summaries.append(summary)

    return summaries
