#!/usr/bin/env python3
"""Daily data collection pipeline.

Collects yesterday's game results, current Barttorvik ratings,
and today's odds. Recomputes features for teams that played.

Run manually or via cron:
    0 6 * * * /path/to/.venv/bin/python /path/to/scripts/daily_collect.py
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.barttorvik_collector import collect_current_ratings
from src.data.cbbpy_collector import collect_date
from src.data.odds_api_collector import collect_current_odds
from src.features.compute import compute_all_features_for_date
from src.utils.config import CURRENT_SEASON
from src.utils.db import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/daily_collect.log"),
    ],
)
logger = logging.getLogger(__name__)


def collect_yesterdays_games(db: FeatureStore) -> int:
    """Collect yesterday's game results and insert into DB."""
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime("%m/%d/%Y")
    date_iso = yesterday.strftime("%Y-%m-%d")

    logger.info("Collecting games for %s", date_iso)
    games = collect_date(date_str)

    inserted = 0
    for game in games:
        home_id = db.upsert_team(game["home_team_name"], CURRENT_SEASON)
        away_id = db.upsert_team(game["away_team_name"], CURRENT_SEASON)
        game["home_team_id"] = home_id
        game["away_team_id"] = away_id
        db.upsert_game(game)
        inserted += 1

    logger.info("Inserted %d games for %s", inserted, date_iso)
    return inserted


def collect_ratings(db: FeatureStore) -> int:
    """Collect current Barttorvik ratings."""
    logger.info("Collecting Barttorvik ratings")
    ratings = collect_current_ratings(CURRENT_SEASON)
    if ratings:
        db.upsert_barttorvik_ratings(ratings)
    logger.info("Stored %d Barttorvik ratings", len(ratings))
    return len(ratings)


def collect_odds(db: FeatureStore) -> int:
    """Collect today's odds from The Odds API."""
    logger.info("Collecting live odds")
    odds_records, quota = collect_current_odds()
    if odds_records:
        db.upsert_live_odds(odds_records)
    logger.info(
        "Stored %d odds records (API quota remaining: %s)",
        len(odds_records), quota.get("remaining", "?"),
    )
    return len(odds_records)


def recompute_features(db: FeatureStore):
    """Recompute features for teams that played yesterday."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    # Compute features as of today (using yesterday's results)
    logger.info("Recomputing features as of %s", today)
    compute_all_features_for_date(db, yesterday, CURRENT_SEASON)

    # Also compute features for today's games (for predictions)
    compute_all_features_for_date(db, today, CURRENT_SEASON)


def main():
    db = FeatureStore()
    start = datetime.now()
    logger.info("=== Daily collection starting at %s ===", start.isoformat())

    try:
        n_games = collect_yesterdays_games(db)
        n_ratings = collect_ratings(db)
        n_odds = collect_odds(db)
        recompute_features(db)

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(
            "=== Daily collection complete in %.1fs: %d games, %d ratings, %d odds ===",
            elapsed, n_games, n_ratings, n_odds,
        )
    except Exception as e:
        logger.exception("Daily collection failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
