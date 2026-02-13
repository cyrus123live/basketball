#!/usr/bin/env python3
"""One-time historical dataset assembly.

Loops through seasons, collects all games via cbbpy + ESPN API,
computes features, and loads SBRO odds. Saves CSV checkpoints per season.

WARNING: This is slow (~3-4 hours per season due to rate limiting).
Run with: python scripts/build_historical.py [start_season] [end_season]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cbbpy_collector import collect_season
from src.data.sbro_loader import load_all_sbro, match_odds_to_games
from src.features.compute import compute_all_features_for_date
from src.utils.config import CURRENT_SEASON, FIRST_SEASON
from src.utils.db import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/build_historical.log"),
    ],
)
logger = logging.getLogger(__name__)


def build_games(db: FeatureStore, start_season: int, end_season: int):
    """Collect all games for a range of seasons."""
    for season in range(start_season, end_season + 1):
        existing = db.count_games(season)
        if existing > 4000:
            logger.info(
                "Season %d already has %d games, skipping collection",
                season, existing,
            )
            continue

        logger.info("=== Collecting season %d ===", season)
        games = collect_season(season)

        # Insert into DB
        for game in games:
            # Ensure team records exist
            home_id = db.upsert_team(game["home_team_name"], season)
            away_id = db.upsert_team(game["away_team_name"], season)
            game["home_team_id"] = home_id
            game["away_team_id"] = away_id
            db.upsert_game(game)

        logger.info(
            "Season %d: inserted %d games (total in DB: %d)",
            season, len(games), db.count_games(season),
        )


def build_features(db: FeatureStore, start_season: int, end_season: int):
    """Compute features for all games in the database."""
    logger.info("=== Computing features ===")

    for season in range(start_season, end_season + 1):
        games_df = db.get_games_for_season(season)
        if games_df.empty:
            continue

        dates = sorted(games_df["game_date"].unique())
        logger.info("Season %d: computing features for %d game dates", season, len(dates))

        for i, game_date in enumerate(dates):
            compute_all_features_for_date(db, game_date, season)
            if (i + 1) % 20 == 0:
                logger.info("  %d/%d dates processed", i + 1, len(dates))


def build_odds(db: FeatureStore):
    """Load SBRO historical odds and match to games."""
    logger.info("=== Loading SBRO odds ===")
    odds_records = load_all_sbro()
    if not odds_records:
        logger.info("No SBRO files found, skipping odds loading")
        return

    odds_records = match_odds_to_games(odds_records, db)
    db.upsert_historical_odds(odds_records)
    logger.info("Inserted %d historical odds records", len(odds_records))


def run_quality_checks(db: FeatureStore):
    """Run data quality checks on the assembled dataset."""
    logger.info("=== Running quality checks ===")

    # Games per season
    counts = db.game_counts_by_season()
    logger.info("Games per season:\n%s", counts.to_string(index=False))

    # Check for reasonable game counts
    for _, row in counts.iterrows():
        season = row["season"]
        count = row["game_count"]
        if count < 3000:
            logger.warning("Season %d has only %d games (expected 4500+)", season, count)
        elif count > 6000:
            logger.warning("Season %d has %d games (unusually high)", season, count)

    # Box score consistency check (sample)
    with db._connect() as conn:
        # Check: (FGM - 3PM)*2 + 3PM*3 + FTM == score
        mismatches = conn.execute("""
            SELECT game_id, home_team_name, home_score,
                   (home_fgm - home_3pm) * 2 + home_3pm * 3 + home_ftm as computed
            FROM games
            WHERE home_fgm IS NOT NULL
              AND home_3pm IS NOT NULL
              AND home_ftm IS NOT NULL
              AND (home_fgm - home_3pm) * 2 + home_3pm * 3 + home_ftm != home_score
            LIMIT 10
        """).fetchall()

        if mismatches:
            logger.warning(
                "Found %d box score mismatches (showing first 10):", len(mismatches)
            )
            for m in mismatches:
                logger.warning(
                    "  %s %s: ESPN=%d, computed=%d",
                    m["game_id"], m["home_team_name"], m["home_score"], m["computed"],
                )
        else:
            logger.info("Box score consistency: PASS")

    # Duplicate check
    with db._connect() as conn:
        dupes = conn.execute(
            "SELECT game_id, COUNT(*) as cnt FROM games GROUP BY game_id HAVING cnt > 1"
        ).fetchall()
        if dupes:
            logger.warning("Found %d duplicate game_ids", len(dupes))
        else:
            logger.info("Duplicate check: PASS")

    total = db.count_games()
    logger.info("Total games in database: %d", total)


def main():
    parser = argparse.ArgumentParser(description="Build historical basketball dataset")
    parser.add_argument(
        "start_season", type=int, nargs="?", default=FIRST_SEASON,
        help=f"First season to collect (default: {FIRST_SEASON})",
    )
    parser.add_argument(
        "end_season", type=int, nargs="?", default=CURRENT_SEASON,
        help=f"Last season to collect (default: {CURRENT_SEASON})",
    )
    parser.add_argument(
        "--skip-games", action="store_true",
        help="Skip game collection (use existing data)",
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip feature computation",
    )
    parser.add_argument(
        "--skip-odds", action="store_true",
        help="Skip SBRO odds loading",
    )
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Skip quality checks",
    )
    args = parser.parse_args()

    db = FeatureStore()
    logger.info(
        "Building historical dataset for seasons %d-%d",
        args.start_season, args.end_season,
    )

    if not args.skip_games:
        build_games(db, args.start_season, args.end_season)

    if not args.skip_features:
        build_features(db, args.start_season, args.end_season)

    if not args.skip_odds:
        build_odds(db)

    if not args.skip_checks:
        run_quality_checks(db)

    logger.info("=== Build complete ===")


if __name__ == "__main__":
    main()
