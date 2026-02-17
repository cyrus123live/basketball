#!/usr/bin/env python3
"""One-time scrape of historical Barttorvik end-of-season ratings.

Scrapes seasons 2015-2020 to provide prior-season quality signals
for training seasons 2016-2021. ~45 seconds runtime.

Usage:
    python scripts/scrape_barttorvik_historical.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.barttorvik_collector import collect_historical_ratings
from src.utils.db import FeatureStore
from src.utils.name_match import resolve_barttorvik_team_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEASONS = [2015, 2016, 2017, 2018, 2019, 2020]


def main():
    db = FeatureStore()

    print(f"Scraping Barttorvik historical ratings for seasons {SEASONS}")
    print("=" * 60)

    total_matched = 0
    total_ratings = 0

    for season in SEASONS:
        print(f"\nSeason {season}...")
        ratings = collect_historical_ratings(season)

        if not ratings:
            print(f"  WARNING: No ratings returned for season {season}")
            continue

        # Resolve team_ids via fuzzy matching
        summary = resolve_barttorvik_team_ids(db, ratings, season)
        print(
            f"  Scraped {len(ratings)} teams, "
            f"matched {summary['matched']}/{summary['total']} "
            f"({summary['matched'] / summary['total'] * 100:.1f}%)"
        )

        if summary["unmatched_names"]:
            print(f"  Unmatched: {summary['unmatched_names'][:10]}")

        # Upsert to database
        db.upsert_barttorvik_ratings(ratings)
        total_matched += summary["matched"]
        total_ratings += len(ratings)

    print("\n" + "=" * 60)
    print(f"Done. Total: {total_ratings} ratings, {total_matched} matched.")

    # Verify data in DB
    with db._connect() as conn:
        counts = conn.execute(
            "SELECT season, COUNT(*) as n, "
            "SUM(CASE WHEN team_id IS NOT NULL THEN 1 ELSE 0 END) as matched "
            "FROM barttorvik_ratings "
            "WHERE season BETWEEN 2015 AND 2020 "
            "GROUP BY season ORDER BY season"
        ).fetchall()

    print("\nDatabase verification:")
    print(f"{'Season':<8} {'Total':<8} {'Matched':<10}")
    print("-" * 26)
    for row in counts:
        print(f"{row['season']:<8} {row['n']:<8} {row['matched']:<10}")


if __name__ == "__main__":
    main()
