"""Collect NCAA men's basketball game data via cbbpy and ESPN API.

Uses cbbpy for boxscores and ESPN's scoreboard API for game metadata.
cbbpy's get_game_info has a pandas 3.0 compatibility bug, so we work
around it by combining ESPN API metadata with cbbpy boxscores.
"""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

import cbbpy.mens_scraper as cbbpy

from src.utils.config import (
    CBBPY_CHUNK_DAYS,
    RAW_DIR,
    REQUEST_DELAY,
)

logger = logging.getLogger(__name__)

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)


def _fetch_espn_scoreboard(date_str: str) -> list[dict]:
    """Fetch game metadata from ESPN scoreboard API for a date.

    Args:
        date_str: Date in YYYYMMDD format.

    Returns:
        List of game info dicts with game_id, date, teams, scores, etc.
    """
    resp = requests.get(
        ESPN_SCOREBOARD_URL,
        params={"dates": date_str, "limit": 200, "groups": 50},
        timeout=30,
    )
    resp.raise_for_status()
    events = resp.json().get("events", [])

    games = []
    for event in events:
        game_id = event["id"]
        comp = event["competitions"][0]
        competitors = {
            c["homeAway"]: c for c in comp["competitors"]
        }
        home = competitors.get("home", {})
        away = competitors.get("away", {})

        status = comp.get("status", {}).get("type", {})
        if status.get("name") != "STATUS_FINAL":
            continue  # Skip non-final games

        games.append({
            "game_id": game_id,
            "game_date": comp["date"][:10],  # YYYY-MM-DD
            "home_team_name": home.get("team", {}).get("displayName", ""),
            "away_team_name": away.get("team", {}).get("displayName", ""),
            "home_score": int(home.get("score", 0)),
            "away_score": int(away.get("score", 0)),
            "neutral_site": comp.get("neutralSite", False),
            "conference_game": comp.get("conferenceCompetition", False),
        })

    return games


def _fetch_boxscore(game_id: str) -> dict | None:
    """Fetch and aggregate player boxscores into team totals via cbbpy.

    Returns dict with home/away box score stats, or None on failure.
    """
    try:
        box_df = cbbpy.get_game_boxscore(game_id)
    except Exception as e:
        logger.warning("Failed to get boxscore for %s: %s", game_id, e)
        return None

    if box_df is None or len(box_df) == 0:
        logger.warning("Empty boxscore for %s", game_id)
        return None

    # Get team totals (TEAM rows)
    totals = box_df[box_df["player"] == "TEAM"].copy()
    if len(totals) != 2:
        logger.warning(
            "Expected 2 team total rows for %s, got %d", game_id, len(totals)
        )
        return None

    teams = totals["team"].tolist()
    result = {"teams": teams}

    stat_cols = [
        "fgm", "fga", "3pm", "3pa", "ftm", "fta",
        "oreb", "dreb", "reb", "ast", "stl", "blk", "pf",
    ]
    # cbbpy uses 'to' for turnovers
    for i, prefix in enumerate(["team1_", "team2_"]):
        row = totals.iloc[i]
        for col in stat_cols:
            result[f"{prefix}{col}"] = _safe_int(row.get(col))
        result[f"{prefix}tov"] = _safe_int(row.get("to"))
        result[f"{prefix}pts"] = _safe_int(row.get("pts"))
        result[f"{prefix}name"] = row["team"]

    return result


def _safe_int(val) -> int | None:
    """Convert value to int, returning None for NaN/None."""
    if pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _merge_game_data(
    espn_info: dict, box_data: dict, season: int
) -> dict | None:
    """Merge ESPN game info with cbbpy boxscore into a single game record.

    Matches teams by comparing ESPN names to cbbpy names and assigns
    home/away box score stats accordingly.
    """
    if box_data is None:
        return None

    espn_home = espn_info["home_team_name"].lower()
    espn_away = espn_info["away_team_name"].lower()
    box_team1 = box_data["team1_name"].lower()

    # Determine which cbbpy team is home vs away
    # cbbpy lists away team first, home team second in the boxscore
    # But verify by name matching
    if _names_match(box_team1, espn_home):
        home_prefix, away_prefix = "team1_", "team2_"
    elif _names_match(box_team1, espn_away):
        home_prefix, away_prefix = "team2_", "team1_"
    else:
        # Try team2
        box_team2 = box_data["team2_name"].lower()
        if _names_match(box_team2, espn_home):
            home_prefix, away_prefix = "team2_", "team1_"
        elif _names_match(box_team2, espn_away):
            home_prefix, away_prefix = "team1_", "team2_"
        else:
            logger.warning(
                "Cannot match teams for %s: ESPN=%s/%s, cbbpy=%s/%s",
                espn_info["game_id"],
                espn_info["home_team_name"],
                espn_info["away_team_name"],
                box_data["team1_name"],
                box_data["team2_name"],
            )
            # Fall back: cbbpy order is typically [away, home]
            home_prefix, away_prefix = "team2_", "team1_"

    stat_cols = [
        "fgm", "fga", "3pm", "3pa", "ftm", "fta",
        "oreb", "dreb", "reb", "ast", "tov", "stl", "blk", "pf",
    ]

    game = {
        "game_id": espn_info["game_id"],
        "season": season,
        "game_date": espn_info["game_date"],
        "home_team_name": espn_info["home_team_name"],
        "away_team_name": espn_info["away_team_name"],
        "home_score": espn_info["home_score"],
        "away_score": espn_info["away_score"],
    }

    for col in stat_cols:
        game[f"home_{col}"] = box_data.get(f"{home_prefix}{col}")
        game[f"away_{col}"] = box_data.get(f"{away_prefix}{col}")

    # Detect OT from score vs box score pts if available
    game["num_ot"] = 0  # Will be updated if we can detect it

    return game


def _names_match(name1: str, name2: str) -> bool:
    """Check if two team names likely refer to the same team.

    Handles common variations like 'Duke Blue Devils' vs 'Duke'.
    """
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if n1 == n2:
        return True
    # Check if one contains the other
    if n1 in n2 or n2 in n1:
        return True
    # Check first word match (university name)
    w1 = n1.split()[0] if n1 else ""
    w2 = n2.split()[0] if n2 else ""
    if len(w1) > 3 and w1 == w2:
        return True
    return False


def _date_to_season(date_str: str) -> int:
    """Convert a game date to season label (end year).

    Games Nov-Dec belong to the season ending the following year.
    Games Jan-Apr belong to the season ending that year.
    """
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    if dt.month >= 9:  # Sep-Dec
        return dt.year + 1
    return dt.year


def collect_date(date: str, season: int = None) -> list[dict]:
    """Collect all games for a single date.

    Args:
        date: Date in MM/DD/YYYY format (for cbbpy) or YYYY-MM-DD.
        season: Season label. If None, inferred from date.

    Returns:
        List of game dicts ready for FeatureStore.upsert_game().
    """
    # Normalize date format
    if "/" in date:
        dt = datetime.strptime(date, "%m/%d/%Y")
    else:
        dt = datetime.strptime(date, "%Y-%m-%d")

    date_espn = dt.strftime("%Y%m%d")
    date_iso = dt.strftime("%Y-%m-%d")

    if season is None:
        season = _date_to_season(date_iso)

    logger.info("Collecting games for %s (season %d)", date_iso, season)

    # Step 1: Get game list from ESPN
    try:
        espn_games = _fetch_espn_scoreboard(date_espn)
    except Exception as e:
        logger.error("ESPN scoreboard failed for %s: %s", date_iso, e)
        return []

    if not espn_games:
        logger.info("No final games found for %s", date_iso)
        return []

    logger.info("Found %d final games on %s", len(espn_games), date_iso)

    # Step 2: Get boxscores via cbbpy
    results = []
    for espn_info in espn_games:
        game_id = espn_info["game_id"]
        box_data = _fetch_boxscore(game_id)
        time.sleep(REQUEST_DELAY)

        game = _merge_game_data(espn_info, box_data, season)
        if game:
            results.append(game)
        else:
            logger.warning("Skipped game %s (no boxscore data)", game_id)

    logger.info("Collected %d/%d games for %s", len(results), len(espn_games), date_iso)
    return results


def collect_date_range(
    start: str, end: str, season: int = None, save_raw: bool = True
) -> list[dict]:
    """Collect games for a date range, chunked with delays.

    Args:
        start: Start date in MM/DD/YYYY format.
        end: End date in MM/DD/YYYY format.
        season: Season label. If None, inferred from each game date.
        save_raw: If True, save raw data to CSV checkpoints.

    Returns:
        List of all game dicts collected.
    """
    start_dt = datetime.strptime(start, "%m/%d/%Y")
    end_dt = datetime.strptime(end, "%m/%d/%Y")

    all_games = []
    current = start_dt

    while current <= end_dt:
        date_str = current.strftime("%m/%d/%Y")
        try:
            day_games = collect_date(date_str, season=season)
            all_games.extend(day_games)
        except Exception as e:
            logger.error("Failed to collect %s: %s", date_str, e)

        current += timedelta(days=1)

    if save_raw and all_games:
        _save_checkpoint(all_games, start_dt, end_dt)

    logger.info(
        "Collected %d total games from %s to %s",
        len(all_games), start, end,
    )
    return all_games


def collect_yesterday() -> list[dict]:
    """Collect yesterday's games. Convenience for daily scheduler."""
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime("%m/%d/%Y")
    return collect_date(date_str)


def collect_season(season: int, save_raw: bool = True) -> list[dict]:
    """Collect all games for a full season.

    This is slow (~3-4 hours per season due to rate limiting).
    Saves CSV checkpoints per week.

    Args:
        season: Season end year (e.g. 2026 for 2025-26 season).
        save_raw: If True, save weekly CSV checkpoints.
    """
    start_year = season - 1
    start_dt = datetime(start_year, 11, 1)
    end_dt = datetime(season, 4, 15)

    # If current season, don't go past today
    today = datetime.now()
    if end_dt > today:
        end_dt = today - timedelta(days=1)

    all_games = []
    chunk_start = start_dt

    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=CBBPY_CHUNK_DAYS - 1), end_dt)

        logger.info(
            "Season %d: collecting %s to %s",
            season,
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )

        chunk_games = collect_date_range(
            chunk_start.strftime("%m/%d/%Y"),
            chunk_end.strftime("%m/%d/%Y"),
            season=season,
            save_raw=False,
        )
        all_games.extend(chunk_games)

        if save_raw and chunk_games:
            _save_checkpoint(chunk_games, chunk_start, chunk_end, season=season)

        chunk_start = chunk_end + timedelta(days=1)

    logger.info("Season %d: collected %d total games", season, len(all_games))
    return all_games


def _save_checkpoint(
    games: list[dict], start_dt: datetime, end_dt: datetime, season: int = None
):
    """Save games to CSV checkpoint file."""
    df = pd.DataFrame(games)
    prefix = f"s{season}_" if season else ""
    filename = (
        f"{prefix}games_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
    )
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved checkpoint: %s (%d games)", path, len(games))
