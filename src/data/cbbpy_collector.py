"""Collect NCAA men's basketball game data via ESPN JSON APIs.

Uses ESPN's undocumented but reliable JSON endpoints:
- Scoreboard API: all games + scores for a date (1 request per day)
- Summary API: full boxscore for a game (1 request per game)

This is ~8x faster than the original cbbpy HTML scraping approach:
~45 min/season vs ~4 hrs/season (0.5s delay vs 3s + HTML parsing).
"""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from src.utils.config import RAW_DIR

logger = logging.getLogger(__name__)

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/summary"
)

# Faster than cbbpy's 3s — ESPN JSON API handles this fine
FAST_DELAY = 0.5


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


def _fetch_boxscore_espn(game_id: str) -> dict | None:
    """Fetch full boxscore from ESPN summary API (JSON, no HTML parsing).

    Returns dict with home/away box score stats, or None on failure.
    """
    try:
        resp = requests.get(
            ESPN_SUMMARY_URL,
            params={"event": game_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to get ESPN summary for %s: %s", game_id, e)
        return None

    boxscore = data.get("boxscore", {})
    teams = boxscore.get("teams", [])
    if len(teams) != 2:
        logger.warning("Expected 2 teams in boxscore for %s, got %d", game_id, len(teams))
        return None

    result = {}
    for team_data in teams:
        side = team_data.get("homeAway", "")  # "home" or "away"
        if side not in ("home", "away"):
            continue

        # Parse team-level statistics (name: displayValue pairs)
        stats = {}
        for stat in team_data.get("statistics", []):
            stats[stat["name"]] = stat.get("displayValue", "")

        result[f"{side}_fgm"], result[f"{side}_fga"] = _parse_made_attempted(
            stats.get("fieldGoalsMade-fieldGoalsAttempted", "")
        )
        result[f"{side}_3pm"], result[f"{side}_3pa"] = _parse_made_attempted(
            stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted", "")
        )
        result[f"{side}_ftm"], result[f"{side}_fta"] = _parse_made_attempted(
            stats.get("freeThrowsMade-freeThrowsAttempted", "")
        )
        result[f"{side}_oreb"] = _safe_int(stats.get("offensiveRebounds"))
        result[f"{side}_dreb"] = _safe_int(stats.get("defensiveRebounds"))
        result[f"{side}_reb"] = _safe_int(stats.get("totalRebounds"))
        result[f"{side}_ast"] = _safe_int(stats.get("assists"))
        result[f"{side}_tov"] = _safe_int(stats.get("totalTurnovers"))
        result[f"{side}_stl"] = _safe_int(stats.get("steals"))
        result[f"{side}_blk"] = _safe_int(stats.get("blocks"))
        result[f"{side}_pf"] = _safe_int(stats.get("fouls"))

    # Detect overtime from header
    header = data.get("header", {})
    comps = header.get("competitions", [{}])
    if comps:
        period = comps[0].get("status", {}).get("period", 2)
        result["num_ot"] = max(0, (period or 2) - 2)

    return result


def _parse_made_attempted(val: str) -> tuple[int | None, int | None]:
    """Parse '26-60' format into (made, attempted)."""
    if not val or "-" not in val:
        return None, None
    parts = val.split("-")
    if len(parts) != 2:
        return None, None
    return _safe_int(parts[0]), _safe_int(parts[1])


def _safe_int(val) -> int | None:
    """Convert value to int, returning None for NaN/None/empty."""
    if val is None or val == "":
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _date_to_season(date_str: str) -> int:
    """Convert a game date to season label (end year).

    Games Nov-Dec belong to the season ending the following year.
    Games Jan-Apr belong to the season ending that year.
    """
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    if dt.month >= 9:  # Sep-Dec
        return dt.year + 1
    return dt.year


def _names_match(name1: str, name2: str) -> bool:
    """Check if two team names likely refer to the same team."""
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True
    w1 = n1.split()[0] if n1 else ""
    w2 = n2.split()[0] if n2 else ""
    if len(w1) > 3 and w1 == w2:
        return True
    return False


def collect_date(date: str, season: int = None) -> list[dict]:
    """Collect all games for a single date.

    Args:
        date: Date in MM/DD/YYYY or YYYY-MM-DD format.
        season: Season label. If None, inferred from date.

    Returns:
        List of game dicts ready for FeatureStore.upsert_game().
    """
    if "/" in date:
        dt = datetime.strptime(date, "%m/%d/%Y")
    else:
        dt = datetime.strptime(date, "%Y-%m-%d")

    date_espn = dt.strftime("%Y%m%d")
    date_iso = dt.strftime("%Y-%m-%d")

    if season is None:
        season = _date_to_season(date_iso)

    logger.info("Collecting games for %s (season %d)", date_iso, season)

    # Step 1: Get game list from ESPN scoreboard (1 request)
    try:
        espn_games = _fetch_espn_scoreboard(date_espn)
    except Exception as e:
        logger.error("ESPN scoreboard failed for %s: %s", date_iso, e)
        return []

    if not espn_games:
        logger.info("No final games found for %s", date_iso)
        return []

    logger.info("Found %d final games on %s", len(espn_games), date_iso)

    # Step 2: Get boxscores from ESPN summary API (1 request per game)
    results = []
    for espn_info in espn_games:
        game_id = espn_info["game_id"]

        box_data = _fetch_boxscore_espn(game_id)
        time.sleep(FAST_DELAY)

        if box_data is None:
            logger.warning("Skipped game %s (no boxscore)", game_id)
            continue

        game = {
            "game_id": game_id,
            "season": season,
            "game_date": espn_info["game_date"],
            "home_team_name": espn_info["home_team_name"],
            "away_team_name": espn_info["away_team_name"],
            "home_score": espn_info["home_score"],
            "away_score": espn_info["away_score"],
            "num_ot": box_data.get("num_ot", 0),
            "home_fgm": box_data.get("home_fgm"),
            "home_fga": box_data.get("home_fga"),
            "home_3pm": box_data.get("home_3pm"),
            "home_3pa": box_data.get("home_3pa"),
            "home_ftm": box_data.get("home_ftm"),
            "home_fta": box_data.get("home_fta"),
            "home_oreb": box_data.get("home_oreb"),
            "home_dreb": box_data.get("home_dreb"),
            "home_reb": box_data.get("home_reb"),
            "home_ast": box_data.get("home_ast"),
            "home_tov": box_data.get("home_tov"),
            "home_stl": box_data.get("home_stl"),
            "home_blk": box_data.get("home_blk"),
            "home_pf": box_data.get("home_pf"),
            "away_fgm": box_data.get("away_fgm"),
            "away_fga": box_data.get("away_fga"),
            "away_3pm": box_data.get("away_3pm"),
            "away_3pa": box_data.get("away_3pa"),
            "away_ftm": box_data.get("away_ftm"),
            "away_fta": box_data.get("away_fta"),
            "away_oreb": box_data.get("away_oreb"),
            "away_dreb": box_data.get("away_dreb"),
            "away_reb": box_data.get("away_reb"),
            "away_ast": box_data.get("away_ast"),
            "away_tov": box_data.get("away_tov"),
            "away_stl": box_data.get("away_stl"),
            "away_blk": box_data.get("away_blk"),
            "away_pf": box_data.get("away_pf"),
        }
        results.append(game)

    logger.info("Collected %d/%d games for %s", len(results), len(espn_games), date_iso)
    return results


def collect_date_range(
    start: str, end: str, season: int = None, save_raw: bool = True
) -> list[dict]:
    """Collect games for a date range.

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

    ~45 minutes per season with 0.5s delay.
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
    chunk_days = 7
    chunk_start = start_dt

    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end_dt)

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
