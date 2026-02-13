"""Feature computation: tempo-free stats and EWMA team features.

Computes possessions, pace, offensive/defensive ratings, and Dean Oliver's
Four Factors from raw box score data. Team features use exponential weighted
moving averages (EWMA) with strict anti-lookahead enforcement.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.utils.config import (
    EWMA_ALPHA_DEFENSE,
    EWMA_ALPHA_OFFENSE,
    EWMA_ALPHA_PACE,
    FTA_COEFF,
    OT_MINUTES,
    REGULATION_MINUTES,
)

logger = logging.getLogger(__name__)


def possessions(fga: float, oreb: float, tov: float, fta: float) -> float:
    """Estimate possessions from box score stats.

    Uses the standard formula with college FTA coefficient (0.40).
    """
    return fga - oreb + tov + FTA_COEFF * fta


def game_minutes(num_ot: int = 0) -> float:
    """Total game minutes (team-level, not player-minutes)."""
    return REGULATION_MINUTES + num_ot * OT_MINUTES


def compute_game_stats(game: dict) -> dict | None:
    """Compute tempo-free stats for a single game from raw box score.

    Args:
        game: Dict with box score fields (home_fga, away_fga, etc.)

    Returns:
        Dict with computed stats for both teams, or None if data is insufficient.
    """
    required = ["home_fga", "away_fga", "home_tov", "away_tov",
                 "home_fta", "away_fta", "home_score", "away_score"]
    if any(game.get(k) is None for k in required):
        return None

    num_ot = game.get("num_ot", 0) or 0
    minutes = game_minutes(num_ot)

    # Possessions for each team
    home_poss = possessions(
        game["home_fga"], game.get("home_oreb", 0) or 0,
        game["home_tov"], game["home_fta"]
    )
    away_poss = possessions(
        game["away_fga"], game.get("away_oreb", 0) or 0,
        game["away_tov"], game["away_fta"]
    )

    # Average possessions (both teams should have roughly equal)
    avg_poss = (home_poss + away_poss) / 2
    if avg_poss <= 0:
        return None

    pace = avg_poss / (minutes / REGULATION_MINUTES)

    # Offensive/Defensive ratings (points per 100 possessions)
    home_ortg = (game["home_score"] / home_poss) * 100 if home_poss > 0 else None
    away_ortg = (game["away_score"] / away_poss) * 100 if away_poss > 0 else None

    home_drtg = away_ortg  # Home defense = away offense
    away_drtg = home_ortg  # Away defense = home offense

    result = {
        "home_poss": home_poss,
        "away_poss": away_poss,
        "pace": pace,
        "home_ortg": home_ortg,
        "home_drtg": home_drtg,
        "away_ortg": away_ortg,
        "away_drtg": away_drtg,
    }

    # Four Factors — Offense
    for prefix in ("home", "away"):
        fgm = game.get(f"{prefix}_fgm")
        fga = game.get(f"{prefix}_fga")
        tpm = game.get(f"{prefix}_3pm")
        fta = game.get(f"{prefix}_fta")
        tov = game.get(f"{prefix}_tov")
        oreb = game.get(f"{prefix}_oreb")

        # Opponent's DREB for ORB% calculation
        opp = "away" if prefix == "home" else "home"
        opp_dreb = game.get(f"{opp}_dreb")

        if fga and fga > 0 and fgm is not None and tpm is not None:
            result[f"{prefix}_efg_pct"] = (fgm + 0.5 * tpm) / fga
        else:
            result[f"{prefix}_efg_pct"] = None

        if fga is not None and fta is not None and tov is not None:
            denom = fga + FTA_COEFF * fta + tov
            result[f"{prefix}_tov_pct"] = tov / denom if denom > 0 else None
        else:
            result[f"{prefix}_tov_pct"] = None

        if oreb is not None and opp_dreb is not None:
            total = oreb + opp_dreb
            result[f"{prefix}_orb_pct"] = oreb / total if total > 0 else None
        else:
            result[f"{prefix}_orb_pct"] = None

        if fga and fga > 0 and fta is not None:
            result[f"{prefix}_ftr"] = fta / fga
        else:
            result[f"{prefix}_ftr"] = None

    return result


def compute_team_features_as_of(
    db, team_id: int, as_of_date: str, season: int = None
) -> dict | None:
    """Compute EWMA features for a team using only pre-game data.

    This is the core anti-lookahead function. It only uses games that
    occurred strictly before as_of_date.

    Args:
        db: FeatureStore instance.
        team_id: Team ID in the database.
        as_of_date: Compute features as of this date (exclusive).
        season: Optional season filter.

    Returns:
        Feature dict ready for FeatureStore.upsert_team_features(), or None.
    """
    games_df = db.get_games_before(team_id, as_of_date)
    if games_df.empty:
        return None

    if season is not None:
        games_df = games_df[games_df["season"] == season]
        if games_df.empty:
            return None

    # Compute per-game stats for each game
    game_stats = []
    for _, game in games_df.iterrows():
        game_dict = game.to_dict()
        stats = compute_game_stats(game_dict)
        if stats is None:
            continue

        # Determine if this team was home or away
        is_home = game["home_team_id"] == team_id
        prefix = "home" if is_home else "away"
        opp_prefix = "away" if is_home else "home"

        game_stats.append({
            "game_date": game["game_date"],
            "pace": stats["pace"],
            "ortg": stats[f"{prefix}_ortg"],
            "drtg": stats[f"{prefix}_drtg"],
            "efg_pct": stats[f"{prefix}_efg_pct"],
            "efg_pct_d": stats[f"{opp_prefix}_efg_pct"],
            "tov_pct": stats[f"{prefix}_tov_pct"],
            "tov_pct_d": stats[f"{opp_prefix}_tov_pct"],
            "orb_pct": stats[f"{prefix}_orb_pct"],
            "orb_pct_d": stats[f"{opp_prefix}_orb_pct"],
            "ftr": stats[f"{prefix}_ftr"],
            "ftr_d": stats[f"{opp_prefix}_ftr"],
        })

    if not game_stats:
        return None

    df = pd.DataFrame(game_stats).sort_values("game_date")

    # EWMA with different alphas for different stat categories
    pace_cols = ["pace"]
    off_cols = ["ortg", "efg_pct", "tov_pct", "orb_pct", "ftr"]
    def_cols = ["drtg", "efg_pct_d", "tov_pct_d", "orb_pct_d", "ftr_d"]

    features = {}
    for col in pace_cols:
        series = df[col].dropna()
        if len(series) > 0:
            features[col] = _ewma_last(series, EWMA_ALPHA_PACE)
    for col in off_cols:
        series = df[col].dropna()
        if len(series) > 0:
            features[col] = _ewma_last(series, EWMA_ALPHA_OFFENSE)
    for col in def_cols:
        series = df[col].dropna()
        if len(series) > 0:
            features[col] = _ewma_last(series, EWMA_ALPHA_DEFENSE)

    # Net rating
    if "ortg" in features and "drtg" in features:
        features["net_rtg"] = features["ortg"] - features["drtg"]
    else:
        features["net_rtg"] = None

    # Rest days: days since last game
    last_game_date = df["game_date"].iloc[-1]
    try:
        last_dt = datetime.strptime(str(last_game_date), "%Y-%m-%d")
        as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
        features["rest_days"] = (as_of_dt - last_dt).days
    except (ValueError, TypeError):
        features["rest_days"] = None

    # Look up team name
    team_name = None
    with db._connect() as conn:
        row = conn.execute(
            "SELECT name, season FROM teams WHERE team_id = ?", (team_id,)
        ).fetchone()
        if row:
            team_name = row["name"]
            if season is None:
                season = row["season"]

    return {
        "team_id": team_id,
        "team_name": team_name,
        "season": season,
        "as_of_date": as_of_date,
        "games_played": len(game_stats),
        "pace": features.get("pace"),
        "ortg": features.get("ortg"),
        "drtg": features.get("drtg"),
        "net_rtg": features.get("net_rtg"),
        "efg_pct": features.get("efg_pct"),
        "efg_pct_d": features.get("efg_pct_d"),
        "tov_pct": features.get("tov_pct"),
        "tov_pct_d": features.get("tov_pct_d"),
        "orb_pct": features.get("orb_pct"),
        "orb_pct_d": features.get("orb_pct_d"),
        "ftr": features.get("ftr"),
        "ftr_d": features.get("ftr_d"),
        "rest_days": features.get("rest_days"),
    }


def compute_all_features_for_date(db, game_date: str, season: int = None):
    """Compute and store features for all teams playing on a given date.

    For each game on game_date, computes features for both teams using
    only data from before that date.
    """
    games_df = db.get_games_for_date(game_date)
    if games_df.empty:
        logger.info("No games on %s", game_date)
        return

    team_ids = set()
    for _, game in games_df.iterrows():
        if game["home_team_id"]:
            team_ids.add(int(game["home_team_id"]))
        if game["away_team_id"]:
            team_ids.add(int(game["away_team_id"]))

    computed = 0
    for team_id in team_ids:
        features = compute_team_features_as_of(db, team_id, game_date, season)
        if features:
            db.upsert_team_features(features)
            computed += 1

    logger.info(
        "Computed features for %d/%d teams on %s",
        computed, len(team_ids), game_date,
    )


def _ewma_last(series: pd.Series, alpha: float) -> float:
    """Compute EWMA and return the last value.

    Uses pandas ewm with adjust=False for a true recursive EWMA.
    """
    if len(series) == 0:
        return np.nan
    return series.ewm(alpha=1 - alpha, adjust=False).mean().iloc[-1]
