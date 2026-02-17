"""Training and evaluation data builders for totals prediction.

Builds feature matrices from the SQLite database. Training data uses ALL games
(labels are just scores, no odds needed). Evaluation data adds SBRO closing
lines for CLV analysis.

Features are COMBINED (averages of both teams), not differentials — totals
prediction cares about the sum of scoring, not which team is better.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# SQL for SBRO total_close with swap fix: if |total_close| < |spread_close|,
# they're swapped. NCAA totals > 100, spreads < 50.
_SWAP_FIX_TOTAL = """
    CASE WHEN ABS(ho.total_close) < ABS(ho.spread_close)
         THEN ho.spread_close
         ELSE ho.total_close
    END
"""

# Features we build for totals prediction (all combined/averaged, not diffs)
TOTALS_FEATURES = [
    "avg_pace",
    "avg_ortg",
    "avg_drtg",
    "combined_efg",
    "combined_efg_d",
    "combined_tov",
    "combined_tov_d",
    "combined_orb",
    "combined_orb_d",
    "combined_ftr",
    "combined_ftr_d",
    "avg_rest",
    "min_games",
]


def build_totals_training_data(db, seasons: list[int]) -> pd.DataFrame:
    """Build training data for totals model — ALL games with features.

    No odds required. Label is actual_total (home_score + away_score).

    Args:
        db: FeatureStore instance.
        seasons: List of season end years to include.

    Returns:
        DataFrame with combined features and actual_total label.
    """
    placeholders = ",".join("?" * len(seasons))
    sql = f"""
        SELECT
            g.game_id,
            g.game_date,
            g.season,
            g.home_team_name,
            g.away_team_name,
            g.home_score,
            g.away_score,
            g.home_score + g.away_score AS actual_total,
            hf.pace       AS home_pace,
            af.pace       AS away_pace,
            hf.ortg       AS home_ortg,
            hf.drtg       AS home_drtg,
            af.ortg       AS away_ortg,
            af.drtg       AS away_drtg,
            hf.efg_pct    AS home_efg,
            hf.efg_pct_d  AS home_efg_d,
            af.efg_pct    AS away_efg,
            af.efg_pct_d  AS away_efg_d,
            hf.tov_pct    AS home_tov,
            hf.tov_pct_d  AS home_tov_d,
            af.tov_pct    AS away_tov,
            af.tov_pct_d  AS away_tov_d,
            hf.orb_pct    AS home_orb,
            hf.orb_pct_d  AS home_orb_d,
            af.orb_pct    AS away_orb,
            af.orb_pct_d  AS away_orb_d,
            hf.ftr        AS home_ftr,
            hf.ftr_d      AS home_ftr_d,
            af.ftr        AS away_ftr,
            af.ftr_d      AS away_ftr_d,
            hf.rest_days  AS home_rest,
            af.rest_days  AS away_rest,
            hf.games_played AS home_games,
            af.games_played AS away_games
        FROM games g
        JOIN team_features hf
            ON g.home_team_id = hf.team_id
            AND hf.as_of_date = g.game_date
        JOIN team_features af
            ON g.away_team_id = af.team_id
            AND af.as_of_date = g.game_date
        WHERE g.season IN ({placeholders})
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND hf.pace IS NOT NULL
          AND af.pace IS NOT NULL
        ORDER BY g.game_date
    """
    with db._connect() as conn:
        df = pd.read_sql_query(sql, conn, params=seasons)

    if df.empty:
        logger.warning("No training data for seasons %s", seasons)
        return df

    df = _add_combined_features(df)

    logger.info(
        "Built training data: %d games across seasons %s",
        len(df), seasons,
    )
    return df


def build_totals_eval_data(db, seasons: list[int]) -> pd.DataFrame:
    """Build evaluation data — games with features AND SBRO closing lines.

    Includes swap-fixed total_close via SQL CASE expression.

    Args:
        db: FeatureStore instance.
        seasons: List of season end years to include.

    Returns:
        DataFrame with combined features, actual_total, and total_close.
    """
    placeholders = ",".join("?" * len(seasons))
    sql = f"""
        SELECT
            g.game_id,
            g.game_date,
            g.season,
            g.home_team_name,
            g.away_team_name,
            g.home_score,
            g.away_score,
            g.home_score + g.away_score AS actual_total,
            {_SWAP_FIX_TOTAL} AS total_close,
            hf.pace       AS home_pace,
            af.pace       AS away_pace,
            hf.ortg       AS home_ortg,
            hf.drtg       AS home_drtg,
            af.ortg       AS away_ortg,
            af.drtg       AS away_drtg,
            hf.efg_pct    AS home_efg,
            hf.efg_pct_d  AS home_efg_d,
            af.efg_pct    AS away_efg,
            af.efg_pct_d  AS away_efg_d,
            hf.tov_pct    AS home_tov,
            hf.tov_pct_d  AS home_tov_d,
            af.tov_pct    AS away_tov,
            af.tov_pct_d  AS away_tov_d,
            hf.orb_pct    AS home_orb,
            hf.orb_pct_d  AS home_orb_d,
            af.orb_pct    AS away_orb,
            af.orb_pct_d  AS away_orb_d,
            hf.ftr        AS home_ftr,
            hf.ftr_d      AS home_ftr_d,
            af.ftr        AS away_ftr,
            af.ftr_d      AS away_ftr_d,
            hf.rest_days  AS home_rest,
            af.rest_days  AS away_rest,
            hf.games_played AS home_games,
            af.games_played AS away_games
        FROM games g
        JOIN team_features hf
            ON g.home_team_id = hf.team_id
            AND hf.as_of_date = g.game_date
        JOIN team_features af
            ON g.away_team_id = af.team_id
            AND af.as_of_date = g.game_date
        JOIN historical_odds ho
            ON ho.game_id = g.game_id
        WHERE g.season IN ({placeholders})
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND hf.pace IS NOT NULL
          AND af.pace IS NOT NULL
          AND ho.total_close IS NOT NULL
        ORDER BY g.game_date
    """
    with db._connect() as conn:
        df = pd.read_sql_query(sql, conn, params=seasons)

    if df.empty:
        logger.warning("No eval data for seasons %s", seasons)
        return df

    # Filter out rows where swap-fixed total_close is still unreasonable
    df = df[(df["total_close"] > 80) & (df["total_close"] < 250)].copy()

    df = _add_combined_features(df)

    logger.info(
        "Built eval data: %d games with SBRO lines across seasons %s",
        len(df), seasons,
    )
    return df


def _add_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined/averaged features for totals prediction."""
    df = df.copy()
    df["avg_pace"] = (df["home_pace"] + df["away_pace"]) / 2
    df["avg_ortg"] = (df["home_ortg"] + df["away_ortg"]) / 2
    df["avg_drtg"] = (df["home_drtg"] + df["away_drtg"]) / 2
    df["combined_efg"] = (df["home_efg"] + df["away_efg"]) / 2
    df["combined_efg_d"] = (df["home_efg_d"] + df["away_efg_d"]) / 2
    df["combined_tov"] = (df["home_tov"] + df["away_tov"]) / 2
    df["combined_tov_d"] = (df["home_tov_d"] + df["away_tov_d"]) / 2
    df["combined_orb"] = (df["home_orb"] + df["away_orb"]) / 2
    df["combined_orb_d"] = (df["home_orb_d"] + df["away_orb_d"]) / 2
    df["combined_ftr"] = (df["home_ftr"] + df["away_ftr"]) / 2
    df["combined_ftr_d"] = (df["home_ftr_d"] + df["away_ftr_d"]) / 2
    df["avg_rest"] = (df["home_rest"] + df["away_rest"]) / 2
    df["min_games"] = df[["home_games", "away_games"]].min(axis=1)
    return df
