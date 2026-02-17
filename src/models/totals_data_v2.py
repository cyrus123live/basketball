"""Training and evaluation data builders with Barttorvik prior-season ratings.

Extends totals_data.py by LEFT JOINing prior-season Barttorvik end-of-season
ratings as "preseason quality" features. For a game in season S, we use each
team's S-1 final ratings — no lookahead bias.

Games where either team lacks prior-season Barttorvik data get NaN in the
Barttorvik columns. The walk-forward framework's dropna() handles this.
"""

import logging

import pandas as pd

from src.models.totals_data import (
    TOTALS_FEATURES,
    _add_combined_features,
    _SWAP_FIX_TOTAL,
)

logger = logging.getLogger(__name__)

# Barttorvik-derived combined features
BART_FEATURES = [
    "bart_avg_adj_t",
    "bart_avg_adj_o",
    "bart_avg_adj_d",
    "bart_avg_barthag",
    "bart_adj_t_diff",
]

# All features: original EWMA (13) + Barttorvik (5) = 18
EWMA_FEATURES = list(TOTALS_FEATURES)
ALL_FEATURES = EWMA_FEATURES + BART_FEATURES

# SQL subquery: get the latest (end-of-season) Barttorvik rating for each
# team+season combination. Joins through the teams table to resolve team
# names, since team_ids are per-season (a team gets a new id each season).
_BART_SUBQUERY = """
    SELECT t.name AS team_name, br1.season, br1.adj_o, br1.adj_d, br1.adj_t, br1.barthag
    FROM barttorvik_ratings br1
    JOIN teams t ON br1.team_id = t.team_id
    WHERE br1.team_id IS NOT NULL
      AND br1.as_of_date = (
          SELECT MAX(br2.as_of_date)
          FROM barttorvik_ratings br2
          WHERE br2.team_id = br1.team_id AND br2.season = br1.season
      )
"""


def build_totals_training_data_v2(
    db, seasons: list[int], include_bart: bool = True
) -> pd.DataFrame:
    """Build training data with optional Barttorvik prior-season features.

    Args:
        db: FeatureStore instance.
        seasons: List of season end years to include.
        include_bart: If True, LEFT JOIN Barttorvik ratings from season - 1.

    Returns:
        DataFrame with combined features and actual_total label.
    """
    placeholders = ",".join("?" * len(seasons))

    bart_joins = ""
    bart_selects = ""
    if include_bart:
        bart_selects = """,
            hb.adj_o AS home_bart_adj_o,
            hb.adj_d AS home_bart_adj_d,
            hb.adj_t AS home_bart_adj_t,
            hb.barthag AS home_bart_barthag,
            ab.adj_o AS away_bart_adj_o,
            ab.adj_d AS away_bart_adj_d,
            ab.adj_t AS away_bart_adj_t,
            ab.barthag AS away_bart_barthag"""
        bart_joins = f"""
        LEFT JOIN ({_BART_SUBQUERY}) hb
            ON hb.team_name = g.home_team_name AND hb.season = g.season - 1
        LEFT JOIN ({_BART_SUBQUERY}) ab
            ON ab.team_name = g.away_team_name AND ab.season = g.season - 1"""

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
            {bart_selects}
        FROM games g
        JOIN team_features hf
            ON g.home_team_id = hf.team_id
            AND hf.as_of_date = g.game_date
        JOIN team_features af
            ON g.away_team_id = af.team_id
            AND af.as_of_date = g.game_date
        {bart_joins}
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
    if include_bart:
        df = _add_barttorvik_features(df)

    logger.info(
        "Built v2 training data: %d games across seasons %s (bart=%s)",
        len(df), seasons, include_bart,
    )
    return df


def build_totals_eval_data_v2(
    db, seasons: list[int], include_bart: bool = True
) -> pd.DataFrame:
    """Build evaluation data with SBRO lines and optional Barttorvik features.

    Args:
        db: FeatureStore instance.
        seasons: List of season end years to include.
        include_bart: If True, LEFT JOIN Barttorvik ratings from season - 1.

    Returns:
        DataFrame with combined features, actual_total, and total_close.
    """
    placeholders = ",".join("?" * len(seasons))

    bart_joins = ""
    bart_selects = ""
    if include_bart:
        bart_selects = """,
            hb.adj_o AS home_bart_adj_o,
            hb.adj_d AS home_bart_adj_d,
            hb.adj_t AS home_bart_adj_t,
            hb.barthag AS home_bart_barthag,
            ab.adj_o AS away_bart_adj_o,
            ab.adj_d AS away_bart_adj_d,
            ab.adj_t AS away_bart_adj_t,
            ab.barthag AS away_bart_barthag"""
        bart_joins = f"""
        LEFT JOIN ({_BART_SUBQUERY}) hb
            ON hb.team_name = g.home_team_name AND hb.season = g.season - 1
        LEFT JOIN ({_BART_SUBQUERY}) ab
            ON ab.team_name = g.away_team_name AND ab.season = g.season - 1"""

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
            {bart_selects}
        FROM games g
        JOIN team_features hf
            ON g.home_team_id = hf.team_id
            AND hf.as_of_date = g.game_date
        JOIN team_features af
            ON g.away_team_id = af.team_id
            AND af.as_of_date = g.game_date
        JOIN historical_odds ho
            ON ho.game_id = g.game_id
        {bart_joins}
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

    # Filter unreasonable totals (same as v1)
    df = df[(df["total_close"] > 80) & (df["total_close"] < 250)].copy()

    df = _add_combined_features(df)
    if include_bart:
        df = _add_barttorvik_features(df)

    logger.info(
        "Built v2 eval data: %d games with SBRO lines across seasons %s (bart=%s)",
        len(df), seasons, include_bart,
    )
    return df


def _add_barttorvik_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute combined Barttorvik features from raw home/away columns.

    These are averages (for totals prediction) and absolute differences
    (for mismatch detection). NaN propagates naturally when a team
    lacks prior-season data.
    """
    df = df.copy()
    df["bart_avg_adj_t"] = (df["home_bart_adj_t"] + df["away_bart_adj_t"]) / 2
    df["bart_avg_adj_o"] = (df["home_bart_adj_o"] + df["away_bart_adj_o"]) / 2
    df["bart_avg_adj_d"] = (df["home_bart_adj_d"] + df["away_bart_adj_d"]) / 2
    df["bart_avg_barthag"] = (
        df["home_bart_barthag"] + df["away_bart_barthag"]
    ) / 2
    df["bart_adj_t_diff"] = (
        df["home_bart_adj_t"] - df["away_bart_adj_t"]
    ).abs()
    return df
