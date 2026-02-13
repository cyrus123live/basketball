#!/usr/bin/env python3
"""Naive paper trading pipeline.

Uses logistic regression with 2 features (net rating differential + home
advantage) to generate win probabilities. Compares against market implied
probabilities and logs predictions.

This is deliberately simple — it validates the full pipeline works end-to-end,
not that we have an edge.

Run: python scripts/paper_trade.py [--date YYYY-MM-DD] [--backtest]
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.odds_api_collector import american_to_implied_prob
from src.utils.db import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/paper_trade.log"),
    ],
)
logger = logging.getLogger(__name__)

MODEL_NAME = "naive_logreg_v1"


def build_training_data(db: FeatureStore, before_date: str) -> pd.DataFrame:
    """Build training dataset from historical games with features.

    Only uses data available before the prediction date.
    """
    with db._connect() as conn:
        sql = """
            SELECT
                g.game_id,
                g.game_date,
                g.home_team_id,
                g.away_team_id,
                g.home_team_name,
                g.away_team_name,
                g.home_score,
                g.away_score,
                hf.net_rtg as home_net_rtg,
                hf.ortg as home_ortg,
                hf.drtg as home_drtg,
                af.net_rtg as away_net_rtg,
                af.ortg as away_ortg,
                af.drtg as away_drtg
            FROM games g
            JOIN team_features hf
                ON g.home_team_id = hf.team_id
                AND hf.as_of_date = g.game_date
            JOIN team_features af
                ON g.away_team_id = af.team_id
                AND af.as_of_date = g.game_date
            WHERE g.game_date < ?
              AND g.home_score IS NOT NULL
              AND hf.net_rtg IS NOT NULL
              AND af.net_rtg IS NOT NULL
            ORDER BY g.game_date
        """
        df = pd.read_sql_query(sql, conn, params=(before_date,))

    if not df.empty:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        df["net_rtg_diff"] = df["home_net_rtg"] - df["away_net_rtg"]
        df["is_home"] = 1  # Always 1 (home court advantage feature)

    return df


def train_model(train_df: pd.DataFrame) -> LogisticRegression | None:
    """Train a simple logistic regression model."""
    if len(train_df) < 50:
        logger.warning("Not enough training data (%d games)", len(train_df))
        return None

    X = train_df[["net_rtg_diff", "is_home"]].values
    y = train_df["home_win"].values

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Log model performance on training data
    train_acc = model.score(X, y)
    logger.info(
        "Trained on %d games, training accuracy: %.3f",
        len(train_df), train_acc,
    )
    logger.info(
        "Coefficients: net_rtg_diff=%.4f, is_home=%.4f, intercept=%.4f",
        model.coef_[0][0], model.coef_[0][1], model.intercept_[0],
    )

    return model


def predict_games(
    db: FeatureStore, model: LogisticRegression, game_date: str
) -> list[dict]:
    """Generate predictions for all games on a date."""
    games_df = db.get_games_for_date(game_date)
    if games_df.empty:
        logger.info("No games found for %s", game_date)
        return []

    predictions = []
    for _, game in games_df.iterrows():
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        if home_id is None or away_id is None:
            continue

        matchup = db.get_matchup_features(int(home_id), int(away_id), game_date)
        if matchup is None:
            continue

        net_rtg_diff = matchup["net_rtg_diff"]
        X = np.array([[net_rtg_diff, 1]])
        home_prob = model.predict_proba(X)[0][1]

        # Get market odds if available
        market_prob, market_spread = _get_market_odds(db, game["game_id"])

        edge = home_prob - market_prob if market_prob else None

        pred = {
            "game_id": game["game_id"],
            "game_date": game_date,
            "model_name": MODEL_NAME,
            "home_team_name": game["home_team_name"],
            "away_team_name": game["away_team_name"],
            "predicted_home_prob": round(home_prob, 4),
            "market_home_prob": round(market_prob, 4) if market_prob else None,
            "market_spread": market_spread,
            "edge": round(edge, 4) if edge else None,
            "bet_recommended": 1 if edge and abs(edge) > 0.05 else 0,
            "actual_home_win": (
                int(game["home_score"] > game["away_score"])
                if game["home_score"] is not None and game["away_score"] is not None
                else None
            ),
            "clv": None,
        }
        predictions.append(pred)

    return predictions


def _get_market_odds(db: FeatureStore, game_id: str) -> tuple[float | None, float | None]:
    """Get market implied probability and spread for a game."""
    with db._connect() as conn:
        # Try live odds first
        row = conn.execute("""
            SELECT price, point FROM live_odds
            WHERE game_id = ? AND market_type = 'spreads'
            ORDER BY snapshot_time DESC
            LIMIT 1
        """, (game_id,)).fetchone()

        if row and row["price"]:
            spread = row["point"]
            # Convert spread to approximate probability
            # Rule of thumb: each point of spread ≈ 3% probability
            market_prob = 0.5 + (-(spread or 0)) * 0.03
            market_prob = max(0.05, min(0.95, market_prob))
            return market_prob, spread

        # Try historical odds
        row = conn.execute("""
            SELECT spread_close FROM historical_odds
            WHERE game_id = ?
            LIMIT 1
        """, (game_id,)).fetchone()

        if row and row["spread_close"] is not None:
            spread = row["spread_close"]
            market_prob = 0.5 + (-spread) * 0.03
            market_prob = max(0.05, min(0.95, market_prob))
            return market_prob, spread

    return None, None


def run_paper_trade(db: FeatureStore, game_date: str):
    """Run the full paper trading pipeline for a date."""
    logger.info("=== Paper trading for %s ===", game_date)

    # Build training data using everything before this date
    train_df = build_training_data(db, game_date)
    if train_df.empty:
        logger.warning("No training data available before %s", game_date)
        return

    # Train model
    model = train_model(train_df)
    if model is None:
        return

    # Generate predictions
    predictions = predict_games(db, model, game_date)

    # Log and store predictions
    for pred in predictions:
        db.upsert_prediction(pred)
        logger.info(
            "  %s @ %s: P(home)=%.3f, market=%.3f, edge=%s, actual=%s",
            pred["away_team_name"],
            pred["home_team_name"],
            pred["predicted_home_prob"],
            pred["market_home_prob"] or 0,
            f"{pred['edge']:.3f}" if pred["edge"] else "N/A",
            pred["actual_home_win"],
        )

    logger.info("Logged %d predictions for %s", len(predictions), game_date)

    # Summary stats
    if predictions:
        probs = [p["predicted_home_prob"] for p in predictions]
        logger.info(
            "Prediction range: [%.3f, %.3f], mean=%.3f",
            min(probs), max(probs), np.mean(probs),
        )


def main():
    parser = argparse.ArgumentParser(description="Naive paper trading pipeline")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date to predict (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run backtest over all available dates",
    )
    parser.add_argument(
        "--backtest-start", type=str, default=None,
        help="Backtest start date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    db = FeatureStore()

    if args.backtest:
        # Run over historical dates
        start = args.backtest_start or "2018-01-01"
        logger.info("Running backtest from %s", start)

        with db._connect() as conn:
            dates = [
                r["game_date"]
                for r in conn.execute(
                    "SELECT DISTINCT game_date FROM games WHERE game_date >= ? ORDER BY game_date",
                    (start,),
                ).fetchall()
            ]

        for game_date in dates:
            try:
                run_paper_trade(db, game_date)
            except Exception as e:
                logger.error("Failed for %s: %s", game_date, e)
    else:
        game_date = args.date or datetime.now().strftime("%Y-%m-%d")
        run_paper_trade(db, game_date)


if __name__ == "__main__":
    main()
