"""Walk-forward validation for totals models.

Train on all prior seasons, predict on the next season, evaluate CLV against
SBRO closing lines. This is the only valid way to backtest temporal data —
no k-fold, no random splits.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.models.evaluation import (
    clv_metrics,
    regression_metrics,
    significance_test,
    simulate_betting,
)
from src.models.totals_data import TOTALS_FEATURES, build_totals_eval_data, build_totals_training_data

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results for a single validation fold (season)."""
    season: int
    n_train: int
    n_eval: int
    regression: dict
    clv: dict
    betting: dict
    closing_line_mae: float  # MAE of closing line vs actual (benchmark)


@dataclass
class WalkForwardResult:
    """Aggregate results across all folds."""
    folds: list[FoldResult]
    feature_names: list[str]

    @property
    def total_eval_games(self) -> int:
        return sum(f.n_eval for f in self.folds)

    @property
    def avg_mae(self) -> float:
        if not self.folds:
            return 0.0
        return np.mean([f.regression["mae"] for f in self.folds])

    @property
    def avg_closing_mae(self) -> float:
        if not self.folds:
            return 0.0
        return np.mean([f.closing_line_mae for f in self.folds])

    @property
    def avg_directional_accuracy(self) -> float:
        if not self.folds:
            return 0.0
        # Weighted by n_evaluable
        total_correct = sum(
            f.clv["directional_accuracy"] * f.clv["n_evaluable"]
            for f in self.folds
        )
        total_games = sum(f.clv["n_evaluable"] for f in self.folds)
        return total_correct / total_games if total_games > 0 else 0.0

    @property
    def total_betting(self) -> dict:
        """Aggregate betting stats across all folds."""
        total_wins = sum(f.betting["wins"] for f in self.folds)
        total_losses = sum(f.betting["losses"] for f in self.folds)
        total_pushes = sum(f.betting["pushes"] for f in self.folds)
        total_profit = sum(f.betting["profit"] for f in self.folds)
        total_bets = sum(f.betting["n_bets"] for f in self.folds)
        return {
            "n_bets": total_bets,
            "wins": total_wins,
            "losses": total_losses,
            "pushes": total_pushes,
            "profit": total_profit,
            "roi": total_profit / total_bets if total_bets > 0 else 0.0,
            "win_rate": total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0.0,
        }

    @property
    def significance(self) -> dict:
        b = self.total_betting
        return significance_test(b["wins"], b["losses"])


def walk_forward_validate(
    db,
    model_factory: Callable,
    feature_names: list[str],
    train_seasons_list: list[list[int]],
    val_seasons: list[int],
    min_edge: float = 1.0,
    train_builder: Callable = None,
    eval_builder: Callable = None,
) -> WalkForwardResult:
    """Run walk-forward validation across multiple folds.

    Args:
        db: FeatureStore instance.
        model_factory: Callable that returns a fresh sklearn-compatible model
            (must have .fit(X, y) and .predict(X)).
        feature_names: List of column names to use as features.
        train_seasons_list: List of training season lists, one per fold.
            E.g., [[2016,2017,2018], [2016,2017,2018,2019], ...]
        val_seasons: Validation season for each fold (same length).
        min_edge: Minimum model-line difference to simulate a bet.
        train_builder: Custom function(db, seasons) -> DataFrame for training.
            Defaults to build_totals_training_data.
        eval_builder: Custom function(db, seasons) -> DataFrame for eval.
            Defaults to build_totals_eval_data.

    Returns:
        WalkForwardResult with per-fold and aggregate results.
    """
    if train_builder is None:
        train_builder = build_totals_training_data
    if eval_builder is None:
        eval_builder = build_totals_eval_data

    folds = []

    for train_seasons, val_season in zip(train_seasons_list, val_seasons):
        logger.info(
            "Fold: train %s → validate %d", train_seasons, val_season,
        )

        # Build datasets
        train_df = train_builder(db, train_seasons)
        eval_df = eval_builder(db, [val_season])

        if train_df.empty or eval_df.empty:
            logger.warning(
                "Skipping fold %d: train=%d, eval=%d",
                val_season, len(train_df), len(eval_df),
            )
            continue

        # Drop rows with NaN in feature columns
        train_clean = train_df.dropna(subset=feature_names + ["actual_total"])
        eval_clean = eval_df.dropna(subset=feature_names + ["actual_total", "total_close"])

        if train_clean.empty or eval_clean.empty:
            logger.warning(
                "Skipping fold %d after NaN drop: train=%d, eval=%d",
                val_season, len(train_clean), len(eval_clean),
            )
            continue

        X_train = train_clean[feature_names].values
        y_train = train_clean["actual_total"].values
        X_eval = eval_clean[feature_names].values
        y_eval = eval_clean["actual_total"].values
        closing = eval_clean["total_close"].values

        # Train and predict
        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_eval)

        # Evaluate
        reg = regression_metrics(y_eval, y_pred)
        clv = clv_metrics(y_pred, closing, y_eval)
        bet = simulate_betting(y_pred, closing, y_eval, min_edge=min_edge)
        closing_mae = regression_metrics(y_eval, closing)["mae"]

        fold = FoldResult(
            season=val_season,
            n_train=len(train_clean),
            n_eval=len(eval_clean),
            regression=reg,
            clv=clv,
            betting=bet,
            closing_line_mae=closing_mae,
        )
        folds.append(fold)

        logger.info(
            "  Season %d: MAE=%.2f (line=%.2f), Dir.Acc=%.1f%%, CLV=%.2f, "
            "Bets=%d, ROI=%.1f%%",
            val_season, reg["mae"], closing_mae,
            clv["directional_accuracy"] * 100, clv["avg_clv"],
            bet["n_bets"], bet["roi"] * 100,
        )

    return WalkForwardResult(folds=folds, feature_names=feature_names)
