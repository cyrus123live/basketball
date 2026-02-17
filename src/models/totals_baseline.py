"""Totals baseline model: incremental feature build-up with walk-forward validation.

Adds features one at a time and measures marginal improvement in MAE and CLV.
This follows the "one feature at a time" philosophy — stop when improvement plateaus.
"""

import logging
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge

from src.models.walk_forward import WalkForwardResult, walk_forward_validate

logger = logging.getLogger(__name__)

# Feature build-up order, following the plan:
# pace → ratings → Four Factors → context
FEATURE_STEPS = [
    (["avg_pace"], "pace"),
    (["avg_ortg"], "+ortg"),
    (["avg_drtg"], "+drtg"),
    (["combined_efg", "combined_efg_d"], "+efg"),
    (["combined_tov", "combined_tov_d"], "+tov"),
    (["combined_orb", "combined_orb_d"], "+orb"),
    (["combined_ftr", "combined_ftr_d"], "+ftr"),
    (["avg_rest"], "+rest"),
    (["min_games"], "+games"),
]

# Default walk-forward folds: train on prior seasons, validate on next
DEFAULT_TRAIN_SEASONS = [
    [2016, 2017, 2018],
    [2016, 2017, 2018, 2019],
    [2016, 2017, 2018, 2019, 2020],
]
DEFAULT_VAL_SEASONS = [2019, 2020, 2021]


@dataclass
class StepResult:
    """Result of one feature build-up step."""
    step: int
    label: str
    feature_names: list[str]
    wf_result: WalkForwardResult
    mae: float
    closing_mae: float
    directional_accuracy: float
    avg_clv: float
    n_bets: int
    roi: float
    mae_improvement: float  # vs previous step (negative = better)


def run_feature_buildup(
    db,
    train_seasons_list: list[list[int]] = None,
    val_seasons: list[int] = None,
    min_edge: float = 1.0,
) -> list[StepResult]:
    """Run incremental feature build-up with walk-forward validation.

    At each step, adds the next feature group and measures marginal improvement.

    Args:
        db: FeatureStore instance.
        train_seasons_list: Training seasons per fold (default: 2016-18/19/20).
        val_seasons: Validation seasons (default: 2019/2020/2021).
        min_edge: Min points edge for simulated betting.

    Returns:
        List of StepResult, one per feature step.
    """
    if train_seasons_list is None:
        train_seasons_list = DEFAULT_TRAIN_SEASONS
    if val_seasons is None:
        val_seasons = DEFAULT_VAL_SEASONS

    cumulative_features = []
    results = []
    prev_mae = None

    for step_idx, (new_features, label) in enumerate(FEATURE_STEPS, 1):
        cumulative_features = cumulative_features + new_features

        logger.info(
            "=== Step %d: %s (features: %s) ===",
            step_idx, label, cumulative_features,
        )

        wf = walk_forward_validate(
            db=db,
            model_factory=lambda: LinearRegression(),
            feature_names=cumulative_features,
            train_seasons_list=train_seasons_list,
            val_seasons=val_seasons,
            min_edge=min_edge,
        )

        mae = wf.avg_mae
        mae_improvement = (mae - prev_mae) if prev_mae is not None else 0.0
        betting = wf.total_betting

        step = StepResult(
            step=step_idx,
            label=label,
            feature_names=list(cumulative_features),
            wf_result=wf,
            mae=mae,
            closing_mae=wf.avg_closing_mae,
            directional_accuracy=wf.avg_directional_accuracy,
            avg_clv=sum(f.clv["avg_clv"] for f in wf.folds) / len(wf.folds) if wf.folds else 0,
            n_bets=betting["n_bets"],
            roi=betting["roi"],
            mae_improvement=mae_improvement,
        )
        results.append(step)

        improvement_pct = abs(mae_improvement / prev_mae * 100) if prev_mae else 0
        logger.info(
            "  MAE=%.2f (delta=%.2f, %.1f%%), Dir.Acc=%.1f%%, CLV=%.2f, "
            "Bets=%d, ROI=%.1f%%",
            mae, mae_improvement, improvement_pct,
            step.directional_accuracy * 100, step.avg_clv,
            step.n_bets, step.roi * 100,
        )

        if prev_mae is not None and improvement_pct < 0.5 and step_idx > 3:
            logger.info("  *** Marginal improvement < 0.5%% — diminishing returns")

        prev_mae = mae

    return results


def compare_models(
    db,
    feature_names: list[str],
    train_seasons_list: list[list[int]] = None,
    val_seasons: list[int] = None,
    min_edge: float = 1.0,
) -> dict[str, WalkForwardResult]:
    """Compare LinearRegression vs Ridge at the best feature set.

    Args:
        db: FeatureStore instance.
        feature_names: Features to use.
        train_seasons_list: Training seasons per fold.
        val_seasons: Validation seasons.
        min_edge: Min points edge for simulated betting.

    Returns:
        Dict mapping model name to WalkForwardResult.
    """
    if train_seasons_list is None:
        train_seasons_list = DEFAULT_TRAIN_SEASONS
    if val_seasons is None:
        val_seasons = DEFAULT_VAL_SEASONS

    models = {
        "LinearRegression": lambda: LinearRegression(),
        "Ridge(alpha=1.0)": lambda: Ridge(alpha=1.0),
        "Ridge(alpha=10.0)": lambda: Ridge(alpha=10.0),
    }

    results = {}
    for name, factory in models.items():
        logger.info("=== %s with %d features ===", name, len(feature_names))
        wf = walk_forward_validate(
            db=db,
            model_factory=factory,
            feature_names=feature_names,
            train_seasons_list=train_seasons_list,
            val_seasons=val_seasons,
            min_edge=min_edge,
        )
        results[name] = wf
        betting = wf.total_betting
        logger.info(
            "  MAE=%.2f, Dir.Acc=%.1f%%, Bets=%d, ROI=%.1f%%",
            wf.avg_mae, wf.avg_directional_accuracy * 100,
            betting["n_bets"], betting["roi"] * 100,
        )

    return results
