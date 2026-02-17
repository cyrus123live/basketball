"""Phase 2 experiment runner: Barttorvik integration + XGBoost.

Runs a matrix of model configurations against different feature sets
to determine if opponent-adjusted ratings and/or non-linear models
can beat the closing line on totals.
"""

import logging
from dataclasses import dataclass
from functools import partial

from sklearn.linear_model import LinearRegression, Ridge

from src.models.totals_data_v2 import (
    ALL_FEATURES,
    BART_FEATURES,
    EWMA_FEATURES,
    build_totals_eval_data_v2,
    build_totals_training_data_v2,
)
from src.models.walk_forward import WalkForwardResult, walk_forward_validate
from src.models.xgb_wrapper import XGB_CONFIGS, XGBEarlyStopping

logger = logging.getLogger(__name__)

# Walk-forward folds (same as Phase 1)
DEFAULT_TRAIN_SEASONS = [
    [2016, 2017, 2018],
    [2016, 2017, 2018, 2019],
    [2016, 2017, 2018, 2019, 2020],
]
DEFAULT_VAL_SEASONS = [2019, 2020, 2021]


@dataclass
class ConfigResult:
    """Results for one configuration in the backtest matrix."""
    name: str
    features: list[str]
    model_type: str
    wf_result: WalkForwardResult
    feature_importance: dict = None


def _make_train_builder(include_bart: bool):
    """Create a train builder with the right include_bart setting."""
    return partial(build_totals_training_data_v2, include_bart=include_bart)


def _make_eval_builder(include_bart: bool):
    """Create an eval builder with the right include_bart setting."""
    return partial(build_totals_eval_data_v2, include_bart=include_bart)


def run_backtest_matrix(
    db,
    train_seasons_list: list[list[int]] = None,
    val_seasons: list[int] = None,
    min_edge: float = 1.0,
) -> list[ConfigResult]:
    """Run the full comparison matrix of feature sets × models.

    Configs:
        1. EWMA + LR (baseline repro)
        2. EWMA + XGB-conservative
        3. Bart + LR
        4. Bart + XGB-conservative
        5. All + LR
        6. All + Ridge(alpha=10)
        7. All + XGB-conservative

    Args:
        db: FeatureStore instance.
        train_seasons_list: Training seasons per fold.
        val_seasons: Validation seasons.
        min_edge: Minimum points edge for simulated betting.

    Returns:
        List of ConfigResult, one per configuration.
    """
    if train_seasons_list is None:
        train_seasons_list = DEFAULT_TRAIN_SEASONS
    if val_seasons is None:
        val_seasons = DEFAULT_VAL_SEASONS

    xgb_params = XGB_CONFIGS["XGB-conservative"]

    configs = [
        ("EWMA + LR", EWMA_FEATURES, lambda: LinearRegression(), False),
        ("EWMA + XGB", EWMA_FEATURES, lambda: XGBEarlyStopping(**xgb_params), False),
        ("Bart + LR", BART_FEATURES, lambda: LinearRegression(), True),
        ("Bart + XGB", BART_FEATURES, lambda: XGBEarlyStopping(**xgb_params), True),
        ("All + LR", ALL_FEATURES, lambda: LinearRegression(), True),
        ("All + Ridge", ALL_FEATURES, lambda: Ridge(alpha=10.0), True),
        ("All + XGB", ALL_FEATURES, lambda: XGBEarlyStopping(**xgb_params), True),
    ]

    results = []

    for name, features, model_factory, needs_bart in configs:
        logger.info("=" * 60)
        logger.info("Config: %s (%d features)", name, len(features))
        logger.info("=" * 60)

        wf = walk_forward_validate(
            db=db,
            model_factory=model_factory,
            feature_names=features,
            train_seasons_list=train_seasons_list,
            val_seasons=val_seasons,
            min_edge=min_edge,
            train_builder=_make_train_builder(needs_bart),
            eval_builder=_make_eval_builder(needs_bart),
        )

        # Get feature importance for XGB configs
        feat_imp = None
        if "XGB" in name and wf.folds:
            # Train one final model on all training data for importance
            try:
                all_train_seasons = train_seasons_list[-1]
                builder = _make_train_builder(needs_bart)
                train_df = builder(db, all_train_seasons)
                train_clean = train_df.dropna(subset=features + ["actual_total"])
                if not train_clean.empty:
                    model = XGBEarlyStopping(**xgb_params)
                    model.fit(
                        train_clean[features].values,
                        train_clean["actual_total"].values,
                    )
                    feat_imp = model.feature_importances(features)
            except Exception as e:
                logger.warning("Could not compute feature importance: %s", e)

        betting = wf.total_betting
        logger.info(
            "  MAE=%.2f, Line MAE=%.2f, Dir.Acc=%.1f%%, "
            "Bets=%d, ROI=%.1f%%",
            wf.avg_mae, wf.avg_closing_mae,
            wf.avg_directional_accuracy * 100,
            betting["n_bets"], betting["roi"] * 100,
        )

        results.append(ConfigResult(
            name=name,
            features=features,
            model_type=name.split(" + ")[1] if " + " in name else name,
            wf_result=wf,
            feature_importance=feat_imp,
        ))

    return results
