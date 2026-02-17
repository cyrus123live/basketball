#!/usr/bin/env python3
"""Backtest totals model with walk-forward validation.

Runs feature build-up, prints results table, saves to backtests/.

Usage:
    python scripts/backtest_totals.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.totals_baseline import compare_models, run_feature_buildup
from src.utils.config import PROJECT_ROOT
from src.utils.db import FeatureStore

BACKTESTS_DIR = PROJECT_ROOT / "backtests"
BACKTESTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "backtest_totals.log"),
    ],
)
logger = logging.getLogger(__name__)


def print_buildup_table(results):
    """Print feature build-up results as a formatted table."""
    print("\n" + "=" * 100)
    print("FEATURE BUILD-UP RESULTS (LinearRegression, walk-forward 2019-2021)")
    print("=" * 100)
    header = (
        f"{'Step':<5} {'Features':<12} {'N feat':<7} "
        f"{'MAE':<8} {'Line MAE':<10} {'MAE Δ':<8} "
        f"{'Dir.Acc':<9} {'Avg CLV':<9} "
        f"{'Bets':<6} {'W-L':<10} {'ROI':<8}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        betting = r.wf_result.total_betting
        w, l = betting["wins"], betting["losses"]
        print(
            f"{r.step:<5} {r.label:<12} {len(r.feature_names):<7} "
            f"{r.mae:<8.2f} {r.closing_mae:<10.2f} {r.mae_improvement:<+8.2f} "
            f"{r.directional_accuracy * 100:<9.1f} {r.avg_clv:<+9.2f} "
            f"{r.n_bets:<6} {f'{w}-{l}':<10} {r.roi * 100:<+8.1f}%"
        )

    print("=" * 100)

    # Per-season breakdown for the best step
    best = min(results, key=lambda r: r.mae)
    print(f"\nBest step: {best.step} ({best.label}) — MAE={best.mae:.2f}")
    print("\nPer-season breakdown:")
    print(f"{'Season':<8} {'N eval':<8} {'MAE':<8} {'Line MAE':<10} {'Dir.Acc':<9} {'CLV':<8} {'Bets':<6} {'ROI':<8}")
    print("-" * 65)
    for fold in best.wf_result.folds:
        print(
            f"{fold.season:<8} {fold.n_eval:<8} {fold.regression['mae']:<8.2f} "
            f"{fold.closing_line_mae:<10.2f} "
            f"{fold.clv['directional_accuracy'] * 100:<9.1f} "
            f"{fold.clv['avg_clv']:<+8.2f} "
            f"{fold.betting['n_bets']:<6} {fold.betting['roi'] * 100:<+8.1f}%"
        )

    # Significance test
    sig = best.wf_result.significance
    print(f"\nSignificance test (vs 52.4% break-even):")
    print(f"  Observed win rate: {sig['observed_rate']:.1%}")
    print(f"  p-value: {sig['p_value']:.4f}")
    print(f"  Significant at 95%: {sig['significant_95']}")
    print(f"  Significant at 99%: {sig['significant_99']}")
    print(f"  Total bets: {sig['n_total']}")


def print_model_comparison(comparisons):
    """Print model comparison results."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (best feature set)")
    print("=" * 80)
    print(f"{'Model':<25} {'MAE':<8} {'Dir.Acc':<9} {'Bets':<6} {'ROI':<8} {'Win Rate':<10}")
    print("-" * 80)

    for name, wf in comparisons.items():
        betting = wf.total_betting
        print(
            f"{name:<25} {wf.avg_mae:<8.2f} "
            f"{wf.avg_directional_accuracy * 100:<9.1f} "
            f"{betting['n_bets']:<6} {betting['roi'] * 100:<+8.1f}% "
            f"{betting['win_rate'] * 100:<10.1f}"
        )
    print("=" * 80)


def save_results(results, comparisons, filepath):
    """Save results to JSON for later analysis."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "feature_buildup": [
            {
                "step": r.step,
                "label": r.label,
                "feature_names": r.feature_names,
                "n_features": len(r.feature_names),
                "mae": r.mae,
                "closing_mae": r.closing_mae,
                "mae_improvement": r.mae_improvement,
                "directional_accuracy": r.directional_accuracy,
                "avg_clv": r.avg_clv,
                "n_bets": r.n_bets,
                "roi": r.roi,
                "per_season": [
                    {
                        "season": f.season,
                        "n_train": f.n_train,
                        "n_eval": f.n_eval,
                        "mae": f.regression["mae"],
                        "closing_mae": f.closing_line_mae,
                        "directional_accuracy": f.clv["directional_accuracy"],
                        "avg_clv": f.clv["avg_clv"],
                        "n_bets": f.betting["n_bets"],
                        "roi": f.betting["roi"],
                    }
                    for f in r.wf_result.folds
                ],
            }
            for r in results
        ],
        "model_comparison": {
            name: {
                "mae": wf.avg_mae,
                "directional_accuracy": wf.avg_directional_accuracy,
                "total_bets": wf.total_betting["n_bets"],
                "roi": wf.total_betting["roi"],
                "win_rate": wf.total_betting["win_rate"],
            }
            for name, wf in comparisons.items()
        },
    }
    filepath.write_text(json.dumps(data, indent=2))
    logger.info("Results saved to %s", filepath)


def main():
    db = FeatureStore()

    # Check data availability
    print("Checking data availability...")
    with db._connect() as conn:
        game_counts = conn.execute(
            "SELECT season, COUNT(*) as n FROM games "
            "WHERE season BETWEEN 2016 AND 2021 GROUP BY season ORDER BY season"
        ).fetchall()
        odds_counts = conn.execute(
            "SELECT season, COUNT(*) as n FROM historical_odds "
            "WHERE game_id IS NOT NULL AND season BETWEEN 2016 AND 2021 "
            "GROUP BY season ORDER BY season"
        ).fetchall()
        feature_counts = conn.execute(
            "SELECT season, COUNT(DISTINCT team_id) as n FROM team_features "
            "WHERE season BETWEEN 2016 AND 2021 GROUP BY season ORDER BY season"
        ).fetchall()

    print("\nData summary:")
    print(f"{'Season':<8} {'Games':<8} {'Odds':<8} {'Teams w/features':<18}")
    print("-" * 42)
    gc = {r["season"]: r["n"] for r in game_counts}
    oc = {r["season"]: r["n"] for r in odds_counts}
    fc = {r["season"]: r["n"] for r in feature_counts}
    for s in range(2016, 2022):
        print(f"{s:<8} {gc.get(s, 0):<8} {oc.get(s, 0):<8} {fc.get(s, 0):<18}")

    # Step 1: Feature build-up
    print("\nRunning feature build-up (this may take a minute)...")
    results = run_feature_buildup(db)

    if not results:
        print("ERROR: No results. Check data availability.")
        sys.exit(1)

    print_buildup_table(results)

    # Step 2: Compare models at best feature set
    best = min(results, key=lambda r: r.mae)
    print(f"\nComparing models with {len(best.feature_names)} features: {best.feature_names}")
    comparisons = compare_models(db, best.feature_names)
    print_model_comparison(comparisons)

    # Step 3: Go/No-Go assessment
    print("\n" + "=" * 60)
    print("GO / NO-GO ASSESSMENT")
    print("=" * 60)
    dir_acc = best.directional_accuracy
    sig = best.wf_result.significance
    avg_roi = best.roi

    if dir_acc > 0.524:
        print(f"  [PASS] Directional accuracy: {dir_acc:.1%} > 52.4%")
    else:
        print(f"  [FAIL] Directional accuracy: {dir_acc:.1%} <= 52.4%")

    all_positive_clv = all(
        f.clv["avg_clv"] > 0 for f in best.wf_result.folds
    )
    if all_positive_clv:
        print(f"  [STRONG] Positive CLV in all {len(best.wf_result.folds)} seasons")
    else:
        pos = sum(1 for f in best.wf_result.folds if f.clv["avg_clv"] > 0)
        print(f"  [MIXED] Positive CLV in {pos}/{len(best.wf_result.folds)} seasons")

    if avg_roi > 0:
        print(f"  [PASS] Simulated ROI: {avg_roi:.1%} > 0%")
    else:
        print(f"  [FAIL] Simulated ROI: {avg_roi:.1%} <= 0%")

    if dir_acc < 0.51:
        print("\n  VERDICT: KILL — model can't beat the market on totals")
    elif dir_acc < 0.524:
        print("\n  VERDICT: MARGINAL — needs more features or better model")
    elif all_positive_clv and avg_roi > 0:
        print("\n  VERDICT: GO — proceed to Phase 2 (more features, XGBoost)")
    else:
        print("\n  VERDICT: PROMISING — directional edge exists, refine model")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(
        results, comparisons,
        BACKTESTS_DIR / f"totals_baseline_{timestamp}.json",
    )

    print(f"\nResults saved to backtests/totals_baseline_{timestamp}.json")


if __name__ == "__main__":
    main()
