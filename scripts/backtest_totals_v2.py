#!/usr/bin/env python3
"""Phase 2 backtest: Barttorvik + XGBoost on totals.

Runs the full comparison matrix, prints results, saves to backtests/.

Usage:
    python scripts/backtest_totals_v2.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.totals_v2 import run_backtest_matrix
from src.utils.config import PROJECT_ROOT
from src.utils.db import FeatureStore

BACKTESTS_DIR = PROJECT_ROOT / "backtests"
BACKTESTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "backtest_totals_v2.log"),
    ],
)
logger = logging.getLogger(__name__)


def check_barttorvik_data(db):
    """Verify Barttorvik data exists for required seasons."""
    with db._connect() as conn:
        counts = conn.execute(
            "SELECT season, COUNT(*) as n, "
            "SUM(CASE WHEN team_id IS NOT NULL THEN 1 ELSE 0 END) as matched "
            "FROM barttorvik_ratings "
            "WHERE season BETWEEN 2015 AND 2020 "
            "GROUP BY season ORDER BY season"
        ).fetchall()

    if not counts:
        print("ERROR: No Barttorvik data found.")
        print("Run: python scripts/scrape_barttorvik_historical.py")
        return False

    print("Barttorvik data:")
    print(f"{'Season':<8} {'Total':<8} {'Matched':<10}")
    print("-" * 26)
    for row in counts:
        print(f"{row['season']:<8} {row['n']:<8} {row['matched']:<10}")

    seasons_found = {row["season"] for row in counts}
    required = {2015, 2016, 2017, 2018, 2019, 2020}
    missing = required - seasons_found
    if missing:
        print(f"\nWARNING: Missing seasons: {sorted(missing)}")
        print("Run: python scripts/scrape_barttorvik_historical.py")
        return False

    return True


def print_matrix_table(results):
    """Print the backtest matrix results."""
    print("\n" + "=" * 110)
    print("PHASE 2 BACKTEST MATRIX: Barttorvik + XGBoost on Totals")
    print("Walk-forward: train → validate 2019, 2020, 2021")
    print("=" * 110)
    header = (
        f"{'Config':<18} {'N feat':<8} "
        f"{'MAE':<8} {'Line MAE':<10} "
        f"{'Dir.Acc':<9} {'Avg CLV':<9} "
        f"{'Bets':<6} {'W-L':<10} {'ROI':<8} {'Win%':<7}"
    )
    print(header)
    print("-" * 110)

    for r in results:
        wf = r.wf_result
        betting = wf.total_betting
        w, l = betting["wins"], betting["losses"]
        avg_clv = (
            sum(f.clv["avg_clv"] for f in wf.folds) / len(wf.folds)
            if wf.folds else 0.0
        )
        print(
            f"{r.name:<18} {len(r.features):<8} "
            f"{wf.avg_mae:<8.2f} {wf.avg_closing_mae:<10.2f} "
            f"{wf.avg_directional_accuracy * 100:<9.1f} {avg_clv:<+9.2f} "
            f"{betting['n_bets']:<6} {f'{w}-{l}':<10} "
            f"{betting['roi'] * 100:<+8.1f}% {betting['win_rate'] * 100:<7.1f}"
        )

    print("=" * 110)


def print_feature_importance(results):
    """Print feature importance for XGB configs."""
    xgb_results = [r for r in results if r.feature_importance]
    if not xgb_results:
        return

    print("\nFEATURE IMPORTANCE (by gain, best XGB config):")
    print("-" * 40)

    # Use the config with most features
    best = max(xgb_results, key=lambda r: len(r.features))
    for name, imp in best.feature_importance.items():
        bar = "█" * int(imp * 50)
        print(f"  {name:<22} {imp:.3f} {bar}")


def print_per_season(results):
    """Print per-season breakdown for the best config."""
    best = max(results, key=lambda r: r.wf_result.avg_directional_accuracy)
    print(f"\nPer-season breakdown for best config: {best.name}")
    print(f"{'Season':<8} {'N train':<9} {'N eval':<8} {'MAE':<8} "
          f"{'Line MAE':<10} {'Dir.Acc':<9} {'CLV':<8} {'Bets':<6} {'ROI':<8}")
    print("-" * 80)
    for fold in best.wf_result.folds:
        print(
            f"{fold.season:<8} {fold.n_train:<9} {fold.n_eval:<8} "
            f"{fold.regression['mae']:<8.2f} {fold.closing_line_mae:<10.2f} "
            f"{fold.clv['directional_accuracy'] * 100:<9.1f} "
            f"{fold.clv['avg_clv']:<+8.2f} "
            f"{fold.betting['n_bets']:<6} {fold.betting['roi'] * 100:<+8.1f}%"
        )


def print_verdict(results):
    """Print go/no-go verdict."""
    print("\n" + "=" * 60)
    print("GO / NO-GO ASSESSMENT")
    print("=" * 60)

    best = max(results, key=lambda r: r.wf_result.avg_directional_accuracy)
    wf = best.wf_result
    betting = wf.total_betting
    dir_acc = wf.avg_directional_accuracy

    print(f"\nBest config: {best.name}")

    if dir_acc > 0.524:
        print(f"  [PASS] Directional accuracy: {dir_acc:.1%} > 52.4%")
    else:
        print(f"  [FAIL] Directional accuracy: {dir_acc:.1%} <= 52.4%")

    if betting["roi"] > 0:
        print(f"  [PASS] Simulated ROI: {betting['roi']:.1%} > 0%")
    else:
        print(f"  [FAIL] Simulated ROI: {betting['roi']:.1%} <= 0%")

    sig = wf.significance
    if sig["significant_95"]:
        print(f"  [PASS] Statistically significant (p={sig['p_value']:.4f})")
    else:
        print(f"  [FAIL] Not significant (p={sig['p_value']:.4f})")

    # Check improvement over EWMA+LR baseline
    baseline = next((r for r in results if r.name == "EWMA + LR"), None)
    if baseline:
        base_acc = baseline.wf_result.avg_directional_accuracy
        improvement = dir_acc - base_acc
        print(f"\n  vs Phase 1 baseline (EWMA+LR): {improvement:+.1%} directional accuracy")

    # Verdict
    if dir_acc < 0.51:
        print("\n  VERDICT: KILL — even with Barttorvik + XGBoost, can't beat the market")
    elif dir_acc < 0.524:
        print("\n  VERDICT: MARGINAL — improvement seen, but not at break-even")
        print("  Next: add contextual features (travel, rivalry, altitude)")
    elif sig["significant_95"]:
        print("\n  VERDICT: GO — statistically significant edge detected")
        print("  Next: paper trade with live odds, monitor CLV in real-time")
    else:
        print("\n  VERDICT: PROMISING — directional edge but needs more data for significance")
        print("  Next: run through 2022-2025 seasons (need Odds API historical)")


def save_results(results, filepath):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "phase": "2",
        "description": "Barttorvik + XGBoost on totals",
        "configs": [
            {
                "name": r.name,
                "features": r.features,
                "n_features": len(r.features),
                "model_type": r.model_type,
                "mae": r.wf_result.avg_mae,
                "closing_mae": r.wf_result.avg_closing_mae,
                "directional_accuracy": r.wf_result.avg_directional_accuracy,
                "total_bets": r.wf_result.total_betting["n_bets"],
                "roi": r.wf_result.total_betting["roi"],
                "win_rate": r.wf_result.total_betting["win_rate"],
                "feature_importance": r.feature_importance,
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
    }
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            return super().default(obj)

    filepath.write_text(json.dumps(data, indent=2, cls=NumpyEncoder))
    logger.info("Results saved to %s", filepath)


def main():
    db = FeatureStore()

    # Verify Barttorvik data
    print("Checking Barttorvik data availability...")
    if not check_barttorvik_data(db):
        sys.exit(1)

    # Also check game/odds/feature data
    print("\nChecking game data...")
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

    print(f"{'Season':<8} {'Games':<8} {'Odds':<8}")
    print("-" * 24)
    gc = {r["season"]: r["n"] for r in game_counts}
    oc = {r["season"]: r["n"] for r in odds_counts}
    for s in range(2016, 2022):
        print(f"{s:<8} {gc.get(s, 0):<8} {oc.get(s, 0):<8}")

    # Run the matrix
    print("\nRunning backtest matrix (this may take a few minutes)...")
    results = run_backtest_matrix(db)

    if not results:
        print("ERROR: No results. Check data availability.")
        sys.exit(1)

    # Print results
    print_matrix_table(results)
    print_feature_importance(results)
    print_per_season(results)
    print_verdict(results)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = BACKTESTS_DIR / f"totals_v2_{timestamp}.json"
    save_results(results, save_path)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
