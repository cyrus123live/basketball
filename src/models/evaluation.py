"""Evaluation framework for totals prediction models.

Stateless functions: arrays in, numbers out. No database dependency.
"""

import numpy as np
from scipy import stats


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics.

    Args:
        y_true: Actual totals.
        y_pred: Predicted totals.

    Returns:
        Dict with mae, rmse, r2, bias.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    bias = np.mean(residuals)
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


def clv_metrics(
    y_pred: np.ndarray,
    closing_line: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Compute Closing Line Value metrics for totals.

    For each game, the model predicts a total. If the model says the total
    should be higher than the closing line, that's an "over" signal (and vice
    versa). CLV measures whether the model's disagreements with the line are
    correct on average.

    Args:
        y_pred: Model predicted totals.
        closing_line: SBRO closing totals.
        y_true: Actual game totals.

    Returns:
        Dict with directional_accuracy, avg_clv, over_record (W-L-P),
        under_record (W-L-P), n_over, n_under, n_push.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    closing_line = np.asarray(closing_line, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    model_edge = y_pred - closing_line  # positive = model says over
    actual_diff = y_true - closing_line  # positive = game went over

    # Only evaluate games where the model disagrees with the line
    has_signal = model_edge != 0
    # Pushes: actual lands exactly on the line
    is_push = actual_diff == 0

    # Directional accuracy: model says over and game goes over, or model says
    # under and game goes under
    correct = ((model_edge > 0) & (actual_diff > 0)) | (
        (model_edge < 0) & (actual_diff < 0)
    )
    # Exclude pushes from directional accuracy
    evaluable = has_signal & ~is_push
    n_evaluable = np.sum(evaluable)
    directional_accuracy = (
        np.sum(correct[evaluable]) / n_evaluable if n_evaluable > 0 else 0.0
    )

    # Average CLV: when model says over, CLV = actual - line; when under,
    # CLV = line - actual. This measures how much the model "beat" the line
    # in the direction it predicted.
    signed_clv = np.where(model_edge > 0, actual_diff, -actual_diff)
    avg_clv = np.mean(signed_clv[has_signal]) if np.sum(has_signal) > 0 else 0.0

    # Over/under records
    over_mask = model_edge > 0
    under_mask = model_edge < 0

    over_wins = int(np.sum(over_mask & (actual_diff > 0)))
    over_losses = int(np.sum(over_mask & (actual_diff < 0)))
    over_pushes = int(np.sum(over_mask & is_push))

    under_wins = int(np.sum(under_mask & (actual_diff < 0)))
    under_losses = int(np.sum(under_mask & (actual_diff > 0)))
    under_pushes = int(np.sum(under_mask & is_push))

    return {
        "directional_accuracy": directional_accuracy,
        "avg_clv": avg_clv,
        "n_evaluable": int(n_evaluable),
        "over_record": f"{over_wins}-{over_losses}-{over_pushes}",
        "under_record": f"{under_wins}-{under_losses}-{under_pushes}",
        "n_over": int(np.sum(over_mask)),
        "n_under": int(np.sum(under_mask)),
        "n_push_on_line": int(np.sum(has_signal & is_push)),
    }


def simulate_betting(
    y_pred: np.ndarray,
    closing_line: np.ndarray,
    y_true: np.ndarray,
    min_edge: float = 1.0,
    kelly_frac: float = 0.25,
    juice: float = -110,
) -> dict:
    """Simulate betting P&L on totals.

    Bets over when model total > closing_line + min_edge, under when
    model total < closing_line - min_edge. Flat 1-unit bets at standard
    -110 juice.

    Args:
        y_pred: Model predicted totals.
        closing_line: SBRO closing totals.
        y_true: Actual game totals.
        min_edge: Minimum points difference to place a bet.
        kelly_frac: Not used for flat betting, reserved for future.
        juice: American odds (default -110).

    Returns:
        Dict with n_bets, wins, losses, pushes, profit, roi, win_rate,
        max_drawdown.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    closing_line = np.asarray(closing_line, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    model_edge = y_pred - closing_line
    actual_diff = y_true - closing_line

    # Convert juice to decimal payout
    if juice < 0:
        win_payout = 100 / abs(juice)  # e.g., -110 -> 0.909
    else:
        win_payout = juice / 100

    bets = np.abs(model_edge) >= min_edge
    if not np.any(bets):
        return {
            "n_bets": 0, "wins": 0, "losses": 0, "pushes": 0,
            "profit": 0.0, "roi": 0.0, "win_rate": 0.0,
            "max_drawdown": 0.0,
        }

    # For each bet, determine outcome
    profits = []
    wins = losses = pushes = 0
    for i in np.where(bets)[0]:
        if actual_diff[i] == 0:
            # Push on closing line
            profits.append(0.0)
            pushes += 1
        elif (model_edge[i] > 0 and actual_diff[i] > 0) or (
            model_edge[i] < 0 and actual_diff[i] < 0
        ):
            # Win
            profits.append(win_payout)
            wins += 1
        else:
            # Loss
            profits.append(-1.0)
            losses += 1

    profits = np.array(profits)
    cumulative = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    total_profit = float(np.sum(profits))
    n_bets = int(np.sum(bets))

    return {
        "n_bets": n_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "profit": total_profit,
        "roi": total_profit / n_bets if n_bets > 0 else 0.0,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
        "max_drawdown": max_drawdown,
    }


def calibration_by_bucket(
    y_pred: np.ndarray,
    closing_line: np.ndarray,
    y_true: np.ndarray,
    n_buckets: int = 5,
) -> list[dict]:
    """Check if bigger model edge → bigger actual edge.

    Buckets games by |model_edge| and checks whether the average actual
    outcome aligns with the model's direction in each bucket.

    Args:
        y_pred: Model predicted totals.
        closing_line: SBRO closing totals.
        y_true: Actual game totals.
        n_buckets: Number of quantile buckets.

    Returns:
        List of dicts per bucket with avg_model_edge, avg_actual_diff,
        directional_accuracy, n_games.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    closing_line = np.asarray(closing_line, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    model_edge = y_pred - closing_line
    actual_diff = y_true - closing_line

    # Only games where model disagrees with line
    has_signal = model_edge != 0
    abs_edge = np.abs(model_edge)

    idx = np.where(has_signal)[0]
    if len(idx) < n_buckets:
        return []

    # Quantile bucket boundaries
    edges = np.quantile(abs_edge[idx], np.linspace(0, 1, n_buckets + 1))

    results = []
    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        if b == n_buckets - 1:
            mask = has_signal & (abs_edge >= lo) & (abs_edge <= hi)
        else:
            mask = has_signal & (abs_edge >= lo) & (abs_edge < hi)

        bucket_idx = np.where(mask)[0]
        if len(bucket_idx) == 0:
            continue

        signed_actual = np.where(
            model_edge[bucket_idx] > 0,
            actual_diff[bucket_idx],
            -actual_diff[bucket_idx],
        )
        correct = signed_actual > 0
        pushes = actual_diff[bucket_idx] == 0

        evaluable = ~pushes
        n_eval = int(np.sum(evaluable))
        acc = float(np.sum(correct[evaluable]) / n_eval) if n_eval > 0 else 0.0

        results.append({
            "bucket": b + 1,
            "edge_range": f"{lo:.1f}-{hi:.1f}",
            "avg_model_edge": float(np.mean(abs_edge[bucket_idx])),
            "avg_actual_diff": float(np.mean(signed_actual)),
            "directional_accuracy": acc,
            "n_games": len(bucket_idx),
        })

    return results


def significance_test(wins: int, losses: int, break_even: float = 0.524) -> dict:
    """Binomial test: is the win rate significantly above break-even?

    At -110 juice, break-even is 52.4%.

    Args:
        wins: Number of winning bets.
        losses: Number of losing bets.
        break_even: Break-even win rate (default 0.524 for -110).

    Returns:
        Dict with observed_rate, p_value, significant_95, significant_99,
        n_total.
    """
    n = wins + losses
    if n == 0:
        return {
            "observed_rate": 0.0,
            "p_value": 1.0,
            "significant_95": False,
            "significant_99": False,
            "n_total": 0,
        }

    observed_rate = wins / n
    # One-sided binomial test: is the true rate > break_even?
    result = stats.binomtest(wins, n, break_even, alternative="greater")
    p_value = float(result.pvalue)

    return {
        "observed_rate": observed_rate,
        "p_value": p_value,
        "significant_95": p_value < 0.05,
        "significant_99": p_value < 0.01,
        "n_total": n,
    }
