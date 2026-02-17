"""Unit tests for totals model evaluation functions.

No database needed — all functions take arrays in, return numbers out.
"""

import numpy as np
import pytest

from src.models.evaluation import (
    calibration_by_bucket,
    clv_metrics,
    regression_metrics,
    significance_test,
    simulate_betting,
)


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([140, 150, 160])
        y_pred = np.array([140, 150, 160])
        m = regression_metrics(y_true, y_pred)
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == 1.0
        assert m["bias"] == 0.0

    def test_constant_offset(self):
        y_true = np.array([140, 150, 160])
        y_pred = np.array([145, 155, 165])  # +5 bias
        m = regression_metrics(y_true, y_pred)
        assert m["mae"] == 5.0
        assert m["rmse"] == 5.0
        assert m["bias"] == 5.0

    def test_symmetric_errors(self):
        y_true = np.array([100, 100, 100, 100])
        y_pred = np.array([90, 110, 95, 105])  # mean error = 0
        m = regression_metrics(y_true, y_pred)
        assert abs(m["bias"]) < 1e-10
        assert m["mae"] == 7.5

    def test_r2_less_than_one(self):
        y_true = np.array([140, 150, 160, 170])
        y_pred = np.array([145, 148, 162, 168])
        m = regression_metrics(y_true, y_pred)
        assert 0 < m["r2"] < 1.0


class TestClvMetrics:
    def test_perfect_clv(self):
        """Model always picks the right side."""
        y_pred = np.array([155, 145, 160, 140])
        closing = np.array([150, 150, 150, 150])
        y_true = np.array([160, 140, 170, 130])
        m = clv_metrics(y_pred, closing, y_true)
        assert m["directional_accuracy"] == 1.0
        assert m["avg_clv"] > 0

    def test_zero_clv(self):
        """Model always picks the wrong side."""
        y_pred = np.array([155, 145])
        closing = np.array([150, 150])
        y_true = np.array([140, 160])  # opposite of model
        m = clv_metrics(y_pred, closing, y_true)
        assert m["directional_accuracy"] == 0.0
        assert m["avg_clv"] < 0

    def test_push_excluded_from_accuracy(self):
        """Pushes (actual == line) shouldn't count in directional accuracy."""
        y_pred = np.array([155, 145, 160])
        closing = np.array([150, 150, 150])
        y_true = np.array([160, 150, 170])  # middle game is a push
        m = clv_metrics(y_pred, closing, y_true)
        # Only 2 evaluable games (game 0 and 2), both correct
        assert m["n_evaluable"] == 2
        assert m["directional_accuracy"] == 1.0
        assert m["n_push_on_line"] == 1

    def test_over_under_records(self):
        y_pred = np.array([155, 145, 155, 145])
        closing = np.array([150, 150, 150, 150])
        y_true = np.array([160, 140, 140, 160])
        # Game 0: over signal, went over -> W
        # Game 1: under signal, went under -> W
        # Game 2: over signal, went under -> L
        # Game 3: under signal, went over -> L
        m = clv_metrics(y_pred, closing, y_true)
        assert m["over_record"] == "1-1-0"
        assert m["under_record"] == "1-1-0"
        assert m["directional_accuracy"] == 0.5


class TestSimulateBetting:
    def test_no_bets_below_edge(self):
        y_pred = np.array([150.5, 149.5])
        closing = np.array([150, 150])
        y_true = np.array([160, 140])
        m = simulate_betting(y_pred, closing, y_true, min_edge=1.0)
        assert m["n_bets"] == 0

    def test_all_wins(self):
        y_pred = np.array([155, 145])
        closing = np.array([150, 150])
        y_true = np.array([160, 140])  # both correct
        m = simulate_betting(y_pred, closing, y_true, min_edge=1.0)
        assert m["n_bets"] == 2
        assert m["wins"] == 2
        assert m["losses"] == 0
        assert m["profit"] > 0
        assert m["roi"] > 0
        assert m["win_rate"] == 1.0

    def test_all_losses(self):
        y_pred = np.array([155, 145])
        closing = np.array([150, 150])
        y_true = np.array([140, 160])  # both wrong
        m = simulate_betting(y_pred, closing, y_true, min_edge=1.0)
        assert m["n_bets"] == 2
        assert m["wins"] == 0
        assert m["losses"] == 2
        assert m["profit"] == -2.0
        assert m["roi"] == -1.0

    def test_push_handling(self):
        y_pred = np.array([155.0])
        closing = np.array([150.0])
        y_true = np.array([150.0])  # lands exactly on closing line
        m = simulate_betting(y_pred, closing, y_true, min_edge=1.0)
        assert m["pushes"] == 1
        assert m["profit"] == 0.0

    def test_juice_at_minus_110(self):
        """At -110 juice, a win pays 100/110 = 0.909 units."""
        y_pred = np.array([155.0])
        closing = np.array([150.0])
        y_true = np.array([160.0])
        m = simulate_betting(y_pred, closing, y_true, min_edge=1.0, juice=-110)
        assert m["wins"] == 1
        assert abs(m["profit"] - (100 / 110)) < 0.001

    def test_max_drawdown(self):
        y_pred = np.array([155, 155, 155, 155])
        closing = np.array([150, 150, 150, 150])
        y_true = np.array([160, 140, 140, 160])  # W, L, L, W
        m = simulate_betting(y_pred, closing, y_true, min_edge=1.0)
        assert m["max_drawdown"] > 0


class TestSignificanceTest:
    def test_zero_bets(self):
        m = significance_test(0, 0)
        assert m["p_value"] == 1.0
        assert m["significant_95"] is False

    def test_clearly_significant(self):
        # 600 wins out of 1000 = 60% (well above 52.4%)
        m = significance_test(600, 400)
        assert m["observed_rate"] == 0.6
        assert m["significant_95"] is True
        assert m["significant_99"] is True

    def test_clearly_not_significant(self):
        # 52 wins out of 100 = 52% (barely below 52.4%)
        m = significance_test(52, 48)
        assert m["observed_rate"] == 0.52
        assert m["significant_95"] is False

    def test_borderline(self):
        # 55% on 600 bets should be borderline significant at 95%
        m = significance_test(330, 270)
        assert abs(m["observed_rate"] - 330 / 600) < 0.001
        assert m["n_total"] == 600


class TestCalibrationByBucket:
    def test_monotonic_calibration(self):
        """Bigger edges should correlate with better accuracy."""
        np.random.seed(42)
        n = 500
        closing = np.full(n, 150.0)
        # Model edges from small to large
        model_edge = np.linspace(0.5, 10, n)
        y_pred = closing + model_edge
        # Actual results: more likely to be correct when edge is larger
        noise = np.random.normal(0, 8, n)
        y_true = closing + model_edge * 0.5 + noise  # signal + noise
        buckets = calibration_by_bucket(y_pred, closing, y_true, n_buckets=3)
        assert len(buckets) == 3
        # Each bucket should have n_games > 0
        for b in buckets:
            assert b["n_games"] > 0

    def test_few_games_returns_empty(self):
        y_pred = np.array([155.0])
        closing = np.array([150.0])
        y_true = np.array([160.0])
        buckets = calibration_by_bucket(y_pred, closing, y_true, n_buckets=5)
        assert buckets == []
