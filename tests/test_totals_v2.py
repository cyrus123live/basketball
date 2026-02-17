"""Tests for Phase 2: name matching, XGBoost wrapper, Barttorvik features."""

import numpy as np
import pandas as pd
import pytest

from src.utils.name_match import normalize_name
from src.models.xgb_wrapper import XGBEarlyStopping, XGB_CONFIGS
from src.models.totals_data_v2 import (
    _add_barttorvik_features,
    BART_FEATURES,
    EWMA_FEATURES,
    ALL_FEATURES,
)


class TestNormalizeName:
    """Test that normalize_name bridges Barttorvik and ESPN naming conventions."""

    @pytest.mark.parametrize("bart_name,espn_name", [
        ("Michigan St.", "Michigan State Spartans"),
        ("N.C. State", "NC State Wolfpack"),
        ("UConn", "Connecticut Huskies"),
        ("UNLV", "UNLV Rebels"),
        ("UCF", "UCF Knights"),
        ("Alabama", "Alabama Crimson Tide"),
        ("Duke", "Duke Blue Devils"),
        ("Kansas", "Kansas Jayhawks"),
        ("Gonzaga", "Gonzaga Bulldogs"),
        ("LSU", "LSU Tigers"),
        ("SMU", "SMU Mustangs"),
        ("VCU", "VCU Rams"),
        ("Pitt", "Pittsburgh Panthers"),
        ("Ole Miss", "Ole Miss Rebels"),
        ("App State", "Appalachian State Mountaineers"),
    ])
    def test_name_pairs_converge(self, bart_name, espn_name):
        """Barttorvik and ESPN names should normalize to similar strings."""
        from thefuzz import fuzz
        bart_norm = normalize_name(bart_name)
        espn_norm = normalize_name(espn_name)
        score = fuzz.token_sort_ratio(bart_norm, espn_norm)
        assert score >= 80, (
            f"normalize_name mismatch: '{bart_name}' -> '{bart_norm}' vs "
            f"'{espn_name}' -> '{espn_norm}' (score={score})"
        )

    def test_abbreviation_expansion(self):
        assert "state" in normalize_name("Michigan St.")
        assert "state" not in normalize_name("Michigan")

    def test_mascot_removal(self):
        norm = normalize_name("Duke Blue Devils")
        assert "blue" not in norm
        assert "devils" not in norm
        assert "duke" in norm

    def test_parenthetical_removal(self):
        norm = normalize_name("Miami (OH) RedHawks")
        assert "oh" not in norm or normalize_name("Miami (FL) Hurricanes") != norm

    def test_unicode_handling(self):
        norm = normalize_name("José State")
        assert "jose" in norm

    def test_empty_string(self):
        assert normalize_name("") == ""
        assert normalize_name("  ") == ""


class TestXGBWrapper:
    """Test XGBEarlyStopping sklearn-compatible wrapper."""

    def test_fit_predict_basic(self):
        """Wrapper should train and produce predictions."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.randn(200) * 0.5

        model = XGBEarlyStopping(
            val_frac=0.2,
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            early_stopping_rounds=10,
            random_state=42,
        )
        model.fit(X, y)
        preds = model.predict(X)

        assert preds.shape == (200,)
        assert model.model is not None
        assert model.best_iteration_ is not None

    def test_early_stopping_activates(self):
        """Early stopping should trigger before max n_estimators."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 3)
        y = X[:, 0] * 2 + np.random.randn(n) * 0.1  # easy signal

        model = XGBEarlyStopping(
            val_frac=0.2,
            max_depth=3,
            learning_rate=0.1,
            n_estimators=500,
            early_stopping_rounds=20,
            random_state=42,
        )
        model.fit(X, y)

        # Early stopping should kick in well before 500 iterations
        assert model.best_iteration_ < 500, (
            f"Expected early stopping, got best_iteration={model.best_iteration_}"
        )

    def test_small_dataset_no_crash(self):
        """Small datasets should train without early stopping split."""
        X = np.random.randn(15, 3)
        y = np.random.randn(15)

        model = XGBEarlyStopping(
            val_frac=0.2,
            max_depth=2,
            n_estimators=10,
            early_stopping_rounds=5,
            random_state=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 15

    def test_feature_importances(self):
        """Feature importance should return sorted dict."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = X[:, 0] * 5 + np.random.randn(200) * 0.1

        model = XGBEarlyStopping(
            max_depth=3, n_estimators=50, random_state=42,
        )
        model.fit(X, y)

        imp = model.feature_importances(["a", "b", "c"])
        assert len(imp) == 3
        # Feature 'a' should be most important (highest coefficient)
        assert list(imp.keys())[0] == "a"

    def test_configs_exist(self):
        """Pre-configured XGB configs should be accessible."""
        assert "XGB-conservative" in XGB_CONFIGS
        assert "XGB-deeper" in XGB_CONFIGS
        assert "XGB-fast" in XGB_CONFIGS
        for name, params in XGB_CONFIGS.items():
            assert "max_depth" in params
            assert "n_estimators" in params


class TestBarttorvikFeatures:
    """Test Barttorvik feature computation."""

    def test_features_computed(self):
        """Combined features should be computed from raw columns."""
        df = pd.DataFrame({
            "home_bart_adj_t": [68.0, 70.0],
            "away_bart_adj_t": [66.0, 72.0],
            "home_bart_adj_o": [110.0, 115.0],
            "away_bart_adj_o": [105.0, 108.0],
            "home_bart_adj_d": [95.0, 98.0],
            "away_bart_adj_d": [100.0, 102.0],
            "home_bart_barthag": [0.85, 0.90],
            "away_bart_barthag": [0.75, 0.80],
        })
        result = _add_barttorvik_features(df)

        assert result["bart_avg_adj_t"].iloc[0] == 67.0
        assert result["bart_avg_adj_o"].iloc[0] == 107.5
        assert result["bart_avg_adj_d"].iloc[0] == 97.5
        assert result["bart_avg_barthag"].iloc[0] == 0.80
        assert result["bart_adj_t_diff"].iloc[0] == 2.0
        # Second row
        assert result["bart_avg_adj_t"].iloc[1] == 71.0
        assert result["bart_adj_t_diff"].iloc[1] == 2.0

    def test_null_handling(self):
        """NaN should propagate when a team lacks prior-season data."""
        df = pd.DataFrame({
            "home_bart_adj_t": [68.0, np.nan],
            "away_bart_adj_t": [66.0, 72.0],
            "home_bart_adj_o": [110.0, np.nan],
            "away_bart_adj_o": [105.0, 108.0],
            "home_bart_adj_d": [95.0, np.nan],
            "away_bart_adj_d": [100.0, 102.0],
            "home_bart_barthag": [0.85, np.nan],
            "away_bart_barthag": [0.75, 0.80],
        })
        result = _add_barttorvik_features(df)

        # First row should be fine
        assert not np.isnan(result["bart_avg_adj_t"].iloc[0])
        # Second row should have NaN (home team missing)
        assert np.isnan(result["bart_avg_adj_t"].iloc[1])
        assert np.isnan(result["bart_avg_adj_o"].iloc[1])
        assert np.isnan(result["bart_avg_barthag"].iloc[1])

    def test_feature_lists_consistent(self):
        """Feature list lengths should add up."""
        assert len(EWMA_FEATURES) == 13
        assert len(BART_FEATURES) == 5
        assert len(ALL_FEATURES) == 18
        assert ALL_FEATURES == EWMA_FEATURES + BART_FEATURES


class TestWalkForwardCustomBuilders:
    """Test that walk_forward_validate accepts custom builders."""

    def test_custom_builders_called(self):
        """Mock builders should be invoked instead of defaults."""
        from unittest.mock import MagicMock, patch

        # Create mock builders that return DataFrames with expected columns
        train_df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "actual_total": [140, 150, 160, 145, 155],
        })
        eval_df = pd.DataFrame({
            "feat1": [2.5, 3.5],
            "actual_total": [148, 158],
            "total_close": [145, 155],
        })

        mock_train_builder = MagicMock(return_value=train_df)
        mock_eval_builder = MagicMock(return_value=eval_df)

        from src.models.walk_forward import walk_forward_validate

        result = walk_forward_validate(
            db=None,  # not used when builders are mocked
            model_factory=lambda: _SimpleModel(),
            feature_names=["feat1"],
            train_seasons_list=[[2016, 2017]],
            val_seasons=[2018],
            train_builder=mock_train_builder,
            eval_builder=mock_eval_builder,
        )

        mock_train_builder.assert_called_once_with(None, [2016, 2017])
        mock_eval_builder.assert_called_once_with(None, [2018])
        assert len(result.folds) == 1

    def test_default_builders_backward_compatible(self):
        """Without custom builders, should use the original functions."""
        from src.models.walk_forward import walk_forward_validate
        import inspect

        sig = inspect.signature(walk_forward_validate)
        assert "train_builder" in sig.parameters
        assert "eval_builder" in sig.parameters
        assert sig.parameters["train_builder"].default is None
        assert sig.parameters["eval_builder"].default is None


class _SimpleModel:
    """Minimal sklearn-compatible model for testing."""
    def fit(self, X, y):
        self._mean = np.mean(y)
        return self

    def predict(self, X):
        return np.full(len(X), self._mean)
