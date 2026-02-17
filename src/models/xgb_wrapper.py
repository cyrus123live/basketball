"""Sklearn-compatible XGBoost wrapper with internal early stopping.

walk_forward_validate() calls model.fit(X, y) with no eval_set parameter.
This wrapper splits training data internally (temporal split: first 80% train,
last 20% for early stopping) to enable early stopping without changing the
walk-forward API.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class XGBEarlyStopping:
    """XGBoost regressor with built-in temporal early stopping split.

    Sklearn-compatible: has fit(X, y) and predict(X) methods.
    """

    def __init__(self, val_frac: float = 0.2, **xgb_params):
        """
        Args:
            val_frac: Fraction of training data to hold out for early stopping
                (taken from the end to preserve temporal ordering).
            **xgb_params: Passed to XGBRegressor. Common ones:
                max_depth, learning_rate, n_estimators, subsample,
                colsample_bytree, early_stopping_rounds, random_state.
        """
        self.val_frac = val_frac
        self.xgb_params = xgb_params
        self.model = None
        self.best_iteration_ = None

    def fit(self, X, y):
        """Fit with internal temporal train/val split for early stopping."""
        from xgboost import XGBRegressor

        X = np.asarray(X)
        y = np.asarray(y)

        # Temporal split: first (1-val_frac) for training, last val_frac for stopping
        n = len(X)
        split = int(n * (1 - self.val_frac))

        # Need at least some data in each split
        if split < 10 or (n - split) < 5:
            # Too small for a split — train without early stopping
            params = dict(self.xgb_params)
            params.pop("early_stopping_rounds", None)
            self.model = XGBRegressor(**params)
            self.model.fit(X, y)
            self.best_iteration_ = self.model.n_estimators
            return self

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.model = XGBRegressor(**self.xgb_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        try:
            self.best_iteration_ = self.model.best_iteration
        except AttributeError:
            # No early stopping rounds configured
            self.best_iteration_ = self.xgb_params.get("n_estimators", 100)
        logger.debug(
            "XGB early stopping: best_iteration=%d / %d",
            self.best_iteration_,
            self.xgb_params.get("n_estimators", 100),
        )
        return self

    def predict(self, X):
        """Predict using the fitted model."""
        return self.model.predict(np.asarray(X))

    def feature_importances(self, feature_names: list[str] = None) -> dict:
        """Return feature importances sorted by gain.

        Args:
            feature_names: Optional list of feature names matching X columns.

        Returns:
            Dict of {feature_name: importance} sorted by importance descending.
        """
        if self.model is None:
            return {}

        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]

        pairs = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return dict(pairs)


# Pre-configured XGBoost configs for the backtest matrix
XGB_CONFIGS = {
    "XGB-conservative": {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
        "random_state": 42,
    },
    "XGB-deeper": {
        "max_depth": 4,
        "learning_rate": 0.03,
        "n_estimators": 800,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
        "random_state": 42,
    },
    "XGB-fast": {
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
        "random_state": 42,
    },
}
