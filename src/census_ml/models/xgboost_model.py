"""XGBoost con configuración fija (sin tuning explícito)."""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from ..config import RANDOM_STATE


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 600,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    """XGBoost con la configuración del proyecto (600 árboles, depth=8, lr=0.05).

    A pesar de no tener tuning con Optuna, en este dataset supera a LightGBM
    tuneado en F1, AUC-PR y AUC-ROC.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model
