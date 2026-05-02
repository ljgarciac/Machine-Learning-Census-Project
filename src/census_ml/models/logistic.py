"""Regresión Logística con regularización ElasticNet (L1 + L2)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..config import RANDOM_STATE


def train_logistic_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Logistic Regression con `penalty='elasticnet'` y solver `saga`.

    Baseline lineal del proyecto. Tiende a tener recall alto (≈0.87) y
    precisión baja por el balanceo SMOTE previo.
    """
    model = LogisticRegression(
        penalty="elasticnet",
        C=C,
        solver="saga",
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
