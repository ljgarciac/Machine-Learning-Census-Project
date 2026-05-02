"""Random Forest con tuning vía RandomizedSearchCV + threshold tuning post-hoc."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV

from ..config import RANDOM_STATE

PARAM_GRID: dict[str, list] = {
    "max_depth":         [10, 15, 20],
    "min_samples_split": [10, 20, 30],
    "min_samples_leaf":  [5, 10, 15],
    "max_features":      ["sqrt", "log2"],
}


def train_random_forest_tuned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "f1",
    random_state: int = RANDOM_STATE,
    verbose: int = 2,
) -> tuple[RandomForestClassifier, dict]:
    """Random Forest con búsqueda aleatoria de hiperparámetros.

    Devuelve (mejor_estimador, mejores_parametros).
    """
    base = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=PARAM_GRID,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_threshold_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Encuentra el threshold que maximiza F1 sobre la curva PR.

    Devuelve (threshold_optimo, predicciones_binarias).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores[:-1]))
    best_threshold = float(thresholds[best_idx])
    y_pred = (y_prob >= best_threshold).astype(int)
    return best_threshold, y_pred
