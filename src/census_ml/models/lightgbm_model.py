"""LightGBM con optimización de hiperparámetros usando Optuna (100 trials, 5-fold CV).

La métrica optimizada es AUC-PR (average_precision_score), que es la métrica
honesta para datasets con desbalance severo (6.3% de la clase positiva).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from ..config import RANDOM_STATE


def _build_objective(X_train: np.ndarray, y_train: np.ndarray, cv_splits: int):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "binary",
            "boosting_type":     "gbdt",
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            model = LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            y_prob = model.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, y_prob))
        return float(np.mean(scores))

    return objective


def train_lightgbm_tuned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 100,
    cv_splits: int = 5,
    show_progress: bool = True,
) -> tuple[LGBMClassifier, optuna.Study]:
    """Optimiza LightGBM con Optuna y reentrena con los mejores parámetros.

    Devuelve (modelo_final, study) — el `study` es serializable y permite
    inspeccionar la historia de búsqueda y los importance scores.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = _build_objective(X_train, y_train, cv_splits)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    best_model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **study.best_params,
    )
    best_model.fit(X_train, y_train)
    return best_model, study
