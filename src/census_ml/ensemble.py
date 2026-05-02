"""Estrategias de ensamble sobre las probabilidades de los 4 modelos base."""

from __future__ import annotations

import numpy as np
import pandas as pd


def soft_voting(
    predictions: dict[str, dict[str, np.ndarray]],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Promedio aritmético simple de las probabilidades.

    Cada modelo aporta peso 1/N. Apropiado cuando todos los modelos tienen
    desempeño comparable.
    """
    probs = np.array([p["y_prob"] for p in predictions.values()])
    y_prob = probs.mean(axis=0)
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred


def weighted_average_by_aucpr(
    predictions: dict[str, dict[str, np.ndarray]],
    auc_pr_scores: pd.Series,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Promedio ponderado por el AUC-PR de cada modelo.

    Pesos = AUC-PR_i / sum(AUC-PR). Si todos los modelos tuvieran AUC-PR igual,
    colapsa a Soft Voting.

    Devuelve (probabilidades, predicciones, pesos).
    """
    weights = auc_pr_scores / auc_pr_scores.sum()
    n_samples = len(next(iter(predictions.values()))["y_prob"])
    y_prob = np.zeros(n_samples)
    for name, p in predictions.items():
        y_prob += weights[name] * p["y_prob"]
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred, weights
