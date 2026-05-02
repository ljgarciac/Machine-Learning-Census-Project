"""Métricas, gráficos comparativos y tabla de resultados."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


METRIC_COLS: list[str] = [
    "Accuracy", "Precision", "Recall", "F1-Score", "AUC-PR", "AUC-ROC",
]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Devuelve las 6 métricas estándar como dict redondeado a 4 decimales."""
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC-PR":    round(average_precision_score(y_true, y_prob), 4),
        "AUC-ROC":   round(roc_auc_score(y_true, y_prob), 4),
    }


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    show_plots: bool = True,
) -> dict[str, float]:
    """Calcula métricas y opcionalmente dibuja matriz de confusión, PR y ROC."""
    metrics = compute_metrics(y_true, y_pred, y_prob)

    if show_plots:
        plot_confusion(name, y_true, y_pred)
        plot_precision_recall(name, y_true, y_prob, metrics["AUC-PR"])
        plot_roc(name, y_true, y_prob, metrics["AUC-ROC"])

    print(f"\nMétricas — {name}")
    print(f"  {'Métrica':<12} {'Valor':>8}")
    print("  " + "-" * 22)
    for k, v in metrics.items():
        print(f"  {k:<12} {v:>8.4f}")
    return metrics


def plot_confusion(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50k", ">50k"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de Confusión — {name}", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.show()


def plot_precision_recall(
    name: str, y_true: np.ndarray, y_prob: np.ndarray, auc_pr: float
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
    baseline = float(np.mean(y_true))
    ax.plot(rec_c, prec_c, color="#D4537E", lw=2, label=f"AUC-PR = {auc_pr:.3f}")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1.5,
               label=f"Baseline (no skill) = {baseline:.3f}")
    ax.fill_between(rec_c, prec_c, baseline, alpha=0.08, color="#D4537E")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Curva Precision-Recall — {name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.show()


def plot_roc(name: str, y_true: np.ndarray, y_prob: np.ndarray, auc_roc: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, color="#1D9E75", lw=2, label=f"AUC-ROC = {auc_roc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Aleatorio")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1D9E75")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"Curva ROC — {name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.show()


def comparison_table(
    predictions: dict[str, dict[str, np.ndarray]],
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Tabla de métricas para todos los modelos en `predictions`.

    `predictions` debe tener forma {nombre: {"y_pred": ..., "y_prob": ...}}.
    """
    rows = []
    for name, p in predictions.items():
        rows.append({"Modelo": name, **compute_metrics(y_test, p["y_pred"], p["y_prob"])})
    return pd.DataFrame(rows).set_index("Modelo").round(4)
