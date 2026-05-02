"""Carga los modelos serializados, calcula las 6 métricas y arma los ensambles.

Imprime una tabla comparativa con los 4 modelos individuales + Soft Voting +
Weighted Average. No reentrena nada — es la verificación rápida del pipeline.

Uso:
    uv run python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from census_ml.config import MODEL_FILES, MODELS_DIR
from census_ml.data import load_census_data, split_train_test
from census_ml.ensemble import soft_voting, weighted_average_by_aucpr
from census_ml.evaluation import comparison_table, compute_metrics
from census_ml.preprocessing import preprocess_full


def main() -> None:
    print("[1/4] Cargando dataset y reproduciendo split + preprocesamiento...")
    X, y = load_census_data()
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    _, _, X_test_proc, _ = preprocess_full(X_train, X_test, y_train)

    print("[2/4] Cargando modelos serializados...")
    models = {}
    for name, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        if not path.exists():
            print(f"      [SKIP] {name}: {path} no existe.")
            continue
        models[name] = joblib.load(path)
        print(f"      OK {name}")

    if not models:
        print("\nNo hay modelos serializados. Corre `python scripts/train_all.py` primero.")
        return

    print("[3/4] Generando predicciones individuales...")
    predictions = {}
    for name, model in models.items():
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        y_pred = model.predict(X_test_proc)
        predictions[name] = {"y_prob": y_prob, "y_pred": y_pred}

    print("[4/4] Calculando métricas y ensambles...\n")
    individual = comparison_table(predictions, y_test.values)

    if len(predictions) >= 2:
        soft_prob, soft_pred = soft_voting(predictions)
        soft_metrics = compute_metrics(y_test.values, soft_pred, soft_prob)

        w_prob, w_pred, weights = weighted_average_by_aucpr(
            predictions, individual["AUC-PR"]
        )
        w_metrics = compute_metrics(y_test.values, w_pred, w_prob)

        print("Pesos del Weighted Average (proporcionales al AUC-PR):")
        for name, w in weights.items():
            print(f"  {name:<25}  AUC-PR={individual.loc[name, 'AUC-PR']:.4f}  ->  peso={w:.4f}")

        ensembles = individual.copy()
        ensembles.loc["Ensemble — Soft Voting"] = soft_metrics
        ensembles.loc["Ensemble — Weighted Average"] = w_metrics
    else:
        ensembles = individual

    print("\nTabla comparativa (test set):\n")
    print(ensembles.to_string())


if __name__ == "__main__":
    main()
