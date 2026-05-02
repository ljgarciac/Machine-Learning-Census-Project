"""Reproduce el pipeline completo end-to-end y guarda los modelos en `models/`.

Uso:
    uv run python scripts/train_all.py                  # entrena todo (lento)
    uv run python scripts/train_all.py --skip-lightgbm  # reusa lgbm_tuned.pkl

El paso más costoso es LightGBM con Optuna (100 trials × 5-fold CV); el flag
`--skip-lightgbm` permite recrear los demás modelos sin re-correr esa búsqueda.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from census_ml.config import MODEL_FILES, MODELS_DIR, LGBM_STUDY_FILE
from census_ml.data import load_census_data, split_train_test
from census_ml.models import (
    train_lightgbm_tuned,
    train_logistic_elasticnet,
    train_random_forest_tuned,
    train_xgboost,
)
from census_ml.preprocessing import preprocess_full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-lightgbm", action="store_true",
                        help="No re-correr Optuna; reusa el .pkl existente.")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Trials de Optuna para LightGBM (default 100).")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Cargando dataset Census Income KDD desde UCI...")
    X, y = load_census_data()
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    print("[2/5] Preprocesando (imputación → encoding → escalado → SMOTE)...")
    X_train_res, y_train_res, _, _ = preprocess_full(X_train, X_test, y_train)
    print(f"      Train balanceado: {X_train_res.shape}")

    print("[3/5] Entrenando Logistic Regression ElasticNet...")
    t0 = time.time()
    logreg = train_logistic_elasticnet(X_train_res, y_train_res)
    joblib.dump(logreg, MODELS_DIR / MODEL_FILES["LogReg ElasticNet"])
    print(f"      OK ({time.time() - t0:.1f}s)")

    print("[3/5] Entrenando Random Forest (RandomizedSearchCV, 20 iter × 3-fold)...")
    t0 = time.time()
    rf, best_params = train_random_forest_tuned(X_train_res, y_train_res, verbose=1)
    joblib.dump(rf, MODELS_DIR / MODEL_FILES["Random Forest (Tuned)"])
    print(f"      OK ({time.time() - t0:.1f}s) — best_params={best_params}")

    print("[4/5] Entrenando XGBoost (config fija: 600 árboles, depth=8)...")
    t0 = time.time()
    xgb = train_xgboost(X_train_res, y_train_res)
    joblib.dump(xgb, MODELS_DIR / MODEL_FILES["XGBoost"])
    print(f"      OK ({time.time() - t0:.1f}s)")

    if args.skip_lightgbm:
        print("[5/5] LightGBM omitido (--skip-lightgbm).")
    else:
        print(f"[5/5] Entrenando LightGBM con Optuna ({args.n_trials} trials × 5-fold CV)...")
        print("      Esto puede tardar varias horas. Usa --skip-lightgbm para omitir.")
        t0 = time.time()
        lgbm, study = train_lightgbm_tuned(X_train_res, y_train_res, n_trials=args.n_trials)
        joblib.dump(lgbm, MODELS_DIR / MODEL_FILES["LightGBM (Tuned)"])
        joblib.dump(study, MODELS_DIR / LGBM_STUDY_FILE)
        print(f"      OK ({time.time() - t0:.1f}s) — best AUC-PR CV: {study.best_value:.4f}")

    print(f"\nListo. Modelos guardados en {MODELS_DIR}/")
    print("Para evaluar y armar los ensambles: uv run python scripts/evaluate.py")


if __name__ == "__main__":
    main()
