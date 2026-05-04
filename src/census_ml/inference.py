"""Aplica el preprocesamiento aprendido a casos nuevos (sin SMOTE).

`preprocess_full` retorna `PreprocessArtifacts` con todos los encoders/scalers
ajustados sobre train. Este módulo los reutiliza para transformar inputs nuevos
de la misma forma — pieza clave para inferencia (la app Streamlit, futuras APIs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import PreprocessArtifacts


def apply_preprocessing(
    X_new: pd.DataFrame,
    artifacts: PreprocessArtifacts,
) -> pd.DataFrame:
    """Transforma `X_new` (1+ filas) usando los artifacts aprendidos en train.

    Aplica las mismas 4 etapas (imputación → encoding alta → encoding baja →
    escalado) en el mismo orden y con los mismos objetos ajustados. **No** aplica
    SMOTE — eso solo va en train.
    """
    X = X_new.copy()
    cat_cols = artifacts.low_card_cols + artifacts.high_card_cols

    X[artifacts.num_cols] = artifacts.num_imputer.transform(X[artifacts.num_cols])
    X[cat_cols] = artifacts.cat_imputer.transform(X[cat_cols])

    X["AHGA"] = artifacts.ordinal_encoder.transform(X[["AHGA"]])
    for col in artifacts.nominal_high_cols:
        means = artifacts.target_encoders[col]
        X[col] = X[col].map(means).fillna(artifacts.target_global_mean)

    arr = artifacts.one_hot_encoder.transform(X[artifacts.low_card_cols])
    df_oh = pd.DataFrame(arr, columns=artifacts.one_hot_cols, index=X.index)
    X = pd.concat([X.drop(columns=artifacts.low_card_cols), df_oh], axis=1)

    X[artifacts.num_cols] = artifacts.scaler.transform(X[artifacts.num_cols])
    return X


def predict_with_ensemble(
    X_proc: pd.DataFrame,
    models: dict,
    weights: pd.Series,
) -> dict[str, np.ndarray]:
    """Devuelve la probabilidad de cada modelo individual + el ensamble ponderado.

    `weights` debe sumar 1; típicamente vienen de `weighted_average_by_aucpr`.
    Las claves del dict resultante son "<model_name>" para los individuales y
    "Ensemble" para el promedio ponderado.
    """
    out: dict[str, np.ndarray] = {}
    ensemble = np.zeros(len(X_proc))
    for name, model in models.items():
        prob = model.predict_proba(X_proc)[:, 1]
        out[name] = prob
        ensemble += weights[name] * prob
    out["Ensemble"] = ensemble
    return out
