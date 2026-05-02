"""Pipeline de preprocesamiento en 5 etapas, sin data leakage.

Cada función se aplica idénticamente a train y test, pero los estadísticos
(mediana, moda, mapeos de target encoding, parámetros de escalado) se
estiman EXCLUSIVAMENTE sobre train.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import (
    CARDINALITY_THRESHOLD,
    EDU_ORDER,
    ORDINAL_VARS,
    RANDOM_STATE,
)


@dataclass
class PreprocessArtifacts:
    """Objetos ajustados durante el preprocesamiento (útil para inferencia futura)."""
    num_imputer: SimpleImputer
    cat_imputer: SimpleImputer
    ordinal_encoder: OrdinalEncoder
    target_encoders: dict[str, pd.Series]
    target_global_mean: float
    one_hot_encoder: OneHotEncoder
    scaler: StandardScaler
    num_cols: list[str]
    low_card_cols: list[str]
    high_card_cols: list[str]
    nominal_high_cols: list[str]
    one_hot_cols: list[str]


def split_columns_by_type(
    X: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Devuelve (numéricas, baja cardinalidad, alta cardinalidad, nominales alta)."""
    cat_vars = X.select_dtypes(include="object").columns.tolist()
    num_vars = X.select_dtypes(exclude=["object"]).columns.tolist()
    low_card = [c for c in cat_vars if X[c].nunique() <= CARDINALITY_THRESHOLD]
    high_card = [c for c in cat_vars if X[c].nunique() > CARDINALITY_THRESHOLD]
    nominal_high = [c for c in high_card if c not in ORDINAL_VARS]
    return num_vars, low_card, high_card, nominal_high


def impute_missing(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer, SimpleImputer]:
    """Mediana para numéricas (robusta a centinelas como 99999) y moda para categóricas."""
    num_imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
    return X_train, X_test, num_imputer, cat_imputer


def encode_high_cardinality(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    nominal_high_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder, dict[str, pd.Series], float]:
    """OrdinalEncoder para AHGA (educación, jerárquica) + Target Encoding para el resto.

    Target encoding: cada categoría se reemplaza por P(income > 50k | categoría),
    calculada solo con y_train. Las categorías no vistas en test caen al global_mean.
    """
    ord_enc = OrdinalEncoder(
        categories=[EDU_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    X_train["AHGA"] = ord_enc.fit_transform(X_train[["AHGA"]])
    X_test["AHGA"] = ord_enc.transform(X_test[["AHGA"]])

    global_mean = float(y_train.mean())
    target_encoders: dict[str, pd.Series] = {}
    for col in nominal_high_cols:
        means = y_train.groupby(X_train[col]).mean()
        target_encoders[col] = means
        X_train[col] = X_train[col].map(means)
        X_test[col] = X_test[col].map(means).fillna(global_mean)

    return X_train, X_test, ord_enc, target_encoders, global_mean


def encode_low_cardinality(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    low_card_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, list[str]]:
    """OneHotEncoder denso con `handle_unknown='ignore'` para robustez en producción."""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_arr = ohe.fit_transform(X_train[low_card_cols])
    test_arr = ohe.transform(X_test[low_card_cols])
    cols = ohe.get_feature_names_out(low_card_cols).tolist()

    train_df = pd.DataFrame(train_arr, columns=cols, index=X_train.index)
    test_df = pd.DataFrame(test_arr, columns=cols, index=X_test.index)

    X_train = pd.concat([X_train.drop(columns=low_card_cols), train_df], axis=1)
    X_test = pd.concat([X_test.drop(columns=low_card_cols), test_df], axis=1)
    return X_train, X_test, ohe, cols


def scale_numeric(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """StandardScaler — crítico para Logistic Regression (regularización en escala)."""
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, scaler


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_neighbors: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversampling de la minoría — SOLO en train, nunca en test."""
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train.values, y_train)
    return X_res, np.asarray(y_res)


def preprocess_full(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, PreprocessArtifacts]:
    """Pipeline completo: imputa, codifica, escala y aplica SMOTE.

    Returns
    -------
    X_train_res : np.ndarray
        Train balanceado (post-SMOTE), listo para entrenar.
    y_train_res : np.ndarray
        Target balanceado (post-SMOTE).
    X_test_proc : pd.DataFrame
        Test preprocesado (sin SMOTE, conserva la distribución natural 6.3%/93.7%).
    artifacts : PreprocessArtifacts
        Encoders y scalers ajustados (para reproducir transformaciones futuras).
    """
    X_train_p = X_train.copy()
    X_test_p = X_test.copy()

    num_cols, low_card, high_card, nominal_high = split_columns_by_type(X_train_p)
    cat_cols = low_card + high_card

    X_train_p, X_test_p, num_imp, cat_imp = impute_missing(
        X_train_p, X_test_p, num_cols, cat_cols
    )
    X_train_p, X_test_p, ord_enc, tgt_enc, gmean = encode_high_cardinality(
        X_train_p, X_test_p, y_train, nominal_high
    )
    X_train_p, X_test_p, ohe, ohe_cols = encode_low_cardinality(
        X_train_p, X_test_p, low_card
    )
    X_train_p, X_test_p, scaler = scale_numeric(X_train_p, X_test_p, num_cols)

    X_train_res, y_train_res = apply_smote(X_train_p, y_train)

    artifacts = PreprocessArtifacts(
        num_imputer=num_imp,
        cat_imputer=cat_imp,
        ordinal_encoder=ord_enc,
        target_encoders=tgt_enc,
        target_global_mean=gmean,
        one_hot_encoder=ohe,
        scaler=scaler,
        num_cols=num_cols,
        low_card_cols=low_card,
        high_card_cols=high_card,
        nominal_high_cols=nominal_high,
        one_hot_cols=ohe_cols,
    )
    return X_train_res, y_train_res, X_test_p, artifacts
