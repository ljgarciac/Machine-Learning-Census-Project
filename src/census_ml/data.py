"""Carga del dataset Census Income KDD desde UCI y división train/test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from .config import RANDOM_STATE, TEST_SIZE, UCI_DATASET_ID


def load_census_data() -> tuple[pd.DataFrame, pd.Series]:
    """Descarga el Census Income KDD desde UCI y devuelve (X, y_binary).

    El target original viene como string ('-50000', '50000+') y se convierte
    a binario: 1 si '50000+', 0 en otro caso. También limpia el centinela
    ' ?' del censo reemplazándolo por NaN.
    """
    repo = fetch_ucirepo(id=UCI_DATASET_ID)
    X = repo.data.features.copy()
    y = repo.data.targets.copy()

    df = pd.concat([X, y], axis=1)
    df.columns = list(X.columns) + ["income"]
    df.replace(" ?", np.nan, inplace=True)
    df["income"] = df["income"].str.strip().str.rstrip(".")

    y_binary = df["income"].apply(lambda v: 1 if v == "50000+" else 0)
    X_features = df.drop(columns=["income"])
    return X_features, y_binary


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Wrapper de `train_test_split` con los defaults del proyecto."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
