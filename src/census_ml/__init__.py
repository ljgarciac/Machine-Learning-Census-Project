"""Pipeline modular para el proyecto Census Income KDD.

Submódulos principales:
    - data:          carga del dataset desde UCI y división train/test.
    - preprocessing: imputación, encoding (target/ordinal/OHE), escalado, SMOTE.
    - models:        entrenamiento de los 4 modelos (LogReg, RF, XGB, LGBM).
    - evaluation:    métricas, gráficos y tabla comparativa.
    - ensemble:      Soft Voting y Weighted Average ponderado por AUC-PR.
"""

__version__ = "0.1.0"
