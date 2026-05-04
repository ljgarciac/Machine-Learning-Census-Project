# CLAUDE.md

Guía para Claude Code al trabajar en este repositorio. El usuario y la documentación están en español — responde y comenta en español.

## Qué es este proyecto

Pipeline de ML para clasificación binaria sobre **Census Income KDD** (UCI id=117) con desbalance severo (6.3% positivos). Entrena 4 modelos (Logistic Regression ElasticNet, Random Forest, XGBoost, LightGBM) y los combina con un **ensamble por promedio ponderado por AUC-PR**. Trabajo académico de la Maestría en IA — no es un producto en producción.

## Comandos esenciales

Todo se ejecuta con **uv**, no con `pip`/`venv` directamente.

```bash
uv sync                               # crea .venv e instala deps
uv sync --group dev                   # incluye jupyterlab + ruff
uv run python scripts/evaluate.py     # carga modelos .pkl y muestra métricas (rápido)
uv run python scripts/train_all.py --skip-lightgbm   # reentrenar todo menos LGBM
uv run python scripts/train_all.py    # reentrenar TODO (lento: Optuna 100 trials)
uv run jupyter lab notebooks/         # abrir el notebook
uv run streamlit run app/streamlit_app.py    # demo Streamlit (3 pestañas)
uv run ruff check src/ scripts/ app/  # linter
```

`scripts/evaluate.py` requiere que `models/*.pkl` existan; si fallan, correr `train_all.py` primero.

## Arquitectura

```
src/census_ml/        # Paquete con todo el pipeline (NO instalable, importado vía sys.path)
  config.py           # RANDOM_STATE=42, paths, EDU_ORDER (jerarquía educativa)
  data.py             # load_census_data() vía ucimlrepo + split_train_test()
  preprocessing.py    # 5 etapas: impute → encode → scale → SMOTE. Función entry: preprocess_full()
  models/             # 1 archivo por modelo, con función train_*() que devuelve modelo entrenado
  evaluation.py       # compute_metrics(), comparison_table(), evaluate_model() con plots
  ensemble.py         # soft_voting() y weighted_average_by_aucpr()
  inference.py        # apply_preprocessing() — reusa artifacts de train para casos nuevos

scripts/              # Orquestación end-to-end (parser CLI con argparse)
app/streamlit_app.py  # Demo interactiva: resumen + métricas + predicción de un caso
notebooks/            # Notebook orquestador — importa desde census_ml, NO duplica lógica
models/*.pkl          # Modelos serializados (commiteados, ~127 MB)
INFORME_PROYECTO.md   # Informe técnico de 213 líneas con justificaciones
```

`pyproject.toml` declara `[tool.uv] package = false` — el código se accede vía `sys.path.insert(0, "src")` en los scripts y el notebook. **No** convertirlo en paquete instalable salvo que el usuario lo pida.

## Convenciones del código

- **Sin data leakage**: cualquier `fit()` de imputers/encoders/scalers se hace SOLO sobre train. SMOTE también solo en train.
- **Las funciones de entrenamiento** (`train_*`) reciben arrays/DataFrames y devuelven el modelo. Las que tunean devuelven `(modelo, params)` o `(modelo, study)`.
- **`compute_metrics()` redondea a 4 decimales**; las tablas usan `.round(4)` para coincidir.
- **Métrica principal**: AUC-PR. NO presentar Accuracy como métrica principal — es engañosa con desbalance 6.3%/93.7%.
- **Idioma**: docstrings y comentarios en español. Identificadores en inglés.

## Modelos serializados — cuidado al cargar

Los `.pkl` (`joblib.dump/load`) requieren la **misma versión exacta** de scikit-learn / xgboost / lightgbm con la que se entrenaron. `uv.lock` fija las versiones. Si una versión cambia y un `.pkl` deja de cargar, **no es bug del código** — es incompatibilidad de pickle. Reentrenar con `train_all.py`.

`models/lgbm_study.pkl` es el `optuna.Study` completo (no un modelo). Útil para inspeccionar la búsqueda; no se usa en `evaluate.py`.

## Pitfalls conocidos

- **`fetch_ucirepo(id=117)` requiere internet** y a veces UCI corta la conexión a mitad de descarga (`http.client.IncompleteRead`). Reintentar.
- **Reentrenar LightGBM toma horas** (100 trials × 5-fold CV). Usar `--skip-lightgbm` en desarrollo.
- **El notebook usa `sys.path.insert(0, str(ROOT / "src"))`** — funciona desde `notebooks/` o desde la raíz, pero si se ejecuta desde otro directorio el import puede fallar.
- **`AHGA` (educación) tiene un orden manual** definido en `config.EDU_ORDER`. Si UCI cambia las categorías, `OrdinalEncoder` tirará error — actualizar la lista.

## Cómo evolucionar el proyecto

- **Agregar un modelo nuevo**: crear `src/census_ml/models/<nombre>.py` con función `train_<nombre>()`, exportarla en `models/__init__.py`, agregarla a `MODEL_FILES` en `config.py`, y a `train_all.py`.
- **Cambiar el preprocesamiento**: modificar `preprocessing.py`. La función pública es `preprocess_full()`; las etapas individuales son reusables.
- **Probar otro ensamble**: agregar función a `ensemble.py` y referenciarla desde `evaluate.py` y el notebook.

## Lo que NO hay que hacer

- No instalar paquetes con `pip install` directamente — usar `uv add <paquete>`.
- No commitear `data/` (dataset descargado) — está en `.gitignore`.
- No re-pickle modelos sin actualizar `uv.lock` con las versiones de las libs.
- No traducir identificadores de columnas del dataset (`AHGA`, `AMJIND`, etc.) — son los nombres oficiales del Census KDD.
- No reescribir el `INFORME_PROYECTO.md` — es la entrega académica oficial; ajustes mínimos solo si el usuario lo pide explícitamente.
