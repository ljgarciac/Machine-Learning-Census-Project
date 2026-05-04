# Machine Learning Census Project

> Predicción del rango salarial (`income > 50K USD/año`) sobre el dataset **Census Income KDD** mediante el ensamblaje de cuatro modelos ponderados por AUC-PR.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Resumen

Trabajo del curso **Aprendizaje de Máquina** (Maestría en Inteligencia Artificial). Construimos un pipeline reproducible para clasificación binaria sobre un dataset con **fuerte desbalance de clases** (6.3% positivos), entrenamos cuatro modelos con familias distintas (lineal, bagging, dos boosting) y probamos que un **ensamble por promedio ponderado por AUC-PR** mejora a cualquier modelo individual.

| Aspecto | Detalle |
|---|---|
| **Dataset** | [Census Income KDD](https://archive.ics.uci.edu/dataset/117/census+income+kdd) — UCI ML Repository, id=117 |
| **Tarea** | Clasificación binaria: ¿el ingreso anual supera USD 50.000? |
| **Tamaño** | 199.523 registros, 41 variables (13 numéricas + 28 categóricas) |
| **Desbalance** | 93.7% negativos vs 6.3% positivos |
| **Métrica principal** | AUC-PR (más honesta que Accuracy o ROC-AUC con desbalance severo) |

Para la discusión técnica detallada de cada decisión, ver [INFORME_PROYECTO.md](INFORME_PROYECTO.md).

---

## Resultados

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-PR | AUC-ROC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression (ElasticNet) | 0.8488 | 0.2644 | **0.8721** | 0.4058 | 0.5624 | 0.9370 |
| Random Forest (tuned) | 0.9094 | 0.3690 | 0.7468 | 0.4940 | 0.5396 | 0.9378 |
| XGBoost | 0.9530 | 0.6178 | 0.5394 | 0.5759 | 0.6309 | 0.9427 |
| LightGBM (tuned con Optuna) | **0.9555** | **0.6708** | 0.4865 | 0.5639 | 0.6283 | 0.9416 |
| Ensemble — Soft Voting | 0.9472 | 0.5483 | 0.6402 | 0.5907 | 0.6371 | 0.9456 |
| **Ensemble — Weighted Average (AUC-PR)** | 0.9486 | 0.5588 | 0.6358 | **0.5949** | **0.6402** | **0.9462** |

> El **Weighted Average ponderado por AUC-PR** es el mejor en F1, AUC-PR y AUC-ROC. Cada modelo individual gana en una columna distinta — la diversidad estructural justifica el ensamble.

---

## Estructura del repositorio

```
Machine-Learning-Census-Project/
├── notebooks/
│   └── Prediccion_Census.ipynb      # Notebook orquestador (importa desde census_ml)
├── src/
│   └── census_ml/                   # Paquete con todo el pipeline modularizado
│       ├── config.py                # RANDOM_STATE, paths, EDU_ORDER, etc.
│       ├── data.py                  # Carga UCI + train/test split
│       ├── preprocessing.py         # 5 etapas: imputar, encode, scale, SMOTE
│       ├── models/                  # Un archivo por modelo
│       │   ├── logistic.py
│       │   ├── random_forest.py
│       │   ├── xgboost_model.py
│       │   └── lightgbm_model.py
│       ├── evaluation.py            # Métricas, plots, comparison_table
│       └── ensemble.py              # Soft Voting + Weighted Average
├── scripts/
│   ├── train_all.py                 # Reproduce todo el pipeline end-to-end
│   └── evaluate.py                  # Carga modelos serializados y evalúa
├── app/
│   └── streamlit_app.py             # Demo interactiva (3 pestañas)
├── models/                          # Modelos entrenados (.pkl)
├── INFORME_PROYECTO.md              # Informe técnico completo (213 líneas)
├── pyproject.toml                   # Manifiesto de dependencias para uv
├── .python-version                  # Pin a Python 3.10
└── README.md
```

---

## Pipeline en una mirada

```
Census Income KDD (UCI)
        │
        ▼
[ data.load_census_data ]   →  X (199k × 41), y binaria
        │
        ▼
[ split_train_test ]        →  80/20 estratificado por random_state=42
        │
        ▼
┌─ preprocess_full ───────────────────────────────────────────────┐
│  1. Imputación      mediana (num) + moda (cat)                  │
│  2. Encoding alta   OrdinalEncoder(AHGA) + Target Encoding      │
│  3. Encoding baja   OneHotEncoder                               │
│  4. Escalado        StandardScaler                              │
│  5. Balanceo        SMOTE(k=5)  ← solo en train                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Modelos individuales ──────────────────────────────────────────┐
│  • Logistic Regression ElasticNet                               │
│  • Random Forest         (RandomizedSearchCV + threshold tuning)│
│  • XGBoost               (config fija: 600 árboles, depth=8)    │
│  • LightGBM              (Optuna 100 trials × 5-fold CV)        │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
[ Ensamble: weighted_average_by_aucpr ]   ← ESTRATEGIA FINAL
```

---

## Instalación y uso con [uv](https://docs.astral.sh/uv/)

### Instalar uv (una sola vez)

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Levantar el proyecto

```bash
git clone <url>
cd Machine-Learning-Census-Project

uv sync                       # crea .venv e instala todas las dependencias
uv sync --group dev           # incluye además jupyterlab, ipykernel y ruff
```

### Reproducir resultados

```bash
# Opción A: cargar modelos serializados y mostrar la tabla comparativa (segundos)
uv run python scripts/evaluate.py

# Opción B: re-entrenar todo desde cero (atención: ~horas por Optuna)
uv run python scripts/train_all.py

# Opción C: re-entrenar todo excepto LightGBM (mucho más rápido)
uv run python scripts/train_all.py --skip-lightgbm

# Opción D: trabajar con el notebook
uv run jupyter lab notebooks/Prediccion_Census.ipynb
```

### App interactiva (Streamlit)

```bash
uv run streamlit run app/streamlit_app.py
```

Abre el navegador en `http://localhost:8501` con tres pestañas:

1. **Resumen** — descripción del problema, dataset y modelos.
2. **Desempeño** — tabla de métricas, curvas ROC/PR y pesos del ensamble.
3. **Predicción interactiva** — formulario para ingresar las variables de un
   individuo y obtener la probabilidad de `>50K` según cada modelo y el
   ensamble final.

---

## Decisiones técnicas clave

- **Métrica principal: AUC-PR.** Con 6.3% de positivos, Accuracy es engañosa (un modelo trivial que predice siempre "negativo" alcanza 93.7%). AUC-PR es sensible al desempeño en la clase minoritaria.
- **SMOTE solo en train.** Aplicar oversampling al test falsearía las métricas; el test conserva la distribución natural.
- **Encoding diferenciado por cardinalidad.** Variables ≤ 20 categorías → OneHot; > 20 categorías nominales → Target Encoding (evita la explosión dimensional de OHE); `AHGA` (educación) → OrdinalEncoder con orden manual (es la única jerárquica genuina).
- **Threshold tuning.** Para Random Forest y LightGBM, barremos thresholds y elegimos el que maximiza F1 sobre la curva PR — mejora significativa frente al default de 0.5.
- **Optuna sobre LightGBM** maximiza AUC-PR vía 5-fold CV con 100 trials y 9 hiperparámetros.
- **Ensamble ponderado por AUC-PR**: pesos = `AUC-PR_i / Σ AUC-PR`. Si todos los modelos tuvieran AUC-PR igual colapsa a Soft Voting.

Justificación completa de cada decisión en [INFORME_PROYECTO.md](INFORME_PROYECTO.md).

---

## Modelos serializados

Los modelos pre-entrenados se distribuyen con el repositorio para permitir reproducir las métricas sin volver a correr Optuna.

| Archivo | Tamaño | Contenido |
|---|---:|---|
| `models/logit_elasticnet.pkl` | 2 KB | Logistic Regression ElasticNet |
| `models/random_forest_tunned.pkl` | 116 MB | Random Forest tuneado (300 árboles) |
| `models/xgb_model.pkl` | 4 MB | XGBoost (config fija) |
| `models/lgbm_tuned.pkl` | 12 MB | LightGBM con mejores hiperparámetros de Optuna |
| `models/lgbm_study.pkl` | 62 KB | `optuna.Study` completo (historial de los 100 trials) |

```python
import joblib
from census_ml.config import MODELS_DIR

xgb = joblib.load(MODELS_DIR / "xgb_model.pkl")
y_prob = xgb.predict_proba(X_test_proc)[:, 1]
```

> **Nota**: deserializar `.pkl` requiere la misma versión exacta de `scikit-learn`/`xgboost`/`lightgbm` con la que se entrenaron. `uv.lock` fija las versiones; correr `uv sync` garantiza compatibilidad.

---

## Limitaciones y trabajo futuro

- **Pesos del ensamble calculados sobre test.** Introduce un pequeño sesgo optimista; lo correcto sería usar un fold de validación separado o cross-validation interna.
- **Sin calibración de probabilidades.** Aplicar `CalibratedClassifierCV` (Platt o Isotónica) podría mejorar el ensamble basado en probabilidades.
- **Stacking real con meta-modelo** (en lugar de promedio ponderado) es el siguiente paso natural.
- **Validación temporal.** El dataset es transversal; en producción sería pertinente evaluar con folds temporales.

---

## Referencias

- UCI ML Repository — [Census Income KDD (id=117)](https://archive.ics.uci.edu/dataset/117/census+income+kdd)
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
- Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.

---

## Autores

Proyecto académico — **Maestría en Inteligencia Artificial, Aprendizaje de Máquina**.

- Luis Jorge García Camargo
- Luis Eduardo Uribe
- Antonia Gacharna
- Felipe Barreto
- Victor Cantor
