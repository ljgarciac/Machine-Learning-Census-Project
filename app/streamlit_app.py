"""Streamlit app del proyecto Census Income KDD.

Tres pestañas:
    1) Resumen del proyecto.
    2) Desempeño de los modelos (tablas + gráficos).
    3) POC de predicción interactiva (formulario con valores por defecto).

Ejecutar con:
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from census_ml.config import MODEL_FILES, MODELS_DIR
from census_ml.data import load_census_data, split_train_test
from census_ml.ensemble import weighted_average_by_aucpr
from census_ml.evaluation import comparison_table, compute_metrics
from census_ml.inference import apply_preprocessing, predict_with_ensemble
from census_ml.preprocessing import preprocess_full


# Campos que el usuario edita explícitamente. El resto se rellena con la moda /
# mediana del train. Elegidos por interpretabilidad y poder predictivo.
USER_FACING_FIELDS: list[str] = [
    "AAGE",      # edad
    "ASEX",      # sexo
    "AHGA",      # educación (jerárquica)
    "AMARITL",   # estado civil
    "ARACE",     # raza
    "AMJOCC",    # ocupación mayor
    "AMJIND",    # industria mayor
    "ACLSWKR",   # clase de trabajador
    "WKSWORK",   # semanas trabajadas en el año
    "CAPGAIN",   # ganancias de capital
    "CAPLOSS",   # pérdidas de capital
    "DIVVAL",    # dividendos
]

FIELD_LABELS: dict[str, str] = {
    "AAGE":    "Edad",
    "ASEX":    "Sexo",
    "AHGA":    "Nivel educativo",
    "AMARITL": "Estado civil",
    "ARACE":   "Raza",
    "AMJOCC":  "Ocupación principal",
    "AMJIND":  "Industria principal",
    "ACLSWKR": "Clase de trabajador",
    "WKSWORK": "Semanas trabajadas (año)",
    "CAPGAIN": "Ganancias de capital (USD)",
    "CAPLOSS": "Pérdidas de capital (USD)",
    "DIVVAL":  "Dividendos recibidos (USD)",
}


# ---------------------------------------------------------------------------
# Carga única (cacheada) del pipeline completo
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando dataset y reproduciendo el preprocesamiento...")
def load_pipeline():
    X, y = load_census_data()
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    _, _, X_test_proc, artifacts = preprocess_full(X_train, X_test, y_train)

    models = {}
    for name, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        if path.exists():
            models[name] = joblib.load(path)

    predictions = {}
    for name, model in models.items():
        predictions[name] = {
            "y_prob": model.predict_proba(X_test_proc)[:, 1],
            "y_pred": model.predict(X_test_proc),
        }

    individual = comparison_table(predictions, y_test.values)
    _, _, weights = weighted_average_by_aucpr(predictions, individual["AUC-PR"])

    # Defaults por columna a partir de X_train
    defaults: dict[str, object] = {}
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            mode_val = X_train[col].mode()
            defaults[col] = mode_val.iloc[0] if len(mode_val) else ""
        else:
            defaults[col] = float(X_train[col].median())

    # Categorías observadas (para los selectbox)
    cat_options: dict[str, list[str]] = {}
    for col in X_train.select_dtypes(include="object").columns:
        vals = sorted(v for v in X_train[col].dropna().unique() if isinstance(v, str))
        cat_options[col] = vals

    # Rangos numéricos (para los number_input)
    num_ranges: dict[str, tuple[float, float]] = {}
    for col in X_train.select_dtypes(exclude="object").columns:
        num_ranges[col] = (float(X_train[col].min()), float(X_train[col].max()))

    return {
        "X_train": X_train,
        "y_test": y_test,
        "X_test_proc": X_test_proc,
        "artifacts": artifacts,
        "models": models,
        "predictions": predictions,
        "individual": individual,
        "weights": weights,
        "defaults": defaults,
        "cat_options": cat_options,
        "num_ranges": num_ranges,
    }


# ---------------------------------------------------------------------------
# Tab 1 — Resumen
# ---------------------------------------------------------------------------

def render_summary_tab() -> None:
    st.header("Resumen del proyecto")

    st.markdown(
        """
**Objetivo.** Predecir si un individuo tiene un ingreso anual superior a
**USD 50.000** a partir de variables sociodemográficas y laborales del
**Census Income KDD** (UCI Machine Learning Repository, id=117).

**Reto principal.** Desbalance severo de clases: solo el **6.3%** de los
registros pertenece a la clase positiva (>50K), frente al 93.7% restante.
Esto descarta Accuracy como métrica útil — usamos **AUC-PR** como métrica
principal, más honesta para datasets desbalanceados.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros", "199,523")
    col2.metric("Variables", "41")
    col3.metric("Clase positiva", "6.3%")
    col4.metric("Modelos entrenados", "4 + 2 ensambles")

    st.markdown(
        """
### Pipeline en cinco pasos

1. **Imputación.** Mediana para numéricas, moda para categóricas.
2. **Encoding alta cardinalidad.** `OrdinalEncoder` para `AHGA` (educación
   jerárquica) y **Target Encoding** para industria, ocupación y nacionalidad.
3. **Encoding baja cardinalidad.** `OneHotEncoder` para variables con
   ≤ 20 categorías.
4. **Escalado.** `StandardScaler` para todas las numéricas.
5. **Balanceo.** `SMOTE(k_neighbors=5)` aplicado **solo en train**.

### Modelos

| Modelo | Configuración |
|---|---|
| Logistic Regression | ElasticNet (L1+L2), C=1.0, l1_ratio=0.5 |
| Random Forest | 300 árboles, RandomizedSearchCV (20 iter × 3-fold) + threshold tuning |
| XGBoost | 600 árboles, depth=8, lr=0.05 (sin tuning) |
| LightGBM | Optuna (100 trials × 5-fold CV) optimizando AUC-PR |

### Estrategia final

Las predicciones de los 4 modelos se combinan en un **promedio ponderado por
AUC-PR**. Cada modelo aporta proporcionalmente a su desempeño individual en
la métrica más relevante para el problema.

> Para la justificación detallada de cada decisión técnica, consulta
> `INFORME_PROYECTO.md` en la raíz del repositorio.
        """
    )


# ---------------------------------------------------------------------------
# Tab 2 — Desempeño
# ---------------------------------------------------------------------------

def render_performance_tab(pipeline: dict) -> None:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    st.header("Desempeño de los modelos")

    individual = pipeline["individual"]
    predictions = pipeline["predictions"]
    weights = pipeline["weights"]
    y_test = pipeline["y_test"]

    # Tabla con ensambles incluidos
    from census_ml.ensemble import soft_voting

    soft_prob, soft_pred = soft_voting(predictions)
    w_prob, w_pred, _ = weighted_average_by_aucpr(predictions, individual["AUC-PR"])

    full = individual.copy()
    full.loc["Ensemble — Soft Voting"] = compute_metrics(y_test.values, soft_pred, soft_prob)
    full.loc["Ensemble — Weighted Average"] = compute_metrics(y_test.values, w_pred, w_prob)

    st.subheader("Tabla comparativa (test set)")
    st.dataframe(
        full.style.background_gradient(cmap="Greens", axis=0).format("{:.4f}"),
        use_container_width=True,
    )

    st.subheader("Métricas por modelo")
    metrics_long = full.reset_index().melt(
        id_vars="Modelo", var_name="Métrica", value_name="Valor"
    )
    fig_bar = px.bar(
        metrics_long, x="Métrica", y="Valor", color="Modelo", barmode="group",
        height=450, template="plotly_white",
    )
    fig_bar.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_bar, use_container_width=True)

    col_left, col_right = st.columns(2)

    # Curva ROC
    with col_left:
        st.subheader("Curvas ROC")
        fig_roc = go.Figure()
        for name, p in predictions.items():
            fpr, tpr, _ = roc_curve(y_test, p["y_prob"])
            auc = roc_auc_score(y_test, p["y_prob"])
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"{name} ({auc:.3f})"))
        # Ensembles
        for label, prob in [("Soft Voting", soft_prob), ("Weighted Avg", w_prob)]:
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"{label} ({auc:.3f})",
                                          line=dict(dash="dash")))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(color="gray", dash="dot"),
                                      showlegend=False))
        fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                              template="plotly_white", height=450)
        st.plotly_chart(fig_roc, use_container_width=True)

    # Curva Precision-Recall
    with col_right:
        st.subheader("Curvas Precision-Recall")
        fig_pr = go.Figure()
        for name, p in predictions.items():
            prec, rec, _ = precision_recall_curve(y_test, p["y_prob"])
            ap = average_precision_score(y_test, p["y_prob"])
            fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                         name=f"{name} ({ap:.3f})"))
        for label, prob in [("Soft Voting", soft_prob), ("Weighted Avg", w_prob)]:
            prec, rec, _ = precision_recall_curve(y_test, prob)
            ap = average_precision_score(y_test, prob)
            fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                         name=f"{label} ({ap:.3f})",
                                         line=dict(dash="dash")))
        baseline = float(y_test.mean())
        fig_pr.add_hline(y=baseline, line=dict(color="gray", dash="dot"),
                         annotation_text=f"Baseline ({baseline:.3f})")
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                             template="plotly_white", height=450)
        st.plotly_chart(fig_pr, use_container_width=True)

    st.subheader("Pesos del ensamble (Weighted Average)")
    weights_df = pd.DataFrame({
        "Modelo":  weights.index,
        "AUC-PR":  individual.loc[weights.index, "AUC-PR"].values,
        "Peso":    weights.values,
    })
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_w = px.bar(weights_df, x="Modelo", y="Peso", color="Modelo",
                       text="Peso", template="plotly_white", height=350)
        fig_w.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_w.update_layout(showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)
    with col_b:
        st.dataframe(
            weights_df.set_index("Modelo").style.format({"AUC-PR": "{:.4f}", "Peso": "{:.4f}"}),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 3 — POC de predicción interactiva
# ---------------------------------------------------------------------------

def render_prediction_tab(pipeline: dict) -> None:
    st.header("POC — Predicción de un caso")
    st.markdown(
        "Ajusta los valores del individuo. Los campos no mostrados se rellenan "
        "automáticamente con la moda (categóricas) o la mediana (numéricas) de "
        "los datos de entrenamiento."
    )

    defaults = pipeline["defaults"]
    cat_options = pipeline["cat_options"]
    num_ranges = pipeline["num_ranges"]

    inputs: dict[str, object] = {}

    cols = st.columns(3)
    for i, field in enumerate(USER_FACING_FIELDS):
        col = cols[i % 3]
        label = FIELD_LABELS.get(field, field)
        default = defaults.get(field)

        with col:
            if field in cat_options:
                options = cat_options[field]
                idx = options.index(default) if default in options else 0
                inputs[field] = st.selectbox(label, options, index=idx, key=f"in_{field}")
            else:
                lo, hi = num_ranges.get(field, (0.0, 1e6))
                inputs[field] = st.number_input(
                    label,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=1.0,
                    key=f"in_{field}",
                )

    with st.expander("Mostrar valores completos del registro (incluye campos rellenados automáticamente)"):
        full_record = {**defaults, **inputs}
        st.json(full_record, expanded=False)

    if st.button("Predecir", type="primary", use_container_width=True):
        full_record = {**defaults, **inputs}
        X_new = pd.DataFrame([full_record], columns=pipeline["X_train"].columns)
        X_proc = apply_preprocessing(X_new, pipeline["artifacts"])

        probs = predict_with_ensemble(X_proc, pipeline["models"], pipeline["weights"])

        st.markdown("---")
        st.subheader("Probabilidades de la clase positiva (>50K)")

        rows = [{"Modelo": name, "P(>50K)": float(prob[0])} for name, prob in probs.items()]
        df_probs = pd.DataFrame(rows)

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            fig = px.bar(df_probs, x="Modelo", y="P(>50K)", color="Modelo",
                         text="P(>50K)", template="plotly_white", height=420)
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(yaxis_range=[0, 1], showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_table:
            st.dataframe(
                df_probs.set_index("Modelo").style.format("{:.4f}"),
                use_container_width=True,
            )

        ensemble_p = float(probs["Ensemble"][0])
        prediction_label = ">50K" if ensemble_p >= 0.5 else "<=50K"
        st.markdown(f"### Predicción del ensamble: **{prediction_label}**")
        st.progress(ensemble_p, text=f"P(>50K) = {ensemble_p:.4f}  ({prediction_label})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Census Income — ML Project",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Predicción de ingresos — Census Income KDD")
    st.caption(
        "Ensamble ponderado por AUC-PR sobre Logistic Regression, Random Forest, "
        "XGBoost y LightGBM."
    )

    pipeline = load_pipeline()

    if not pipeline["models"]:
        st.error(
            "No se encontraron modelos serializados en `models/`. Corre primero "
            "`uv run python scripts/train_all.py`."
        )
        return

    tab1, tab2, tab3 = st.tabs([
        "1. Resumen",
        "2. Desempeño",
        "3. Predicción interactiva",
    ])
    with tab1:
        render_summary_tab()
    with tab2:
        render_performance_tab(pipeline)
    with tab3:
        render_prediction_tab(pipeline)


if __name__ == "__main__":
    main()
