"""Constantes compartidas por todo el pipeline."""

from __future__ import annotations

from pathlib import Path

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
UCI_DATASET_ID: int = 117

CARDINALITY_THRESHOLD: int = 20

ORDINAL_VARS: list[str] = ["AHGA"]

EDU_ORDER: list[str] = [
    "Children",
    "Less than 1st grade",
    "1st 2nd 3rd or 4th grade",
    "5th or 6th grade",
    "7th and 8th grade",
    "9th grade",
    "10th grade",
    "11th grade",
    "12th grade no diploma",
    "High school graduate",
    "Some college but no degree",
    "Associates degree-occup /vocational",
    "Associates degree-academic program",
    "Bachelors degree(BA AB BS)",
    "Masters degree(MA MS MEng MEd MSW MBA)",
    "Prof school degree (MD DDS DVM LLB JD)",
    "Doctorate degree(PhD EdD)",
]

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODELS_DIR: Path = PROJECT_ROOT / "models"

MODEL_FILES: dict[str, str] = {
    "LogReg ElasticNet":     "logit_elasticnet.pkl",
    "Random Forest (Tuned)": "random_forest_tunned.pkl",
    "XGBoost":               "xgb_model.pkl",
    "LightGBM (Tuned)":      "lgbm_tuned.pkl",
}
LGBM_STUDY_FILE: str = "lgbm_study.pkl"
