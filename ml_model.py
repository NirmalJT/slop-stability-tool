from math import sqrt
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from train_model import (
    DATASETS,
    FEATURE_COLUMNS,
    load_and_preprocess_dataset,
    model_filename,
    train,
   
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# =============================
# 🔹 Ensure models exist
# =============================
def _ensure_models_exist():
    if not (MODEL_DIR / "metadata.joblib").exists():
        train()


def _load_metadata():
    _ensure_models_exist()
    return joblib.load(MODEL_DIR / "metadata.joblib")


# =============================
# 🔹 Helpers
# =============================
def _normalize_condition(condition):
    text = str(condition or "drained").strip().lower()
    if text not in {"drained", "undrained"}:
        raise ValueError("Condition must be 'drained' or 'undrained'.")
    return text


def _load_model(condition, model_name):
    condition = _normalize_condition(condition)
    selected_model = model_name or "Random Forest"

    return None, selected_model, joblib.load(model_filename(condition, selected_model))

def get_model_names():
    return [
        "Random Forest",
        "Gradient Boosting",
        "SVR",
        "XGBoost"
    ]
# =============================
# 🔹 Validation Metrics (UPDATED)
# =============================

@lru_cache(maxsize=4)
def _validation_metrics(condition):
    condition = _normalize_condition(condition)
    X, y = load_and_preprocess_dataset(DATASETS[condition])
    metrics = {}

    model_names = get_model_names()

    for model_name in model_names:
        model = joblib.load(model_filename(condition, model_name))

        predictions = model.predict(X)
        errors = [a - p for a, p in zip(y, predictions)]

        rmse = sqrt(sum(e**2 for e in errors) / len(errors))
        mae = sum(abs(e) for e in errors) / len(errors)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

        metrics[model_name] = {
            "r2": round(float(cv_scores.mean()), 4),
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std": round(float(cv_scores.std()), 4),
            "cv_scores": [round(float(s), 4) for s in cv_scores],
        }

    # ranking
    ranked = sorted(metrics, key=lambda x: metrics[x]["r2"], reverse=True)
    for i, name in enumerate(ranked, 1):
        metrics[name]["rank"] = i

    return metrics

# =============================
# 🔹 Feature Input Builder
# =============================
def make_feature_frame(
    unsaturated_unit_weight=None,
    saturated_unit_weight=None,
    void_ratio=None,
    c=None,
    phi=None,
    H=None,
    slope_angle=None,
):
    import numpy as np

    eps = 1e-6

    # Base features
    data = {
        "unsaturated_unit_weight": unsaturated_unit_weight,
        "saturated_unit_weight": saturated_unit_weight,
        "void_ratio": void_ratio,
        "c": c,
        "phi": phi,
        "H": H,
        "slope_angle": slope_angle,
    }

    df = pd.DataFrame([data])

    # ✅ ADD SAME FEATURES AS TRAINING
    phi_rad = np.radians(df["phi"].fillna(0))

    df["stability_number"] = (
        df["c"] / ((df["saturated_unit_weight"] * df["H"]) + eps)
    ) ** 0.5

    df["friction_factor"] = np.tan(phi_rad)

    return df

# =============================
# 🔹 Single Prediction
# =============================
def predict_fos(condition="drained", model_name=None, **kwargs):
    _, _, model = _load_model(condition, model_name)
    features = make_feature_frame(**kwargs)
    return float(model.predict(features)[0])


# =============================
# 🔹 All Models Prediction
# =============================
def predict_all_models(condition="drained", selected_model=None, **kwargs):
    condition = _normalize_condition(condition)
    features = make_feature_frame(**kwargs)
    validation_metrics = _validation_metrics(condition)

    results = []
    model_names = get_model_names()

    for model_name in model_names:
        model = joblib.load(model_filename(condition, model_name))
        prediction = float(model.predict(features)[0])

        m = validation_metrics[model_name]

        results.append({
            "model": model_name,
            "fos": round(prediction, 3),
            "r2": m["r2"],
            "rmse": m["rmse"],
            "mae": m["mae"],
            "cv_r2_mean": m["cv_r2_mean"],
            "cv_r2_std": m["cv_r2_std"],
            "cv_scores": m["cv_scores"],
            "rank": m["rank"],
            "best": m["rank"] == 1,
            "selected": model_name == selected_model,
        })

    return sorted(results, key=lambda x: x["rank"])

# =============================
# 🔹 Diagnostics (UPDATED)
# =============================
def get_model_diagnostics(condition="drained", model_name=None):
    condition = _normalize_condition(condition)
    selected_model = model_name or "Random Forest"

    validation_metrics = _validation_metrics(condition)[selected_model]

    return {
        "r2": validation_metrics["r2"],
        "rmse": validation_metrics["rmse"],
        "mae": validation_metrics["mae"],
        "cv_r2_mean": validation_metrics["cv_r2_mean"],
        "cv_r2_std": validation_metrics["cv_r2_std"],
        "cv_scores": validation_metrics["cv_scores"],
        "rank": validation_metrics["rank"],
    }                               
    