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


MODEL_DIR = Path("models")


def _ensure_models_exist():
    if not (MODEL_DIR / "metadata.joblib").exists():
        train()


def _load_metadata():
    _ensure_models_exist()
    return joblib.load(MODEL_DIR / "metadata.joblib")


def _normalize_condition(condition):
    text = str(condition or "drained").strip().lower()
    if text not in {"drained", "undrained"}:
        raise ValueError("Condition must be 'drained' or 'undrained'.")
    return text


def _load_model(condition, model_name):
    metadata = _load_metadata()
    condition = _normalize_condition(condition)
    condition_info = metadata["conditions"][condition]
    selected_model = model_name or condition_info["best_model"]

    if selected_model not in condition_info["scores"]:
        available = ", ".join(condition_info["scores"])
        raise ValueError(f"Unknown model '{selected_model}'. Available models: {available}")

    return metadata, selected_model, joblib.load(model_filename(condition, selected_model))


def get_model_names():
    return _load_metadata()["model_names"]


@lru_cache(maxsize=4)
def _validation_metrics(condition):
    metadata = _load_metadata()
    condition = _normalize_condition(condition)
    X, y = load_and_preprocess_dataset(DATASETS[condition])
    metrics = {}

    for model_name, score in metadata["conditions"][condition]["scores"].items():
        model = joblib.load(model_filename(condition, model_name))
        predictions = model.predict(X)
        errors = [actual - predicted for actual, predicted in zip(y, predictions)]
        rmse = sqrt(sum(error**2 for error in errors) / len(errors))
        mae = sum(abs(error) for error in errors) / len(errors)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        metrics[model_name] = {
            "r2": score,
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std": round(float(cv_scores.std()), 4),
        }

    ranked_names = sorted(metrics, key=lambda name: metrics[name]["r2"], reverse=True)
    for index, model_name in enumerate(ranked_names, start=1):
        metrics[model_name]["rank"] = index

    return metrics


def make_feature_frame(
    unsaturated_unit_weight=None,
    saturated_unit_weight=None,
    void_ratio=None,
    c=None,
    phi=None,
    H=None,
    slope_angle=None,
):
    values = {
        "unsaturated_unit_weight": unsaturated_unit_weight,
        "saturated_unit_weight": saturated_unit_weight,
        "void_ratio": void_ratio,
        "c": c,
        "phi": phi,
        "H": H,
        "slope_angle": slope_angle,
    }
    return pd.DataFrame([{column: values.get(column) for column in FEATURE_COLUMNS}])


def predict_fos(condition="drained", model_name=None, **kwargs):
    _, _, model = _load_model(condition, model_name)
    features = make_feature_frame(**kwargs)
    return float(model.predict(features)[0])


def predict_all_models(condition="drained", selected_model=None, **kwargs):
    metadata = _load_metadata()
    condition = _normalize_condition(condition)
    features = make_feature_frame(**kwargs)
    validation_metrics = _validation_metrics(condition)
    results = []

    for model_name, score in metadata["conditions"][condition]["scores"].items():
        model = joblib.load(model_filename(condition, model_name))
        prediction = float(model.predict(features)[0])
        model_metrics = validation_metrics[model_name]
        results.append(
            {
                "model": model_name,
                "fos": round(prediction, 3),
                "r2": score,
                "rmse": model_metrics["rmse"],
                "mae": model_metrics["mae"],
                "cv_r2_mean": model_metrics["cv_r2_mean"],
                "cv_r2_std": model_metrics["cv_r2_std"],
                "rank": model_metrics["rank"],
                "best": model_name == metadata["conditions"][condition]["best_model"],
                "selected": model_name == selected_model,
            }
        )

    return sorted(results, key=lambda item: item["rank"])


def get_model_diagnostics(condition="drained", model_name=None):
    metadata, selected_model, model = _load_model(condition, model_name)
    condition = _normalize_condition(condition)
    condition_info = metadata["conditions"][condition]
    validation_metrics = _validation_metrics(condition)[selected_model]

    return {
        "r2": condition_info["scores"].get(selected_model),
        "rmse": validation_metrics["rmse"],
        "mae": validation_metrics["mae"],
        "cv_r2_mean": validation_metrics["cv_r2_mean"],
        "cv_r2_std": validation_metrics["cv_r2_std"],
        "rank": validation_metrics["rank"],
        "rows": condition_info["rows"],
        "best_model": condition_info["best_model"],
        "best_model_r2": condition_info["scores"][condition_info["best_model"]],
    }
