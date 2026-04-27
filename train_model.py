from plots import plot_actual_vs_predicted, plot_feature_importance, plot_shap
from sklearn.model_selection import cross_val_score, KFold
from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


DATASETS = {
    "drained": Path("drained.csv"),
    "undrained": Path("undrained.csv"),
}

MODEL_DIR = Path("models")

def model_filename(condition, name):
    safe_name = name.replace(" ", "_")
    return MODEL_DIR / f"{condition}_{safe_name}.joblib"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "unsaturated_unit_weight",
    "saturated_unit_weight",
    "void_ratio",
    "c",
    "phi",
    "H",
    "slope_angle",
]


def add_dimensionless_features(X):
    X = X.copy()
    eps = 1e-6

    phi_rad = np.radians(X["phi"].fillna(0))

   
    X["stability_number"] = (
        X["c"] / ((X["saturated_unit_weight"] * X["H"]) + eps)
    ) ** 0.5


    X["friction_factor"] = np.tan(phi_rad)

    return X

# =============================
# 🔹 Noise
# =============================
def add_noise(X, noise_level=0.03):
    noise = np.random.normal(0, noise_level * X.std(), X.shape)
    return X + noise


# =============================
# 🔹 Cleaning helpers
# =============================
def clean_column_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def extract_slope_angle(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else None


def clean_fos(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return None
    return numeric_value / 1000 if numeric_value > 10 else numeric_value


# =============================
# 🔹 Load + preprocess
# =============================
def read_dataset(path):
    return pd.read_csv(path)


def load_and_preprocess_dataset(path):
    data = read_dataset(path)
    data.columns = [clean_column_name(col) for col in data.columns]

    data = data.rename(
        columns={
            "unsaturated_unit_wt": "unsaturated_unit_weight",
            "saturated_unit_wt": "saturated_unit_weight",
            "cohesion": "c",
            "angle_of_friction": "phi",
            "height": "H",
            "slope_ratio_slope_angle": "slope_angle",
        }
    )

    data["slope_angle"] = data["slope_angle"].apply(extract_slope_angle)

    X = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = data["fos"].apply(clean_fos)

    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    # Add features + noise
    X = add_dimensionless_features(X)
    X = add_noise(X)

    # Clean invalid values
    X = X[(X["H"] > 0) & (X["phi"] >= 0) & (X["c"] >= 0)]
    y = y.loc[X.index]

    return X, y


# =============================
# 🔹 Models
# =============================
def build_models():
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer()),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer()),
            ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))
        ]),
        "SVR": Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", SVR())
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer()),
            ("model", DecisionTreeRegressor(max_depth=3,random_state=RANDOM_STATE))
        ]),
        "XGBoost": Pipeline([
             ("imputer", SimpleImputer()),
             ("model", XGBRegressor(objective="reg:squarederror",random_state=RANDOM_STATE,verbosity=0,
             n_jobs=-1
))
]),
    }


# =============================
# 🔹 Training
# =============================
def train_condition(condition, path):
    X, y = load_and_preprocess_dataset(path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = {}

    for name, model in build_models().items():

        # Parameter grids
        if name == "Random Forest":
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 4]
            }

        elif name == "Gradient Boosting":
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 4]
            }

        elif name == "SVR":
            param_grid = {
                "model__C": [1, 10],
                "model__epsilon": [0.1, 0.2]
            }
        elif name == "XGBoost":
            param_grid = {
               "model__n_estimators": [100, 200],
               "model__learning_rate": [0.05, 0.1],
               "model__max_depth": [3, 4]
    }

        else:
            param_grid = {}

        grid = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        # =============================
        # 5-Fold CV Evaluation (REPORTING)
        # =============================
     

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="r2")
        print(f"\n{condition} - {name} 5-Fold CV R² Scores:")
        for i, score in enumerate(cv_scores, 1):
            print(f"Fold {i}: {round(score, 4)}")

        print(f"Mean R²: {round(cv_scores.mean(), 4)} ± {round(cv_scores.std(), 4)}")

        # Predictions
        y_pred = best_model.predict(X_test)
         # =============================
        # Save predictions
        # =============================
        csv_path = f"models/{condition}_{name}_test.csv"

        pd.DataFrame({
            "actual": y_test,
            "predicted": y_pred
        }).to_csv(csv_path, index=False)

        # =============================
        # Create plots
        # =============================
        plot_actual_vs_predicted(
            csv_path,
            name,
            f"static/{condition}_{name}_actual_vs_pred.png"
        )

        # Feature importance (only for tree models)
        if name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            plot_feature_importance(
                best_model,
                X_train.columns,
                f"static/{condition}_{name}_importance.png"
            )

        # SHAP plot
        if name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            plot_shap(
                best_model,
                X_test,
                f"static/{condition}_{name}_shap.png"
    )
            
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Save predictions for graphs
        pd.DataFrame({
            "actual": y_test,
            "predicted": y_pred
        }).to_csv(f"models/{condition}_{name}_test.csv", index=False)

        safe_name = name.replace(" ", "_")
        joblib.dump(best_model, MODEL_DIR / f"{condition}_{safe_name}.joblib")

        results[name] = {
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "best_params": grid.best_params_
        }

    best_model = max(results, key=lambda k: results[k]["r2"])

    return {
        "results": results,
        "best_model": best_model
    }


# =============================
# 🔹 Main
# =============================
def train():
    MODEL_DIR.mkdir(exist_ok=True)

    metadata = {"conditions": {}}

    for cond, path in DATASETS.items():
        metadata["conditions"][cond] = train_condition(cond, path)

    # joblib.dump(metadata, MODEL_DIR / "metadata.joblib")

    print("\nTraining Complete\n")
    print(metadata)


if __name__ == "__main__":
    train()