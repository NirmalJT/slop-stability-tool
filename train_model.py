from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


DATASETS = {
    "drained": Path("drained.csv"),
    "undrained": Path("undrained.csv"),
}
MODEL_DIR = Path("models")
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


def read_dataset(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        try:
            return pd.read_excel(path)
        except ImportError:
            return read_xlsx_with_stdlib(path)


def read_xlsx_with_stdlib(path):
    with ZipFile(path) as workbook:
        shared_strings = []
        namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", namespace):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//a:t", namespace))
                )

        sheet_root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet_root.findall(".//a:sheetData/a:row", namespace):
            values = []
            for cell in row.findall("a:c", namespace):
                value = cell.find("a:v", namespace)
                cell_value = "" if value is None else value.text
                if cell.get("t") == "s" and cell_value != "":
                    cell_value = shared_strings[int(cell_value)]
                values.append(cell_value)
            rows.append(values)

    return pd.DataFrame(rows[1:], columns=rows[0])


def extract_slope_angle(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    if "OR" in text.upper():
        text = re.split(r"\bor\b", text, flags=re.IGNORECASE)[-1]

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    return float(numbers[-1])


def clean_fos(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return None
    if numeric_value > 10:
        return numeric_value / 1000
    return numeric_value


def load_and_preprocess_dataset(path):
    data = read_dataset(path)
    data.columns = [clean_column_name(column) for column in data.columns]

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

    for column in FEATURE_COLUMNS + ["fos"]:
        if column not in data.columns:
            data[column] = None

    X = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = data["fos"].apply(clean_fos)
    valid_rows = y.notna()
    return X.loc[valid_rows], y.loc[valid_rows]


def build_models():
    return {
        "Linear Regression": Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
            ]
        ),
        "SVR": Pipeline(
            [("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", SVR())]
        ),
        "KNN": Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5)),
            ]
        ),
    }


def model_filename(condition, model_name):
    safe_model_name = model_name.lower().replace(" ", "_")
    return MODEL_DIR / f"{condition}_{safe_model_name}.joblib"


def train_condition(condition, dataset_path):
    X, y = load_and_preprocess_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scores = {}
    trained_models = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = round(float(r2_score(y_test, predictions)), 3)

        scores[name] = score
        trained_models[name] = model
        joblib.dump(model, model_filename(condition, name))

    best_model_name = max(scores, key=scores.get)
    joblib.dump(trained_models[best_model_name], MODEL_DIR / f"{condition}_best_model.joblib")

    return {
        "dataset": str(dataset_path),
        "scores": scores,
        "best_model": best_model_name,
        "rows": int(len(X)),
    }


def train():
    MODEL_DIR.mkdir(exist_ok=True)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "conditions": {},
        "model_names": list(build_models().keys()),
    }

    for condition, dataset_path in DATASETS.items():
        metadata["conditions"][condition] = train_condition(condition, dataset_path)

    joblib.dump(metadata, MODEL_DIR / "metadata.joblib")

    print("\nModel Performance")
    for condition, info in metadata["conditions"].items():
        print(f"\n{condition.title()} ({info['rows']} rows)")
        for name, score in info["scores"].items():
            print(f"{name}: R2 = {score}")
        print(f"Best Model: {info['best_model']}")


if __name__ == "__main__":
    train()
