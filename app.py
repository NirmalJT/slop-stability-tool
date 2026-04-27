from flask import Flask, jsonify, render_template, request

from fos_calculation import plot_ml_vs_slope_angle
from ml_model import (
    get_model_diagnostics,
    get_model_names,
    predict_all_models,
    predict_fos,
)


app = Flask(__name__)


def as_float(data, key, default=0):
    value = data.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def build_recommendations(fos, condition, model_input):
    slope_angle = model_input["slope_angle"]
    height = model_input["H"]
    cohesion = model_input["c"]
    friction = model_input["phi"]

    if fos < 1.0:
        headline = "FoS < 1.0: Active Failure State — Immediate Emergency Intervention Required"
        actions = [
            "Evacuate all personnel, vehicles, and heavy equipment from the embankment crest and toe immediately.",
            "Mobilize emergency flood-fighting measures (e.g., dropping geo-bags or boulders at the toe to provide a stabilizing counter-weight).",
            "Identify and plug any active piping (muddy water seeping from the toe) using inverted sand filters.",
            "Consult a senior geotechnical engineer immediately to design emergency stabilizing berms or sheet pile walls.",
        ]
    elif fos < 1.2:
        headline = "FoS 1.0-1.2: Marginal Stability — High Risk of Creep or Localized Failure"
        actions = [
            "Strictly prohibit all surcharge loads (vehicles, construction materials, heavy machinery) on or near the crest.",
            "Initiate daily visual inspections specifically looking for longitudinal tension cracks on the crest and bulging at the toe.",
            "Install temporary relief trenches or clear existing toe drains to quickly dissipate excess internal pore water pressures.",
            "Prepare emergency stockpiles of sandbags or geo-bags near the site for rapid deployment if the FoS drops further.",
        ]
    elif fos < 1.5:
        headline = "FoS 1.2-1.5: Temporary Stability — Acceptable for Drawdown or End-of-Construction"
        actions = [
            "Monitor piezometers closely to ensure trapped pore water pressures are dissipating as expected during drawdown.",
            "Inspect the riverside slope for shallow sloughing or localized shear failures immediately after a rapid drop in river water level.",
            "Maintain surface vegetation and ensure weep holes are clear to prevent monsoon surface runoff from infiltrating the clay core.",
            "Verify that actual field soil conditions (unit weight, moisture content) match your Plaxis design parameters before assuming long-term safety.",
        ]
    else:
        headline = "FoS >= 1.5: Long-Term Stable — Meets Standard Geotechnical Design Criteria"
        actions = [
            "Conduct routine pre-monsoon and post-monsoon visual inspections to document any minor surface erosion.",
            "Maintain a healthy, deep-rooted grass cover and actively fill any animal burrows to prevent future internal piping.",
            "Keep surface drainage channels perfectly clear to prevent rainwater from pooling on the embankment crest.",
            "Re-evaluate the Machine Learning FoS model only if there is a massive change to slope geometry (e.g., severe river scouring) or new crest loads.",
        ]
    if condition == "undrained":
        actions.append("For undrained checks, review short-term loading and rapid drawdown scenarios.")
    if slope_angle >= 40:
        actions.append("The entered slope angle is steep; compare alternatives at lower angles.")
    if height >= 10:
        actions.append("The entered height is high; consider staged berms or intermediate benches.")
    if cohesion < 10 or friction < 10:
        actions.append("Low strength inputs are controlling the result; confirm lab/test values.")

    return {"headline": headline, "actions": actions[:6]}


@app.route("/")
def home():
    return render_template("index.html", model_names=get_model_names())


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        condition = data.get("condition", "drained")
        model_name = data.get("model_name") or "Random Forest"
        model_input = {
            "unsaturated_unit_weight": as_float(data, "unsaturated_unit_weight", 18),
            "saturated_unit_weight": as_float(data, "saturated_unit_weight", 20),
            "void_ratio": as_float(data, "void_ratio", 0.7),
            "c": as_float(data, "c"),
            "phi": as_float(data, "phi"),
            "H": as_float(data, "H"),
            "slope_angle": as_float(data, "slope_angle"),
        }

        ml_fos = predict_fos(condition=condition, model_name=model_name, **model_input)
        models = predict_all_models(
            condition=condition, selected_model=model_name, **model_input
        )
        diagnostics = get_model_diagnostics(condition=condition, model_name=model_name)
        selected = next((item for item in models if item["model"] == model_name), None)

        plot_ml_vs_slope_angle(
            condition=condition, model_name=model_name, **model_input
        )

        if ml_fos < 1:
            status, color = "Unstable", "red"
        elif ml_fos < 1.3:
            status, color = "Marginally Stable", "orange"
        else:
            status, color = "Stable", "green"

        return jsonify(
            {
                "ml_fos": round(ml_fos, 3),
                "condition": condition,
                "site_location": data.get("site_location", ""),
                "selected_model": model_name,
                "selected_model_r2": selected["r2"] if selected else None,
                "selected_model_rmse": diagnostics["rmse"],
                "selected_model_mae": diagnostics["mae"],
                "selected_model_cv_r2_mean": diagnostics["cv_r2_mean"],
                "selected_model_cv_r2_std": diagnostics["cv_r2_std"],
                "diagnostics": diagnostics,
                "models": models,
                "status": status,
                "color": color,
                "recommendations": build_recommendations(ml_fos, condition, model_input),
                "inputs": model_input,
                "graphs": {"ml_graph": "/static/ml_graph.png"},
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

if __name__ == "__main__":
    # host='0.0.0.0' allows external devices to connect
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)