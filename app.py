from flask import Flask, render_template, request
from fos_calculation import calculate_fos, plot_slope, plot_fos_vs_slope, plot_ml_vs_slope
from ml_model import predict_fos
import math

app = Flask(__name__)

# 🔥 Convert slope ratio → angle
def slope_to_angle(slope):
    try:
        val = float(slope.split(":")[0])
        return math.degrees(math.atan(1 / val))
    except:
        return 30  # default

@app.route("/", methods=["GET","POST"])
def home():

    fos = None
    ml_fos = None
    status = None

    if request.method == "POST":

        c = float(request.form["c"])
        phi = float(request.form["phi"])
        H = float(request.form["H"])
        water = float(request.form["water"]) if request.form["water"] else 0
        slope = request.form["slope"]   # string (1.5:1)
        soil = request.form["soil"]

        # 🔥 Convert slope for analytical formula
        slope_angle = slope_to_angle(slope)

        # Analytical FOS
        gamma = 18  # default assumption
        fos = calculate_fos(c, phi, gamma, H, slope_angle)

        # ML Prediction
        ml_fos = round(predict_fos(c, phi, H, water, slope, soil), 3)

        # Graphs
        plot_slope(H, slope_angle)
        plot_fos_vs_slope(c, phi, gamma, H)
        plot_ml_vs_slope(c, phi, H, water, soil)

        # Status based on ML (better)
        if ml_fos < 1:
            status = "Unstable (Failure likely)"
        elif ml_fos < 1.3:
            status = "Marginally Stable"
        else:
            status = "Stable"

    return render_template("index.html", fos=fos, ml_fos=ml_fos, status=status)

if __name__ == "__main__":
    app.run(debug=True)