from flask import Flask, render_template, request
from fos_calculation import calculate_fos, plot_slope, plot_fos_vs_slope
from ml_model import predict_fos

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():

    fos = None
    ml_fos = None
    status = None

    if request.method == "POST":

        c = float(request.form["c"])
        phi = float(request.form["phi"])
        gamma = float(request.form["gamma"])
        H = float(request.form["H"])
        slope = float(request.form["slope"])

        fos = calculate_fos(c, phi, gamma, H, slope)

        ml_fos = round(predict_fos(c, phi, gamma, H, slope), 3)

        plot_slope(H, slope)
        plot_fos_vs_slope(c, phi, gamma, H)

        # slope stability classification
        if fos < 1:
            status = "Unstable (Failure likely)"
        elif fos < 1.3:
            status = "Marginally Stable"
        else:
            status = "Stable"

    return render_template("index.html", fos=fos, ml_fos=ml_fos, status=status)


if __name__ == "__main__":
    app.run(debug=True)