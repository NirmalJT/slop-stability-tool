from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_model import predict_fos


STATIC_DIR = Path("static")


def plot_ml_vs_slope_angle(condition="drained", model_name=None, **kwargs):
    STATIC_DIR.mkdir(exist_ok=True)
    angles = [20, 25, 30, 35, 40, 45, 50]
    predictions = []

    for angle in angles:
        model_input = dict(kwargs)
        model_input["slope_angle"] = angle
        predictions.append(
            predict_fos(condition=condition, model_name=model_name, **model_input)
        )

    plt.figure(figsize=(6, 4))
    plt.plot(angles, predictions, marker="o", color="#0f766e")
    plt.xlabel("Slope angle (degrees)")
    plt.ylabel("ML predicted FOS")
    plt.title("ML FOS vs Slope Angle")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(STATIC_DIR / "ml_graph.png", dpi=140)
    plt.close()
