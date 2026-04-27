import pandas as pd
import matplotlib.pyplot as plt
import shap


# =============================
# 🔹 Actual vs Predicted Plot
# =============================
def plot_actual_vs_predicted(csv_path, model_name, save_path):
    df = pd.read_csv(csv_path)

    actual = df["actual"]
    predicted = df["predicted"]
    error = abs(actual - predicted)

    plt.figure()

    # Color by error
    colors = []
    for e in error:
        if e < 0.05:
            colors.append("green")
        elif e < 0.15:
            colors.append("orange")
        else:
            colors.append("red")

    plt.scatter(actual, predicted, c=colors)

    # Perfect line
    plt.plot([actual.min(), actual.max()],
             [actual.min(), actual.max()],
             linestyle="--")

    plt.xlabel("Actual FoS")
    plt.ylabel("Predicted FoS")
    plt.title(f"Actual vs Predicted — {model_name}")

    plt.savefig(save_path)
    plt.close()


# =============================
# 🔹 Feature Importance
# =============================


def plot_feature_importance(model, feature_names, save_path):
    model_obj = model.named_steps["model"]

    # Some models don't have feature_importances_
    if not hasattr(model_obj, "feature_importances_"):
        print("Skipping feature importance (not supported)")
        return

    importances = model_obj.feature_importances_

    plt.figure(figsize=(8, 6))  

    plt.barh(feature_names, importances)

    plt.xlabel("Importance")
    plt.title("Feature Importance")

    plt.tight_layout()  

    plt.savefig(save_path)
    plt.close()

# =============================
# 🔹 SHAP Plot
# =============================
def plot_shap(model, X_test, save_path):
    import shap
    import matplotlib.pyplot as plt

    model_obj = model.named_steps["model"]

    # Only allow tree-based models
    if model_obj.__class__.__name__ not in [
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "XGBRegressor"
    ]:
        print(f"Skipping SHAP for {model_obj.__class__.__name__}")
        return

    try:
        explainer = shap.TreeExplainer(model_obj)
        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        print(f"SHAP failed: {e}")