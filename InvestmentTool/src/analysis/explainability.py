import warnings
warnings.filterwarnings("ignore")
import shap
import numpy as np
import matplotlib.pyplot as plt
from src.modeling.fuzzy_logic import get_fuzzy_risk

def make_shap_plots(model, X_test, features):
    print("Строим SHAP графики...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar", show=False)
    plt.savefig("data/shap_bar.png", bbox_inches='tight')
    plt.close()
    object_index = 0
    class_index = 0
    if isinstance(shap_values, list):
        vals = shap_values[class_index][object_index]
        base = explainer.expected_value[class_index]
    elif len(shap_values.shape) == 3:
        vals = shap_values[object_index, :, class_index]
        base = explainer.expected_value[class_index]
    else:
        vals = shap_values[object_index]
        base = explainer.expected_value

    shap.waterfall_plot(
        shap.Explanation(
            values=vals,
            base_values=base,
            data=X_test[object_index],
            feature_names=features
        ),
        show=False
    )
    plt.savefig("data/shap_waterfall.png", bbox_inches='tight')
    plt.close()

def run_monte_carlo(row, n=500):
    print(f"Запускаем Монте-Карло ({n} прогонов)...")
    results = []
    for _ in range(n):
        noisy_row = row.copy()
        noisy_row['market_volatility'] *= np.random.normal(1, 0.1)
        noisy_row['project_cost'] *= np.random.normal(1, 0.05)
        risk = get_fuzzy_risk(noisy_row)
        results.append(risk)
    print(f"Средний риск: {np.mean(results):.1f}")
    print(f"Разброс (std): {np.std(results):.1f}")