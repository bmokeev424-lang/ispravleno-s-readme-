import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_fuzzy_risk(row):
    prof = ctrl.Antecedent(np.arange(0, 101, 1), 'prof')
    pay = ctrl.Antecedent(np.arange(1, 11, 1), 'pay')
    fin = ctrl.Consequent(np.arange(0, 101, 1), 'fin')

    prof['low'] = fuzz.trimf(prof.universe, [0, 0, 50])
    prof['high'] = fuzz.trimf(prof.universe, [50, 100, 100])
    pay['short'] = fuzz.trimf(pay.universe, [1, 1, 5])
    pay['long'] = fuzz.trimf(pay.universe, [5, 10, 10])
    fin['low'] = fuzz.trimf(fin.universe, [0, 0, 50])
    fin['high'] = fuzz.trimf(fin.universe, [50, 100, 100])

    rules_fin = [ctrl.Rule(prof['high'] & pay['short'], fin['low']),
                 ctrl.Rule(prof['low'] & pay['long'], fin['high']),
                 ctrl.Rule(prof['high'] & pay['long'], fin['low']),
                 ctrl.Rule(prof['low'] & pay['short'], fin['high'])]

    sim_fin = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_fin))
    try:
        sim_fin.input['prof'] = row['profitability']
        sim_fin.input['pay'] = row['payback_period']
        sim_fin.compute()
        fin_val = sim_fin.output['fin']
    except:
        fin_val = 50

    mkt_val = (row['market_volatility'] + row['competition_level'] * 10) / 2
    org_val = 100 - row['capability_score'] / 10
    final_risk = (fin_val * 0.4) + (mkt_val * 0.4) + (org_val * 0.2)
    return min(max(final_risk, 0), 100)


def train_models(df):
    print("Обучаем модели (это займет около минуты)...")
    df['fuzzy_risk'] = df.apply(get_fuzzy_risk, axis=1)

    features = ['project_cost', 'expected_profit', 'payback_period', 'team_experience',
                'market_volatility', 'technology_readiness', 'competition_level',
                'strategic_fit', 'regulatory_risk', 'profitability', 'roi_index',
                'risk_density', 'capability_score', 'fuzzy_risk']

    X = df[features].values
    y = df['risk_class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def objective(trial):
        layers = trial.suggest_int('layers', 16, 64)
        model = MLPClassifier(hidden_layer_sizes=(layers,), max_iter=3000, random_state=42,
                              early_stopping=True, n_iter_no_change=30, tol=1e-4)
        return cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    mlp = MLPClassifier(hidden_layer_sizes=(study.best_params['layers'],), max_iter=3000, random_state=42,
                        early_stopping=True, n_iter_no_change=30, tol=1e-4)

    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    results = []
    for name, model in [("MLP", mlp), ("RandomForest", rf), ("GradientBoosting", gb)]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, preds),
            'F1': f1_score(y_test, preds, average='macro'),
            'ROC_AUC': roc_auc_score(y_test, proba, multi_class='ovr')
        })

    os.makedirs("data/results", exist_ok=True)
    pd.DataFrame(results).to_csv("data/results/model_comparison.csv", index=False)
    print("Метрики сохранены в data/results/model_comparison.csv")

    joblib.dump(rf, "data/best_model.joblib")
    return df, X_train, X_test, y_test, rf