import pandas as pd
import numpy as np
import os

def generate_data():
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)
    n = 2000
    print("1. Генерируем сырые данные...")
    df = pd.DataFrame({
        'project_cost': np.random.uniform(1, 100, n),
        'expected_profit': np.random.uniform(0.5, 150, n),
        'payback_period': np.random.uniform(1, 10, n),
        'team_experience': np.random.uniform(0, 20, n),
        'market_volatility': np.random.uniform(0, 100, n),
        'technology_readiness': np.random.randint(1, 10, n),
        'competition_level': np.random.randint(1, 11, n),
        'strategic_fit': np.random.randint(1, 11, n),
        'regulatory_risk': np.random.uniform(0, 100, n),
        'sector': np.random.choice(['производство', 'ИТ', 'Логистика', 'Энергетика'], n)
    })
    risk_score = ((df['market_volatility'] / 100) * 0.4 +
                  (df['regulatory_risk'] / 100) * 0.3 +
                  (df['competition_level'] / 10) * 0.3)
    df['risk_class'] = pd.cut(risk_score, bins=[-1, 0.45, 0.65, 2], labels=[0, 1, 2]).astype(int)
    for col in ['expected_profit', 'team_experience', 'regulatory_risk']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan

    df.loc[df.sample(frac=0.03).index, 'project_cost'] = 9999  # Выброс
    df.to_csv("data/data_raw.csv", index=False)
    print("Сырые данные сохранены в data/data_raw.csv")