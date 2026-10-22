import pandas as pd
import numpy as np

def clean_data():
    print("2. Чистим данные...")
    df = pd.read_csv("data/data_raw.csv")

    for col in ['expected_profit', 'team_experience', 'regulatory_risk']:
        df[col] = df[col].fillna(df[col].median())

    for col in ['project_cost', 'payback_period']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
    df['profitability'] = (df['expected_profit'] / df['project_cost']) * 100
    df['roi_index'] = df['expected_profit'] / (df['project_cost'] / df['payback_period'])
    df['risk_density'] = (df['market_volatility'] + df['regulatory_risk'] + df['competition_level'] * 10) / 3
    df['capability_score'] = df['team_experience'] * df['technology_readiness'] * df['strategic_fit']

    df.to_csv("data/data_clean.csv", index=False)
    print("Очищенные данные сохранены в data/data_clean.csv")
    return df