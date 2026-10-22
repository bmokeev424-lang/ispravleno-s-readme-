import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_eda_plots(df):
    print("3. Строим графики EDA...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include='number').corr(), cmap='coolwarm')
    plt.savefig("data/eda_corr.png")
    plt.close()
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='risk_class', y='project_cost', data=df)
    plt.savefig("data/eda_box.png")
    plt.close()
    plt.figure(figsize=(8, 5))
    df['profitability'].hist()
    plt.savefig("data/eda_hist.png")
    plt.close()
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='risk_density', y='capability_score', hue='risk_class', data=df)
    plt.savefig("data/eda_scatter.png")
    plt.close()
    print("Графики сохранены в папку data/")