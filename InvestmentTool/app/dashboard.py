import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import run_pipeline_step

warnings.filterwarnings("ignore")
st.set_page_config(page_title="ИнвестКомитет", layout="wide", page_icon="📊")

st.title("📊 Система поддержки решений Инвестиционного комитета")

menu = st.sidebar.radio(
    "Навигация",
    ["Управление пайплайном", "Данные", "Графики (EDA)", "Метрики ML", "Что-если", "Новый проект"]
)

if menu == "Управление пайплайном":
    st.header("Управление данными и моделями")
    st.write("Нажимайте кнопки по порядку, чтобы подготовить систему к работе.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("1. Сгенерировать данные", use_container_width=True):
            with st.spinner("Генерация..."):
                res = run_pipeline_step("generate")
                st.success(res)

        if st.button("3. Построить EDA графики", use_container_width=True):
            with st.spinner("Построение графиков..."):
                res = run_pipeline_step("eda")
                st.success(res)

        if st.button("5. Построить SHAP графики", use_container_width=True):
            with st.spinner("Построение SHAP..."):
                res = run_pipeline_step("shap")
                st.info(res)

    with col2:
        if st.button("2. Очистить данные", use_container_width=True):
            with st.spinner("Очистка..."):
                res = run_pipeline_step("clean")
                st.success(res)

        if st.button("4. Обучить ML модели", use_container_width=True):
            with st.spinner("Обучение (займет около минуты)..."):
                res = run_pipeline_step("train")
                st.success(res)

        if st.button("6. Сгенерировать PDF отчет", use_container_width=True):
            with st.spinner("Создание PDF..."):
                res = run_pipeline_step("pdf")
                st.success(res)

elif menu == "Данные":
    st.header("Загруженные данные")
    if os.path.exists("data/data_clean.csv"):
        df = pd.read_csv("data/data_clean.csv")
        st.write(f"Всего записей: **{len(df)}**")
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.warning("⚠️ Данные не найдены. Перейдите во вкладку 'Управление пайплайном'.")

elif menu == "Графики (EDA)":
    st.header("Разведочный анализ данных")
    if os.path.exists("data/data_clean.csv"):
        df = pd.read_csv("data/data_clean.csv")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Стоимость по классам риска")
            fig, ax = plt.subplots()
            sns.boxplot(x='risk_class', y='project_cost', data=df, ax=ax)
            st.pyplot(fig)
            st.subheader("Гистограмма прибыльности")
            fig, ax = plt.subplots()
            df['profitability'].hist(ax=ax, bins=30, color='skyblue')
            st.pyplot(fig)
        with col2:
            st.subheader("Матрица корреляций")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.select_dtypes(include='number').corr(), cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            st.subheader("Плотность риска vs Возможности")
            fig, ax = plt.subplots()
            sns.scatterplot(x='risk_density', y='capability_score', hue='risk_class', data=df, ax=ax)
            st.pyplot(fig)
    else:
        st.warning("⚠️ Данные не найдены.")

elif menu == "Метрики ML":
    st.header("Сравнение моделей машинного обучения")
    if os.path.exists("data/results/model_comparison.csv"):
        metrics = pd.read_csv("data/results/model_comparison.csv")
        st.dataframe(metrics, use_container_width=True)
        fig, ax = plt.subplots()
        sns.barplot(x='Model', y='F1', data=metrics, ax=ax, palette='viridis')
        st.pyplot(fig)
    else:
        st.warning("⚠️ Метрики не найдены. Обучите модели во вкладке 'Управление'.")

elif menu == "Что-если":
    st.header("Сценарный анализ (What-If)")
    col1, col2 = st.columns(2)
    with col1:
        cost = st.slider("Стоимость проекта (млн руб.)", 1.0, 100.0, 50.0)
        profit = st.slider("Ожидаемая прибыль (млн руб.)", 0.5, 150.0, 40.0)
    with col2:
        volatility = st.slider("Волатильность рынка", 0, 100, 50)
        competition = st.slider("Уровень конкуренции", 1, 10, 5)

    st.subheader("Результат оценки:")
    if volatility > 70 or competition > 8:
        st.error(f"🔴 ВЫСОКИЙ РИСК: Проект слишком рискованный при таких параметрах.")
    elif profit / cost > 0.5:
        st.success(f"🟢 НИЗКИЙ РИСК: Отличные показатели, проект рекомендуется.")
    else:
        st.warning(f"🟡 СРЕДНИЙ РИСК: Требуется дополнительный анализ.")

elif menu == "Новый проект":
    st.header("Оценка нового инвестиционного проекта")
    with st.form("new_project_form"):
        name = st.text_input("Название проекта", "Модернизация цеха №5")
        cost = st.number_input("Стоимость (млн руб.)", min_value=1.0, max_value=100.0, value=20.0)
        profit = st.number_input("Ожидаемая прибыль (млн руб.)", min_value=0.5, max_value=150.0, value=30.0)
        payback = st.number_input("Срок окупаемости (лет)", min_value=1.0, max_value=10.0, value=3.0)
        sector = st.selectbox("Отрасль", ["производство", "ИТ", "Логистика", "Энергетика"])
        submitted = st.form_submit_button("Оценить проект")

        if submitted:
            st.success(f"Проект **'{name}'** успешно добавлен в базу!")
            if payback > 7:
                st.warning("Срок окупаемости слишком велик.")
            else:
                st.info("Показатели в норме. Рекомендуется к рассмотрению.")