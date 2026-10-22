import os
import pandas as pd
import sys
from src.data_processing.generator import generate_data
from src.data_processing.preprocessor import clean_data
from src.analysis.eda import make_eda_plots
from src.modeling.ml_models import train_models
from src.analysis.explainability import make_shap_plots, run_monte_carlo
from src.reporting.report_maker import make_pdf_report


def run_pipeline_step(step):
    try:
        if step == "generate":
            generate_data()
            return "Данные успешно сгенерированы!"
        elif step == "clean":
            df = clean_data()
            return f"Данные очищены. Осталось записей: {len(df)}"
        elif step == "eda":
            df = pd.read_csv("data/data_clean.csv")
            make_eda_plots(df)
            return "Графики EDA построены и сохранены!"
        elif step == "train":
            df = pd.read_csv("data/data_clean.csv")
            train_models(df)
            return "Модели обучены! Метрики сохранены."
        elif step == "shap":
            return "SHAP-графики строятся (проверьте папку data)."
        elif step == "pdf":
            make_pdf_report()
            return "PDF отчет сгенерирован!"
        else:
            return "Неизвестная команда"
    except Exception as e:
        return f"Ошибка: {e}"

import PySimpleGUI as sg
import matplotlib.pyplot as plt


def show_plot(filename, title):
    if os.path.exists(filename):
        img = plt.imread(filename)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        ax.axis('off')
        fig.canvas.manager.set_window_title(title)
        plt.show()
    else:
        sg.popup_error(f"Файл {filename} не найден!\nСначала выполните нужное действие.")


def start_gui():
    sg.theme('DarkBlue3')

    layout = [
        [sg.Text("Система поддержки решений ИнвестКомитета", font=("Arial", 16, "bold"))],
        [sg.HorizontalSeparator()],
        [sg.Frame("Управление проектом", [
            [sg.Button("⚡ ОБНОВИТЬ ВСЕ ДАННЫЕ (Генерация + Очистка + Обучение)",
                       key="-UPDATE_ALL-", size=(60, 2), button_color=('white', 'green'))],
            [sg.Button("Открыть папку с файлами (data)", key="-OPEN_FOLDER-", size=(60, 1))]
        ])],
        [sg.HorizontalSeparator()],
        [sg.Frame("1. Действия (Запуск расчетов)", [
            [sg.Button("Сгенерировать данные", key="-GEN-", size=(30, 1)),
             sg.Button("Очистить данные", key="-CLEAN-", size=(30, 1))],
            [sg.Button("Построить EDA (графики)", key="-DO_EDA-", size=(30, 1)),
             sg.Button("Обучить ML модели (Optuna)", key="-TRAIN-", size=(30, 1))],
            [sg.Button("Построить SHAP", key="-DO_SHAP-", size=(30, 1)),
             sg.Button("Создать PDF отчет", key="-PDF-", size=(30, 1))]
        ])],
        [sg.HorizontalSeparator()],
        [sg.Frame("2. Просмотр результатов", [
            [sg.Button("Показать данные", key="-SHOW_DATA-", size=(20, 1)),
             sg.Button("Метрики моделей", key="-SHOW_METRICS-", size=(20, 1))],

            [sg.Text("Графики EDA:"),
             sg.Button("Корреляции", key="-EDA_CORR-", size=(12, 1)),
             sg.Button("Боксплот", key="-EDA_BOX-", size=(12, 1)),
             sg.Button("Гистограмма", key="-EDA_HIST-", size=(12, 1))],

            [sg.Text("SHAP:"),
             sg.Button("Summary", key="-SHAP_BAR-", size=(12, 1)),
             sg.Button("Waterfall", key="-SHAP_WATER-", size=(12, 1))]
        ])],

        [sg.HorizontalSeparator()],
        [sg.Text("Статус:", size=(10, 1)), sg.Text("Готов к работе", key="-STATUS-", size=(60, 1))],
        [sg.Button("Выход", key="-EXIT-", button_color=('white', 'firebrick3'), size=(10, 1))]
    ]

    window = sg.Window("ИнвестКомитет v3.0", layout, size=(750, 750))

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "-EXIT-"):
            break

        if event == "-OPEN_FOLDER-":
            if os.path.exists("data"):
                os.startfile(os.path.abspath("data"))
            else:
                sg.popup_error("Папка data еще не создана!")

        elif event == "-UPDATE_ALL-":
            window["-STATUS-"].update("Шаг 1/3: Генерация данных...")
            window.refresh()
            res1 = run_pipeline_step("generate")

            window["-STATUS-"].update("Шаг 2/3: Очистка данных...")
            window.refresh()
            res2 = run_pipeline_step("clean")

            window["-STATUS-"].update("Шаг 3/3: Обучение моделей (это займет около минуты)...")
            window.refresh()
            res3 = run_pipeline_step("train")

            window["-STATUS-"].update("Все данные успешно обновлены!")
            sg.popup("Все данные успешно обновлены!\nТеперь можно строить графики и отчеты.")

        elif event == "-GEN-":
            window["-STATUS-"].update("Генерация данных...")
            window.refresh()
            res = run_pipeline_step("generate")
            window["-STATUS-"].update(res)

        elif event == "-CLEAN-":
            window["-STATUS-"].update("Очистка данных...")
            window.refresh()
            res = run_pipeline_step("clean")
            window["-STATUS-"].update(res)

        elif event == "-DO_EDA-":
            window["-STATUS-"].update("Построение графиков EDA...")
            window.refresh()
            res = run_pipeline_step("eda")
            window["-STATUS-"].update(res)

        elif event == "-TRAIN-":
            window["-STATUS-"].update("Обучение моделей (пожалуйста, подождите)...")
            window.refresh()
            res = run_pipeline_step("train")
            window["-STATUS-"].update(res)

        elif event == "-DO_SHAP-":
            window["-STATUS-"].update("Построение SHAP...")
            window.refresh()
            res = run_pipeline_step("shap")
            window["-STATUS-"].update(res)

        elif event == "-PDF-":
            window["-STATUS-"].update("Генерация PDF...")
            window.refresh()
            res = run_pipeline_step("pdf")
            window["-STATUS-"].update(res)

        elif event == "-SHOW_DATA-":
            if os.path.exists("data/data_clean.csv"):
                df = pd.read_csv("data/data_clean.csv")
                text = df.head(15).to_string()
                sg.popup_scrolled(text, title="Очищенные данные", size=(80, 20))
            else:
                sg.popup_error("Файл данных не найден!")

        elif event == "-SHOW_METRICS-":
            if os.path.exists("data/results/model_comparison.csv"):
                df = pd.read_csv("data/results/model_comparison.csv")
                text = df.to_string(index=False)
                sg.popup_scrolled(text, title="Метрики моделей", size=(70, 10))
            else:
                sg.popup_error("Метрики не найдены (обучите модели).")

        elif event == "-EDA_CORR-":
            show_plot("data/eda_corr.png", "Матрица корреляций")
        elif event == "-EDA_BOX-":
            show_plot("data/eda_box.png", "Стоимость по классам риска")
        elif event == "-EDA_HIST-":
            show_plot("data/eda_hist.png", "Гистограмма прибыльности")

        elif event == "-SHAP_BAR-":
            show_plot("data/shap_bar.png", "SHAP Summary")
        elif event == "-SHAP_WATER-":
            show_plot("data/shap_waterfall.png", "SHAP Waterfall")

    window.close()

if __name__ == "__main__":
    start_gui()