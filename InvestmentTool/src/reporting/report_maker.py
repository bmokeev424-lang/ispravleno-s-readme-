from reportlab.pdfgen import canvas
import os

def make_pdf_report():
    print("Генерируем PDF отчет...")
    os.makedirs("data", exist_ok=True)
    c = canvas.Canvas("data/report.pdf")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Отчет: Инвестиционный комитет")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, "1. Данные сгенерированы и очищены.")
    c.drawString(50, 750, "2. Нечеткая логика использована для объяснения рисков.")
    c.drawString(50, 730, "3. Модели обучены, лучшая сохранена.")
    c.save()
    print("PDF готов!")