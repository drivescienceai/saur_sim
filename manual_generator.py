"""
manual_generator.py
════════════════════════════════════════════════════════════════════════════════
Генератор PDF-мануала программы «Симуляция тушения пожара РВС — САУР ПСП».

Запуск:
    python -m saur_sim.manual_generator
    python saur_sim/manual_generator.py
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import os
import datetime
import io

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ══════════════════════════════════════════════════════════════════════════════
# ШРИФТЫ
# ══════════════════════════════════════════════════════════════════════════════

def _register_fonts():
    font_paths = [
        ("C:/Windows/Fonts/arial.ttf",   "Arial"),
        ("C:/Windows/Fonts/arialbd.ttf", "Arial-Bold"),
        ("C:/Windows/Fonts/ariali.ttf",  "Arial-Italic"),
        ("C:/Windows/Fonts/cour.ttf",    "Courier-Cyr"),
        ("C:/Windows/Fonts/courbd.ttf",  "Courier-Cyr-Bold"),
    ]
    reg = {}
    for path, name in font_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                reg[name] = True
            except Exception:
                pass
    return reg


_FONTS   = _register_fonts()
_F       = "Arial"       if "Arial"       in _FONTS else "Helvetica"
_FB      = "Arial-Bold"  if "Arial-Bold"  in _FONTS else "Helvetica-Bold"
_FI      = "Arial-Italic"if "Arial-Italic"in _FONTS else "Helvetica-Oblique"
_FM      = "Courier-Cyr" if "Courier-Cyr" in _FONTS else "Courier"
_FMB     = "Courier-Cyr-Bold" if "Courier-Cyr-Bold" in _FONTS else "Courier-Bold"

# Цвета
C_DARK   = colors.HexColor("#1a1f2e")
C_HEADER = colors.HexColor("#2c3e50")
C_FIRE   = colors.HexColor("#c0392b")
C_WATER  = colors.HexColor("#2471a3")
C_OK     = colors.HexColor("#27ae60")
C_WARN   = colors.HexColor("#e67e22")
C_PANEL  = colors.HexColor("#eaf2fb")
C_STRIPE = colors.HexColor("#f4f6f7")
C_BORDER = colors.HexColor("#aab7c4")
C_ACCENT = colors.HexColor("#e67e22")
C_NOTE   = colors.HexColor("#7f8c8d")


# ══════════════════════════════════════════════════════════════════════════════
# СТИЛИ
# ══════════════════════════════════════════════════════════════════════════════

def _make_styles() -> dict:
    return {
        "cover_title": ParagraphStyle("cover_title", fontName=_FB, fontSize=22,
                                       textColor=colors.white, alignment=TA_CENTER,
                                       leading=28, spaceAfter=8),
        "cover_sub":   ParagraphStyle("cover_sub", fontName=_F, fontSize=12,
                                       textColor=C_PANEL, alignment=TA_CENTER,
                                       spaceAfter=4),
        "cover_ver":   ParagraphStyle("cover_ver", fontName=_FI, fontSize=9,
                                       textColor=C_BORDER, alignment=TA_CENTER,
                                       spaceAfter=4),
        "h1": ParagraphStyle("h1", fontName=_FB, fontSize=14,
                              textColor=C_FIRE, spaceBefore=14, spaceAfter=5),
        "h2": ParagraphStyle("h2", fontName=_FB, fontSize=11,
                              textColor=C_HEADER, spaceBefore=10, spaceAfter=4),
        "h3": ParagraphStyle("h3", fontName=_FB, fontSize=9,
                              textColor=C_ACCENT, spaceBefore=7, spaceAfter=3),
        "body": ParagraphStyle("body", fontName=_F, fontSize=9,
                               textColor=colors.black, leading=14,
                               spaceAfter=4, alignment=TA_JUSTIFY),
        "bullet": ParagraphStyle("bullet", fontName=_F, fontSize=9,
                                  textColor=colors.black, leading=14,
                                  spaceAfter=2, leftIndent=14,
                                  bulletIndent=4),
        "mono":  ParagraphStyle("mono", fontName=_FM, fontSize=8,
                                 textColor=C_HEADER, leading=12,
                                 spaceAfter=2, leftIndent=10,
                                 backColor=colors.HexColor("#f4f6f7")),
        "note":  ParagraphStyle("note", fontName=_FI, fontSize=8,
                                 textColor=C_NOTE, spaceAfter=3),
        "warn_box": ParagraphStyle("warn_box", fontName=_FB, fontSize=9,
                                    textColor=C_WARN, spaceAfter=4,
                                    leftIndent=8),
        "ok_box":   ParagraphStyle("ok_box", fontName=_FB, fontSize=9,
                                    textColor=C_OK, spaceAfter=4,
                                    leftIndent=8),
        "toc":  ParagraphStyle("toc", fontName=_F, fontSize=9,
                                textColor=C_HEADER, spaceAfter=3, leftIndent=10),
        "toc_h": ParagraphStyle("toc_h", fontName=_FB, fontSize=10,
                                 textColor=C_DARK, spaceAfter=5),
    }


def _hr():
    return HRFlowable(width="100%", thickness=0.5, color=C_BORDER,
                      spaceBefore=2, spaceAfter=4)


def _table_style_manual(header_rows=1):
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, header_rows-1), C_HEADER),
        ("TEXTCOLOR",    (0, 0), (-1, header_rows-1), colors.white),
        ("FONTNAME",     (0, 0), (-1, header_rows-1), _FB),
        ("FONTSIZE",     (0, 0), (-1, header_rows-1), 8),
        ("ALIGN",        (0, 0), (-1, header_rows-1), "CENTER"),
        ("FONTNAME",     (0, header_rows), (-1, -1), _F),
        ("FONTSIZE",     (0, header_rows), (-1, -1), 8),
        ("ALIGN",        (0, header_rows), (0, -1),  "LEFT"),
        ("ROWBACKGROUNDS",(0, header_rows),(-1,-1), [colors.white, C_STRIPE]),
        ("GRID",         (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ])


def _info_box(text, sty, bg=None, border_color=None):
    """Создать информационный блок (таблица 1×1 с цветной рамкой)."""
    bg = bg or colors.HexColor("#fef9e7")
    border_color = border_color or C_WARN
    data = [[Paragraph(text, sty)]]
    tbl = Table(data, colWidths=[170*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), bg),
        ("BOX",          (0,0), (-1,-1), 1.2, border_color),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
    ]))
    return tbl


class _PageDecor:
    """Колонтитулы страниц мануала."""
    def __call__(self, canv: pdf_canvas.Canvas, doc):
        canv.saveState()
        w, h = A4
        # Верхний колонтитул (кроме первой страницы)
        if doc.page > 1:
            canv.setFillColor(C_HEADER)
            canv.rect(0, h-14*mm, w, 14*mm, fill=1, stroke=0)
            canv.setFont(_FB, 8)
            canv.setFillColor(colors.white)
            canv.drawString(15*mm, h-9*mm,
                            "РУКОВОДСТВО ПОЛЬЗОВАТЕЛЯ — СИМУЛЯЦИЯ ТУШЕНИЯ ПОЖАРА РВС")
            canv.setFont(_F, 7)
            canv.drawRightString(w-15*mm, h-9*mm, "САУР ПСП v1.0")
        # Нижний колонтитул
        canv.setFillColor(C_DARK)
        canv.rect(0, 0, w, 10*mm, fill=1, stroke=0)
        canv.setFont(_F, 7)
        canv.setFillColor(colors.white)
        canv.drawString(15*mm, 3*mm, "© 2025 САУР ПСП. Разработано для научно-исследовательских целей.")
        if doc.page > 1:
            canv.drawRightString(w-15*mm, 3*mm, f"Стр. {doc.page}")
        canv.restoreState()


# ══════════════════════════════════════════════════════════════════════════════
# ОСНОВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ МАНУАЛА
# ══════════════════════════════════════════════════════════════════════════════

def generate_manual(output_path: str = "") -> str:
    """Создать PDF-мануал программы.

    Args:
        output_path: путь к PDF (если пусто — авто в папке пакета)

    Returns:
        Путь к созданному файлу.
    """
    if not output_path:
        output_path = os.path.join(
            os.path.dirname(__file__),
            "SAUR_PSP_Manual.pdf"
        )

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=22*mm, bottomMargin=18*mm,
        title="Руководство пользователя — САУР ПСП",
        author="САУР ПСП v1.0",
    )

    S = _make_styles()
    story = []

    # ══════════════════════════════════════════════════════════
    # ОБЛОЖКА
    # ══════════════════════════════════════════════════════════
    story.append(Spacer(1, 30*mm))

    cover_data = [[
        Paragraph(
            "🔥  СИМУЛЯЦИЯ УПРАВЛЕНИЯ<br/>ТУШЕНИЕМ ПОЖАРА РВС<br/><br/>"
            "<font size='14'>РУКОВОДСТВО ПОЛЬЗОВАТЕЛЯ</font>",
            S["cover_title"]
        )
    ]]
    cover_tbl = Table(cover_data, colWidths=[170*mm])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 18),
        ("BOTTOMPADDING", (0,0), (-1,-1), 18),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
    ]))
    story.append(cover_tbl)
    story.append(Spacer(1, 8*mm))

    story.append(Paragraph("САУР ПСП v1.0", S["cover_sub"]))
    story.append(Paragraph(
        "Система адаптивного управления реагированием пожарно-спасательного подразделения",
        S["cover_sub"]
    ))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(
        "Программа предназначена для исследования сценариев тушения пожаров "
        "резервуарных парков методами дискретно-событийного моделирования (DES) "
        "и обучения с подкреплением (Reinforcement Learning).",
        S["body"]
    ))
    story.append(Spacer(1, 10*mm))

    meta_data = [
        ["Версия программы",  "1.0 (2025)"],
        ["Платформа",         "Python 3.12+, Windows 10/11"],
        ["Зависимости",       "tkinter, matplotlib, numpy, reportlab, python-docx"],
        ["Язык интерфейса",   "Русский"],
        ["Нормативная база",  "ГОСТ Р 51043-2002, СП 155.13130.2014, Справочник РТП"],
        ["Дата документа",    datetime.datetime.now().strftime("%d.%m.%Y")],
    ]
    mt = Table(meta_data, colWidths=[65*mm, 105*mm])
    mt.setStyle(_table_style_manual(0))
    mt.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (-1,-1), _F),
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("FONTNAME",  (0,0), (0,-1),  _FB),
        ("GRID",      (0,0), (-1,-1), 0.4, C_BORDER),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [colors.white, C_STRIPE]),
        ("TOPPADDING",(0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING",(0,0),(-1,-1), 8),
    ]))
    story.append(mt)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # СОДЕРЖАНИЕ
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("СОДЕРЖАНИЕ", S["h1"]))
    story.append(_hr())

    toc_items = [
        ("1.", "Назначение программы"),
        ("2.", "Системные требования и установка"),
        ("3.", "Интерфейс программы — обзор"),
        ("3.1.", "Панель управления"),
        ("3.2.", "Карта пожара (левая панель)"),
        ("3.3.", "Статусная таблица"),
        ("3.4.", "Вкладка «Хронология»"),
        ("3.5.", "Вкладка «Метрики»"),
        ("3.6.", "Вкладка «RL-агент»"),
        ("3.7.", "Вкладка «Справочник действий»"),
        ("3.8.", "Вкладка «Отчёт / Экспорт»"),
        ("3.9.", "Вкладка «Настройки»"),
        ("4.", "Сценарии моделирования"),
        ("4.1.", "Сценарий 1 — РВС №9, Туапсе (2025)"),
        ("4.2.", "Сценарий 2 — РВС №20, Серпухов (ПТП 2015)"),
        ("5.", "Система действий РТП (15 действий)"),
        ("6.", "Q-learning агент — описание алгоритма"),
        ("7.", "Физическая модель пенного тушения"),
        ("8.", "Генерация отчётов и экспорт данных"),
        ("8.1.", "PDF-отчёт для командира"),
        ("8.2.", "JSON-выгрузка для научной статьи"),
        ("8.3.", "DOCX-черновик для научной статьи"),
        ("9.", "Нормативная база"),
        ("10.", "Известные ограничения и дорожная карта"),
    ]
    for num, title in toc_items:
        indent = 10 if "." in num[:-1] else 0
        p = Paragraph(
            f"<b>{num}</b>  {title}",
            ParagraphStyle("toc_item", fontName=_F, fontSize=9,
                           leftIndent=indent, spaceAfter=3,
                           textColor=C_HEADER)
        )
        story.append(p)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # 1. НАЗНАЧЕНИЕ
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("1. Назначение программы", S["h1"]))
    story.append(_hr())
    story.append(Paragraph(
        "Программа «Симуляция тушения пожара РВС — САУР ПСП» предназначена для:",
        S["body"]
    ))
    bullets = [
        "исследования динамики развития и ликвидации пожаров резервуарных парков;",
        "обучения и тестирования RL-агентов управления реагированием ПСП;",
        "анализа нормативных требований (ГОСТ, СП) к расходам ОВ;",
        "подготовки учебных материалов и наглядных пособий для РТП и НШ;",
        "проведения вычислительных экспериментов для написания научных статей;",
        "генерации отчётов по результатам моделирования.",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", S["bullet"]))
    story.append(Spacer(1, 4*mm))

    story.append(_info_box(
        "⚠  Программа является учебно-исследовательским инструментом. "
        "Результаты моделирования не заменяют реальные оперативные решения РТП "
        "и не являются официальными нормативными расчётами.",
        S["warn_box"],
        bg=colors.HexColor("#fef9e7"),
        border_color=C_WARN
    ))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("Реализованные сценарии:", S["h2"]))
    scen_data = [
        ["Сценарий", "Объект", "Ранг", "Продолж.", "Описание"],
        ["Туапсе",
         "РВС №9, V=20 000 м³\nООО «РН-МТ Туапсе»",
         "№4", "81 ч 2 мин",
         "Реальный пожар 14–17.03.2025.\nПлавающая крыша, 6 пенных атак."],
        ["Серпухов (ПТП)",
         "РВС №20, V=2000 м³\nЗАО «Рос-Трейд»",
         "№2", "5 ч",
         "По ПТП 2015 М.А. Пересыпкина.\nКонусная кровля, 1 атака."],
    ]
    st = Table(scen_data, colWidths=[22*mm, 48*mm, 14*mm, 24*mm, 62*mm])
    st.setStyle(_table_style_manual(1))
    story.append(st)
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════
    # 2. СИСТЕМНЫЕ ТРЕБОВАНИЯ И УСТАНОВКА
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("2. Системные требования и установка", S["h1"]))
    story.append(_hr())

    req_data = [
        ["Компонент", "Требование"],
        ["ОС",             "Windows 10 / 11 (64-bit)"],
        ["Python",         "3.10 или выше (рекомендуется 3.12)"],
        ["RAM",            "≥ 4 ГБ (для matplotlib)"],
        ["Дисплей",        "≥ 1280×800 пикселей"],
        ["tkinter",        "Встроен в стандартную поставку Python"],
        ["numpy",          "pip install numpy"],
        ["matplotlib",     "pip install matplotlib"],
        ["reportlab",      "pip install reportlab"],
        ["python-docx",    "pip install python-docx"],
    ]
    rt = Table(req_data, colWidths=[50*mm, 120*mm])
    rt.setStyle(_table_style_manual(1))
    story.append(rt)
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("Установка зависимостей:", S["h3"]))
    story.append(Paragraph(
        "pip install numpy matplotlib reportlab python-docx",
        S["mono"]
    ))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("Запуск программы:", S["h3"]))
    story.append(Paragraph(
        "# Из корневой папки проекта:\npython -m saur_sim.tank_fire_sim\n\n"
        "# Или напрямую:\npython saur_sim/tank_fire_sim.py",
        S["mono"]
    ))
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════
    # 3. ИНТЕРФЕЙС ПРОГРАММЫ
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("3. Интерфейс программы — обзор", S["h1"]))
    story.append(_hr())
    story.append(Paragraph(
        "Главное окно программы состоит из трёх зон: "
        "верхняя строка заголовка (название сценария), "
        "основная рабочая область (разделена на левую и правую панели), "
        "нижняя панель управления (кнопки Пуск/Стоп/Сброс, скорость, прогресс-бар).",
        S["body"]
    ))
    story.append(Spacer(1, 4*mm))

    # 3.1 Панель управления
    story.append(Paragraph("3.1. Панель управления", S["h2"]))
    ctrl_data = [
        ["Элемент", "Функция"],
        ["▶  Пуск",       "Запустить (возобновить) анимированную симуляцию"],
        ["⏸  Стоп",       "Поставить симуляцию на паузу; обновить все графики"],
        ["⏮  Сброс",      "Сбросить симуляцию в начальное состояние (t=0)"],
        ["Скорость 1×–300×","Множитель скорости: сколько шагов (по 5 мин) за один тик GUI"],
        ["Прогресс",      "Текущая минута симуляции / максимум (%)"],
    ]
    cdt = Table(ctrl_data, colWidths=[45*mm, 125*mm])
    cdt.setStyle(_table_style_manual(1))
    story.append(cdt)
    story.append(Spacer(1, 5*mm))

    # 3.2 Карта пожара
    story.append(Paragraph("3.2. Карта пожара (левая панель)", S["h2"]))
    story.append(Paragraph(
        "Анимированный рисунок-схема территории объекта. Отображает:",
        S["body"]
    ))
    map_items = [
        "РВС №9 (горящий) — красно-оранжевый круг с пульсирующим пламенем;",
        "РВС №17 (охлаждаемый) — синий круг с ореолом охлаждения;",
        "Стволы охлаждения — синие стрелки с метками А1–А7;",
        "ПНС/ПАНРК — пунктирные линии водоснабжения;",
        "Боевые участки БУ-1…БУ-3 — полупрозрачные цветные прямоугольники;",
        "Розлив горящего топлива — оранжевое пятно с надписью;",
        "Здания (лаборатория, столовая) — меняют цвет при вторичном пожаре;",
        "Текущее действие РТП — информационная строка над р. Туапсе.",
    ]
    for item in map_items:
        story.append(Paragraph(f"• {item}", S["bullet"]))
    story.append(Spacer(1, 4*mm))

    # 3.3 Статусная таблица
    story.append(Paragraph("3.3. Статусная таблица", S["h2"]))
    status_data = [
        ["Поле", "Описание"],
        ["Время симуляции",   "Текущее время (реальное, от 03:20 14.03)"],
        ["Фаза пожара",       "S1–S5 с описанием фазы"],
        ["Площадь пожара",    "S(t), м² — текущая площадь зеркала горения"],
        ["Расход ОВ",         "Q(t), л/с — суммарный расход огнетушащих веществ"],
        ["Стволов на РВС",    "N_стволов (горящий РВС) + N (соседний РВС)"],
        ["ПНС на воде",       "Число ПНС/ПАНРК, установленных на водоисточники"],
        ["Боевых участков",   "Число созданных БУ из 3 возможных"],
        ["Пенных атак",       "Счётчик пенных атак; значок исхода (✅/🔒/⏳)"],
        ["Риск",              "Уровень (НИЗКИЙ/СРЕДНИЙ/ВЫСОКИЙ/КРИТИЧЕСКИЙ) и значение"],
        ["Действие RL",       "Код и описание последнего действия RL-агента"],
        ["Препятствие крыши", "Доля обструкции каркасом плавающей крыши (ГОСТ)"],
        ["Расход пены",       "Фактический расход пенного раствора vs норматив"],
    ]
    sdt = Table(status_data, colWidths=[45*mm, 125*mm])
    sdt.setStyle(_table_style_manual(1))
    story.append(sdt)
    story.append(Spacer(1, 5*mm))

    # 3.4–3.9
    tabs_info = [
        ("3.4.", "Вкладка «Хронология»",
         "Отображает хронологию реального пожара (из PDF-протокола) и "
         "текущий журнал событий симуляции. Цветовая кодировка: "
         "красный — опасность, оранжевый — предупреждение, "
         "зелёный — успешное событие, синий — информация."),
        ("3.5.", "Вкладка «Метрики»",
         "Четыре динамических графика: площадь пожара S(t), расход ОВ Q(t), "
         "число стволов охлаждения N(t), индекс риска R(t). "
         "Вертикальные пунктирные линии на графике S(t) — моменты пенных атак "
         "(зелёные — успешные, красные — неудачные). "
         "Обновляются каждые ~2 секунды по времени GUI."),
        ("3.6.", "Вкладка «RL-агент»",
         "Три графика: Q-значения действий в текущем состоянии агента "
         "(цветные столбцы: красный=стратегический, оранжевый=тактический, "
         "синий=оперативный; жёлтый контур = выбранное действие), "
         "частота выбора действий за сессию, "
         "кривая накопленной награды с MA-20."),
        ("3.7.", "Вкладка «Справочник действий»",
         "Перечень рекомендуемых действий РТП для каждой из фаз S1–S5. "
         "Выбрать фазу с помощью радиокнопок. Включает код действия, "
         "название и подсказку по реализации."),
        ("3.8.", "Вкладка «Отчёт / Экспорт»",
         "Кнопки формирования итоговых документов по результатам симуляции: "
         "PDF-отчёт для оперативного штаба, "
         "JSON-выгрузка для написания научной статьи, "
         "DOCX-черновик с готовыми текстовыми фрагментами для статьи. "
         "Также содержит краткую сводку итогов текущей сессии."),
        ("3.9.", "Вкладка «Настройки»",
         "Позволяет изменить параметры до нажатия «Применить»: "
         "сценарий моделирования, начальная площадь пожара, скорость распространения, "
         "объём РВС, эффективность пены, надёжность техники, "
         "запас пенообразователя, гиперпараметры RL-агента (ε, α, γ), "
         "режим управления (обучение с подкреплением / эксплуатация агента / ручное)."
         "Раздел «Ручное действие РТП» позволяет выбрать действие из списка "
         "и выполнить его немедленно кнопкой «▶ Выполнить действие»."),
    ]
    for num, title, desc in tabs_info:
        story.append(Paragraph(f"{num} {title}", S["h2"]))
        story.append(Paragraph(desc, S["body"]))
        story.append(Spacer(1, 3*mm))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # 4. СЦЕНАРИИ
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("4. Сценарии моделирования", S["h1"]))
    story.append(_hr())

    story.append(Paragraph("4.1. Сценарий 1 — РВС №9, г. Туапсе (14–17.03.2025)", S["h2"]))
    story.append(Paragraph(
        "Основан на реальном протоколе тушения пожара РВС №9 ООО «РН-Морской "
        "терминал Туапсе», Краснодарский край. Ранг пожара №4 (наивысший). "
        "Продолжительность 81 ч 02 мин (4862 мин симуляции).",
        S["body"]
    ))
    t1_data = [
        ["Параметр", "Значение"],
        ["Объём РВС",           "20 000 м³"],
        ["Диаметр",             "≈ 40 м"],
        ["Продукт",             "Прямогонный бензин"],
        ["S зеркала горения",   "1250 м² (100% зеркала)"],
        ["Плавающая крыша",     "Да (препятствие 70%)"],
        ["Норм. Q_пены",        "81.3 л/с (I=0.065 л/(м²·с))"],
        ["Пенных атак в ПТП",   "6 (5 неудачных + 1 успешная с АКП-50)"],
        ["АКП-50",              "Ключевой ресурс: снижает препятствие до 20%"],
        ["Число ПНС",           "4 (р. Туапсе, ПАНРК × 2, водоём)"],
    ]
    tt1 = Table(t1_data, colWidths=[60*mm, 110*mm])
    tt1.setStyle(_table_style_manual(1))
    story.append(tt1)
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("4.2. Сценарий 2 — РВС №20, г. Серпухов (ПТП 2015)", S["h2"]))
    story.append(Paragraph(
        "Основан на плане тушения пожара ЗАО «Рос-Трейд», г. Серпухов. "
        "Вариант №2 ПТП: пожар РВС №20, V=2000 м³. Ранг пожара №2. "
        "Продолжительность по ПТП: 300 мин.",
        S["body"]
    ))
    t2_data = [
        ["Параметр", "Значение"],
        ["Объём РВС",           "2 000 м³"],
        ["Диаметр",             "14.62 м"],
        ["Продукт",             "Бензин АИ-92/95"],
        ["S зеркала горения",   "168 м²"],
        ["Тип кровли",          "Конусная (нет плавающей крыши, препятствие 0%)"],
        ["Норм. Q_пены",        "8.4 л/с (I=0.05 л/(м²·с))"],
        ["ГПС-600 (3 шт.)",     "Q=3×5.64=16.9 л/с — достаточно с первой атаки"],
        ["Пожарные гидранты",   "ПГ-106, ПГ-108 (≈15 л/с каждый)"],
        ["Источник",            "ПТП, сост. М.А. Пересыпкин, утв. 26.05.2015"],
    ]
    tt2 = Table(t2_data, colWidths=[60*mm, 110*mm])
    tt2.setStyle(_table_style_manual(1))
    story.append(tt2)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # 5. СИСТЕМА ДЕЙСТВИЙ
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("5. Система действий РТП (15 действий)", S["h1"]))
    story.append(_hr())
    story.append(Paragraph(
        "Агент (и оператор в ручном режиме) управляет тушением через 15 "
        "дискретных действий трёх уровней. Доступность действий ограничена "
        "маской фазы пожара и оперативной обстановкой.",
        S["body"]
    ))
    story.append(Spacer(1, 3*mm))

    act_data = [
        ["Код", "Уровень", "Описание", "Доступна в фазах"],
        ["S1", "Стратег.", "Спасение людей (РН угрозы жизни)", "S1, S4"],
        ["S2", "Стратег.", "Защита соседнего РВС №17", "S2, S3"],
        ["S3", "Стратег.", "Локализация горения в контуре РВС №9", "S1–S3"],
        ["S4", "Стратег.", "Ликвидация горения — пенная атака", "S3, S4"],
        ["S5", "Стратег.", "Предотвращение вскипания нефти", "S3, S4"],
        ["T1", "Тактич.",  "Создать боевые участки по секторам", "S2–S3"],
        ["T2", "Тактич.",  "Перегруппировать силы и средства", "S3–S5"],
        ["T3", "Тактич.",  "Вызов дополнительных С и С", "S1, S3, S4"],
        ["T4", "Тактич.",  "Установить ПНС/ПАНРК на водоисточник", "S1–S3"],
        ["O1", "Операт.",  "Подать ствол Антенор на охлаждение РВС №9", "S1–S3"],
        ["O2", "Операт.",  "Охлаждение РВС №17 (орошение + стволы)", "S2–S4"],
        ["O3", "Операт.",  "Пенная атака (Акрон/Муссон/ЛС-С330/ГПС)", "S2–S4"],
        ["O4", "Операт.",  "Разведка пожара — уточнение обстановки", "S1–S5"],
        ["O5", "Операт.",  "Ликвидация розлива горящего топлива", "S3 (при розливе)"],
        ["O6", "Операт.",  "Сигнал отхода — экстренный вывод ЛС", "При R>0.85"],
    ]
    at = Table(act_data, colWidths=[14*mm, 18*mm, 90*mm, 48*mm])
    at.setStyle(_table_style_manual(1))
    story.append(at)
    story.append(Spacer(1, 5*mm))

    story.append(_info_box(
        "ℹ  Награды за действия: наивысшую награду (+12.0) получает успешная "
        "пенная атака (O3/S4). Действие «Разведка» (O4) даёт r=+0.05 каждый шаг, "
        "что обеспечивает базовый сигнал обучения. "
        "Отход (O6) штрафуется −1.0.",
        S["warn_box"],
        bg=colors.HexColor("#eaf2fb"),
        border_color=C_WATER
    ))
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════
    # 6. RL-АГЕНТ
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("6. Q-learning агент — описание алгоритма", S["h1"]))
    story.append(_hr())
    story.append(Paragraph(
        "Управление симуляцией осуществляет табличный Q-learning агент:",
        S["body"]
    ))

    rl_data = [
        ["Параметр", "Значение по умолчанию", "Назначение"],
        ["Пространство состояний", "128 дискретных состояний",
         "Фаза (5) × стволы (4) × ПНС (4) × пена (2) × розлив (2) × ..."],
        ["Пространство действий", "15 действий (S1-S5, T1-T4, O1-O6)",
         "Иерархическая структура: стратег./тактич./оперативн."],
        ["Скорость обучения α", "0.15", "Шаг обновления Q-таблицы"],
        ["Дисконтирование γ",  "0.95", "Предпочтение будущих наград"],
        ["Начальный ε",        "0.90", "Исследование vs эксплуатация"],
        ["Затухание ε",        "×0.99 за эпизод", "После каждого вызова end_episode()"],
        ["Минимальный ε",      "0.05", "Гарантированный уровень исследования"],
    ]
    rlt = Table(rl_data, colWidths=[48*mm, 42*mm, 80*mm])
    rlt.setStyle(_table_style_manual(1))
    story.append(rlt)
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("Формула обновления Q-таблицы:", S["h3"]))
    story.append(Paragraph(
        "Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') − Q(s, a)]",
        S["mono"]
    ))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "Для перехода от обучения к эксплуатации установите ε = 0.05 "
        "в разделе Настройки → «Параметры RL-агента» и выберите режим "
        "«RL эксплуатация (greedy)».",
        S["body"]
    ))
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════
    # 7. ФИЗИЧЕСКАЯ МОДЕЛЬ ПЕННОГО ТУШЕНИЯ
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("7. Физическая модель пенного тушения", S["h1"]))
    story.append(_hr())
    story.append(Paragraph(
        "Модель основана на ГОСТ Р 51043-2002 (Изменение №1) и "
        "СП 155.13130.2014. Ключевой принцип: каркас плавающей крыши "
        "блокирует часть расхода пены — не весь поданный расход достигает "
        "зеркала горения.",
        S["body"]
    ))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("Расчётная схема:", S["h3"]))
    formulas = [
        "Q_эфф  = Q_total × (1 − k_обструкции)   [л/с]",
        "Q_норм = I_норм × S_зеркала              [л/с]",
        "Атака успешна:  Q_эфф ≥ Q_норм",
        "",
        "Условие блокировки (отдельная проверка):",
        "  k_обструкции ≥ 0.70  →  атака невозможна независимо от Q_total",
    ]
    for f in formulas:
        story.append(Paragraph(f, S["mono"]))
    story.append(Spacer(1, 3*mm))

    foam_cases = [
        ["Ситуация", "k_обструкции", "Пример", "Исход"],
        ["Конусная кровля (Серпухов)",
         "0%", "РВС №20 (ПТП)", "Атака успешна с 1-й попытки"],
        ["Плавающая крыша, норм. состояние",
         "30–40%", "Теоретическая"],
         ["Плавающая крыша, деформация",
         "65–70%", "РВС №9 Туапсе, атаки №1–5", "❌ Атака неудачна"],
        ["АКП-50 + ГПС-1000 через люк",
         "20%", "РВС №9 Туапсе, атака №6", "✅ Атака успешна"],
    ]
    ft = Table(foam_cases, colWidths=[52*mm, 25*mm, 55*mm, 38*mm])
    ft.setStyle(_table_style_manual(1))
    story.append(ft)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # 8. ГЕНЕРАЦИЯ ОТЧЁТОВ И ЭКСПОРТ
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("8. Генерация отчётов и экспорт данных", S["h1"]))
    story.append(_hr())
    story.append(Paragraph(
        "Вкладка «Отчёт / Экспорт» позволяет сохранить результаты симуляции "
        "в нескольких форматах. Все файлы сохраняются в папку пакета "
        "(saur_sim/) с автоматическим именем, содержащим сценарий и временну́ю метку.",
        S["body"]
    ))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("8.1. PDF-отчёт", S["h2"]))
    story.append(Paragraph(
        "Полный PDF-отчёт для командира подразделения или руководства МЧС. "
        "Содержит: титульный лист с итогами, нормативные требования, "
        "таблицу результатов, 4 графика динамики, 3 графика RL-агента, "
        "хронологию событий, выводы и рекомендации.",
        S["body"]
    ))
    story.append(Paragraph("Имя файла: report_<сценарий>_<дата_время>.pdf", S["mono"]))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("8.2. JSON-выгрузка для научной статьи", S["h2"]))
    story.append(Paragraph(
        "Структурированный JSON-файл с метаданными, параметрами сценария, "
        "итоговыми показателями, временны́ми рядами, статистикой RL-агента "
        "и нормативным анализом. Предназначен для программной обработки "
        "и вставки в «Результаты» статьи.",
        S["body"]
    ))
    story.append(Paragraph("Имя файла: article_data_<сценарий>_<дата_время>.json", S["mono"]))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("8.3. DOCX-черновик для научной статьи", S["h2"]))
    story.append(Paragraph(
        "Документ Word с готовыми текстовыми фрагментами для разделов: "
        "«Объект и методология», «Результаты», «Обсуждение», "
        "а также шаблонами подписей к рисункам и таблицам. "
        "Автор использует документ как основу для редактирования статьи.",
        S["body"]
    ))
    story.append(Paragraph("Имя файла: article_draft_<сценарий>_<дата_время>.docx", S["mono"]))
    story.append(Spacer(1, 5*mm))

    story.append(_info_box(
        "💡 Совет: для получения качественного отчёта рекомендуется запустить "
        "симуляцию на скорости 60× или 300× до завершения (т=4862 или т=300), "
        "затем нажать «Стоп» и перейти на вкладку «Отчёт / Экспорт».",
        S["ok_box"],
        bg=colors.HexColor("#eafaf1"),
        border_color=C_OK
    ))
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════
    # 9. НОРМАТИВНАЯ БАЗА
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("9. Нормативная база", S["h1"]))
    story.append(_hr())

    norms_list = [
        ["Документ", "Применение в программе"],
        ["ГОСТ Р 51043-2002 (Изм. №1, 2020)",
         "Коэффициент производительности K, интенсивность орошения, "
         "формулы расхода пены. Основа физической модели тушения."],
        ["СП 155.13130.2014",
         "Нормативный расход охлаждения стенок РВС "
         "(0.8 л/(с·м) — горящий, 0.3 л/(с·м) — соседний). "
         "Требования к пенному тушению РВС."],
        ["Справочник РТП (ВНИИПО, 2021), стр. 104–106",
         "Нормы интенсивности подачи пенного раствора по видам топлива: "
         "бензин I=0.05 л/(м²·с), нефть I=0.06. Нормы охлаждения."],
        ["ПТП нефтебазы ЗАО «Рос-Трейд», г. Серпухов, 2015",
         "Параметры сценария «Серпухов»: Q_тр=66 л/с, N_ГПС=3, "
         "N_ств_А=7. Составитель: М.А. Пересыпкин."],
        ["Протокол тушения РВС №9, Туапсе, 14–17.03.2025",
         "Параметры сценария «Туапсе»: 81 ч 02 мин, ранг №4, "
         "хронология 48 событий, 6 пенных атак."],
    ]
    nt = Table(norms_list, colWidths=[55*mm, 115*mm])
    nt.setStyle(_table_style_manual(1))
    story.append(nt)
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════════
    # 10. ОГРАНИЧЕНИЯ И ДОРОЖНАЯ КАРТА
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("10. Известные ограничения и дорожная карта", S["h1"]))
    story.append(_hr())

    story.append(Paragraph("Текущие ограничения:", S["h2"]))
    limits = [
        "Карта пожара отрисована для сценария «Туапсе»; для «Серпухов» используется та же схема;",
        "RL-агент является табличным (128 состояний); глубокое обучение (DQN) не реализовано;",
        "Метеорологические условия (направление ветра, скорость) не влияют на модель;",
        "Водоснабжение моделируется упрощённо (нет гидравлического расчёта сети);",
        "Только два сценария; пользовательские сценарии не поддерживаются в GUI;",
        "Многопоточность отсутствует — GUI может «подвисать» при скорости 300×.",
    ]
    for lim in limits:
        story.append(Paragraph(f"• {lim}", S["bullet"]))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("Планируемые улучшения:", S["h2"]))
    roadmap = [
        "Добавить карту для сценария «Серпухов» с правильным масштабом РВС;",
        "Реализовать DQN-агент на базе PyTorch для обучения по опыту;",
        "Добавить параметр ветра и его влияние на скорость горения и эффективность пены;",
        "Поддержка пользовательских сценариев через JSON-конфигурацию;",
        "Веб-интерфейс (Flask/Dash) для доступа без установки Python;",
        "Интеграция с базой данных МЧС для автоматической загрузки ПТП.",
    ]
    for item in roadmap:
        story.append(Paragraph(f"→ {item}", S["bullet"]))
    story.append(Spacer(1, 6*mm))

    story.append(_hr())
    story.append(Paragraph(
        f"Документ сгенерирован автоматически | САУР ПСП v1.0 | "
        f"{datetime.datetime.now().strftime('%d.%m.%Y')}",
        S["note"]
    ))

    # Сборка PDF
    page_decor = _PageDecor()
    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out = generate_manual()
    print(f"✅ Мануал сохранён: {out}")
