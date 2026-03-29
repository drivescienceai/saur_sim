"""
generate_dissertation_scheme.py — Структурно-логическая схема диссертации.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "figures")
os.makedirs(_OUT, exist_ok=True)


def generate_scheme(filename: str = "dissertation_scheme.png") -> str:

    fig, ax = plt.subplots(figsize=(22, 30), facecolor="white")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 30)
    ax.axis("off")

    # ── Цвета ─────────────────────────────────────────────────────────────
    C_GOAL    = "#c0392b"   # цель
    C_TASK    = "#2980b9"   # задачи
    C_MODEL   = "#27ae60"   # модели
    C_METHOD  = "#e67e22"   # методы
    C_ALGO    = "#8e44ad"   # алгоритмы
    C_SW      = "#1abc9c"   # программное обеспечение
    C_RESULT  = "#d4ac0d"   # результаты
    C_CHAPTER = "#ecf0f1"   # главы
    C_ARROW   = "#7f8c8d"
    C_TEXT    = "#2c3e50"

    def box(x, y, w, h, text, color, fontsize=8, text_color="white", alpha=0.9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor="white",
                              linewidth=1.5, alpha=alpha, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold",
                wrap=True, zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", facecolor=color,
                          edgecolor="none", alpha=0.0))

    def arrow(x1, y1, x2, y2, color=C_ARROW, style="-|>", lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                   lw=lw, connectionstyle="arc3,rad=0.0"),
                    zorder=2)

    def label(x, y, text, color=C_TEXT, fontsize=7, ha="center"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
                color=color, style="italic", zorder=5)

    def section_label(x, y, text, color="#95a5a6"):
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                color=color, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa",
                          edgecolor=color, linewidth=1.5))

    # ══════════════════════════════════════════════════════════════════════
    # УРОВЕНЬ 1: ЦЕЛЬ ИССЛЕДОВАНИЯ
    # ══════════════════════════════════════════════════════════════════════
    Y = 28.5
    box(5, Y, 12, 1.0, "ЦЕЛЬ: Разработка моделей и методов адаптивного\n"
        "управления реагированием ПСП при тушении пожаров РВС", C_GOAL, 9)

    # ══════════════════════════════════════════════════════════════════════
    # УРОВЕНЬ 2: ЗАДАЧИ ИССЛЕДОВАНИЯ
    # ══════════════════════════════════════════════════════════════════════
    Y_TASKS = 26.5
    section_label(2, Y_TASKS + 0.5, "ЗАДАЧИ")

    tasks = [
        (0.5,  "З1: Анализ проблемы\nуправления ПСП\nи обзор подходов"),
        (4.0,  "З2: Разработка\nполумарковской модели\nи стат. анализ"),
        (7.5,  "З3: Разработка ЭС\nи прецедентного\nанализа"),
        (11.0, "З4: Разработка\nметодов ОП для\nадаптивного управления"),
        (14.5, "З5: Создание\nпрограммной платформы\nи эксперименты"),
        (18.0, "З6: Практическое\nприменение и\nрекомендации"),
    ]
    for x, text in tasks:
        box(x, Y_TASKS - 0.6, 3.2, 1.1, text, C_TASK, 7)
        arrow(11, Y, 11, Y_TASKS + 0.5, C_GOAL, lw=2)

    # Стрелки от цели к задачам
    for x, _ in tasks:
        arrow(11, Y, x + 1.6, Y_TASKS + 0.5, C_GOAL, lw=1)

    # ══════════════════════════════════════════════════════════════════════
    # УРОВЕНЬ 3: МОДЕЛИ (Глава 2)
    # ══════════════════════════════════════════════════════════════════════
    Y_MOD = 24.0
    section_label(2, Y_MOD + 0.5, "МОДЕЛИ")

    models = [
        (0.5,  "Полумарковская\nмодель фаз пожара\nS1→S2→S3→S4→S5"),
        (4.2,  "Модель Вейбулла\nF(t)=1−exp[−(t/λ)^k]\nкалибровка на N=300+"),
        (7.9,  "Модель риска\nR = 0.40·staleness +\n0.35·fire + 0.25·deficit"),
        (11.6, "Модель адаптации\nΦ: δ_s → режим →\nN→T→O→M→D"),
        (15.3, "Модель автономности\nα = 1−I(a∈MAP[g])\nα ≈ 1/(1+3λ)"),
        (19.0, "Модель качества\nL7 = Π(ρ_i^w_i)×\n(1−deficit)×phase_f"),
    ]
    for x, text in models:
        box(x, Y_MOD - 0.6, 3.2, 1.1, text, C_MODEL, 7)

    # Связи задачи → модели
    arrow(5.6, Y_TASKS - 0.6, 2.1, Y_MOD + 0.5, C_TASK, lw=1)
    arrow(5.6, Y_TASKS - 0.6, 5.8, Y_MOD + 0.5, C_TASK, lw=1)
    arrow(12.6, Y_TASKS - 0.6, 9.5, Y_MOD + 0.5, C_TASK, lw=1)
    arrow(12.6, Y_TASKS - 0.6, 13.2, Y_MOD + 0.5, C_TASK, lw=1)
    arrow(12.6, Y_TASKS - 0.6, 16.9, Y_MOD + 0.5, C_TASK, lw=1)
    arrow(12.6, Y_TASKS - 0.6, 20.6, Y_MOD + 0.5, C_TASK, lw=1)

    # ══════════════════════════════════════════════════════════════════════
    # УРОВЕНЬ 4: МЕТОДЫ (Главы 2, 3, 4)
    # ══════════════════════════════════════════════════════════════════════
    Y_METH = 21.3
    section_label(2, Y_METH + 0.5, "МЕТОДЫ")

    methods = [
        (0.3,  "Статистический\nанализ: корреляции,\nANOVA, регрессия"),
        (3.5,  "Обнаружение\nточек разладки:\nCUSUM, Байес"),
        (6.7,  "Экспертная система\n27 правил БУПО\n«ЕСЛИ-ТО»"),
        (9.9,  "Прецедентный\nанализ (CBR):\nk-NN, K-Means"),
        (13.1, "Обучение с\nподкреплением:\n10 режимов"),
        (16.3, "Обратное обучение\nMaxEnt IRL:\nвосстановление R(t)"),
        (19.5, "Мультиагентное\nОП: 5 агентов\nРТП+НБУ×3+НТ"),
    ]
    for x, text in methods:
        box(x, Y_METH - 0.6, 2.9, 1.1, text, C_METHOD, 6.5)

    # Связи модели → методы
    for i, (xm, _) in enumerate(models):
        for j, (xme, _) in enumerate(methods):
            if abs(i - j) <= 1 or (i >= 3 and j >= 4):
                arrow(xm + 1.6, Y_MOD - 0.6, xme + 1.45, Y_METH + 0.5,
                      C_MODEL, lw=0.7)

    # ══════════════════════════════════════════════════════════════════════
    # УРОВЕНЬ 5: АЛГОРИТМЫ (Глава 4)
    # ══════════════════════════════════════════════════════════════════════
    Y_ALG = 18.5
    section_label(2, Y_ALG + 0.5, "АЛГОРИТМЫ")

    algorithms = [
        (0.5,  "Q-learning\nтабличный\n315×15, ε-greedy"),
        (3.7,  "Иерархический\nHRL моноагент\nL3→L2→L1"),
        (6.9,  "Мультиагентный\nМАОП 5 агентов\nлокальные набл."),
        (10.1, "Клонирование\nповедения\nP(a|s) supervised"),
        (13.3, "Обратное ОП\nMaxEnt IRL\n4 признака φ(s,a)"),
        (16.5, "Пакетное ОП\nFitted Q-Iteration\noffline данные"),
        (19.7, "ОП по предпочт.\nBradley-Terry\nиз СППР"),
    ]
    for x, text in algorithms:
        box(x, Y_ALG - 0.6, 2.8, 1.1, text, C_ALGO, 6.5)

    # Связи методы → алгоритмы
    arrow(14.55, Y_METH - 0.6, 1.9, Y_ALG + 0.5, C_METHOD, lw=0.8)
    arrow(14.55, Y_METH - 0.6, 5.1, Y_ALG + 0.5, C_METHOD, lw=0.8)
    arrow(20.95, Y_METH - 0.6, 8.3, Y_ALG + 0.5, C_METHOD, lw=0.8)
    arrow(17.75, Y_METH - 0.6, 11.5, Y_ALG + 0.5, C_METHOD, lw=0.8)
    arrow(17.75, Y_METH - 0.6, 14.7, Y_ALG + 0.5, C_METHOD, lw=0.8)
    arrow(14.55, Y_METH - 0.6, 17.9, Y_ALG + 0.5, C_METHOD, lw=0.8)
    arrow(14.55, Y_METH - 0.6, 21.1, Y_ALG + 0.5, C_METHOD, lw=0.8)

    # ══════════════════════════════════════════════════════════════════════
    # УРОВЕНЬ 6: ПРОГРАММНОЕ ОБЕСПЕЧЕНИЕ (Глава 5)
    # ══════════════════════════════════════════════════════════════════════
    Y_SW = 15.5
    section_label(2, Y_SW + 0.5, "ПРОГРАММНОЕ\nОБЕСПЕЧЕНИЕ")

    sw_modules = [
        (0.3,  "Симулятор\ntank_fire_sim.py\nГОСТ, СП 155"),
        (3.3,  "Агенты ОП\nrl_agent.py\nhrl_agents.py"),
        (6.3,  "Мультиагент\nmulti_agent.py\n5 агентов"),
        (9.3,  "ЭС + CBR\nexpert_system.py\ncbr_engine.py"),
        (12.3, "Анализ\nstat_analysis.py\ntimeseries.py"),
        (15.3, "Адаптация\nadaptation_model\nautonomy_analysis"),
        (18.3, "Данные + отчёты\nptp_parser.py\nresults_db.py"),
    ]
    for x, text in sw_modules:
        box(x, Y_SW - 0.6, 2.7, 1.1, text, C_SW, 6.5, text_color="white")

    # Стрелки алгоритмы → ПО
    for xa, _ in algorithms:
        for xs, _ in sw_modules:
            if abs(xa - xs) < 4:
                arrow(xa + 1.4, Y_ALG - 0.6, xs + 1.35, Y_SW + 0.5,
                      C_ALGO, lw=0.5)

    # Блок платформы
    box(0.2, Y_SW - 2.0, 21.6, 0.8,
        "ПЛАТФОРМА САУР-ПСП  —  42 модуля  |  ~25 000 строк Python  |  "
        "10 режимов  |  15+ визуализаций  |  118 тестов  |  "
        "централизованная БД результатов",
        "#34495e", 8, text_color="#ecf0f1")

    # ══════════════════════════════════════════════════════════════════════
    # УРОВЕНЬ 7: ЭКСПЕРИМЕНТАЛЬНЫЕ РЕЗУЛЬТАТЫ
    # ══════════════════════════════════════════════════════════════════════
    Y_RES = 11.5
    section_label(2, Y_RES + 0.5, "РЕЗУЛЬТАТЫ")

    results = [
        (0.3,  "Калиброванная\nполумарковская\nмодель (k, λ)\nна N=300+ случаях"),
        (3.8,  "Стат. значимость:\nранг→длит. p<0.001\nη²=0.526\nR²=0.55"),
        (7.3,  "Сравнение 4 агентов:\nЭС vs Таблич. vs\nИерарх. vs Мульти\nМанн-Уитни, d Коэна"),
        (10.8, "Автономность α:\nmax α=0.73 в S3\nα(λ)≈1/(1+3λ)\nранг 4: 0.53 vs 2: 0.20"),
        (14.3, "IRL: восстановл.\nвеса приоритетов\nw_тяжесть=−0.42\nw_стоимость=−0.03"),
        (17.8, "Точки разладки:\nCUSUM + Байес\nсовпадение с\nмоментами адаптации"),
    ]
    for x, text in results:
        box(x, Y_RES - 0.8, 3.2, 1.4, text, C_RESULT, 6.5, text_color=C_TEXT)

    arrow(11, Y_SW - 2.0, 11, Y_RES + 0.5, "#34495e", lw=2)

    # ══════════════════════════════════════════════════════════════════════
    # УРОВЕНЬ 8: НАУЧНАЯ НОВИЗНА
    # ══════════════════════════════════════════════════════════════════════
    Y_NOV = 8.5
    section_label(2, Y_NOV + 0.5, "НАУЧНАЯ\nНОВИЗНА")

    novelty = [
        (0.5,  "Н1: Формальная модель\nадаптации Φ с критерием δ_s\n"
               "и теоремами о монотонности\nи сходимости"),
        (5.8,  "Н2: Калиброванная\nполумарковская модель\n"
               "пожара РВС (k, λ для S1-S5)\nна 300+ реальных случаях"),
        (11.1, "Н3: Измеримый коэффициент\nавтономности α ∈ [0;1]\n"
               "с зависимостью α≈1/(1+kλ)\nот фазы и ранга"),
        (16.4, "Н4: Метод восстановления\nскрытых приоритетов РТП\n"
               "(IRL) + сравнение 10 методов\nуправления на единой области"),
    ]
    for x, text in novelty:
        box(x, Y_NOV - 0.8, 4.8, 1.3, text, C_GOAL, 7, alpha=0.85)

    # Связи результаты → новизна
    for xr, _ in results:
        for xn, _ in novelty:
            if abs(xr - xn) < 6:
                arrow(xr + 1.6, Y_RES - 0.8, xn + 2.4, Y_NOV + 0.5,
                      C_RESULT, lw=0.7)

    # ══════════════════════════════════════════════════════════════════════
    # УРОВЕНЬ 9: ПРАКТИЧЕСКАЯ ЗНАЧИМОСТЬ
    # ══════════════════════════════════════════════════════════════════════
    Y_PRAC = 5.5
    section_label(2, Y_PRAC + 0.5, "ПРАКТИЧЕСКАЯ\nЗНАЧИМОСТЬ")

    practical = [
        (0.5,  "Тренажёр РТП:\nбалльная оценка решений,\nдебрифинг, PDF-отчёт"),
        (5.5,  "СППР для штаба:\nрекомендации агента,\nпрецедентный поиск"),
        (10.5, "Нормативный анализ:\nавтопроверка ГОСТ/СП,\nвыявление отклонений"),
        (15.5, "Научная платформа:\n300+ сценариев, 10 режимов,\nполная статистика"),
    ]
    for x, text in practical:
        box(x, Y_PRAC - 0.6, 4.5, 1.1, text, C_SW, 7, alpha=0.8)

    for xn, _ in novelty:
        for xp, _ in practical:
            if abs(xn - xp) < 7:
                arrow(xn + 2.4, Y_NOV - 0.8, xp + 2.25, Y_PRAC + 0.5,
                      C_GOAL, lw=0.7)

    # ══════════════════════════════════════════════════════════════════════
    # УРОВЕНЬ 10: ГЛАВЫ ДИССЕРТАЦИИ (привязка)
    # ══════════════════════════════════════════════════════════════════════
    Y_CH = 3.0
    chapters = [
        (0.3,  "Глава 1\nАнализ проблемы", "#bdc3c7"),
        (3.7,  "Глава 2\nМарковские модели\n+ статистика", "#bdc3c7"),
        (7.1,  "Глава 3\nЭС + прецеденты", "#bdc3c7"),
        (10.5, "Глава 4\nМодели и методы ОП", "#bdc3c7"),
        (13.9, "Глава 5\nПлатформа +\nэксперименты", "#bdc3c7"),
        (17.3, "Глава 6\nПрименение +\nрекомендации", "#bdc3c7"),
    ]
    for x, text, c in chapters:
        box(x, Y_CH - 0.4, 3.1, 0.9, text, c, 7, text_color=C_TEXT, alpha=0.6)

    # Связи от глав вверх
    chapter_links = [
        (0, [0]),        # Гл.1 → Задача 1
        (1, [0, 1]),     # Гл.2 → Модели 0,1, Методы 0,1
        (2, [2, 3]),     # Гл.3 → Методы 2,3
        (3, [4, 5, 6]),  # Гл.4 → Алгоритмы
        (4, []),         # Гл.5 → ПО
        (5, []),         # Гл.6 → Практика
    ]

    # ══════════════════════════════════════════════════════════════════════
    # ЛЕГЕНДА
    # ══════════════════════════════════════════════════════════════════════
    Y_LEG = 1.0
    legend_items = [
        (1, C_GOAL, "Цель / Новизна"),
        (4, C_TASK, "Задачи"),
        (7, C_MODEL, "Модели"),
        (10, C_METHOD, "Методы"),
        (13, C_ALGO, "Алгоритмы"),
        (16, C_SW, "Программное обеспечение"),
        (19, C_RESULT, "Результаты"),
    ]
    for x, color, text in legend_items:
        rect = plt.Rectangle((x, Y_LEG), 0.4, 0.4, facecolor=color,
                              edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.6, Y_LEG + 0.2, text, fontsize=7, color=C_TEXT,
                va="center")

    # ══════════════════════════════════════════════════════════════════════
    # ЗАГОЛОВОК
    # ══════════════════════════════════════════════════════════════════════
    ax.text(11, 29.7,
            "СТРУКТУРНО-ЛОГИЧЕСКАЯ СХЕМА ДИССЕРТАЦИИ",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_TEXT)
    ax.text(11, 29.3,
            "«Модели и методы адаптивного управления реагированием "
            "пожарно-спасательных подразделений»",
            ha="center", va="center", fontsize=9, color="#7f8c8d",
            style="italic")

    path = os.path.join(_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    path = generate_scheme()
    print(f"Схема сохранена: {path}")
