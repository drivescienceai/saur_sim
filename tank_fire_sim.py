"""
tank_fire_sim.py
════════════════════════════════════════════════════════════════════════════════
Интерактивная симуляция управления тушением пожара резервуарного парка.
Сценарии: пожар РВС (V=20 000 м³, ранг №4, 81 ч) и РВС (V=2 000 м³, ранг №2, 5 ч).

Запуск:  python -m saur_sim.tank_fire_sim
         python saur_sim/tank_fire_sim.py
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math, random, os, sys, json, threading, csv
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

# ── Директория хранения чекпоинтов и данных ──────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_DATA_DIR, exist_ok=True)
_CHECKPOINT_DIR  = os.path.join(_DATA_DIR, "checkpoints")
_FLAT_QTABLE_PATH = os.path.join(_CHECKPOINT_DIR, "flat_qtable.npz")
_RUNS_DB_PATH    = os.path.join(_DATA_DIR, "runs.json")
_TRAINER_LOG_PATH = os.path.join(_DATA_DIR, "trainer_history.json")
os.makedirs(_CHECKPOINT_DIR, exist_ok=True)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# Нормативные данные ГОСТ Р 51043-2002 / СП 155.13130.2014
try:
    from .norms_gost import foam_attack_feasibility, NOZZLE_DB, RVS_9 as _RVS9_PARAMS
except ImportError:
    from norms_gost import foam_attack_feasibility, NOZZLE_DB, RVS_9 as _RVS9_PARAMS

# Условные графические обозначения БУПО (Приложение №1, Приказ МЧС №777)
try:
    from .bupo_symbols import (draw_bupo_symbol, LABEL_TO_BUPO,
        BUPO_AC, BUPO_APT, BUPO_AKP, BUPO_PNS, BUPO_ASH, BUPO_AR,
        BUPO_PANRK, BUPO_PP, BUPO_SKMP, BUPO_OSH, BUPO_KPP, BUPO_BU,
        BUPO_STVOL, BUPO_STVOL_LAF, BUPO_PG, BUPO_GPS, BUPO_OCHAG,
        BUPO_OBVAL, BUPO_PIRS, BUPO_VODOYOM)
except ImportError:
    from bupo_symbols import (draw_bupo_symbol, LABEL_TO_BUPO,
        BUPO_AC, BUPO_APT, BUPO_AKP, BUPO_PNS, BUPO_ASH, BUPO_AR,
        BUPO_PANRK, BUPO_PP, BUPO_SKMP, BUPO_OSH, BUPO_KPP, BUPO_BU,
        BUPO_STVOL, BUPO_STVOL_LAF, BUPO_PG, BUPO_GPS, BUPO_OCHAG,
        BUPO_OBVAL, BUPO_PIRS, BUPO_VODOYOM)

# ══════════════════════════════════════════════════════════════════════════════
# ЦВЕТОВАЯ ПАЛИТРА И КОНСТАНТЫ
# ══════════════════════════════════════════════════════════════════════════════
P = dict(
    bg="#f5f6fa", panel="#ffffff", panel2="#eef0f4", canvas="#f0f2f8",
    fire="#ff4500", fire2="#ff8c00", flame="#ffcc00",
    rvs9="#c0392b", rvs17="#2471a3", rvs9_cool="#884422",
    water="#00aaff", foam="#27ae60", smoke="#a0b0b8",
    building="#95a5a6", ground="#a8d5a2", road="#b0b8c8",
    unit_ac="#e74c3c", unit_apt="#e67e22", unit_pns="#3498db",
    unit_panrk="#8e44ad", unit_train="#16a085", unit_ash="#f39c12",
    hydrant="#1abc9c", river="#85c1e9",
    success="#27ae60", warn="#e67e22", danger="#c0392b", info="#2980b9",
    text="#2c3e50", text2="#7f8c8d", hi="#c0392b",
    strat="#c0392b", tact="#e67e22", oper="#2980b9",
    grid="#bdc3c7", accent="#e67e22",
    phase_s1="#e74c3c", phase_s2="#e67e22", phase_s3="#f39c12",
    phase_s4="#27ae60", phase_s5="#2980b9",
)

MAP_W, MAP_H = 825, 675
TOTAL_MIN    = 4862   # 81 ч 2 мин

# ── Физические константы (нормативные) ───────────────────────────────────────
FOAM_INTENSITY_NORM_LS_M2 = 0.05  # л/(с·м²) — ГОСТ Р 51043-2002, п. 5.1.1
COOLING_NORM_LS_M         = 0.8   # л/(с·м)  — Справочник РТП ВНИИПО, 2021
FOAM_RESERVE_COEFF        = 5     # кратность запаса пенообразователя
FOAM_APPLICATION_TIME_MIN = 15    # мин — расчётное время подачи пены
RISK_CRITICAL_THRESHOLD   = 0.75  # значение индекса риска → переход в "критический"

PHASE_COLORS = {
    "S1": P["phase_s1"], "S2": P["phase_s2"], "S3": P["phase_s3"],
    "S4": P["phase_s4"], "S5": P["phase_s5"],
}
PHASE_NAMES = {
    "S1": "S1 — Обнаружение / Выезд",
    "S2": "S2 — Боевое развёртывание",
    "S3": "S3 — Активное горение / Локализация",
    "S4": "S4 — Пенная атака / Ликвидация",
    "S5": "S5 — Ликвидация последствий",
}

# ── 15 действий РТП (S1-S5, T1-T4, O1-O6) ────────────────────────────────────
ACTIONS: List[Tuple[str, str, str]] = [
    # idx  code   уровень       описание
    ("S1", "стратег.",  "Спасение людей — РН по угрозе жизни"),
    ("S2", "стратег.",  "Защита соседнего РВС от воспламенения"),
    ("S3", "стратег.",  "Локализация горения в контуре горящего РВС"),
    ("S4", "стратег.",  "Ликвидация горения — пенная атака"),
    ("S5", "стратег.",  "Предотвращение вскипания / выброса нефти"),
    ("T1", "тактич.",   "Создать боевые участки (БУ) по секторам"),
    ("T2", "тактич.",   "Перегруппировать силы и средства"),
    ("T3", "тактич.",   "Вызов доп. С и С (повышение ранга пожара)"),
    ("T4", "тактич.",   "Установить ПНС/ПАНРК на водоисточник"),
    ("O1", "оперативн.", "Подача Антенор-1500 на охлаждение горящего РВС"),
    ("O2", "оперативн.", "Охлаждение соседнего РВС (орошение + стволы)"),
    ("O3", "оперативн.", "Пенная атака (Акрон Аполло/Муссон/ЛС-С330)"),
    ("O4", "оперативн.", "Разведка пожара — уточнение обстановки"),
    ("O5", "оперативн.", "Ликвидация розлива горящего топлива"),
    ("O6", "оперативн.", "Сигнал отхода — экстренный вывод ЛС"),
]
N_ACT = len(ACTIONS)
LEVEL_C = {"стратег.": P["strat"], "тактич.": P["tact"], "оперативн.": P["oper"]}

# ── Режимы работы приложения ──────────────────────────────────────────────────
# (label, accent_color, bg_color)
APP_MODES: Dict[str, tuple] = {
    "trainer":  ("🎓  ТРЕНАЖЁР РТП",    "#8e44ad", "#f3e5f5"),
    "sppр":     ("🧭  РЕЖИМ СППР",      "#1a6da8", "#e3f0fb"),
    "research": ("🔬  ИССЛЕДОВАНИЕ RL",  "#1e8449", "#e8f8f0"),
}
# Индексы видимых вкладок для каждого режима (0=Настройки … 7=Отчёт)
MODE_TABS: Dict[str, set] = {
    "trainer":  {0, 1, 3, 7},
    "sppр":     {0, 1, 3, 7},
    "research": set(range(8)),
}
# Баллы тренажёра: насколько выбранное действие близко к оптимальному
_T_PERFECT = 20   # точное совпадение с greedy-выбором агента
_T_GOOD    = 10   # в top-3 по Q-значению
_T_VALID   = 2    # допустимое действие, но не лучшее
_T_WRONG   = -5   # недопустимое (нарушение маски фазы)

# ── Допустимые действия по фазам ──────────────────────────────────────────────
PHASE_VALID: Dict[str, List[int]] = {
    "S1": [0, 2, 7, 8, 12],
    "S2": [1, 2, 5, 8, 9, 10, 12],
    "S3": [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13],
    "S4": [0, 3, 4, 6, 11, 12, 13, 14],
    "S5": [10, 12, 13, 14],
}

# ── Хронология событий (мин от начала, метка времени Ч+N, описание, цвет) ────
TIMELINE: List[Tuple[int, str, str, str]] = [
    (0,    "Ч+0",   "Сообщение о загорании горящего РВС (V=20 000 м³, прямогонный бензин)", P["warn"]),
    (1,    "Ч+1",   "Направлены подразделения по рангу пожара №4 (АЦ-16, АПТ-4, АР-1, ПНС-1, ППП-1)", P["info"]),
    (2,    "Ч+2",   "Направлены скорая помощь, ДПС; объявлен сбор личного состава", P["info"]),
    (5,    "Ч+5",   "РТП-1 прибыл: горение по всей площади зеркала, S=1250 м²; ранг №4 подтверждён", P["danger"]),
    (10,   "Ч+10",  "Прибытие 2 АЦ, 1 АШ (РТП-1), 3 АЦ + 2 АПТ + ППП", P["info"]),
    (11,   "Ч+11",  "Поданы 3 ствола Антенор-1500: охлаждение горящего РВС с ЮГА, ВОСТОКА, ЗАПАДА", P["info"]),
    (12,   "Ч+12",  "Создан оперативный штаб (ОШ); назначены НБУ-1,2, НТ, ОТ", P["info"]),
    (15,   "Ч+15",  "РТП-1: 3 ствола поданы; охлаждение соседнего РВС — кольца орошения", P["info"]),
    (20,   "Ч+20",  "РТП-1 запросил: ПНС-110, АР-2, АКП-50", P["warn"]),
    (50,   "Ч+50",  "Прибыли: ПНС-110, АНР-130", P["info"]),
    (55,   "Ч+55",  "ПНС-110 → открытый водоисточник; 4-й ствол Антенор на охлаждение (ЮГ)", P["info"]),
    (110,  "Ч+110", "ПНС-110 → открытый водоисточник; 4-ходовое разветвление; 5-й ствол (ВОСТОК)", P["info"]),
    (125,  "Ч+125", "АНР-130 → открытый водоисточник; 6-й ствол Антенор (ЗАПАД)", P["info"]),
    (159,  "Ч+159", "Прибытие дежурной смены СПТ на АШ (РТП-2)", P["info"]),
    (160,  "Ч+160", "РТП-2 принял руководство; ранг №4 подтверждён; штаб переназначен", P["info"]),
    (231,  "Ч+231", "РТП-3 принял руководство", P["info"]),
    (283,  "Ч+283", "Прибыл ПАНРК СПСЧ", P["info"]),
    (330,  "Ч+330", "НБУ-2 докладывает: готовность к пенной атаке", P["warn"]),
    (340,  "Ч+340", "⚡ ПЕННАЯ АТАКА №1: 2 Акрон Аполло с ППП", P["warn"]),
    (360,  "Ч+360", "❌ Атака №1 ПРЕКРАЩЕНА: выход из строя ППП", P["danger"]),
    (361,  "Ч+361", "Замена ППП", P["info"]),
    (370,  "Ч+370", "ПНС контейнерного типа → водоём; подвоз воды АЦ", P["info"]),
    (400,  "Ч+400", "ПАНРК → водоисточник; перекачка через АР-2", P["info"]),
    (440,  "Ч+440", "НТ докладывает: бесперебойная подача ОВ обеспечена", P["success"]),
    (470,  "Ч+470", "Прибыл ПАНРК", P["info"]),
    (495,  "Ч+495", "Переназначены НБУ-1 (ЮГ+лаб.), НБУ-2 (ВОСТОК), НБУ-3 (ЗАПАД); 3 БУ", P["info"]),
    (502,  "Ч+502", "НБУ-2: установлен ЛС-С330 (расход 330 л/с)", P["info"]),
    (505,  "Ч+505", "6 лафетных стволов на горящий РВС, 2+1 на соседний РВС; 6 ед. на открытые водоисточники", P["info"]),
    (548,  "Ч+548", "Прибыл пожарный поезд → эстакада", P["info"]),
    (557,  "Ч+557", "🚨 СВИЩ! Розлив горящего бензина 300 м²; S=1550 м²", P["danger"]),
    (558,  "Ч+558", "2 ствола Антенор-1500 → тушение розлива", P["warn"]),
    (580,  "Ч+580", "✅ Розлив ликвидирован (300 м²); общая S вернулась к 1250 м²", P["success"]),
    (758,  "Ч+758", "⚡ ПЕННАЯ АТАКА №2: ЛС-С330 + Акрон Аполло + Муссон-125", P["warn"]),
    (775,  "Ч+775", "❌ Атака №2 ПРЕКРАЩЕНА: каркас крыши внутри РВС, разрушение пены", P["danger"]),
    (924,  "Ч+924", "Прибыл пожарный поезд → эстакада", P["info"]),
    (1242, "Ч+1242","🔥 Возгорание вторичного очага 50 м² (тепловое воздействие)", P["danger"]),
    (1243, "Ч+1243","Звено ГДЗС → тушение вторичного очага (ствол Дельта-500)", P["warn"]),
    (1247, "Ч+1247","✅ Вторичный очаг ликвидирован (50 м²)", P["success"]),
    (1260, "Ч+1260","7 стволов на горящий РВС, 5 на соседний РВС; 3 БУ; 10 магистральных линий", P["info"]),
    (1820, "Ч+1820","⚡ ПЕННАЯ АТАКА №3: Акрон Аполло + Муссон-125 с ППП", P["warn"]),
    (1848, "Ч+1848","❌ Атака №3 ПРЕКРАЩЕНА: каркасы, карманы, высокая интенсивность горения", P["danger"]),
    (3375, "Ч+3375","⚡ ПЕННАЯ АТАКА №4: Акрон Аполло + Муссон-125", P["warn"]),
    (3395, "Ч+3395","❌ Атака №4 ПРЕКРАЩЕНА: аналогично — каркасы и карманы", P["danger"]),
    (3480, "Ч+3480","⚡ ПЕННАЯ АТАКА №5: Акрон Аполло + Муссон-125", P["warn"]),
    (3508, "Ч+3508","❌ Атака №5 ПРЕКРАЩЕНА", P["danger"]),
    (3510, "Ч+3510","🔒 ПОЖАР ЛОКАЛИЗОВАН на площади 1250 м²", P["success"]),
    (4740, "Ч+4740","⚡ ПЕННАЯ АТАКА №6: Антенор-1500 + Муссон-125 + 2×ГПС-1000 с АКП-50", P["warn"]),
    (4760, "Ч+4760","✅ Видимое горение ОТСУТСТВУЕТ! Подача ОВ продолжается", P["success"]),
    (4862, "Ч+4862","🏁 ПОЖАР ЛИКВИДИРОВАН. Продолжается охлаждение резервуаров.", P["success"]),
]
_TL_LOOKUP: Dict[int, List] = {}
for _ev in TIMELINE:
    _TL_LOOKUP.setdefault(_ev[0], []).append(_ev)

# ── Структурированные эффекты хронологии (tuapse) ────────────────────────────
# Ключ = время_мин: dict с явными изменениями состояния симуляции.
# Используется в _apply_scripted вместо парсинга строк описания.
_SCRIPTED_EFFECTS_TUAPSE: Dict[int, dict] = {
    11:   {"n_trunks_burn_min": 3, "water_flow_min": 105},
    12:   {"has_shtab": True},
    55:   {"n_pns_add": 1, "water_flow_add": 110.0},
    110:  {"n_pns_add": 1, "water_flow_add": 110.0},
    125:  {"n_pns_add": 1, "water_flow_add": 110.0},
    330:  {"foam_ready": True, "foam_conc_min": 4.0},
    340:  {"foam_attack_start": True},
    360:  {"foam_attack_fail": True},
    495:  {"n_bu_min": 3},
    505:  {"n_trunks_burn_min": 6, "water_flow_min": 600},
    557:  {"spill": True, "spill_area": 300.0, "fire_area_set": 1550.0},
    580:  {"spill": False, "fire_area_set": 1250.0},
    758:  {"foam_attack_start": True},
    775:  {"foam_attack_fail": True},
    1242: {"secondary_fire": True},
    1247: {"secondary_fire": False},
    1260: {"n_trunks_burn_min": 7, "water_flow_min": 700, "n_bu_min": 3},
    1820: {"foam_attack_start": True},
    1848: {"foam_attack_fail": True},
    3375: {"foam_attack_start": True},
    3395: {"foam_attack_fail": True},
    3480: {"foam_attack_start": True},
    3508: {"foam_attack_fail": True},
    3510: {"localized": True},
    4740: {"foam_attack_start": True, "akp50": True, "roof_obstruction": 0.20},
    4760: {"foam_ready": True},
    4862: {"extinguished": True, "fire_area_set": 0.0},
}

# ══════════════════════════════════════════════════════════════════════════════
# СЦЕНАРИЙ 2 — РВС СРЕДНИЙ (V=2000 м³, бензин, ранг №2)
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_MIN_SERP = 300   # Ч+300 мин — окончание работ

TIMELINE_SERP: List[Tuple[int, str, str, str]] = [
    (0,   "Ч+0",   "Загорание РВС (V=2000 м³, бензин). S=168 м², S_обвал=3410 м²", P["danger"]),
    (10,  "Ч+10",  "Обнаружение, вызов 01. Начало тушения силами ДПД с первичными средствами", P["warn"]),
    (14,  "Ч+14",  "Прибытие ПЧ-6. РТП-1: разведка пожара. S=168 м². Q_тр=66 л/с. Развёртывание", P["info"]),
    (16,  "Ч+16",  "Прибытие ПЧ-330. 2 стволов А (РС-70) → охлаждение. 1 ГПС-600 → тушение. Q_ф=40 л/с", P["info"]),
    (18,  "Ч+18",  "4 стволов А (РС-70) + 4 ГПС-600. Прибытие ПЧ-304. АЦ установить на пожарные гидранты (ПГ-106, ПГ-108). Готовность к пенной атаке", P["warn"]),
    (20,  "Ч+20",  "Начало тушения ПЧ-304: 6 ГПС-600 введены на тушение и охлаждение обвалования", P["info"]),
    (32,  "Ч+32",  "Прибытие ПЧ-52. Всего 6 стволов А + 3 ГПС-600 на позициях. Расход воды обеспечен", P["info"]),
    (40,  "Ч+40",  "Создан оперативный штаб тушения пожара. ОП-2 ПЧ-3, ПЧ-2, СЧ-17. 9 ГПС-600 + 3 ствола А", P["info"]),
    (60,  "Ч+60",  "Прибытие резервных АЦ. 7 стволов А + 3 ГПС-600 на охлаждение. Контроль обстановки", P["info"]),
    (90,  "Ч+90",  "🔒 Пожар локализован на площади 168 м²", P["success"]),
    (240, "Ч+240", "✅ Ликвидировано горение в горящем РВС. 2 ствола А на охлаждение и контроль", P["success"]),
    (300, "Ч+300", "🏁 Пожар ликвидирован. Охлаждение резервуаров ≥ 6 ч. ПЧ-6 — дежурство", P["success"]),
]
_TL_SERP_LOOKUP: Dict[int, List] = {}
for _ev in TIMELINE_SERP:
    _TL_SERP_LOOKUP.setdefault(_ev[0], []).append(_ev)

# ── Структурированные эффекты хронологии (serp) ──────────────────────────────
_SCRIPTED_EFFECTS_SERP: Dict[int, dict] = {
    16:  {"n_trunks_burn_min": 2, "water_flow_min": 14, "n_pns_add": 1, "water_flow_add": 15.0},
    18:  {"n_trunks_burn_min": 4, "foam_ready": True, "n_pns_add": 1, "water_flow_add": 15.0},
    20:  {"water_flow_min": 600},
    32:  {"n_trunks_burn_min": 6, "water_flow_min": 600},
    40:  {"has_shtab": True, "n_trunks_burn_min": 7, "water_flow_min": 700},
    60:  {"n_trunks_burn_min": 7, "water_flow_min": 700},
    90:  {"localized": True},
    240: {"extinguished": True, "fire_area_set": 0.0},
    300: {"extinguished": True, "fire_area_set": 0.0},
}

# Справочник действий РТП — сценарий РВС средний
ACTIONS_BY_PHASE_SERP = {
    "S1": [
        ("O4", "Разведка пожара",        "Установить тип РВС, объём, продукт, S зеркала. Вызвать ДПД"),
        ("S3", "Подтвердить ранг пожара", "Ранг №2 по расписанию выезда (РВС V=1000–2000 м³)"),
        ("T3", "Вызов по расписанию",     "АЦ-2, АЛ-1 ПЧ-6; дополнительно — ПЧ-330, ПЧ-304"),
        ("S1", "Уведомить ЕДДС и главу МО", "Направить скорую, ДПС; охрана отключает электроснабжение"),
    ],
    "S2": [
        ("O1", "Охлаждение горящего РВС",            "2–4 ствола А (РС-70) по периметру РВС с наветренной стороны"),
        ("O2", "Охлаждение соседних РВС",            "Стволы А на РВС в зоне теплового воздействия"),
        ("T4", "Установка АЦ на ПГ",                "АЦ-1 → ПГ-1; АЦ-2 → ПГ-2; Q_ПГ ≈ 15 л/с каждый"),
        ("T1", "Создать БУ по секторам",             "БУ-1 (тушение горящего РВС), БУ-2 (охлаждение смежных)"),
    ],
    "S3": [
        ("T4", "Установка ПА на водоисточники",      "Все АЦ на ПГ; при нехватке — водоём 500 м³ (сухотруб)"),
        ("O1", "Наращивание стволов до 7–9",          "6–9 ГПС-600 на тушение + 7 стволов А на охлаждение"),
        ("T3", "Запросить дополнительные СиС",        "ОП-2 ПЧ-3, ПЧ-2, СЧ-17. Обеспечить запас пенообразователя"),
        ("T1", "Создать оперативный штаб",            "НШ, НТ, НБУ-1,2; расчёт: N_ГПС=3, N_ств_А=7, Q=66 л/с"),
        ("O3", "Подготовить пенную атаку",            "3 ГПС-600 на тушение: Q_пена = 3×5.64 = 16.9 л/с ≥ 8.4 л/с"),
    ],
    "S4": [
        ("O3", "Пенная атака ГПС-600",               "Одновременная подача от 3 ГПС-600 через пеноподъёмники"),
        ("O2", "Непрерывное охлаждение стенок",       "Не прекращать охлаждение во время атаки и после"),
        ("O6", "Сигнал отхода при угрозе вскипания", "Наблюдатель контролирует уровень и температуру продукта"),
        ("T2", "Обеспечить резерв сил",               "50% личного состава в резерве за обвалованием"),
    ],
    "S5": [
        ("O4", "Контроль остаточного горения",        "Убедиться в отсутствии горения. ИКС-контроль температуры"),
        ("O2", "Охлаждение резервуаров ≥ 6 ч",        "Норма: охлаждение после локализации не менее 6 часов"),
        ("T2", "Поэтапная демобилизация",             "Возврат СиС после письменного разрешения на свёртывание"),
    ],
}

# ── Реестр сценариев симуляции ────────────────────────────────────────────────
SCENARIOS: Dict[str, dict] = {
    "tuapse": {
        "name":              "РВС крупный (V=20 000 м³, бензин, ранг №4, 81 ч)",
        "short":             "РВС-20000 (ранг 4)",
        "total_min":         TOTAL_MIN,
        "initial_fire_area": 1250.0,
        "fuel":              "бензин",
        "rvs_name":          "РВС крупный (V=20 000 м³)",
        "rvs_diameter_m":    40.0,
        "fire_rank_default": 4,
        "roof_obstruction_init": 0.70,
        "foam_intensity":    0.065,      # ГОСТ Р 51043-2002
        "tl_lookup":         _TL_LOOKUP,
        "scripted_effects":  _SCRIPTED_EFFECTS_TUAPSE,
        "actions_by_phase":  None,       # ACTIONS_BY_PHASE используется глобально
    },
    "serp": {
        "name":              "РВС средний (V=2 000 м³, бензин, ранг №2, 5 ч)",
        "short":             "РВС-2000 (ранг 2)",
        "total_min":         TOTAL_MIN_SERP,
        "initial_fire_area": 168.0,
        "fuel":              "бензин",
        "rvs_name":          "РВС средний (V=2000 м³)",
        "rvs_diameter_m":    14.62,
        "fire_rank_default": 2,
        "roof_obstruction_init": 0.0,   # нет плавающей крыши (конусная кровля)
        "foam_intensity":    0.05,       # ПТП / Справочник РТП, стр. 104
        "tl_lookup":         _TL_SERP_LOOKUP,
        "scripted_effects":  _SCRIPTED_EFFECTS_SERP,
        "actions_by_phase":  ACTIONS_BY_PHASE_SERP,
    },
}

# ── Действия РТП по фазам пожара (для вкладки «Справочник») ──────────────────
ACTIONS_BY_PHASE = {
    "S1": [
        ("O4", "Разведка пожара", "Установить: тип РВС, объём, продукт, площадь зеркала, угрозу людям"),
        ("S3", "Подтвердить ранг пожара", "Ранг №4 по расписанию выезда для РВС V>5000 м³"),
        ("T3", "Вызов сил по расписанию", "АЦ-16, АПТ-4, АР-1, ПНС-1, ППП-1"),
        ("S1", "Уведомить главу МО и ЕДДС", "Направить скорую, ДПС, привести в готовность АЦ КС"),
    ],
    "S2": [
        ("T1", "Создать штаб пожаротушения", "Назначить НШ, НБУ-1,2, НТ, ОТ, наблюдателя за воздухом"),
        ("O1", "Подать Антенор-1500 на охлаждение горящего РВС", "3 ствола: с ЮГА, ВОСТОКА, ЗАПАДА одновременно"),
        ("O2", "Охлаждение соседнего РВС", "Стационарные кольца орошения + 1 ствол с ВОСТОКА"),
        ("T4", "Установить ПА на водоисточники", "АЦ-1 → ПГ-1; АПТ-1 → ПГ-2; АПТ-2 → ПГ-3"),
        ("T2", "Определить границы боевых участков", "БУ-1 (ЮГ+лаб), БУ-2 (ВОСТОК+соседний РВС), задача НБУ"),
    ],
    "S3": [
        ("T4", "Установить ПНС на водоисточник", "ПНС-110 → 4-ходовое разветвление → Антенор-1500"),
        ("O1", "Наращивать стволы до 6–7 лафетных", "Задействовать ПАНРК"),
        ("S2", "Усилить защиту соседнего РВС", "3 ствола + кольца орошения; НБУ-2 контролирует температуру"),
        ("T3", "Запросить ПАНРК, пожарные поезда", "ПАНРК → водоисточник; поезда → эстакада"),
        ("T1", "Создать 3-й боевой участок (ЗАПАД)", "НБУ-3 организует охлаждение с западной стороны"),
        ("O3", "Подготовить пенную атаку", "Подвоз пенообразователя; проверка ППП; обеспечить 9 ч подготовки"),
        ("S3", "Организовать подвоз воды", "АЦ резерва → водоём очистных сооружений; пополнение 24/7"),
        ("O5", "Ликвидация розлива горящего топлива", "При свище: 2 ствола Антенор на площадь 300 м², преграды"),
        ("S5", "Предотвращение вскипания нефти", "Максимальное охлаждение стенок; наблюдение за уровнем"),
    ],
    "S4": [
        ("O3", "Пенная атака — Акрон Аполло с ППП", "Подача пены сверху через пеноподъёмник ПЧ-23/ПЧ-18"),
        ("O3", "Пенная атака — Муссон-125 с ППП", "Дополнительная подача пены с западной стороны"),
        ("O3", "Пенная атака — ЛС-С330 (330 л/с)", "Передвижной лафетный ствол высокой производительности"),
        ("O3", "Пенная атака — ГПС-1000 с АКП-50", "Генератор пены с коленчатого автоподъёмника; наиболее эффективна"),
        ("S4", "Прекратить пенную атаку (нет результата)", "Если каркас крыши внутри РВС, карманы или разрушение пены"),
        ("T2", "Перегруппировка перед следующей атакой", "Смена позиций ППП; дополнительный пенообразователь"),
        ("O6", "Обеспечить готовность к экстренному отходу", "Постоянное наблюдение за целостностью стенок РВС"),
        ("T3", "Привлечь АКП-50 для подачи ГПС-1000", "Ключевое решение: пена через люк в крышу с подъёмника"),
    ],
    "S5": [
        ("O4", "Контроль после пенной атаки", "Убедиться в отсутствии видимого горения; продолжать подачу ОВ"),
        ("O2", "Охлаждение до полного остывания", "Не менее 3 ч после ликвидации; температура стенок < 80°С"),
        ("O6", "Тушение вторичных очагов через ГДЗС", "Звено ГДЗС → столовая, лаборатория; ствол Дельта-500"),
        ("S3", "Ограничение розлива — преграды", "Мешки с песком от лаборатории до столовой и бетонного забора"),
        ("T2", "Поэтапная демобилизация сил", "Поэтапный возврат С и С после подтверждения ликвидации"),
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# Q-LEARNING АГЕНТ (автономный)
# ══════════════════════════════════════════════════════════════════════════════
class QLAgent:
    """Q-обучение с дискретной таблицей Q[состояние, действие].

    Пространство состояний кодируется в 128 индексов функцией state_to_idx()
    по 7 признакам (фаза, стволы, ПНС, готовность пены, розлив, кол-во атак, БУ).
    Пространство действий — 15 возможных решений РТП (см. ACTIONS).
    Стратегия выбора действий: ε-жадная (epsilon-greedy), ε убывает по эпизодам.
    """

    def __init__(self, n_states: int = 128, n_actions: int = N_ACT,
                 alpha: float = 0.15, gamma: float = 0.95,
                 epsilon: float = 0.90, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma = alpha, gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.episode_rewards: List[float] = []
        self._ep_reward: float = 0.0
        self.action_counts = np.zeros(n_actions, dtype=int)

    def state_to_idx(self, s: dict) -> int:
        # Номер фазы (S1..S5 → 0..4) — ключевой признак смены тактики
        ph  = {"S1": 0,"S2": 1,"S3": 2,"S4": 3,"S5": 4}.get(s.get("phase","S1"), 0)
        # Число стволов квантуется попарно (0–1→0, 2–3→1, 4–5→2, 6+→3)
        tr  = min(3, s.get("n_trunks", 0) // 2)          # 0–3
        pns = min(3, s.get("n_pns", 0))                   # 0–3
        fr  = int(s.get("foam_ready", False))              # 0–1
        sp  = int(s.get("spill", False))                   # 0–1
        fa  = min(3, s.get("foam_attacks", 0))             # 0–3
        bu  = min(3, s.get("n_bu", 0))                     # 0–3
        # Хэш-свёртка в 128 ячеек таблицы Q (возможны коллизии — допустимо для обучения)
        return int((ph*64 + tr*16 + pns*4 + fr*2 + sp + fa*8 + bu*2) % 128)

    def select_action(self, s: dict, mask: Optional[np.ndarray] = None,
                      training: bool = True) -> int:
        # С вероятностью ε — случайное исследование среди допустимых действий
        if training and self.rng.random() < self.epsilon:
            valid = np.where(mask)[0] if mask is not None else np.arange(self.n_actions)
            return int(self.rng.choice(valid)) if len(valid) > 0 else 0
        # Иначе — жадный выбор: действие с максимальным Q, недопустимые занижены до -∞
        q = self.Q[self.state_to_idx(s)].copy()
        if mask is not None:
            q[~mask] = -1e9
        return int(np.argmax(q))

    def update(self, s: dict, a: int, r: float, s_next: dict, done: bool):
        i, j = self.state_to_idx(s), self.state_to_idx(s_next)
        # Цель TD: r + γ·max_a'Q(s',a'), при завершении эпизода — только r
        td = r + (0.0 if done else self.gamma * np.max(self.Q[j]))
        # Обновление Q по правилу: Q(s,a) ← Q(s,a) + α·(td_цель − Q(s,a))
        self.Q[i, a] += self.alpha * (td - self.Q[i, a])
        self._ep_reward += r
        self.action_counts[a] += 1

    def end_episode(self):
        self.episode_rewards.append(self._ep_reward)
        self._ep_reward = 0.0
        self.epsilon = max(0.05, self.epsilon * 0.99)

    def q_values(self, s: dict) -> np.ndarray:
        return self.Q[self.state_to_idx(s)].copy()

    def coverage(self) -> float:
        """Доля состояний, в которых хотя бы одно действие отличается от 0."""
        return float(np.any(self.Q != 0, axis=1).mean())

    def save(self, path: str):
        """Сохранить Q-таблицу и метаданные обучения в .npz файл."""
        np.savez_compressed(
            path,
            Q=self.Q,
            action_counts=self.action_counts,
            episode_rewards=np.array(self.episode_rewards, dtype=np.float32),
            epsilon=np.array([self.epsilon]),
            alpha=np.array([self.alpha]),
            gamma=np.array([self.gamma]),
        )

    def load(self, path: str) -> bool:
        """Загрузить Q-таблицу из .npz файла. Возвращает True при успехе."""
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=False)
            if data["Q"].shape == self.Q.shape:
                self.Q             = data["Q"].copy()
                self.action_counts = data["action_counts"].copy()
                self.episode_rewards = data["episode_rewards"].tolist()
                self.epsilon       = float(data["epsilon"][0])
            return True
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════════════════════
# СИМУЛЯЦИЯ
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SimSnapshot:
    t: int
    phase: str
    fire_area: float
    water_flow: float
    n_trunks_burn: int
    n_trunks_neighbor: int
    n_pns: int
    n_bu: int
    has_shtab: bool
    foam_attacks: int
    foam_ready: bool
    spill: bool
    secondary_fire: bool
    localized: bool
    extinguished: bool
    risk: float
    last_action: int
    reward: float
    roof_obstruction: float = 0.70  # доля блокировки каркасом крыши [0..1]
    foam_flow_ls: float = 0.0       # суммарный расход пены, л/с


class TankFireSim:
    """Дискретно-событийная симуляция тушения пожара РВС.

    На каждом шаге (dt мин): воспроизводятся скриптованные события из хронологии,
    обновляется физика пожара, определяется фаза, RL-агент выбирает и применяет
    действие. Поддерживает два сценария: «tuapse» (РВС крупный, ранг №4) и «serp»
    (РВС средний, ранг №2). Нормативные данные берутся из модуля norms_gost.
    """

    def __init__(self, seed: int = 42, training: bool = True,
                 scenario: str = "tuapse"):
        self.rng      = random.Random(seed)
        self.np_rng   = np.random.RandomState(seed)
        self.agent    = QLAgent(seed=seed)
        self.training = training
        self.scenario = scenario
        self._cfg     = SCENARIOS[scenario]
        self.reset()

    # ── Инициализация ──────────────────────────────────────────────────────────
    def reset(self):
        cfg = self._cfg
        self.t              = 0
        self.phase          = "S1"
        self.fire_area      = cfg["initial_fire_area"]
        self.n_trunks_burn  = 0
        self.n_trunks_nbr   = 0
        self.n_pns          = 0          # ПНС на водоисточниках
        self.n_bu           = 0          # число боевых участков (макс. 3)
        self.has_shtab      = False      # создан оперативный штаб
        self.foam_attacks   = 0          # счётчик выполненных пенных атак
        self.foam_ready     = False      # флаг: все условия для атаки выполнены
        self.foam_conc      = 12.0       # запас пенообразователя (т)
        self.spill          = False      # активный розлив горящего топлива
        self.spill_area     = 0.0
        self.secondary_fire = False      # вторичный очаг (столовая / лаборатория)
        self.localized      = False      # пожар локализован (площадь не растёт)
        self.extinguished   = False      # видимое горение ликвидировано
        self.water_flow     = 0.0        # суммарный расход ОВ, л/с
        self.last_action    = 12         # O4: разведка по умолчанию

        # Физика пенной атаки (нормы ГОСТ Р 51043-2002 / Справочник РТП)
        self.roof_obstruction = cfg["roof_obstruction_init"]
        self.akp50_available  = False   # АКП-50 (коленчатый подъёмник) на сцене
        self.foam_flow_ls     = 0.0     # суммарный расход пенного раствора, л/с

        # История для графиков
        self.h_fire:      List[Tuple[int,float]] = [(0, cfg["initial_fire_area"])]
        self.h_water:     List[Tuple[int,float]] = [(0, 0.0)]
        self.h_risk:      List[Tuple[int,float]] = [(0, 0.3)]
        self.h_trunks:    List[Tuple[int,int]]   = [(0, 0)]
        self.h_reward:    List[float]            = []
        self.h_foam_conc: List[Tuple[int,float]] = [(0, self.foam_conc)]
        self.h_phase:     List[Tuple[int,int]]   = [(0, 1)]  # фазы S1..S5 → числа 1..5

        # Журнал событий (t, цвет, текст)
        self.events: List[Tuple[int,str,str]] = [
            (0, P["warn"], "Поступило сообщение о загорании РВС")
        ]
        self._last_state: Optional[dict] = None
        self._scripted_triggered = set()

    # ── Состояние для агента ───────────────────────────────────────────────────
    def _state(self) -> dict:
        # Вектор признаков, который агент передаёт в state_to_idx() и затем в Q-таблицу.
        # roof_low=True означает, что каркас крыши уже не блокирует пенную атаку (< 40%)
        return dict(
            phase=self.phase,
            n_trunks=self.n_trunks_burn,
            n_pns=self.n_pns,
            foam_ready=self.foam_ready,
            spill=self.spill,
            foam_attacks=self.foam_attacks,
            n_bu=self.n_bu,
            roof_low=self.roof_obstruction < 0.40,   # True когда препятствие снижено
        )

    def _mask(self) -> np.ndarray:
        # Базовое ограничение — только действия, допустимые для текущей фазы
        m = np.zeros(N_ACT, dtype=bool)
        valid = PHASE_VALID.get(self.phase, list(range(N_ACT)))
        for i in valid:
            m[i] = True
        # Дополнительные ограничения
        if not (self.foam_ready and self.foam_conc > 0):
            m[3] = m[11] = False   # S4, O3 — нужна пена
        if not self.spill:
            m[13] = False           # O5 — только при розливе
        if not self.secondary_fire:
            m[0] = False            # S1 — только при угрозе людям
        # Сигнал отхода (O6) разрешён лишь при действительно критическом риске
        m[14] = self._risk() > 0.85
        if not m.any():
            m[12] = True            # запасной: разведка
        return m

    def _risk(self) -> float:
        # Базовый риск зависит от фазы: S3 (активное горение) — наиболее опасна
        base = {"S1":0.25,"S2":0.4,"S3":0.65,"S4":0.5,"S5":0.2}.get(self.phase,0.3)
        if self.spill:          base += 0.20   # розлив резко увеличивает угрозу
        if self.secondary_fire: base += 0.10
        if self.foam_attacks >= 3 and not self.localized: base += 0.10  # многократные неудачи
        # Вклад площади пожара: при 6000 м² максимальная надбавка +0.25
        area_f = min(0.25, self.fire_area / 6000)
        return min(1.0, base + area_f)

    # ── Применение действия RL ─────────────────────────────────────────────────
    def _apply(self, a: int) -> float:
        code, level, desc = ACTIONS[a]
        r = 0.0

        if code == "O1":   # Добавить ствол на горящий РВС
            if self.n_trunks_burn < 10:
                self.n_trunks_burn += 1
                self.water_flow += 35.0
                r = 0.4 if self.n_trunks_burn <= 7 else 0.1
                self._log(P["info"], f"O1: подан ствол Антенор-1500 → итого {self.n_trunks_burn} на горящий РВС")

        elif code == "O2":  # Охлаждение соседнего
            if self.n_trunks_nbr < 5:
                self.n_trunks_nbr += 1
                self.water_flow += 30.0
                r = 0.3
                self._log(P["info"], f"O2: охлаждение соседнего РВС — {self.n_trunks_nbr} ствола")

        elif code in ("O3","S4"):  # Пенная атака: проверяем выполнимость по нормам ГОСТ
            if self.foam_ready and self.foam_conc > 0:
                self.foam_attacks += 1
                self.foam_conc -= 1.8   # расход пенообразователя на одну атаку, т
                q_foam = self._compute_foam_flow()
                self.foam_flow_ls = q_foam
                # Нормативная проверка: Q_эфф ≥ Q_треб с учётом препятствия крыши
                result = foam_attack_feasibility(
                    self.fire_area, q_foam, self.roof_obstruction,
                    self._cfg["fuel"]
                )
                ok = result["feasible"]
                if ok:
                    self.extinguished = True
                    r = 12.0
                    self._log(P["success"],
                              f"⚡ ПЕННАЯ АТАКА №{self.foam_attacks}: УСПЕШНА — горение ликвидировано! "
                              f"Q_эфф={result['q_effective']:.0f} л/с ≥ норм. {result['q_required']:.0f} л/с")
                else:
                    r = -0.4
                    self._log(P["danger"],
                              f"❌ Атака №{self.foam_attacks} не дала результата: {result['reason']}")
                self.foam_ready = False
            else:
                r = -0.5
                self._log(P["warn"], "O3: пенная атака недоступна — нет пены или не готово")

        elif code == "O4":  # Разведка
            r = 0.05
            self._log(P["text2"], "O4: разведка — уточнены границы зоны горения и наличие людей")

        elif code == "O5":  # Ликвидация розлива
            if self.spill:
                self.spill = False
                self.fire_area = max(self._cfg["initial_fire_area"], self.fire_area - 300)
                r = 1.5
                self._log(P["success"], "✅ O5: розлив горящего топлива ликвидирован (−300 м²)")

        elif code == "O6":  # Отход
            r = -1.0
            self._log(P["danger"], "O6: сигнал отхода — экстренный вывод ЛС из опасной зоны")

        elif code == "T1":  # Создать БУ
            if self.n_bu < 3:
                self.n_bu += 1
                r = 0.5
                names = ["БУ-1 (ЮГ + лаборатория)", "БУ-2 (ВОСТОК + соседний РВС)", "БУ-3 (ЗАПАД)"]
                self._log(P["info"], f"T1: создан {names[self.n_bu-1]}")

        elif code == "T2":  # Перегруппировка
            r = 0.2
            self._log(P["info"], "T2: перегруппировка С и С по секторам; оптимизация позиций стволов")

        elif code == "T3":  # Вызов подкрепления
            r = 0.4
            self._log(P["warn"], "T3: запрос доп. С и С — ПНС, ПАНРК, пожарный поезд")

        elif code == "T4":  # Наладить водоснабжение: каждый ПНС-110 добавляет ~110 л/с
            if self.n_pns < 4:
                self.n_pns += 1
                self.water_flow += 110.0
                r = 0.7
                srcs = ["открытый водоисточник (ПНС)", "открытый водоисточник (ПАНРК)", "водоисточник (ПАНРК СПСЧ)", "водоём очистных сооружений"]
                self._log(P["info"], f"T4: ПНС/ПАНРК №{self.n_pns} → {srcs[self.n_pns-1]}")

        elif code == "S1":  # Спасение людей
            r = 0.8
            self._log(P["danger"], "S1: РН — спасение людей; АСА и АЛ направлены к очагу угрозы")

        elif code == "S2":  # Защита соседнего
            r = 0.5
            if not self.has_shtab:
                self.has_shtab = True
            self._log(P["info"], "S2: РН — защита соседнего РВС; усиление охлаждения")

        elif code == "S3":  # Локализация
            r = 0.3
            self._log(P["info"], "S3: РН — локализация; удерживать огонь в контуре горящего РВС")

        elif code == "S5":  # Предотвращение вскипания
            r = 0.6
            self._log(P["warn"], "S5: РН — предотвращение вскипания; максимальное охлаждение стенок")

        # Пенная атака возможна только при наличии пенообразователя, воды и стволов охлаждения
        self.foam_ready = (
            self.foam_conc > 0
            and self.n_pns >= 1
            and self.n_trunks_burn >= 3
        )

        # Постоянный штраф: чем больше площадь пожара, тем ниже каждое вознаграждение
        r -= 0.005 * self.fire_area / 1000
        return r

    def _compute_foam_flow(self) -> float:
        """Суммарный расход пенного раствора для текущей атаки, л/с.

        Сценарий «tuapse» (ГОСТ Р 51043-2002, РВС крупный V=20 000 м³):
          - Базовая: 2 × Акрон-Аполло (66.6 л/с)
          - Со 2-й: + Муссон-125 (+125 л/с)
          - С АКП-50: + 2×ГПС-1000 + Антенор-1500 → roof_obstruction=0.20

        Сценарий «serp» (Справочник РТП, РВС средний V=2000 м³):
          - Базовая: 3 × ГПС-600 (3 × 5.64 = 16.9 л/с)
          - Со 2-й атаки: + 2 × ГПС-600 (до 5 × ГПС-600 = 28.2 л/с)
          - Нет препятствия крыши (конусная кровля)
        """
        if self.scenario == "serp":
            n_gps = min(5, 3 + max(0, self.foam_attacks - 1))
            return n_gps * 5.64   # ПТП: q_ГПС-600_факт = 5.64 л/с
        # tuapse
        q = 2.0 * NOZZLE_DB["Акрон-Аполло"].flow_ls
        if self.foam_attacks >= 2:
            q += NOZZLE_DB["Муссон-125"].flow_ls
        if self.akp50_available:
            q += 2.0 * NOZZLE_DB["ГПС-1000"].flow_ls
            q += NOZZLE_DB["Антенор-1500"].flow_ls
            self.roof_obstruction = 0.20
        return q

    def _log(self, color: str, text: str):
        self.events.append((self.t, color, text))

    # ── Основной шаг симуляции ─────────────────────────────────────────────────
    def step(self, dt: int = 5, action: Optional[int] = None) -> SimSnapshot:
        """Один шаг симуляции.

        action: если задано — использовать это действие вместо агента
                (применяется иерархическим агентом из hrl_sim.py).
        """
        self.t += dt

        # Применить скриптованные события из хронологии
        tl = self._cfg["tl_lookup"]
        for step_t in range(self.t - dt + 1, self.t + 1):
            for ev in tl.get(step_t, []):
                if step_t not in self._scripted_triggered:
                    self._scripted_triggered.add(step_t)
                    self._apply_scripted(step_t, ev[2])
                    self._log(ev[3], f"[{ev[1]}] {ev[2]}")

        # Обновить физику пожара
        self._update_fire()

        # Обновить фазу
        self._update_phase()

        # RL-решение: внешнее (иерарх. агент) или внутреннее (flat агент)
        state = self._state()
        mask  = self._mask()
        if action is not None:
            a = int(action)   # иерархический агент передал действие извне
        else:
            a = self.agent.select_action(state, mask, training=self.training)
        r     = self._apply(a)

        # Обучение
        if self._last_state is not None:
            self.agent.update(self._last_state, self.last_action, r,
                              state, self.extinguished)
        self._last_state = state
        self.last_action = a
        self.h_reward.append(r)

        # Записать историю метрик
        self.h_fire.append((self.t, self.fire_area))
        self.h_water.append((self.t, self.water_flow))
        self.h_risk.append((self.t, self._risk()))
        self.h_trunks.append((self.t, self.n_trunks_burn))
        _phase_num = {"S1":1,"S2":2,"S3":3,"S4":4,"S5":5}.get(self.phase, 1)
        self.h_phase.append((self.t, _phase_num))
        self.h_foam_conc.append((self.t, self.foam_conc))

        if self.extinguished:
            self.fire_area = 0.0
            self.agent.end_episode()

        return SimSnapshot(
            t=self.t, phase=self.phase, fire_area=self.fire_area,
            water_flow=self.water_flow,
            n_trunks_burn=self.n_trunks_burn, n_trunks_neighbor=self.n_trunks_nbr,
            n_pns=self.n_pns, n_bu=self.n_bu, has_shtab=self.has_shtab,
            foam_attacks=self.foam_attacks, foam_ready=self.foam_ready,
            spill=self.spill, secondary_fire=self.secondary_fire,
            localized=self.localized, extinguished=self.extinguished,
            risk=self._risk(), last_action=a, reward=r,
            roof_obstruction=self.roof_obstruction,
            foam_flow_ls=self.foam_flow_ls,
        )

    def _apply_scripted(self, t: int, desc: str):
        """Применить автоматические изменения из хронологии.

        Сначала пробует структурированный словарь effects (cfg["scripted_effects"]),
        затем — fallback: парсинг строки описания (для кастомных сценариев).
        """
        effects: dict = self._cfg.get("scripted_effects", {}).get(t, {})
        if effects:
            self._apply_effects(effects)
            return
        # ── Fallback: парсинг строки для кастомных сценариев ─────────────────
        lo = desc.lower()
        if "3 ствола антенор" in lo:
            self.n_trunks_burn = max(self.n_trunks_burn, 3)
            self.water_flow    = max(self.water_flow, 3*35)
        if "6 лафетных" in lo or "6 стволов" in lo:
            self.n_trunks_burn = max(self.n_trunks_burn, 6)
            self.water_flow    = max(self.water_flow, 600)
        if "7 стволов" in lo or "7 лафетных" in lo:
            self.n_trunks_burn = max(self.n_trunks_burn, 7)
            self.water_flow    = max(self.water_flow, 700)
        if ("штаб" in lo and "создан" in lo) or "ош" in lo:
            self.has_shtab = True
        if "3 бу" in lo or "3-х боевых" in lo or "3 боевых" in lo:
            self.n_bu = max(self.n_bu, 3)
        if "пнс" in lo and ("туапсе" in lo or "реке" in lo or "мост" in lo or "водоисточник" in lo):
            self.n_pns = min(4, self.n_pns + 1)
            self.water_flow += 110.0
        if "свищ" in lo or ("розлив" in lo and "горящего" in lo and "бензина" in lo):
            self.spill      = True
            self.spill_area = 300.0
            self.fire_area  = 1550.0
        if "розлив ликвидирован" in lo or ("розлив" in lo and "ликвидир" in lo):
            self.spill      = False
            self.fire_area  = 1250.0
        if "возгорание столовой" in lo or "возгорание вторичного очага" in lo:
            self.secondary_fire = True
        if "столовая потушена" in lo or "столовой ликвидирован" in lo or "вторичный очаг ликвидирован" in lo:
            self.secondary_fire = False
        if "локализован" in lo and "пожар" in lo:
            self.localized = True
        if "ликвидирован" in lo and ("пожар" in lo or "горение" in lo) and "розлив" not in lo:
            self.extinguished = True
            self.fire_area    = 0.0
        if "готовность к пенной" in lo:
            self.foam_ready = (self.foam_conc > 0 and self.n_pns >= 1)
            self.foam_conc  = max(self.foam_conc, 4.0)
        if "акп-50" in lo or "акп50" in lo:
            self.akp50_available  = True
            self.roof_obstruction = 0.20
        if "муссон" in lo and "пенная" in lo:
            self.foam_flow_ls = max(self.foam_flow_ls,
                                    NOZZLE_DB["Муссон-125"].flow_ls +
                                    2.0 * NOZZLE_DB["Акрон-Аполло"].flow_ls)
        if "гпс-600" in lo and self.foam_conc > 0:
            self.foam_ready = True
        if ("пожарные гидранты" in lo or ("пг-" in lo and "установить" in lo)):
            self.n_pns = min(4, self.n_pns + 1)
            self.water_flow += 15.0
        if "стволов а" in lo or "ствола а" in lo:
            import re as _re
            m = _re.search(r'(\d+)\s*стволов?\s*а', lo)
            if m:
                self.n_trunks_burn = max(self.n_trunks_burn, int(m.group(1)))
                self.water_flow = max(self.water_flow, self.n_trunks_burn * 7.0)

    def _apply_effects(self, fx: dict):
        """Применить структурированный словарь эффектов к состоянию симуляции."""
        if "n_trunks_burn_min" in fx:
            self.n_trunks_burn = max(self.n_trunks_burn, fx["n_trunks_burn_min"])
        if "water_flow_min" in fx:
            self.water_flow = max(self.water_flow, fx["water_flow_min"])
        if "water_flow_add" in fx:
            self.water_flow += fx["water_flow_add"]
        if "n_pns_add" in fx:
            self.n_pns = min(4, self.n_pns + fx["n_pns_add"])
        if "n_bu_min" in fx:
            self.n_bu = max(self.n_bu, fx["n_bu_min"])
        if "has_shtab" in fx:
            self.has_shtab = fx["has_shtab"]
        if "spill" in fx:
            self.spill = fx["spill"]
        if "spill_area" in fx:
            self.spill_area = fx["spill_area"]
        if "fire_area_set" in fx:
            self.fire_area = fx["fire_area_set"]
        if "secondary_fire" in fx:
            self.secondary_fire = fx["secondary_fire"]
        if "localized" in fx:
            self.localized = fx["localized"]
        if "extinguished" in fx:
            self.extinguished = fx["extinguished"]
        if fx.get("foam_ready"):
            self.foam_ready = (self.foam_conc > 0 and self.n_pns >= 1) or fx["foam_ready"] is True
            self.foam_conc  = max(self.foam_conc, fx.get("foam_conc_min", 0.0))
        if fx.get("akp50"):
            self.akp50_available  = True
            self.roof_obstruction = fx.get("roof_obstruction", 0.20)
        if "roof_obstruction" in fx and not fx.get("akp50"):
            self.roof_obstruction = fx["roof_obstruction"]
        if fx.get("foam_attack_start"):
            pass   # пенная атака запускается через _apply() действием S4/O3
        if fx.get("foam_attack_fail"):
            pass   # прерывание атаки — визуальное событие, физика через step()

    def _update_fire(self):
        if self.extinguished:
            self.fire_area = 0.0
            return
        area0 = self._cfg["initial_fire_area"]
        if self.localized:
            # При локализации площадь медленно убывает; нижняя граница — 50% или 800 м²
            min_area = area0 * 0.5 if self.scenario == "serp" else 800.0
            self.fire_area = max(min_area, self.fire_area - self.rng.uniform(0, 1))
        elif self.n_trunks_burn < 4:
            # При нехватке стволов охлаждения пожар распространяется (до 2.5×начальной площади)
            max_area = area0 * 2.5
            self.fire_area = min(max_area, self.fire_area + self.rng.uniform(0, 3))

    def _update_phase(self):
        # Фаза S5 наступает только после полного тушения (extinguished)
        if self.extinguished:
            self.phase = "S5"
        elif self.scenario == "serp":
            # Серпухов: пороги по реальной хронологии ПТП (мин от начала)
            if   self.t >= 90:   self.phase = "S4"   # локализация → пенная атака
            elif self.t >= 20:   self.phase = "S3"   # активное горение
            elif self.t >= 14:   self.phase = "S2"   # прибытие первых подразделений
            else:                self.phase = "S1"
        else:
            # tuapse: пороги по хронологии крупного пожара РВС
            if   self.t >= 4740: self.phase = "S4"   # 6-я пенная атака с АКП-50
            elif self.t >= 160:  self.phase = "S3"   # смена РТП-2, активное горение
            elif self.t >= 10:   self.phase = "S2"   # первые подразделения прибыли
            else:                self.phase = "S1"


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# ДИАЛОГ ВЫБОРА РЕЖИМА
# ══════════════════════════════════════════════════════════════════════════════
class ModeSelectDialog(tk.Tk):
    """Стартовый экран САУР-ПСП: пользователь выбирает режим работы.

    После клика диалог уничтожается и запускается TankFireApp в нужном режиме.
    """

    _MODE_DATA = [
        (
            "trainer",
            "🎓",
            "ТРЕНАЖЁР РТП",
            "#8e44ad",
            "#f3e5f5",
            (
                "Вы принимаете решения самостоятельно\n"
                "Программа оценивает каждое действие\n"
                "В конце — подробный разбор ошибок"
            ),
            [
                "Выбор действия вручную из кнопок",
                "Немедленная оценка (+/– баллы)",
                "Сравнение с рекомендацией агента",
                "Итоговый дебрифинг по сценарию",
            ],
        ),
        (
            "sppр",
            "🧭",
            "РЕЖИМ СППР",
            "#1a6da8",
            "#e3f0fb",
            (
                "Агент рекомендует оптимальное действие\n"
                "Вы принимаете или отклоняете совет\n"
                "Отчёт покажет все отклонения от RL"
            ),
            [
                "Рекомендация агента + уверенность %",
                "Кнопки «Принять» / «Переопределить»",
                "Граф Q-значений в реальном времени",
                "Журнал отклонений в финальном отчёте",
            ],
        ),
        (
            "research",
            "🔬",
            "ИССЛЕДОВАНИЕ RL",
            "#1e8449",
            "#e8f8f0",
            (
                "Полный доступ ко всем инструментам\n"
                "Обучение Flat RL и HRL агентов\n"
                "Массовые эксперименты и PDF-отчёт"
            ),
            [
                "Все 8 вкладок и настроек",
                "Обучение Flat Q-learning агента",
                "Трёхуровневый HRL (НГ/РТП/НБТП)",
                "N-прогонов, CI, Mann-Whitney, PDF",
            ],
        ),
    ]

    def __init__(self):
        super().__init__()
        self.title("САУР-ПСП — Выбор режима работы")
        self.configure(bg="#1c2833")
        self.resizable(False, False)
        self._selected: Optional[str] = None
        self._build()
        # Центрировать на экране
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _build(self):
        # ── Заголовок ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg="#17202a", pady=20)
        hdr.pack(fill="x")
        tk.Label(
            hdr,
            text="🔥  САУР-ПСП",
            font=("Arial", 22, "bold"),
            bg="#17202a", fg="#e74c3c",
        ).pack()
        tk.Label(
            hdr,
            text="Система автоматизированного управления реагированием пожарно-спасательного подразделения",
            font=("Arial", 10),
            bg="#17202a", fg="#aab7b8",
        ).pack()
        tk.Label(
            hdr,
            text="Выберите режим работы:",
            font=("Arial", 12, "bold"),
            bg="#17202a", fg="#ecf0f1",
        ).pack(pady=(12, 0))

        # ── Карточки режимов ─────────────────────────────────────────────────
        cards_row = tk.Frame(self, bg="#1c2833")
        cards_row.pack(padx=24, pady=20)

        for mode_key, icon, title, accent, bg_light, desc, features in self._MODE_DATA:
            self._make_card(cards_row, mode_key, icon, title, accent, bg_light, desc, features)

        # ── Подвал ───────────────────────────────────────────────────────────
        footer = tk.Frame(self, bg="#17202a", pady=6)
        footer.pack(fill="x")
        tk.Label(
            footer,
            text="Режим можно сменить: меню Файл → Сменить режим  |  ГОСТ Р 51043-2002 / СП 155.13130.2014",
            font=("Arial", 8),
            bg="#17202a", fg="#5d6d7e",
        ).pack()

    def _make_card(self, parent, mode_key, icon, title, accent, bg_light,
                   desc, features):
        """Одна карточка режима с hover-эффектом."""
        BG_NORMAL = "#2c3e50"
        BG_HOVER  = "#34495e"

        outer = tk.Frame(parent, bg=accent, bd=0)
        outer.pack(side="left", padx=10, pady=4)

        card = tk.Frame(outer, bg=BG_NORMAL, cursor="hand2", padx=16, pady=14, bd=0)
        card.pack(padx=2, pady=2)

        # Иконка
        tk.Label(card, text=icon, font=("Arial", 32), bg=BG_NORMAL, fg=accent).pack()

        # Заголовок
        tk.Label(card, text=title, font=("Arial", 13, "bold"),
                 bg=BG_NORMAL, fg="#ecf0f1").pack(pady=(4, 0))

        # Описание
        tk.Label(card, text=desc, font=("Arial", 9), bg=BG_NORMAL,
                 fg="#aab7b8", justify="center", wraplength=210).pack(pady=(6, 8))

        # Список возможностей
        feats_frame = tk.Frame(card, bg=BG_NORMAL)
        feats_frame.pack(anchor="w", fill="x")
        for feat in features:
            row = tk.Frame(feats_frame, bg=BG_NORMAL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text="✓", font=("Arial", 8, "bold"),
                     bg=BG_NORMAL, fg=accent).pack(side="left")
            tk.Label(row, text=f" {feat}", font=("Arial", 8),
                     bg=BG_NORMAL, fg="#bdc3c7").pack(side="left")

        # Кнопка выбора
        btn = tk.Button(
            card,
            text=f"  Выбрать →",
            font=("Arial", 10, "bold"),
            bg=accent, fg="white",
            relief="flat", padx=16, pady=6,
            cursor="hand2",
            command=lambda k=mode_key: self._select(k),
        )
        btn.pack(pady=(12, 0), fill="x")

        # Hover эффекты
        def _enter(e, c=card, bg=BG_HOVER):
            for w in c.winfo_children():
                try:
                    w.config(bg=bg)
                except Exception:
                    pass
            c.config(bg=bg)

        def _leave(e, c=card, bg=BG_NORMAL):
            for w in c.winfo_children():
                try:
                    w.config(bg=bg)
                except Exception:
                    pass
            c.config(bg=bg)

        card.bind("<Enter>", _enter)
        card.bind("<Leave>", _leave)

    def _select(self, mode: str):
        self._selected = mode
        self.destroy()

    def run(self) -> str:
        """Показать диалог и вернуть выбранный ключ режима."""
        self.mainloop()
        return self._selected or "research"


# ══════════════════════════════════════════════════════════════════════════════
# ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ══════════════════════════════════════════════════════════════════════════════
class TankFireApp(tk.Tk):
    """Главное окно приложения: GUI-оболочка вокруг TankFireSim.

    Компоновка: заголовок → горизонтальный PanedWindow (левая панель с картой
    и статусом + правая с вкладками) → нижняя панель управления.
    Анимационный цикл запускается через tk.after() с периодом TICK_MS мс;
    за один тик выполняется _speed шагов симуляции по STEP_MIN мин каждый.
    """

    SPEEDS = {"1×": 1, "5×": 5, "15×": 15, "60×": 60, "300×": 300}
    STEP_MIN = 5   # минут на шаг симуляции
    TICK_MS  = 80  # мс между шагами GUI

    def __init__(self, mode: str = "research"):
        super().__init__()
        self._mode = mode   # "trainer" | "sppр" | "research"
        _label, _accent, _bg = APP_MODES[mode]
        self.title(f"САУР-ПСП — {_label}")
        self.configure(bg=P["bg"])
        self.resizable(True, True)
        self.minsize(1100, 760)

        self._scenario_key = "tuapse"
        self.sim       = TankFireSim(seed=42, training=True, scenario=self._scenario_key)
        self._running  = False
        self._after_id: Optional[str] = None
        self._speed    = 1
        self._anim_t   = 0          # animation counter for fire flicker
        self._snap: Optional[SimSnapshot] = None

        # ── Блокировка для доступа к shared-данным из фоновых потоков ──────────
        self._train_lock = threading.Lock()

        # ── Авто-загрузка чекпоинта Flat RL (если существует) ────────────────
        if os.path.exists(_FLAT_QTABLE_PATH):
            _pre_agent = TankFireSim(seed=42, training=True, scenario="tuapse")
            if _pre_agent.agent.load(_FLAT_QTABLE_PATH):
                self._hrl_flat_sim     = _pre_agent
                self._hrl_flat_trained = True

        # ── Состояние тренажёра ──────────────────────────────────────────────
        self._trainer_score: int  = 0
        self._trainer_steps: int  = 0
        self._trainer_log:   list = []   # [{"t","user_a","best_a","pts","phase"}]
        self._trainer_active: bool = False

        # ── Состояние СППР ───────────────────────────────────────────────────
        self._sppр_deviations: int  = 0
        self._sppр_total:      int  = 0
        self._sppр_override_pending: Optional[int] = None  # индекс override-действия
        self._sppр_log:        list = []  # [{"t","rec_a","user_a","accepted"}]

        self._build_ui()
        self._draw_map()
        self.after(200, self._update_charts)

    # ─────────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────
    def _on_close(self):
        """Корректно закрыть приложение: освободить matplotlib-фигуры, остановить цикл."""
        import matplotlib.pyplot as plt
        self._on_pause()
        for attr in ("_fig_metrics", "_fig_rl", "_hrl_fig", "_batch_fig", "_sppр_qfig"):
            fig = getattr(self, attr, None)
            if fig is not None:
                try:
                    plt.close(fig)
                except Exception:
                    pass
        self.destroy()

    def _build_ui(self):
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # ── Меню ──────────────────────────────────────────────────────────────
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Сменить режим…",
                              command=self._switch_mode)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self._on_close)
        menubar.add_cascade(label="Файл", menu=file_menu)
        self.config(menu=menubar)

        # ── Полоса режима (цветная, под заголовком) ───────────────────────────
        _label, _accent, _bg_clr = APP_MODES[self._mode]
        mode_bar = tk.Frame(self, bg=_accent, height=28)
        mode_bar.pack(fill="x")
        mode_bar.pack_propagate(False)
        tk.Label(mode_bar, text=f"  {_label}",
                 font=("Arial", 10, "bold"),
                 bg=_accent, fg="white").pack(side="left", padx=10, pady=4)
        _mode_hints = {
            "trainer":  "Выберите действие РТП кнопками ниже карты",
            "sppр":     "Примите или отклоните рекомендацию агента",
            "research": "Полный доступ ко всем инструментам и агентам",
        }
        tk.Label(mode_bar, text=f"│  {_mode_hints[self._mode]}",
                 font=("Arial", 9),
                 bg=_accent, fg="#dde8f0").pack(side="left")
        # Кнопка смены режима в правом углу
        tk.Button(mode_bar, text="⇄  Сменить режим",
                  command=self._switch_mode,
                  bg="#ffffff", fg=_accent,
                  font=("Arial", 8, "bold"),
                  relief="flat", padx=8, pady=2).pack(side="right", padx=10, pady=4)

        # ── Заголовок: фиксированная полоса с названием сценария ─────────────
        hdr = tk.Frame(self, bg=P["panel"], height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🔥  САУР-ПСП — СИМУЛЯЦИЯ УПРАВЛЕНИЯ ТУШЕНИЕМ ПОЖАРА РВС",
                 font=("Arial", 15, "bold"), bg=P["panel"], fg=P["hi"]).pack(side="left", padx=16, pady=8)
        self._hdr_var = tk.StringVar(value=SCENARIOS[self._scenario_key]["name"])
        tk.Label(hdr, textvariable=self._hdr_var,
                 font=("Arial", 9), bg=P["panel"], fg=P["text2"]).pack(side="left")

        # ── Нижняя панель управления (ОБЯЗАТЕЛЬНО до paned!) ─────────────────
        # Tkinter pack: side="bottom" виден только если упакован ДО fill/expand
        self._build_controls()

        # ── Гид по порядку работы ─────────────────────────────────────────────
        self._build_workflow_guide()

        # ── Главная панель: карта слева, вкладки справа ───────────────────────
        paned = tk.PanedWindow(self, orient="horizontal", bg=P["bg"],
                               sashwidth=5, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        left = self._build_left(paned)
        right = self._build_right(paned)
        paned.add(left,  minsize=420)
        paned.add(right, minsize=620)
        paned.paneconfig(left,  width=MAP_W + 20)
        paned.paneconfig(right, width=640)

    # ── Левая панель: карта + статус ─────────────────────────────────────────
    def _build_left(self, parent) -> tk.Frame:
        frm = tk.Frame(parent, bg=P["panel"], bd=0)

        # Canvas — карта пожара
        lbl = tk.Label(frm, text=" КАРТА ПОЖАРА ", font=("Arial", 9, "bold"),
                       bg=P["panel2"], fg=P["text2"])
        lbl.pack(fill="x", padx=2, pady=(2,0))

        self.canvas = tk.Canvas(frm, width=MAP_W, height=MAP_H,
                                bg=P["canvas"], highlightthickness=1,
                                highlightbackground=P["grid"])
        self.canvas.pack(padx=4, pady=(4, 0))

        # ── Перетаскивание объектов на карте ──────────────────────────────────
        # _map_drag_objects: {name: (cx, cy, w, h)} — позиции перемещаемых объектов
        self._map_drag_objects: Dict[str, Tuple[int,int,int,int]] = {}
        # _map_drag_offsets: {name: (dx, dy)} — пользовательские смещения
        self._map_drag_offsets: Dict[str, Tuple[int,int]] = {}
        self._drag_target: Optional[str] = None
        self._drag_start: Optional[Tuple[int,int]] = None
        self.canvas.bind("<ButtonPress-1>",   self._map_drag_press)
        self.canvas.bind("<B1-Motion>",        self._map_drag_motion)
        self.canvas.bind("<ButtonRelease-1>",  self._map_drag_release)

        # ── Панель легенды и действия РТП (ниже карты) ───────────────────────
        self._map_info = tk.Frame(frm, bg=P["panel2"], bd=0)
        self._map_info.pack(fill="x", padx=4, pady=(1, 2))

        # Легенда (левая сторона)
        leg_frame = tk.Frame(self._map_info, bg=P["panel2"])
        leg_frame.pack(side="left", padx=8, pady=3)
        tk.Label(leg_frame, text="Легенда:", bg=P["panel2"], fg=P["text2"],
                 font=("Arial", 8, "bold")).pack(side="left", padx=(0,6))
        self._legend_items = []   # list of (canvas_widget, text_label)
        _leg_colors = [
            (P["rvs9"],     "Горящий РВС"),
            (P["rvs17"],    "Соседний РВС"),
            (P["water"],    "Стволы: — "),
            (P["unit_pns"], "ПНС: — "),
        ]
        for clr, txt in _leg_colors:
            dot = tk.Canvas(leg_frame, width=12, height=12, bg=P["panel2"],
                            highlightthickness=0)
            dot.create_oval(1, 1, 11, 11, fill=clr, outline="")
            dot.pack(side="left")
            lvar = tk.StringVar(value=txt)
            self._legend_items.append(lvar)
            tk.Label(leg_frame, textvariable=lvar, bg=P["panel2"], fg=P["text2"],
                     font=("Arial", 8)).pack(side="left", padx=(1, 8))

        # Действие РТП (правая сторона)
        act_frame = tk.Frame(self._map_info, bg=P["panel2"])
        act_frame.pack(side="right", padx=8, pady=3)
        self._map_action_var = tk.StringVar(value="Действие РТП: —")
        self._map_action_lbl = tk.Label(act_frame, textvariable=self._map_action_var,
                                         bg=P["panel2"], fg=P["hi"],
                                         font=("Arial", 8, "bold"))
        self._map_action_lbl.pack(side="top", anchor="e")
        self._map_phase_var = tk.StringVar(value="")
        tk.Label(act_frame, textvariable=self._map_phase_var,
                 bg=P["panel2"], fg=P["text2"],
                 font=("Arial", 8)).pack(side="top", anchor="e")

        # ── Панель СППР (под легендой, в левой колонке) ───────────────────────
        if self._mode == "sppр":
            self._build_sppр_panel(frm)

        # Статусная таблица
        sf = tk.Frame(frm, bg=P["panel2"])
        sf.pack(fill="x", padx=4, pady=2)
        self._status_vars = {}
        rows = [
            ("Время симуляции",  "sim_time",   "Ч+0"),
            ("Фаза пожара",      "phase",      "S1 — Обнаружение"),
            ("Площадь пожара",   "fire_area",  "1250 м²"),
            ("Расход ОВ",        "flow",       "0 л/с"),
            ("Стволов на РВС",   "trunks",     "0 / 0"),
            ("ПНС на воде",      "pns",        "0"),
            ("Боевых участков",  "bu",         "0 из 3"),
            ("Пенных атак",      "foam",       "0"),
            ("Риск",             "risk",       "НИЗКИЙ"),
            ("Действие RL",      "action",     "O4 — Разведка"),
            ("Препятствие крыши","roof_obs",   "70% (каркас)"),
            ("Расход пены",      "foam_flow",  "0 л/с"),
        ]
        for i, (lname, key, default) in enumerate(rows):
            r, c = divmod(i, 2)
            tk.Label(sf, text=f"{lname}:", font=("Arial", 8),
                     bg=P["panel2"], fg=P["text2"], anchor="e", width=16
                     ).grid(row=r, column=c*2, padx=(4,0), pady=1, sticky="e")
            var = tk.StringVar(value=default)
            self._status_vars[key] = var
            lbl2 = tk.Label(sf, textvariable=var, font=("Arial", 8, "bold"),
                            bg=P["panel2"], fg=P["text"], anchor="w", width=18)
            lbl2.grid(row=r, column=c*2+1, padx=(2,4), pady=1, sticky="w")

        return frm

    # ── Правая панель: вкладки ────────────────────────────────────────────────
    def _build_right(self, parent) -> tk.Frame:
        frm = tk.Frame(parent, bg=P["bg"])

        # ── Тренажёр: панель выбора действий над вкладками ────────────────────
        if self._mode == "trainer":
            self._build_trainer_panel(frm)

        nb = ttk.Notebook(frm)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",     background=P["panel"],  borderwidth=0)
        style.configure("TNotebook.Tab", background=P["panel2"], foreground=P["text2"],
                        padding=(10,4), font=("Arial", 9))
        style.map("TNotebook.Tab",
                  background=[("selected", P["accent"])],
                  foreground=[("selected", "#000000")])

        # ── Вкладки в строгом порядке шагов работы ──────────────────────────
        # Шаг 1 — Настройки (конфигурация перед стартом)
        t_cfg = tk.Frame(nb, bg=P["bg"])
        nb.add(t_cfg, text="⚙️  1. Настройки")
        self._build_settings_tab(t_cfg)

        # Шаг 2 — Хронология (журнал событий во время симуляции)
        t_log = tk.Frame(nb, bg=P["bg"])
        nb.add(t_log, text="📋  2. Хронология")
        self._build_timeline_tab(t_log)

        # Шаг 3 — Метрики (графики во время и после симуляции)
        t_met = tk.Frame(nb, bg=P["bg"])
        nb.add(t_met, text="📊  3. Метрики")
        self._build_metrics_tab(t_met)

        # Шаг 4 — Справочник действий (таблица допустимых действий РТП)
        t_ref = tk.Frame(nb, bg=P["bg"])
        nb.add(t_ref, text="📖  4. Справочник")
        self._build_reference_tab(t_ref)

        # Шаг 5 — Flat RL (обучение плоского Q-learning агента)
        t_rl = tk.Frame(nb, bg=P["bg"])
        nb.add(t_rl, text="🤖  5. Flat RL")
        self._build_rl_tab(t_rl)

        # Шаг 6 — Иерархический RL (обучение 3-уровневого агента)
        t_hrl = tk.Frame(nb, bg=P["bg"])
        nb.add(t_hrl, text="🏛  6. Иерарх. RL")
        self._build_hrl_tab(t_hrl)

        # Шаг 7 — Массовое моделирование (сравнительный анализ N эпизодов)
        t_bat = tk.Frame(nb, bg=P["bg"])
        nb.add(t_bat, text="🔬  7. Массовое")
        self._build_batch_tab(t_bat)

        # Шаг 8 — Отчёт / Экспорт (PDF, DOCX, JSON)
        t_rep = tk.Frame(nb, bg=P["bg"])
        nb.add(t_rep, text="📄  8. Отчёт")
        self._build_report_tab(t_rep)

        self._nb = nb

        # Скрыть вкладки, не нужные в текущем режиме
        visible = MODE_TABS.get(self._mode, set(range(8)))
        for i in range(nb.index("end")):
            if i not in visible:
                nb.tab(i, state="hidden")

        # По умолчанию всегда открыта вкладка «Настройки» (шаг 1)
        nb.select(0)

        return frm

    # ── Tab: Хронология ──────────────────────────────────────────────────────
    def _build_timeline_tab(self, parent):
        hdr = tk.Frame(parent, bg=P["panel2"])
        hdr.pack(fill="x", padx=4, pady=(4,0))
        tk.Label(hdr, text="Журнал событий симуляции",
                 font=("Arial", 9, "bold"), bg=P["panel2"], fg=P["hi"]
                 ).pack(side="left", padx=8, pady=4)
        tk.Label(hdr, text="(Жирным — текущие события симуляции)",
                 font=("Arial", 8), bg=P["panel2"], fg=P["text2"]
                 ).pack(side="left")

        tf = tk.Frame(parent, bg=P["bg"])
        tf.pack(fill="both", expand=True, padx=4, pady=4)
        sb = tk.Scrollbar(tf)
        sb.pack(side="right", fill="y")
        self._log_text = tk.Text(
            tf, yscrollcommand=sb.set, bg=P["canvas"],
            fg=P["text"], font=("Consolas", 8), state="disabled",
            wrap="word", bd=0, padx=6, pady=4, spacing1=2,
        )
        self._log_text.pack(fill="both", expand=True)
        sb.config(command=self._log_text.yview)

        # Тэги для цветов
        for color_key, color_val in [
            ("warn",    P["warn"]), ("danger",  P["danger"]),
            ("success", P["success"]), ("info",   P["info"]),
            ("neutral", P["text2"]), ("hi",     P["hi"]),
        ]:
            self._log_text.tag_config(color_key, foreground=color_val)
        self._log_text.tag_config("bold", font=("Consolas", 8, "bold"))

        # Наполнить хронологией из PDF
        self._log_text.config(state="normal")
        self._log_text.insert("end", "═"*68 + "\n", "neutral")
        self._log_text.insert("end", " ХРОНОЛОГИЯ СОБЫТИЙ СИМУЛЯЦИИ\n", "hi")
        self._log_text.insert("end", "═"*68 + "\n\n", "neutral")
        for t_ev, time_lbl, desc, color in TIMELINE:
            tag = "warn" if color == P["warn"] else \
                  "danger" if color == P["danger"] else \
                  "success" if color == P["success"] else "info"
            date_str = f"[{time_lbl}]"
            self._log_text.insert("end", f"{date_str:18s} ", "neutral")
            self._log_text.insert("end", f"{desc}\n", tag)
        self._log_text.config(state="disabled")
        self._log_text.see("end")

    # ── Tab: Метрики ──────────────────────────────────────────────────────────
    def _build_metrics_tab(self, parent):
        self._fig_metrics = Figure(figsize=(8, 7.5), facecolor=P["bg"])
        self._fig_metrics.subplots_adjust(left=0.10, right=0.97, top=0.95,
                                          bottom=0.07, hspace=0.55, wspace=0.40)
        gs = gridspec.GridSpec(3, 2, figure=self._fig_metrics)

        ax_kw = dict(facecolor=P["canvas"])
        self._ax_fire      = self._fig_metrics.add_subplot(gs[0, 0], **ax_kw)
        self._ax_water     = self._fig_metrics.add_subplot(gs[0, 1], **ax_kw)
        self._ax_trunks    = self._fig_metrics.add_subplot(gs[1, 0], **ax_kw)
        self._ax_risk      = self._fig_metrics.add_subplot(gs[1, 1], **ax_kw)
        self._ax_phase     = self._fig_metrics.add_subplot(gs[2, 0], **ax_kw)
        self._ax_foam_conc = self._fig_metrics.add_subplot(gs[2, 1], **ax_kw)

        _ylabels = {
            self._ax_fire:      ("Площадь пожара", "м²"),
            self._ax_water:     ("Расход ОВ", "л/с"),
            self._ax_trunks:    ("Стволов (горящий РВС)", "шт."),
            self._ax_risk:      ("Индекс риска", "0–1"),
            self._ax_phase:     ("Фаза пожара", "1=S1…5=S5"),
            self._ax_foam_conc: ("Запас пенообразователя", "т"),
        }
        for ax, (title, ylabel) in _ylabels.items():
            ax.set_title(title, color=P["text"], fontsize=8, pad=3)
            ax.set_ylabel(ylabel, color=P["text2"], fontsize=7)
            ax.set_xlabel("время, мин", color=P["text2"], fontsize=7)
            ax.tick_params(colors=P["text2"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(P["grid"])
            ax.set_facecolor(P["canvas"])
            ax.grid(True, color=P["grid"], linewidth=0.4, alpha=0.7)

        self._fc_metrics = FigureCanvasTkAgg(self._fig_metrics, master=parent)
        self._fc_metrics.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    # ── Tab: RL-агент ─────────────────────────────────────────────────────────
    def _build_rl_tab(self, parent):
        # ── Панель управления обучением ──────────────────────────────────────
        ctrl = tk.Frame(parent, bg=P["panel2"])
        ctrl.pack(fill="x", padx=4, pady=(4, 0))

        tk.Label(ctrl, text=" Обучение Flat Q-learning агента:",
                 bg=P["panel2"], fg=P["text"], font=("Arial", 9, "bold")
                 ).pack(side="left", padx=8, pady=5)

        tk.Label(ctrl, text="Эпизодов:", bg=P["panel2"], fg=P["text"],
                 font=("Arial", 8)).pack(side="left")
        self._rl_nep_var = tk.StringVar(value="500")
        tk.Entry(ctrl, textvariable=self._rl_nep_var, width=6,
                 bg=P["canvas"], fg=P["text"], relief="flat"
                 ).pack(side="left", padx=(2, 8))

        tk.Button(ctrl, text="▶  Запустить обучение",
                  command=self._rl_train_flat,
                  bg=P["success"], fg="white", font=("Arial", 8, "bold"),
                  relief="flat", padx=10, pady=3
                  ).pack(side="left", padx=4)
        tk.Button(ctrl, text="⏹  Стоп",
                  command=self._rl_stop_training,
                  bg=P["warn"], fg="white", font=("Arial", 8),
                  relief="flat", padx=8, pady=3
                  ).pack(side="left", padx=2)

        tk.Frame(ctrl, bg=P["panel2"], width=1, height=20
                 ).pack(side="left", padx=8, fill="y")
        tk.Button(ctrl, text="💾 Сохранить Q",
                  command=self._rl_save_qtable,
                  bg=P["info"], fg="white", font=("Arial", 8),
                  relief="flat", padx=6, pady=3
                  ).pack(side="left", padx=2)
        tk.Button(ctrl, text="📂 Загрузить Q",
                  command=self._rl_load_qtable,
                  bg=P["info"], fg="white", font=("Arial", 8),
                  relief="flat", padx=6, pady=3
                  ).pack(side="left", padx=2)
        tk.Button(ctrl, text="📊 Экспорт CSV",
                  command=self._rl_export_csv,
                  bg=P["text2"], fg="white", font=("Arial", 8),
                  relief="flat", padx=6, pady=3
                  ).pack(side="left", padx=2)

        _init_status = "Агент не обучен. Нажмите «Запустить обучение»."
        if hasattr(self, "_hrl_flat_trained") and self._hrl_flat_trained:
            cov = self._hrl_flat_sim.agent.coverage()
            _init_status = f"Чекпоинт загружен ✓ | покрытие={cov:.0%}"
        self._rl_status_var = tk.StringVar(value=_init_status)
        tk.Label(ctrl, textvariable=self._rl_status_var,
                 bg=P["panel2"], fg=P["text2"], font=("Arial", 8)
                 ).pack(side="left", padx=10)

        self._rl_progress = ttk.Progressbar(parent, length=300, mode="determinate",
                                             maximum=100)
        self._rl_progress.pack(fill="x", padx=8, pady=(2, 0))

        # ── Графики RL-агента ─────────────────────────────────────────────────
        self._fig_rl = Figure(figsize=(8, 4.8), facecolor=P["bg"])
        self._fig_rl.subplots_adjust(left=0.08, right=0.97, top=0.93,
                                     bottom=0.1, hspace=0.55, wspace=0.38)
        gs = gridspec.GridSpec(2, 2, figure=self._fig_rl)
        ax_kw = dict(facecolor=P["canvas"])

        self._ax_qval   = self._fig_rl.add_subplot(gs[0, :], **ax_kw)
        self._ax_actcnt = self._fig_rl.add_subplot(gs[1, 0], **ax_kw)
        self._ax_reward = self._fig_rl.add_subplot(gs[1, 1], **ax_kw)

        for ax, title in [
            (self._ax_qval,   "Q-значения действий (текущее состояние)"),
            (self._ax_actcnt, "Частота выбора действий"),
            (self._ax_reward, "Накопленная награда (по шагам)"),
        ]:
            ax.set_title(title, color=P["text"], fontsize=8, pad=3)
            ax.tick_params(colors=P["text2"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(P["grid"])
            ax.set_facecolor(P["canvas"])
            ax.grid(True, color=P["grid"], linewidth=0.4, alpha=0.7)

        self._fc_rl = FigureCanvasTkAgg(self._fig_rl, master=parent)
        self._fc_rl.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def _rl_train_flat(self):
        """Запустить обучение Flat Q-learning агента (из вкладки RL-агент)."""
        import threading
        try:
            n = max(1, int(self._rl_nep_var.get()))
        except ValueError:
            messagebox.showerror("САУР-ПСП", "Введите целое число эпизодов.")
            return
        self._rl_status_var.set(f"Обучение... (0 / {n} эп.)")
        self._rl_progress["value"] = 0
        if not hasattr(self, "_hrl_stop"):
            self._hrl_stop = [False]
        self._hrl_stop[0] = False

        def _run():
            rng = __import__("random").Random(42)
            flat_sim = TankFireSim(seed=42, training=True, scenario="tuapse")
            self._hrl_flat_sim = flat_sim
            phases = [
                (max(1, n // 5),    ["serp"]),
                (max(1, n * 3 // 10), ["serp", "tuapse"]),
                (max(1, n // 2),    ["tuapse"]),
            ]
            done_ep = 0
            total_flat = sum(p[0] for p in phases)
            for ep_count, scenarios in phases:
                for _ in range(ep_count):
                    if self._hrl_stop[0]:
                        break
                    scen = rng.choice(scenarios)
                    flat_sim.scenario = scen
                    flat_sim._cfg     = SCENARIOS[scen]
                    flat_sim.reset()
                    steps = 0
                    while (not flat_sim.extinguished
                           and flat_sim.t < flat_sim._cfg["total_min"]
                           and steps < 2000):
                        flat_sim.step()
                        steps += 1
                    done_ep += 1
                    _upd_interval = max(1, total_flat // 100)
                    if done_ep % _upd_interval == 0:
                        pct = done_ep / total_flat * 100
                        status = (f"Flat Q: эп. {done_ep}/{total_flat}  "
                                  f"ε={flat_sim.agent.epsilon:.3f}")
                        self.after(0, lambda p=pct, s=status: [
                            self._rl_progress.configure(value=p),
                            self._rl_status_var.set(s),
                        ])
                    # Автосохранение чекпоинта каждые 10% эпизодов
                    _ckpt_interval = max(10, total_flat // 10)
                    if done_ep % _ckpt_interval == 0:
                        flat_sim.agent.save(_FLAT_QTABLE_PATH)
            self._hrl_flat_trained = True
            flat_sim.agent.save(_FLAT_QTABLE_PATH)
            cov = flat_sim.agent.coverage()
            cov_str = f" | покрытие={cov:.0%}"
            self.after(0, lambda: [
                self._rl_status_var.set(
                    f"Обучен ✓  {total_flat} эп. "
                    f"| ε={flat_sim.agent.epsilon:.3f}{cov_str}  💾"),
                self._rl_progress.configure(value=100),
                self._update_charts(),
            ])

        threading.Thread(target=_run, daemon=True).start()

    def _rl_save_qtable(self):
        """Ручное сохранение Q-таблицы в файл по выбору пользователя."""
        if not hasattr(self, "_hrl_flat_sim"):
            messagebox.showwarning("САУР-ПСП", "Агент ещё не обучен.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy archive", "*.npz")],
            initialfile="flat_qtable.npz",
            title="Сохранить Q-таблицу",
        )
        if path:
            self._hrl_flat_sim.agent.save(path)
            self._rl_status_var.set(f"Q-таблица сохранена: {os.path.basename(path)}")

    def _rl_load_qtable(self):
        """Ручная загрузка Q-таблицы из .npz файла."""
        path = filedialog.askopenfilename(
            filetypes=[("NumPy archive", "*.npz")],
            title="Загрузить Q-таблицу",
        )
        if path:
            if not hasattr(self, "_hrl_flat_sim"):
                self._hrl_flat_sim = TankFireSim(seed=42, training=True, scenario="tuapse")
            ok = self._hrl_flat_sim.agent.load(path)
            if ok:
                self._hrl_flat_trained = True
                cov = self._hrl_flat_sim.agent.coverage()
                self._rl_status_var.set(
                    f"Q-таблица загружена ✓ | покрытие={cov:.0%} | {os.path.basename(path)}")
                self._update_charts()
            else:
                messagebox.showerror("САУР-ПСП", "Не удалось загрузить Q-таблицу.")

    def _rl_export_csv(self):
        """Экспорт Q-таблицы в CSV."""
        if not hasattr(self, "_hrl_flat_sim"):
            messagebox.showwarning("САУР-ПСП", "Агент ещё не обучен.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV файл", "*.csv")],
            initialfile="qtable_export.csv",
            title="Экспорт Q-таблицы в CSV",
        )
        if not path:
            return
        agent = self._hrl_flat_sim.agent
        action_codes = [a[0] for a in ACTIONS]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["state_idx"] + action_codes)
            for s_idx in range(agent.Q.shape[0]):
                writer.writerow([s_idx] + agent.Q[s_idx].tolist())
        self._rl_status_var.set(f"Q-таблица экспортирована: {os.path.basename(path)}")

    def _rl_stop_training(self):
        if hasattr(self, "_hrl_stop"):
            self._hrl_stop[0] = True
        if hasattr(self, "_rl_status_var"):
            self._rl_status_var.set("Обучение остановлено.")

    # ── Tab: Справочник действий ──────────────────────────────────────────────
    def _build_reference_tab(self, parent):
        # Selector по фазе
        hdr = tk.Frame(parent, bg=P["panel2"])
        hdr.pack(fill="x", padx=4, pady=(4,0))
        tk.Label(hdr, text="Фаза:", bg=P["panel2"], fg=P["text"],
                 font=("Arial", 9)).pack(side="left", padx=8)
        self._ref_phase_var = tk.StringVar(value="S2")
        for ph in ["S1","S2","S3","S4","S5"]:
            ttk.Radiobutton(hdr, text=ph, variable=self._ref_phase_var,
                            value=ph, command=self._refresh_reference
                            ).pack(side="left", padx=4)

        tf = tk.Frame(parent, bg=P["bg"])
        tf.pack(fill="both", expand=True, padx=4, pady=4)
        sb = tk.Scrollbar(tf)
        sb.pack(side="right", fill="y")
        self._ref_text = tk.Text(
            tf, yscrollcommand=sb.set, bg=P["canvas"],
            fg=P["text"], font=("Consolas", 8), state="disabled",
            wrap="word", bd=0, padx=8, pady=6, spacing1=3,
        )
        self._ref_text.pack(fill="both", expand=True)
        sb.config(command=self._ref_text.yview)
        for tag, fg in [("code", P["hi"]), ("desc", P["text"]),
                        ("hint", P["text2"]), ("head", P["accent"]),
                        ("strat", P["strat"]), ("tact", P["tact"]),
                        ("oper", P["oper"])]:
            self._ref_text.tag_config(tag, foreground=fg)
        self._ref_text.tag_config("head", font=("Consolas", 9, "bold"))
        self._ref_text.tag_config("code", font=("Consolas", 8, "bold"))
        self._refresh_reference()

    def _refresh_reference(self):
        ph  = self._ref_phase_var.get()
        # Используем словарь действий текущего сценария (если задан), иначе глобальный
        scen_cfg = SCENARIOS.get(getattr(self, "_scenario_key", "tuapse"), {})
        scen_acts = scen_cfg.get("actions_by_phase")
        if scen_acts is None:
            scen_acts = ACTIONS_BY_PHASE
        acts = scen_acts.get(ph, ACTIONS_BY_PHASE.get(ph, []))
        t = self._ref_text
        t.config(state="normal")
        t.delete("1.0","end")
        t.insert("end", f"{'═'*60}\n", "hint")
        t.insert("end", f" Перечень действий РТП | Фаза {ph}: {PHASE_NAMES[ph]}\n", "head")
        t.insert("end", f"{'═'*60}\n\n", "hint")
        for code, name, hint in acts:
            a_code, a_level, a_desc = next((a for a in ACTIONS if a[0]==code), (code,"",""))
            color_tag = {"стратег.":"strat","тактич.":"tact","оперативн.":"oper"}.get(a_level,"desc")
            t.insert("end", f"  [{code}]  ", "code")
            t.insert("end", f"{name}\n", color_tag)
            t.insert("end", f"         {hint}\n\n", "hint")
        t.config(state="disabled")

    # ── Tab: Отчёт / Экспорт ─────────────────────────────────────────────────
    def _build_report_tab(self, parent):
        """Вкладка для генерации отчётов и экспорта данных по итогам симуляции."""
        # Импорт модулей отчётности (отложенный, чтобы не замедлять запуск)
        try:
            from .report_generator import (generate_comprehensive_report,
                                            generate_trainer_report,
                                            generate_sppр_report,
                                            export_for_article_json,
                                            export_for_article_docx)
            from .manual_generator import generate_manual
        except ImportError:
            from report_generator import (generate_comprehensive_report,
                                          generate_trainer_report,
                                          generate_sppр_report,
                                          export_for_article_json,
                                          export_for_article_docx)
            from manual_generator import generate_manual

        self._gen_pdf         = generate_comprehensive_report
        self._gen_pdf_trainer = generate_trainer_report
        self._gen_pdf_sppр    = generate_sppр_report
        self._gen_json   = export_for_article_json
        self._gen_docx   = export_for_article_docx
        self._gen_manual = generate_manual

        self._pdf_path    = ""
        self._json_path   = ""
        self._docx_path   = ""
        self._manual_path = ""

        # Заголовок
        hdr = tk.Frame(parent, bg=P["panel2"])
        hdr.pack(fill="x", padx=4, pady=(4, 0))
        tk.Label(hdr, text="  📄  Отчёт по результатам моделирования",
                 font=("Arial", 10, "bold"), bg=P["panel2"], fg=P["hi"]
                 ).pack(side="left", padx=8, pady=6)

        # Область содержимого с прокруткой
        canvas = tk.Canvas(parent, bg=P["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=P["bg"])
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        def section(title):
            f = tk.LabelFrame(inner, text=f"  {title}  ", bg=P["panel"],
                              fg=P["accent"], font=("Arial", 9, "bold"),
                              bd=1, relief="groove")
            f.pack(fill="x", padx=10, pady=6, ipadx=6, ipady=4)
            return f

        # ── Краткая сводка итогов симуляции ──────────────────────────────────
        f_summary = section("Краткая сводка итогов симуляции")
        self._report_summary_var = tk.StringVar(value="(запустите симуляцию и нажмите «Обновить сводку»)")
        tk.Label(f_summary, textvariable=self._report_summary_var,
                 font=("Consolas", 8), bg=P["panel"], fg=P["text"],
                 justify="left", wraplength=560).pack(anchor="w", padx=10, pady=4)

        tk.Button(f_summary, text="🔄  Обновить сводку",
                  command=self._refresh_report_summary,
                  bg=P["info"], fg="#fff", font=("Arial", 9, "bold"),
                  relief="flat", padx=10, pady=4).pack(anchor="w", padx=10, pady=(0, 4))

        # ── PDF-отчёт ─────────────────────────────────────────────────────────
        f_pdf = section("PDF-отчёт для оперативного штаба")
        tk.Label(f_pdf, text=("Полный отчёт с параметрами сценария, динамикой пожара, RL-агентом,\n"
                "сравнением Flat vs Иерархический RL, хронологией и выводами."),
                 font=("Arial", 8), bg=P["panel"], fg=P["text2"], justify="left"
                 ).pack(anchor="w", padx=10, pady=(4, 2))
        self._pdf_status = tk.StringVar(value="")
        tk.Label(f_pdf, textvariable=self._pdf_status,
                 font=("Consolas", 8), bg=P["panel"], fg=P["success"]
                 ).pack(anchor="w", padx=10)
        pdf_btns = tk.Frame(f_pdf, bg=P["panel"])
        pdf_btns.pack(anchor="w", padx=10, pady=4)
        tk.Button(pdf_btns, text="📄  Сформировать PDF-отчёт",
                  command=self._do_generate_pdf,
                  bg=P["danger"], fg="#fff", font=("Arial", 9, "bold"),
                  relief="flat", padx=12, pady=5).pack(side="left", padx=(0, 6))
        self._pdf_open_btn = tk.Button(pdf_btns, text="📂  Открыть PDF",
                  command=lambda: self._open_file(self._pdf_path),
                  bg=P["info"], fg="#fff", font=("Arial", 9),
                  relief="flat", padx=8, pady=5, state="disabled")
        self._pdf_open_btn.pack(side="left")

        # ── JSON-выгрузка ─────────────────────────────────────────────────────
        f_json = section("JSON-выгрузка для научной статьи")
        tk.Label(f_json,
                 text=("Структурированные данные: метаданные, параметры сценария, результаты,\n"
                       "временны́е ряды, статистика RL-агента, нормативный анализ."),
                 font=("Arial", 8), bg=P["panel"], fg=P["text2"], justify="left"
                 ).pack(anchor="w", padx=10, pady=(4, 2))
        self._json_status = tk.StringVar(value="")
        tk.Label(f_json, textvariable=self._json_status,
                 font=("Consolas", 8), bg=P["panel"], fg=P["success"]
                 ).pack(anchor="w", padx=10)
        json_btns = tk.Frame(f_json, bg=P["panel"])
        json_btns.pack(anchor="w", padx=10, pady=4)
        tk.Button(json_btns, text="📊  Экспортировать JSON",
                  command=self._do_export_json,
                  bg=P["info"], fg="#fff", font=("Arial", 9, "bold"),
                  relief="flat", padx=12, pady=5).pack(side="left", padx=(0, 6))
        self._json_open_btn = tk.Button(json_btns, text="📂  Открыть JSON",
                  command=lambda: self._open_file(self._json_path),
                  bg=P["info"], fg="#fff", font=("Arial", 9),
                  relief="flat", padx=8, pady=5, state="disabled")
        self._json_open_btn.pack(side="left")

        # ── DOCX-черновик ─────────────────────────────────────────────────────
        f_docx = section("DOCX-черновик для написания статьи")
        tk.Label(f_docx,
                 text=("Документ Word с готовыми фрагментами текста для разделов статьи:\n"
                       "«Объект и метод», «Результаты», «Обсуждение», подписи к рисункам."),
                 font=("Arial", 8), bg=P["panel"], fg=P["text2"], justify="left"
                 ).pack(anchor="w", padx=10, pady=(4, 2))
        self._docx_status = tk.StringVar(value="")
        tk.Label(f_docx, textvariable=self._docx_status,
                 font=("Consolas", 8), bg=P["panel"], fg=P["success"]
                 ).pack(anchor="w", padx=10)
        docx_btns = tk.Frame(f_docx, bg=P["panel"])
        docx_btns.pack(anchor="w", padx=10, pady=4)
        tk.Button(docx_btns, text="📝  Экспортировать DOCX",
                  command=self._do_export_docx,
                  bg=P["warn"], fg="#000", font=("Arial", 9, "bold"),
                  relief="flat", padx=12, pady=5).pack(side="left", padx=(0, 6))
        self._docx_open_btn = tk.Button(docx_btns, text="📂  Открыть DOCX",
                  command=lambda: self._open_file(self._docx_path),
                  bg=P["info"], fg="#fff", font=("Arial", 9),
                  relief="flat", padx=8, pady=5, state="disabled")
        self._docx_open_btn.pack(side="left")

    def _refresh_report_summary(self):
        """Обновить краткую сводку итогов на вкладке «Отчёт»."""
        sim = self.sim
        outcome = ("✅ ЛИКВИДИРОВАН" if sim.extinguished else
                   ("🔒 ЛОКАЛИЗОВАН" if sim.localized else "🔥 АКТИВНЫЙ"))
        risk = sim._risk()
        lines = [
            f"Сценарий:         {sim._cfg['name'][:55]}",
            f"Время симуляции:  {sim.t} мин ({self._fmt_time(sim.t)})",
            f"Исход пожара:     {outcome}",
            f"Фаза:             {sim.phase}  |  Площадь: {sim.fire_area:.0f} м²",
            f"Пенных атак:      {sim.foam_attacks}  |  АКП-50: {'Да' if sim.akp50_available else 'Нет'}",
            f"Расход ОВ:        {sim.water_flow:.0f} л/с  |  ПНС: {sim.n_pns}  |  БУ: {sim.n_bu}",
            f"Индекс риска:     {risk:.3f}  ({('КРИТИЧЕСКИЙ' if risk>0.75 else 'ВЫСОКИЙ' if risk>0.5 else 'СРЕДНИЙ' if risk>0.25 else 'НИЗКИЙ')})",
            f"RL ε:             {sim.agent.epsilon:.3f}  |  Шагов: {len(sim.h_reward)}  |  Σ reward: {sum(sim.h_reward):.1f}",
        ]
        self._report_summary_var.set("\n".join(lines))

    def _do_generate_pdf(self):
        """Запустить генерацию PDF-отчёта в отдельном потоке.

        Тип отчёта зависит от режима приложения:
          - trainer  → отчёт тренажёра (счёт, разбор действий)
          - sppр     → отчёт СППР (журнал решений, уровень согласия)
          - research → полный исследовательский отчёт (RL-анализ, сравнение)
        """
        import threading
        self._pdf_status.set("⏳ Генерация PDF…")
        self.update_idletasks()

        mode         = self._mode
        trainer_log  = list(self._trainer_log)
        trainer_score = self._trainer_score
        trainer_steps = self._trainer_steps
        sppр_log     = list(self._sppр_log)
        sppр_total   = self._sppр_total
        sppр_dev     = self._sppр_deviations
        flat_sim     = getattr(self, "_hrl_flat_sim", None)
        hier_sim     = getattr(self, "_hrl_sim", None)

        def _run():
            try:
                if mode == "trainer":
                    path = self._gen_pdf_trainer(
                        self.sim,
                        trainer_log=trainer_log,
                        trainer_score=trainer_score,
                        trainer_steps=trainer_steps,
                    )
                elif mode == "sppр":
                    path = self._gen_pdf_sppр(
                        self.sim,
                        sppр_log=sppр_log,
                        sppр_total=sppр_total,
                        sppр_deviations=sppр_dev,
                    )
                else:
                    path = self._gen_pdf(self.sim, flat_sim=flat_sim, hier_sim=hier_sim)
                self._pdf_path = path
                self.after(0, lambda: [
                    self._pdf_status.set(f"✅ {os.path.basename(path)}"),
                    self._pdf_open_btn.config(state="normal"),
                ])
            except Exception as exc:
                self.after(0, lambda e=exc: self._pdf_status.set(f"❌ {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _do_export_json(self):
        """Экспортировать данные в JSON для научной статьи."""
        import threading
        self._json_status.set("⏳ Экспорт JSON…")
        self.update_idletasks()

        def _run():
            try:
                path = self._gen_json(self.sim)
                self._json_path = path
                self.after(0, lambda: [
                    self._json_status.set(f"✅ {os.path.basename(path)}"),
                    self._json_open_btn.config(state="normal"),
                ])
            except Exception as exc:
                self.after(0, lambda e=exc: self._json_status.set(f"❌ {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _do_export_docx(self):
        """Экспортировать DOCX-черновик для написания статьи."""
        import threading
        self._docx_status.set("⏳ Экспорт DOCX…")
        self.update_idletasks()

        def _run():
            try:
                path = self._gen_docx(self.sim)
                self._docx_path = path
                self.after(0, lambda: [
                    self._docx_status.set(f"✅ {os.path.basename(path)}"),
                    self._docx_open_btn.config(state="normal"),
                ])
            except Exception as exc:
                self.after(0, lambda e=exc: self._docx_status.set(f"❌ {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _do_generate_manual(self):
        """Создать PDF-мануал программы."""
        import threading
        self._manual_status.set("⏳ Создание мануала…")

        def _run():
            try:
                path = self._gen_manual()
                self._manual_path = path
                self.after(0, lambda: [
                    self._manual_status.set(f"✅ {os.path.basename(path)}"),
                    self._manual_open_btn.config(state="normal"),
                ])
            except Exception as exc:
                self.after(0, lambda e=exc: self._manual_status.set(f"❌ {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _open_file(self, path: str):
        """Открыть файл в системном приложении по умолчанию."""
        import subprocess, sys
        if not path or not os.path.exists(path):
            return
        if sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])

    # ── Tab: Иерархический RL ─────────────────────────────────────────────────
    def _build_hrl_tab(self, parent):
        """Вкладка 3-уровневого иерархического RL.

        Структура:
          Верх: параметры (2 колонки) + curriculum + приоры целей
          Низ:  кнопки управления, прогресс, сравнительная таблица, графики
        """
        import threading

        # Состояние вкладки
        self._hrl_sim     = None   # HierarchicalTankFireSim
        self._hrl_trained = False
        self._hrl_stop    = [False]
        self._hrl_flat_trained = False

        # ── Скроллируемый контейнер ──────────────────────────────────────
        outer = tk.Frame(parent, bg=P["bg"])
        outer.pack(fill="both", expand=True)
        canvas_s = tk.Canvas(outer, bg=P["bg"], highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=canvas_s.yview)
        canvas_s.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas_s.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas_s, bg=P["bg"])
        canvas_s.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas_s.configure(
                       scrollregion=canvas_s.bbox("all")))

        # ── Заголовок ─────────────────────────────────────────────────────
        hdr = tk.Frame(inner, bg=P["panel2"])
        hdr.pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(hdr, text="🏛  3-УРОВНЕВЫЙ ИЕРАРХИЧЕСКИЙ Q-LEARNING",
                 font=("Arial", 10, "bold"), bg=P["panel2"], fg=P["hi"]
                 ).pack(side="left", padx=8, pady=5)
        tk.Label(hdr, text="L3: НГ/ГУ МЧС → L2: РТП/НШ → L1: НБТП/командир",
                 font=("Arial", 8), bg=P["panel2"], fg=P["text2"]
                 ).pack(side="left", padx=4)

        # ── Строка параметров (2 блока рядом) ─────────────────────────────
        params_row = tk.Frame(inner, bg=P["bg"])
        params_row.pack(fill="x", padx=6, pady=2)

        # --- Блок «Иерархия» ---
        pf1 = tk.LabelFrame(params_row, text=" Параметры иерархии ",
                             font=("Arial", 8, "bold"),
                             bg=P["panel"], fg=P["text2"], bd=1)
        pf1.pack(side="left", fill="both", expand=True, padx=(0, 3), pady=2)

        hrl_params = [
            ("K3 (шагов, L3 НГ):",          "hrl_k3",     "6"),
            ("K2 (шагов, L2 РТП):",          "hrl_k2",     "2"),
            ("λ интринзик. награды:",         "hrl_lambda", "0.30"),
            ("Мягкое маск-ние (1=да):",       "hrl_soft",   "1"),
        ]
        self._hrl_vars = {}
        for i, (lbl, key, dflt) in enumerate(hrl_params):
            tk.Label(pf1, text=lbl, font=("Arial", 8), bg=P["panel"],
                     fg=P["text2"], anchor="e", width=22
                     ).grid(row=i, column=0, padx=4, pady=2, sticky="e")
            var = tk.StringVar(value=dflt)
            self._hrl_vars[key] = var
            tk.Entry(pf1, textvariable=var, width=7, bg=P["panel2"],
                     fg=P["text"], insertbackground=P["text"],
                     font=("Arial", 8), relief="flat"
                     ).grid(row=i, column=1, padx=4, pady=2, sticky="w")

        # --- Блок «Обучение» ---
        pf2 = tk.LabelFrame(params_row, text=" Параметры Q-learning ",
                             font=("Arial", 8, "bold"),
                             bg=P["panel"], fg=P["text2"], bd=1)
        pf2.pack(side="left", fill="both", expand=True, padx=(3, 3), pady=2)

        rl_params = [
            ("α L3:",  "hrl_al3", "0.10"), ("γ L3:",  "hrl_gl3", "0.90"),
            ("α L2:",  "hrl_al2", "0.12"), ("γ L2:",  "hrl_gl2", "0.92"),
            ("α L1:",  "hrl_al1", "0.15"), ("γ L1:",  "hrl_gl1", "0.95"),
            ("ε нач:", "hrl_eps", "0.90"), ("ε убыв:","hrl_edc", "0.993"),
        ]
        for i, (lbl, key, dflt) in enumerate(rl_params):
            r, c = divmod(i, 2)
            tk.Label(pf2, text=lbl, font=("Arial", 8), bg=P["panel"],
                     fg=P["text2"], anchor="e", width=8
                     ).grid(row=r, column=c*2, padx=4, pady=2, sticky="e")
            var = tk.StringVar(value=dflt)
            self._hrl_vars[key] = var
            tk.Entry(pf2, textvariable=var, width=7, bg=P["panel2"],
                     fg=P["text"], insertbackground=P["text"],
                     font=("Arial", 8), relief="flat"
                     ).grid(row=r, column=c*2+1, padx=4, pady=2, sticky="w")

        # --- Блок «Curriculum» ---
        pf3 = tk.LabelFrame(params_row, text=" Curriculum Learning ",
                             font=("Arial", 8, "bold"),
                             bg=P["panel"], fg=P["text2"], bd=1)
        pf3.pack(side="left", fill="both", expand=True, padx=(3, 0), pady=2)

        cur_params = [
            ("Фаза 1 (эп., РВС-2000):",     "hrl_cur1", "200"),
            ("Фаза 2 (эп., РВС-2000+20000):","hrl_cur2", "300"),
            ("Фаза 3 (эп., РВС-20000):",    "hrl_cur3", "500"),
            ("Flat агент (эп.):",            "hrl_flat_ep", "1000"),
            ("Оценочных прогонов:",          "hrl_neval", "100"),
        ]
        for i, (lbl, key, dflt) in enumerate(cur_params):
            tk.Label(pf3, text=lbl, font=("Arial", 8), bg=P["panel"],
                     fg=P["text2"], anchor="e", width=24
                     ).grid(row=i, column=0, padx=4, pady=2, sticky="e")
            var = tk.StringVar(value=dflt)
            self._hrl_vars[key] = var
            tk.Entry(pf3, textvariable=var, width=7, bg=P["panel2"],
                     fg=P["text"], insertbackground=P["text"],
                     font=("Arial", 8), relief="flat"
                     ).grid(row=i, column=1, padx=4, pady=2, sticky="w")

        # ── Калибровка приоров целей L2 ───────────────────────────────────
        prior_frame = tk.LabelFrame(inner,
                                    text=" Калибровочные веса целей L2 "
                                         "(из актов пожаров; 0 = не задано) ",
                                    font=("Arial", 8, "bold"),
                                    bg=P["panel"], fg=P["text2"], bd=1)
        prior_frame.pack(fill="x", padx=6, pady=2)

        try:
            from hrl_agents import HGoal
        except ImportError:
            from .hrl_agents import HGoal

        self._hrl_prior_vars = {}
        for i, (gidx, gname) in enumerate(HGoal.NAMES.items()):
            col_base = i * 3
            color = HGoal.COLORS.get(gidx, P["text"])
            tk.Label(prior_frame, text=f"{gname}:", font=("Arial", 8),
                     bg=P["panel"], fg=color, width=14, anchor="e"
                     ).grid(row=0, column=col_base, padx=(8, 2), pady=4, sticky="e")
            var = tk.StringVar(value="0")
            self._hrl_prior_vars[gidx] = var
            tk.Entry(prior_frame, textvariable=var, width=6, bg=P["panel2"],
                     fg=color, insertbackground=color, font=("Arial", 8),
                     relief="flat"
                     ).grid(row=0, column=col_base+1, padx=2, pady=4, sticky="w")
        tk.Label(prior_frame,
                 text="(0 во всех полях = равномерное распределение по умолчанию)",
                 font=("Arial", 7), bg=P["panel"], fg=P["text2"]
                 ).grid(row=1, column=0, columnspan=15, padx=8, sticky="w")

        # ── Кнопки управления ─────────────────────────────────────────────
        btn_frame = tk.Frame(inner, bg=P["panel2"])
        btn_frame.pack(fill="x", padx=6, pady=4)

        def _btn(text, cmd, fg=P["text"], width=20):
            return tk.Button(btn_frame, text=text, command=cmd,
                             bg=P["panel"], fg=fg, font=("Arial", 9, "bold"),
                             relief="flat", bd=0, padx=8, pady=5,
                             activebackground=P["accent"], cursor="hand2",
                             width=width)

        # Подсказка: обучение Flat агента — на вкладке 5
        note = tk.Frame(btn_frame, bg=P["panel2"], bd=1, relief="groove")
        note.pack(side="left", padx=(0, 12), pady=3)
        tk.Label(note,
                 text="ℹ  Сначала обучите Flat агента\n    на вкладке «🤖 5. Flat RL»",
                 bg=P["panel2"], fg=P["info"], font=("Arial", 8),
                 justify="left").pack(padx=8, pady=4)

        _btn("🏛 Обучить иерарх. агента", self._hrl_train_hier,
             fg=P["warn"]).pack(side="left", padx=4, pady=3)
        _btn("📊 Сравнить агентов",       self._hrl_compare,
             fg=P["success"]).pack(side="left", padx=4, pady=3)
        _btn("⏹ Остановить",             self._hrl_stop_training,
             fg=P["danger"], width=12).pack(side="left", padx=4, pady=3)
        _btn("💾 Сохранить Q-таблицы",    self._hrl_save_qtables,
             fg=P["text2"], width=18).pack(side="left", padx=4, pady=3)

        # ── Прогресс-бар и статус ─────────────────────────────────────────
        prog_frame = tk.Frame(inner, bg=P["panel"])
        prog_frame.pack(fill="x", padx=6, pady=2)
        self._hrl_status_var = tk.StringVar(value="Агенты не обучены")
        tk.Label(prog_frame, textvariable=self._hrl_status_var,
                 font=("Arial", 8), bg=P["panel"], fg=P["hi"]
                 ).pack(side="left", padx=8)
        self._hrl_progress = ttk.Progressbar(prog_frame, orient="horizontal",
                                              length=300, mode="determinate")
        self._hrl_progress.pack(side="left", padx=8, pady=3)
        self._hrl_phase_var = tk.StringVar(value="")
        tk.Label(prog_frame, textvariable=self._hrl_phase_var,
                 font=("Arial", 8), bg=P["panel"], fg=P["text2"]
                 ).pack(side="left", padx=4)

        # ── Индикаторы текущих целей (отображаются при симуляции) ─────────
        ind_frame = tk.Frame(inner, bg=P["panel"])
        ind_frame.pack(fill="x", padx=6, pady=2)
        tk.Label(ind_frame, text="Текущий режим (L3):",
                 font=("Arial", 8), bg=P["panel"], fg=P["text2"]
                 ).pack(side="left", padx=8)
        self._hrl_mode_var = tk.StringVar(value="—")
        tk.Label(ind_frame, textvariable=self._hrl_mode_var,
                 font=("Arial", 9, "bold"), bg=P["panel"], fg=P["success"],
                 width=22
                 ).pack(side="left", padx=4)
        tk.Label(ind_frame, text="Текущая цель (L2):",
                 font=("Arial", 8), bg=P["panel"], fg=P["text2"]
                 ).pack(side="left", padx=8)
        self._hrl_goal_var = tk.StringVar(value="—")
        tk.Label(ind_frame, textvariable=self._hrl_goal_var,
                 font=("Arial", 9, "bold"), bg=P["panel"], fg=P["warn"],
                 width=22
                 ).pack(side="left", padx=4)

        # ── Таблица сравнения ─────────────────────────────────────────────
        tbl_frame = tk.LabelFrame(inner, text=" Результаты сравнения агентов ",
                                  font=("Arial", 8, "bold"),
                                  bg=P["panel"], fg=P["text2"], bd=1)
        tbl_frame.pack(fill="x", padx=6, pady=4)
        self._hrl_table_text = tk.Text(
            tbl_frame, height=10, bg=P["canvas"], fg=P["text"],
            font=("Consolas", 8), state="disabled", wrap="none",
            bd=0, padx=6, pady=4)
        self._hrl_table_text.pack(fill="x", padx=4, pady=4)
        for tag, col in [("header", P["hi"]), ("sig", P["success"]),
                          ("nosig", P["text2"]), ("good", P["success"]),
                          ("bad", P["danger"])]:
            self._hrl_table_text.tag_config(tag, foreground=col)

        # ── Графики обучения ──────────────────────────────────────────────
        chart_frame = tk.LabelFrame(inner, text=" Кривые обучения (сглаженные) ",
                                    font=("Arial", 8, "bold"),
                                    bg=P["panel"], fg=P["text2"], bd=1)
        chart_frame.pack(fill="both", expand=True, padx=6, pady=4)

        self._hrl_fig = Figure(figsize=(8, 3.2), facecolor=P["panel"])
        self._hrl_canvas_widget = FigureCanvasTkAgg(self._hrl_fig, chart_frame)
        self._hrl_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        self._hrl_draw_empty_charts()

    # ── HRL: вспомогательные методы GUI ──────────────────────────────────────

    def _hrl_draw_empty_charts(self):
        """Нарисовать пустые оси графиков обучения."""
        fig = self._hrl_fig
        fig.clear()
        axes = fig.subplots(1, 3)
        titles = ["Flat: кривая обучения", "Иерарх.: кривая обучения",
                  "Сравнение метрик"]
        for ax, title in zip(axes, titles):
            ax.set_facecolor(P["canvas"])
            ax.set_title(title, color=P["text2"], fontsize=7)
            ax.tick_params(colors=P["text2"], labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(P["grid"])
            ax.set_xlabel("Эпизод", color=P["text2"], fontsize=6)
            ax.set_ylabel("Награда", color=P["text2"], fontsize=6)
            ax.text(0.5, 0.5, "Нет данных", transform=ax.transAxes,
                    ha="center", va="center", color=P["text2"], fontsize=8)
        fig.tight_layout(pad=1.0)
        self._hrl_canvas_widget.draw()

    def _hrl_get_config(self):
        """Собрать HRLConfig из полей GUI."""
        try:
            from hrl_sim import HRLConfig, CurriculumPhase
        except ImportError:
            from .hrl_sim import HRLConfig, CurriculumPhase

        v = self._hrl_vars
        cfg = HRLConfig(
            k3=int(v["hrl_k3"].get()),
            k2=int(v["hrl_k2"].get()),
            lambda_intrinsic=float(v["hrl_lambda"].get()),
            soft_masking=bool(int(v["hrl_soft"].get())),
            alpha_l3=float(v["hrl_al3"].get()),
            alpha_l2=float(v["hrl_al2"].get()),
            alpha_l1=float(v["hrl_al1"].get()),
            gamma_l3=float(v["hrl_gl3"].get()),
            gamma_l2=float(v["hrl_gl2"].get()),
            gamma_l1=float(v["hrl_gl1"].get()),
            epsilon_start=float(v["hrl_eps"].get()),
            epsilon_decay=float(v["hrl_edc"].get()),
            n_eval_episodes=int(v["hrl_neval"].get()),
            curriculum=[
                CurriculumPhase(int(v["hrl_cur1"].get()), ["serp"],
                                "Базовая тактика (ранг 2)"),
                CurriculumPhase(int(v["hrl_cur2"].get()), ["serp","tuapse"],
                                "Обобщение (ранг 2+4)"),
                CurriculumPhase(int(v["hrl_cur3"].get()), ["tuapse"],
                                "Сложный сценарий (ранг 4)"),
            ],
        )
        # Приоры целей (если заданы)
        prior = {}
        for gidx, var in self._hrl_prior_vars.items():
            try:
                w = float(var.get())
                if w > 0:
                    prior[gidx] = w
            except ValueError:
                pass
        if prior:
            cfg.goal_prior = prior
        return cfg

    def _hrl_train_flat(self):
        """Обучить flat Q-learning агента с curriculum (в фоновом потоке)."""
        import threading
        self._hrl_stop[0] = False
        v = self._hrl_vars
        total_ep = int(v["hrl_flat_ep"].get())

        def _run():
            from tank_fire_sim import TankFireSim, SCENARIOS
            rng = __import__("random").Random(42)
            self._hrl_flat_sim = TankFireSim(seed=42, training=True,
                                             scenario="tuapse")
            # Flat обучение: curriculum вручную
            phases = [
                (max(1, total_ep//5),   ["serp"]),
                (max(1, total_ep*3//10),["serp","tuapse"]),
                (max(1, total_ep//2),   ["tuapse"]),
            ]
            done_ep = 0
            total_flat = sum(p[0] for p in phases)
            for ep_count, scenarios in phases:
                for _ in range(ep_count):
                    if self._hrl_stop[0]:
                        break
                    scen = rng.choice(scenarios)
                    self._hrl_flat_sim.scenario = scen
                    self._hrl_flat_sim._cfg     = SCENARIOS[scen]
                    self._hrl_flat_sim.reset()
                    steps = 0
                    while (not self._hrl_flat_sim.extinguished
                           and self._hrl_flat_sim.t < self._hrl_flat_sim._cfg["total_min"]
                           and steps < 2000):
                        self._hrl_flat_sim.step()
                        steps += 1
                    done_ep += 1
                    pct = done_ep / total_flat * 100
                    self.after(0, lambda p=pct, e=done_ep:
                               self._hrl_update_progress(
                                   p, f"Flat: эп. {e}/{total_flat}",
                                   f"Flat Q-learning: ε={self._hrl_flat_sim.agent.epsilon:.3f}"))
            self._hrl_flat_trained = True
            self.after(0, lambda: self._hrl_status_var.set(
                "Flat агент обучен ✓   Запустите иерарх. или нажмите «Сравнить»"))

        threading.Thread(target=_run, daemon=True).start()

    def _hrl_train_hier(self):
        """Обучить иерархического агента с curriculum (в фоновом потоке)."""
        import threading
        self._hrl_stop[0] = False
        cfg = self._hrl_get_config()
        total_ep = cfg.total_train_episodes

        def _run():
            try:
                from hrl_sim import HierarchicalTankFireSim
            except ImportError:
                from .hrl_sim import HierarchicalTankFireSim
            self._hrl_sim = HierarchicalTankFireSim(
                cfg=cfg, seed=42, scenario="tuapse")

            def progress_cb(ep, total, phase_label, result):
                pct = ep / total * 100
                status = (f"Иерарх. RL: эп. {ep}/{total}  "
                          f"ε={self._hrl_sim.l1.epsilon:.3f}  "
                          f"{'✓' if result['extinguished'] else '✗'}")
                self.after(0, lambda p=pct, s=status, ph=phase_label:
                           self._hrl_update_progress(p, ph, s))

            self._hrl_sim.train_curriculum(
                progress_cb=progress_cb,
                stop_flag=self._hrl_stop)

            self._hrl_trained = True
            rewards_l1 = self._hrl_sim.l1.episode_rewards
            self.after(0, lambda: [
                self._hrl_status_var.set(
                    f"Иерарх. агент обучен ✓  L1: {len(rewards_l1)} эп.  "
                    f"Покрытие: L1={self._hrl_sim.l1.coverage():.0%} "
                    f"L2={self._hrl_sim.l2.coverage():.0%} "
                    f"L3={self._hrl_sim.l3.coverage():.0%}"),
                self._hrl_draw_learning_curves(),
            ])

        threading.Thread(target=_run, daemon=True).start()

    def _hrl_stop_training(self):
        self._hrl_stop[0] = True
        self._hrl_status_var.set("Обучение остановлено.")

    def _hrl_update_progress(self, pct: float, phase: str, status: str):
        self._hrl_progress["value"] = pct
        self._hrl_phase_var.set(phase)
        self._hrl_status_var.set(status)

    def _hrl_compare(self):
        """Запустить сравнительный эксперимент (в фоновом потоке)."""
        import threading
        if not self._hrl_flat_trained or not self._hrl_trained:
            self._hrl_status_var.set(
                "⚠ Сначала обучите оба агента (Flat и Иерархический)")
            return

        n_eval = int(self._hrl_vars["hrl_neval"].get())
        self._hrl_status_var.set(f"Запуск {n_eval} оценочных прогонов…")

        def _run():
            try:
                from hrl_metrics import run_full_comparison
            except ImportError:
                from .hrl_metrics import run_full_comparison
            result = run_full_comparison(
                self._hrl_flat_sim,
                self._hrl_sim,
                n_eval=n_eval,
                alpha=0.05,
            )
            self._hrl_comparison = result
            self.after(0, lambda: self._hrl_show_results(result))

        threading.Thread(target=_run, daemon=True).start()

    def _hrl_show_results(self, result):
        """Отобразить таблицу результатов и обновить графики."""
        table = result.summary_table()
        self._hrl_table_text.config(state="normal")
        self._hrl_table_text.delete("1.0", "end")
        # Заголовок
        header_line = "  ".join(f"{c:<28}" for c in table[0]) + "\n"
        self._hrl_table_text.insert("end", header_line, "header")
        self._hrl_table_text.insert("end", "─" * 120 + "\n", "header")
        for row in table[1:]:
            sig = row[-1] == "✓"
            delta_val = row[3]
            is_positive = delta_val.startswith("+")
            line = "  ".join(f"{c:<28}" for c in row) + "\n"
            tag = "sig" if sig else "nosig"
            self._hrl_table_text.insert("end", line, tag)
        self._hrl_table_text.config(state="disabled")
        self._hrl_status_var.set(
            f"Сравнение завершено. Значимых различий: "
            f"{sum(result.significant.values())}/{len(result.significant)}")
        self._hrl_draw_comparison_charts(result)

    def _hrl_draw_learning_curves(self):
        """Нарисовать кривые обучения flat и иерархического агентов."""
        try:
            from hrl_metrics import smooth_rewards, convergence_episode
        except ImportError:
            from .hrl_metrics import smooth_rewards, convergence_episode

        fig = self._hrl_fig
        fig.clear()
        axes = fig.subplots(1, 3)

        colors = {"flat": P["info"], "hier": P["warn"]}

        # Flat: кривая обучения
        ax0 = axes[0]
        ax0.set_facecolor(P["canvas"])
        if hasattr(self, "_hrl_flat_sim") and self._hrl_flat_sim:
            rw = self._hrl_flat_sim.agent.episode_rewards
            if rw:
                sm = smooth_rewards(rw, window=20)
                x  = range(len(sm))
                ax0.plot(x, sm, color=colors["flat"], linewidth=1.2)
                ax0.fill_between(x, sm, alpha=0.15, color=colors["flat"])
        ax0.set_title("Flat Q-learning", color=P["text2"], fontsize=7)
        ax0.set_xlabel("Эпизод", color=P["text2"], fontsize=6)
        ax0.set_ylabel("Награда (сглаж.)", color=P["text2"], fontsize=6)
        ax0.tick_params(colors=P["text2"], labelsize=6)
        for s in ax0.spines.values(): s.set_edgecolor(P["grid"])

        # Hierarchical: кривые всех 3 уровней
        ax1 = axes[1]
        ax1.set_facecolor(P["canvas"])
        if self._hrl_trained and self._hrl_sim:
            for agent, label, color in [
                (self._hrl_sim.l1, "L1 (НБТП)", P["warn"]),
                (self._hrl_sim.l2, "L2 (РТП)", P["success"]),
                (self._hrl_sim.l3, "L3 (НГ)", P["danger"]),
            ]:
                rw = agent.episode_rewards
                if rw:
                    sm = smooth_rewards(rw, window=max(5, len(rw)//50))
                    ax1.plot(range(len(sm)), sm, label=label,
                             color=color, linewidth=1.0)
            ax1.legend(fontsize=6, facecolor=P["panel"], labelcolor=P["text"])
        ax1.set_title("Иерархический RL (3 уровня)", color=P["text2"], fontsize=7)
        ax1.set_xlabel("Эпизод", color=P["text2"], fontsize=6)
        ax1.set_ylabel("Награда (сглаж.)", color=P["text2"], fontsize=6)
        ax1.tick_params(colors=P["text2"], labelsize=6)
        for s in ax1.spines.values(): s.set_edgecolor(P["grid"])

        # Покрытие Q-таблицы
        ax2 = axes[2]
        ax2.set_facecolor(P["canvas"])
        if self._hrl_trained and self._hrl_sim:
            levels  = ["L3 (32×3)", "L2 (64×5)", "L1 (256×15)"]
            covs    = [self._hrl_sim.l3.coverage(),
                       self._hrl_sim.l2.coverage(),
                       self._hrl_sim.l1.coverage()]
            bar_c   = [P["danger"], P["warn"], P["info"]]
            bars    = ax2.bar(levels, [c*100 for c in covs],
                              color=bar_c, alpha=0.8)
            for bar, cov in zip(bars, covs):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 1,
                         f"{cov:.0%}", ha="center", fontsize=7,
                         color=P["text"])
            ax2.axhline(y=80, color=P["success"], linestyle="--",
                        linewidth=0.8, alpha=0.7)
            ax2.set_ylim(0, 105)
        ax2.set_title("Покрытие Q-таблиц (%)", color=P["text2"], fontsize=7)
        ax2.set_ylabel("%", color=P["text2"], fontsize=6)
        ax2.tick_params(colors=P["text2"], labelsize=6)
        for s in ax2.spines.values(): s.set_edgecolor(P["grid"])

        fig.tight_layout(pad=1.0)
        self._hrl_canvas_widget.draw()

    def _hrl_draw_comparison_charts(self, result):
        """Нарисовать сравнительные графики после эксперимента."""
        try:
            from hrl_metrics import EVAL_METRICS, METRIC_LABELS
        except ImportError:
            from .hrl_metrics import EVAL_METRICS, METRIC_LABELS

        fig = self._hrl_fig
        fig.clear()
        axes = fig.subplots(1, 3)

        metrics_show = ["success_rate", "total_reward", "fire_area_reduction"]

        for ax, metric in zip(axes, metrics_show):
            ax.set_facecolor(P["canvas"])
            label = METRIC_LABELS.get(metric, metric)

            fm = result.flat.mean.get(metric, 0)
            hm = result.hier.mean.get(metric, 0)
            fci = result.flat.ci95.get(metric, (fm, fm))
            hci = result.hier.ci95.get(metric, (hm, hm))

            x = [0, 1]
            y = [fm, hm]
            yerr_low  = [fm - fci[0], hm - hci[0]]
            yerr_high = [fci[1] - fm, hci[1] - hm]
            colors = [P["info"], P["warn"]]
            bars = ax.bar(x, y, color=colors, alpha=0.8, width=0.4)
            ax.errorbar(x, y,
                        yerr=[yerr_low, yerr_high],
                        fmt="none", color=P["text"], capsize=4, linewidth=1.2)

            # Метка значимости
            sig = result.significant.get(metric, False)
            d   = result.cohens_d.get(metric, 0.0)
            p   = result.mannwhitney_p.get(metric, 1.0)
            top = max(fci[1], hci[1]) * 1.05
            sig_text = f"{'*' if sig else 'ns'}  d={d:+.2f}  p={p:.3f}"
            ax.text(0.5, top, sig_text, ha="center", fontsize=6,
                    color=P["success"] if sig else P["text2"],
                    transform=ax.get_xaxis_transform())

            ax.set_xticks(x)
            ax.set_xticklabels(["Flat", "Hier"], fontsize=7, color=P["text"])
            ax.set_title(label[:25], color=P["text2"], fontsize=7)
            ax.tick_params(colors=P["text2"], labelsize=6)
            for s in ax.spines.values(): s.set_edgecolor(P["grid"])

        fig.tight_layout(pad=1.0)
        self._hrl_canvas_widget.draw()

    def _hrl_save_qtables(self):
        """Сохранить Q-таблицы иерархического агента."""
        if not self._hrl_trained or not self._hrl_sim:
            self._hrl_status_var.set("⚠ Нет обученного иерарх. агента")
            return
        import os
        path = os.path.join(os.path.dirname(__file__), "hrl_qtables.npz")
        self._hrl_sim.save_qtables(path)
        self._hrl_status_var.set(f"Q-таблицы сохранены → {path}")

    # ── Tab: Массовое моделирование ───────────────────────────────────────────
    def _build_batch_tab(self, parent):
        """Вкладка массового моделирования для научного сравнения агентов."""
        import threading

        self._batch_stop = [False]
        self._batch_results = {}
        self._batch_path = ""

        # ── Заголовок ────────────────────────────────────────────────────────
        hdr = tk.Frame(parent, bg=P["panel2"])
        hdr.pack(fill="x", padx=4, pady=(4, 0))
        tk.Label(hdr, text="🔬  МАССОВОЕ МОДЕЛИРОВАНИЕ — Научное сравнение агентов",
                 font=("Arial", 10, "bold"), bg=P["panel2"], fg=P["hi"]
                 ).pack(side="left", padx=8, pady=5)

        # ── Скроллируемое тело ───────────────────────────────────────────────
        outer = tk.Frame(parent, bg=P["bg"])
        outer.pack(fill="both", expand=True)
        cs = tk.Canvas(outer, bg=P["bg"], highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=cs.yview)
        cs.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        cs.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cs, bg=P["bg"])
        cs.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: cs.configure(scrollregion=cs.bbox("all")))

        def section(title):
            f = tk.LabelFrame(inner, text=f"  {title}  ", bg=P["panel"],
                              fg=P["accent"], font=("Arial", 9, "bold"),
                              bd=1, relief="groove")
            f.pack(fill="x", padx=10, pady=6, ipadx=6, ipady=4)
            return f

        # ── Параметры эксперимента ────────────────────────────────────────────
        f_cfg = section("Параметры эксперимента")
        row1 = tk.Frame(f_cfg, bg=P["panel"])
        row1.pack(fill="x", padx=8, pady=4)

        tk.Label(row1, text="Число прогонов на агента:", bg=P["panel"], fg=P["text"],
                 font=("Arial", 8)).pack(side="left")
        self._batch_n_var = tk.StringVar(value="200")
        tk.Entry(row1, textvariable=self._batch_n_var, width=6,
                 bg=P["canvas"], fg=P["text"], relief="flat").pack(side="left", padx=(4, 20))

        tk.Label(row1, text="Сценарии:", bg=P["panel"], fg=P["text"],
                 font=("Arial", 8)).pack(side="left")
        self._batch_serp_var  = tk.BooleanVar(value=True)
        self._batch_tuapse_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row1, text="Ранг 2", variable=self._batch_serp_var,
                       bg=P["panel"], fg=P["text"], selectcolor=P["panel2"],
                       font=("Arial", 8)).pack(side="left", padx=4)
        tk.Checkbutton(row1, text="Ранг 4", variable=self._batch_tuapse_var,
                       bg=P["panel"], fg=P["text"], selectcolor=P["panel2"],
                       font=("Arial", 8)).pack(side="left", padx=4)

        row2 = tk.Frame(f_cfg, bg=P["panel"])
        row2.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(row2, text="Агенты:", bg=P["panel"], fg=P["text"],
                 font=("Arial", 8)).pack(side="left")
        self._batch_flat_var = tk.BooleanVar(value=True)
        self._batch_hier_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row2, text="Flat Q-learning", variable=self._batch_flat_var,
                       bg=P["panel"], fg=P["text"], selectcolor=P["panel2"],
                       font=("Arial", 8)).pack(side="left", padx=4)
        tk.Checkbutton(row2, text="Иерархический RL", variable=self._batch_hier_var,
                       bg=P["panel"], fg=P["text"], selectcolor=P["panel2"],
                       font=("Arial", 8)).pack(side="left", padx=4)

        tk.Label(row2, text="  ΔЗерно (seed offset):", bg=P["panel"], fg=P["text"],
                 font=("Arial", 8)).pack(side="left", padx=(16, 0))
        self._batch_seed_var = tk.StringVar(value="100")
        tk.Entry(row2, textvariable=self._batch_seed_var, width=6,
                 bg=P["canvas"], fg=P["text"], relief="flat").pack(side="left", padx=4)

        # ── Управление ────────────────────────────────────────────────────────
        f_ctrl = section("Управление")
        ctrl_row = tk.Frame(f_ctrl, bg=P["panel"])
        ctrl_row.pack(padx=8, pady=6, anchor="w")
        tk.Button(ctrl_row, text="▶  Запустить эксперимент",
                  command=self._batch_run,
                  bg=P["success"], fg="#fff", font=("Arial", 9, "bold"),
                  relief="flat", padx=12, pady=5).pack(side="left", padx=(0, 6))
        tk.Button(ctrl_row, text="⏹  Стоп",
                  command=lambda: self._batch_stop.__setitem__(0, True),
                  bg=P["warn"], fg="#fff", font=("Arial", 9),
                  relief="flat", padx=8, pady=5).pack(side="left", padx=(0, 12))
        self._batch_export_btn = tk.Button(ctrl_row, text="📄  Экспорт отчёта",
                  command=self._batch_export,
                  bg=P["info"], fg="#fff", font=("Arial", 9),
                  relief="flat", padx=8, pady=5, state="disabled")
        self._batch_export_btn.pack(side="left")
        self._batch_open_btn = tk.Button(ctrl_row, text="📂  Открыть",
                  command=lambda: self._open_file(self._batch_path),
                  bg=P["info"], fg="#fff", font=("Arial", 9),
                  relief="flat", padx=8, pady=5, state="disabled")
        self._batch_open_btn.pack(side="left", padx=4)

        self._batch_status_var = tk.StringVar(value="Готов к эксперименту")
        tk.Label(f_ctrl, textvariable=self._batch_status_var,
                 bg=P["panel"], fg=P["text2"], font=("Arial", 8)
                 ).pack(anchor="w", padx=8)
        self._batch_progress = ttk.Progressbar(f_ctrl, length=400, mode="determinate",
                                               maximum=100)
        self._batch_progress.pack(fill="x", padx=8, pady=(2, 6))

        # ── Таблица результатов ───────────────────────────────────────────────
        f_table = section("Сравнительная таблица результатов (Bootstrap 95% CI)")
        tf = tk.Frame(f_table, bg=P["canvas"])
        tf.pack(fill="both", expand=True, padx=4, pady=4)
        sb_t = tk.Scrollbar(tf)
        sb_t.pack(side="right", fill="y")
        self._batch_table_text = tk.Text(
            tf, yscrollcommand=sb_t.set, bg=P["canvas"],
            fg=P["text"], font=("Consolas", 8), state="disabled",
            height=10, bd=0, padx=6, pady=4,
        )
        self._batch_table_text.pack(fill="both", expand=True)
        sb_t.config(command=self._batch_table_text.yview)
        for tag, fg_c in [("header", P["hi"]), ("sig", P["success"]), ("nosig", P["text2"])]:
            self._batch_table_text.tag_config(tag, foreground=fg_c)
        self._batch_table_text.tag_config("header", font=("Consolas", 8, "bold"))

        # ── Графики ───────────────────────────────────────────────────────────
        f_charts = section("Графики сравнения")
        self._batch_fig = Figure(figsize=(9, 5), facecolor=P["bg"])
        self._batch_fc  = FigureCanvasTkAgg(self._batch_fig, master=f_charts)
        self._batch_fc.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        # ── Научная новизна ───────────────────────────────────────────────────
        f_nov = section("Научная новизна и интерпретация")
        self._novelty_text = tk.Text(f_nov, bg=P["canvas"], fg=P["text"],
                                     font=("Arial", 8), height=10, state="disabled",
                                     wrap="word", bd=0, padx=8, pady=6)
        self._novelty_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._novelty_text.tag_config("bold", font=("Arial", 8, "bold"), foreground=P["hi"])
        # Статичный текст о научной новизне
        novelty_intro = (
            "НАУЧНАЯ НОВИЗНА РАБОТЫ\n\n"
            "1. ИЕРАРХИЧЕСКАЯ RL-АРХИТЕКТУРА ДЛЯ ПОЖАРОТУШЕНИЯ\n"
            "Впервые предложена 3-уровневая иерархическая Q-сеть для управления тушением пожара РВС: "
            "L3 (НГ/ГУ МЧС) задаёт стратегический режим; L2 (РТП/НШ) формирует макро-цели; "
            "L1 (НБТП) выбирает примитивные действия. Структура соответствует реальной вертикали "
            "управления ICS (Incident Command System) согласно ГОСТ Р 22.7.01-2021.\n\n"
            "2. CURRICULUM LEARNING ДЛЯ ПОЖАРНОЙ ТАКТИКИ\n"
            "Разработана программа постепенного обучения: сначала агент осваивает простые сценарии "
            "(ранг пожара 2, V=2000 м³), затем переходит к сложным (ранг 4, V=20000 м³). "
            "Это ускоряет сходимость и улучшает обобщение по сравнению с прямым обучением.\n\n"
            "3. ИНТРИНСИЧЕСКОЕ ВОЗНАГРАЖДЕНИЕ ДЛЯ СОГЛАСОВАНИЯ ЦЕЛЕЙ\n"
            "Введён механизм r_total = r_env + λ·r_intrinsic, где r_intrinsic поощряет L1 "
            "за выбор действий, согласованных с макро-целью L2. λ настраивается через GUI.\n\n"
            "4. ФИЗИЧЕСКИ ОБОСНОВАННОЕ ПРОСТРАНСТВО СОСТОЯНИЙ\n"
            "Вектор состояния кодирует физические переменные пожара (площадь S(t), расход Q(t), "
            "индекс риска R(t)) совместно с тактическими параметрами (кол-во стволов, ПНС, БУ). "
            "Это отличает подход от абстрактных grid-world RL-бенчмарков.\n\n"
            "5. СТАТИСТИЧЕСКИ ОБОСНОВАННОЕ СРАВНЕНИЕ АГЕНТОВ\n"
            "Применяется Bootstrap 95% CI (N=10000 ресэмплов) + критерий Манна-Уитни "
            "для сравнения агентов без допущений о нормальности распределений. "
            "Размер эффекта оценивается по Cohen's d.\n\n"
            "После завершения эксперимента здесь будут показаны конкретные числа и выводы."
        )
        self._novelty_text.config(state="normal")
        self._novelty_text.insert("end", novelty_intro)
        self._novelty_text.config(state="disabled")

    def _batch_run(self):
        """Запустить массовый эксперимент в фоновом потоке."""
        import threading
        self._batch_stop[0] = False
        try:
            n = max(1, int(self._batch_n_var.get()))
        except ValueError:
            self._batch_status_var.set("⚠ Введите целое число прогонов.")
            return
        scenarios = []
        if self._batch_serp_var.get():
            scenarios.append("serp")
        if self._batch_tuapse_var.get():
            scenarios.append("tuapse")
        if not scenarios:
            self._batch_status_var.set("⚠ Выберите хотя бы один сценарий")
            return
        use_flat = self._batch_flat_var.get()
        use_hier = self._batch_hier_var.get()
        if not use_flat and not use_hier:
            self._batch_status_var.set("⚠ Выберите хотя бы один тип агента")
            return
        try:
            seed_offset = int(self._batch_seed_var.get())
        except ValueError:
            seed_offset = 0
        self._batch_status_var.set(f"Запуск {n} прогонов × {'2' if use_flat and use_hier else '1'} агента…")
        self._batch_progress["value"] = 0

        def _run():
            import random
            rng = random.Random(seed_offset)
            results = {}

            total_runs = n * (int(use_flat) + int(use_hier))
            done = [0]

            def run_episodes(agent_type):
                """Run N evaluation episodes for agent_type ('flat' or 'hier')."""
                eps_list = []
                for i in range(n):
                    if self._batch_stop[0]:
                        break
                    seed = seed_offset + i
                    scen = rng.choice(scenarios)
                    try:
                        ep = self._batch_single_episode(agent_type, scen, seed)
                        eps_list.append(ep)
                    except Exception as ex:
                        eps_list.append({"error": str(ex)})
                    done[0] += 1
                    pct = done[0] / total_runs * 100
                    if done[0] % max(1, total_runs // 100) == 0:
                        self.after(0, lambda p=pct, d=done[0]: [
                            self._batch_progress.configure(value=p),
                            self._batch_status_var.set(
                                f"Прогресс: {d}/{total_runs} прогонов…"),
                        ])
                return eps_list

            if use_flat:
                results["flat"] = run_episodes("flat")
            if use_hier:
                results["hier"] = run_episodes("hier")

            self._batch_results = results
            self.after(0, lambda: self._batch_show_results(results))

        threading.Thread(target=_run, daemon=True).start()

    def _batch_single_episode(self, agent_type: str, scenario: str, seed: int) -> dict:
        """Провести один оценочный прогон и вернуть словарь метрик."""
        sim = TankFireSim(seed=seed, training=False, scenario=scenario)
        sim.reset()

        if agent_type == "flat":
            # Копируем Q-таблицу обученного агента если есть
            if hasattr(self, "_hrl_flat_sim") and self._hrl_flat_sim is not None:
                src = self._hrl_flat_sim.agent
                sim.agent.Q = src.Q.copy()
                sim.agent.epsilon = 0.0   # greedy evaluation

        elif agent_type == "hier":
            if not (hasattr(self, "_hrl_sim") and self._hrl_sim is not None):
                raise RuntimeError("Иерархический агент не обучен")
            # Для иерархического — запускаем через hrl_sim
            try:
                from hrl_sim import HierarchicalTankFireSim
            except ImportError:
                from .hrl_sim import HierarchicalTankFireSim
            hier = self._hrl_sim
            hier_ep = hier.run_episode(training=False)
            return {
                "success":    hier_ep.get("extinguished", False),
                "duration":   hier_ep.get("duration_min", hier.env.t if hasattr(hier, "env") else 0),
                "fire_area":  hier_ep.get("fire_area_final", 0.0),
                "reward":     hier_ep.get("total_reward", 0.0),
                "scenario":   scenario,
                "agent":      "hier",
            }

        steps = 0
        cfg = sim._cfg
        while not sim.extinguished and sim.t < cfg["total_min"] and steps < 3000:
            sim.step()
            steps += 1

        reward_total = float(sum(sim.h_reward)) if sim.h_reward else 0.0
        return {
            "success":   sim.extinguished,
            "duration":  sim.t,
            "fire_area": float(sim.fire_area),
            "reward":    reward_total,
            "scenario":  scenario,
            "agent":     agent_type,
        }

    def _batch_show_results(self, results: dict):
        """Показать таблицу и графики сравнения."""
        import numpy as np

        def bootstrap_ci(data, stat=np.mean, n=2000, alpha=0.05):
            if len(data) < 2:
                return stat(data), stat(data), stat(data)
            boots = [stat(np.random.choice(data, len(data), replace=True)) for _ in range(n)]
            lo, hi = np.percentile(boots, [alpha/2*100, (1-alpha/2)*100])
            return stat(data), lo, hi

        metrics_spec = [
            ("Доля тушений P(ext)", "success",   True,  "{:.3f}"),
            ("Длительность, мин",    "duration",  False, "{:.1f}"),
            ("Финальная пл-дь, м²",  "fire_area", False, "{:.1f}"),
            ("Σ Награда агента",      "reward",    False, "{:.1f}"),
        ]

        lines = []
        sep   = "─" * 100
        hdr   = f"{'Метрика':<30} {'Flat Q: M [95%CI]':<28} {'Иерарх: M [95%CI]':<28} {'Δ%':>8}  p-value  Знч?"
        lines.append(("header", hdr))
        lines.append(("header", sep))

        for label, key, higher_is_better, fmt in metrics_spec:
            flat_data  = [ep[key] for ep in results.get("flat",  []) if key in ep and "error" not in ep]
            hier_data  = [ep[key] for ep in results.get("hier",  []) if key in ep and "error" not in ep]
            flat_raw   = [float(v) for v in flat_data]
            hier_raw   = [float(v) for v in hier_data]

            def fmt_ci(data):
                if not data:
                    return "—"
                m, lo, hi = bootstrap_ci(np.array(data))
                return f"{fmt.format(m)} [{fmt.format(lo)};{fmt.format(hi)}]"

            flat_str = fmt_ci(flat_raw)
            hier_str = fmt_ci(hier_raw)

            delta_str = "—"
            p_str     = "—"
            sig       = False
            if flat_raw and hier_raw:
                m_flat = np.mean(flat_raw)
                m_hier = np.mean(hier_raw)
                if m_flat != 0:
                    delta = (m_hier - m_flat) / abs(m_flat) * 100
                    delta_str = f"{delta:+.1f}%"
                # Mann-Whitney U test (manual if scipy not available)
                try:
                    from scipy.stats import mannwhitneyu
                    _, p = mannwhitneyu(flat_raw, hier_raw, alternative="two-sided")
                except ImportError:
                    # Approximate via normal distribution of U
                    n1, n2 = len(flat_raw), len(hier_raw)
                    U = sum(1 for x in flat_raw for y in hier_raw if x < y) + \
                        0.5 * sum(1 for x in flat_raw for y in hier_raw if x == y)
                    mu_U = n1 * n2 / 2
                    sigma_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                    z = (U - mu_U) / (sigma_U + 1e-9)
                    import math as _math
                    p = 2 * (1 - 0.5 * (1 + _math.erf(abs(z) / _math.sqrt(2))))
                sig = p < 0.05
                p_str = f"{p:.4f}"

            tag = "sig" if sig else "nosig"
            line = f"{label:<30} {flat_str:<28} {hier_str:<28} {delta_str:>8}  {p_str:<8} {'✓' if sig else ''}"
            lines.append((tag, line))

        # Display table
        t = self._batch_table_text
        t.config(state="normal")
        t.delete("1.0", "end")
        for tag, line in lines:
            t.insert("end", line + "\n", tag)
        t.config(state="disabled")

        # Draw charts
        self._batch_draw_charts(results)

        # Update novelty text with results
        self._batch_update_novelty(results)

        # Enable export
        self._batch_export_btn.config(state="normal")
        n_total = sum(len(v) for v in results.values())
        self._batch_status_var.set(f"Завершено: {n_total} прогонов. Результаты готовы.")

    def _batch_draw_charts(self, results: dict):
        """Нарисовать сравнительные графики по результатам массового эксперимента."""
        import numpy as np

        fig = self._batch_fig
        fig.clear()
        fig.patch.set_facecolor(P["bg"])

        axes = fig.subplots(2, 3)
        fig.subplots_adjust(hspace=0.5, wspace=0.4, left=0.08, right=0.97,
                            top=0.92, bottom=0.1)

        colors_map = {"flat": P["info"], "hier": P["warn"]}
        labels_map = {"flat": "Flat Q", "hier": "Иерарх."}

        def style(ax, title):
            ax.set_facecolor(P["canvas"])
            ax.set_title(title, color=P["text"], fontsize=7, pad=3)
            ax.tick_params(colors=P["text2"], labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor(P["grid"])
            ax.grid(True, color=P["grid"], linewidth=0.3, alpha=0.7)

        # 1. Доля тушений (bar)
        ax = axes[0, 0]
        style(ax, "Доля тушений P(ext)")
        for i, (key, data) in enumerate(results.items()):
            succ = [ep["success"] for ep in data if "error" not in ep]
            if succ:
                rate = np.mean(succ)
                n    = len(succ)
                ci   = 1.96 * np.sqrt(rate*(1-rate)/n)
                ax.bar(i, rate, color=colors_map[key], alpha=0.85,
                       label=labels_map[key], width=0.5)
                ax.errorbar(i, rate, yerr=ci, color=P["text"], capsize=4)
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([labels_map.get(k, k) for k in results], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, facecolor=P["panel"])

        # 2. Длительность (boxplot/violin)
        ax = axes[0, 1]
        style(ax, "Длительность прогона, мин")
        box_data = []
        box_labels = []
        box_colors = []
        for key, data in results.items():
            dur = [ep["duration"] for ep in data if "error" not in ep and "duration" in ep]
            if dur:
                box_data.append(dur)
                box_labels.append(labels_map[key])
                box_colors.append(colors_map[key])
        if box_data:
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                            medianprops=dict(color=P["hi"], linewidth=1.5))
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        # 3. Суммарная награда (boxplot)
        ax = axes[0, 2]
        style(ax, "Σ Награда агента")
        rw_data = []
        rw_labels = []
        rw_colors = []
        for key, data in results.items():
            rw = [ep["reward"] for ep in data if "error" not in ep and "reward" in ep]
            if rw:
                rw_data.append(rw)
                rw_labels.append(labels_map[key])
                rw_colors.append(colors_map[key])
        if rw_data:
            bp2 = ax.boxplot(rw_data, tick_labels=rw_labels, patch_artist=True,
                             medianprops=dict(color=P["hi"], linewidth=1.5))
            for patch, color in zip(bp2["boxes"], rw_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        # 4. Финальная площадь пожара (hist)
        ax = axes[1, 0]
        style(ax, "Финальная площадь пожара, м²")
        for key, data in results.items():
            fa = [ep["fire_area"] for ep in data if "error" not in ep and "fire_area" in ep]
            if fa:
                ax.hist(fa, bins=20, alpha=0.6, color=colors_map[key],
                        label=labels_map[key], density=True)
        ax.legend(fontsize=6, facecolor=P["panel"])

        # 5. Кривая успешности по эпизодам (cumulative success rate)
        ax = axes[1, 1]
        style(ax, "Кумулятивная доля тушений")
        for key, data in results.items():
            succ = [float(ep.get("success", False)) for ep in data if "error" not in ep]
            if succ:
                cumrate = np.cumsum(succ) / (np.arange(len(succ)) + 1)
                ax.plot(cumrate, color=colors_map[key], linewidth=1.2,
                        label=labels_map[key])
        ax.set_xlabel("Эпизод", fontsize=6, color=P["text2"])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, facecolor=P["panel"])

        # 6. Сравнение по сценариям (grouped bars)
        ax = axes[1, 2]
        style(ax, "P(ext) по сценариям")
        scen_map = {"serp": "Ранг 2", "tuapse": "Ранг 4"}
        x_pos = 0
        tick_pos = []
        tick_labels = []
        for scen_key, scen_label in scen_map.items():
            for i, (key, data) in enumerate(results.items()):
                succ = [ep["success"] for ep in data
                        if "error" not in ep and ep.get("scenario") == scen_key]
                if succ:
                    rate = np.mean(succ)
                    n    = len(succ)
                    ci   = 1.96 * np.sqrt(rate*(1-rate)/max(n, 1))
                    ax.bar(x_pos, rate, color=colors_map[key], alpha=0.8,
                           width=0.7, label=labels_map[key] if x_pos < 2 else "")
                    ax.errorbar(x_pos, rate, yerr=ci, color=P["text"], capsize=3)
                    tick_pos.append(x_pos)
                    tick_labels.append(f"{labels_map[key]}\n{scen_label}")
                    x_pos += 1
            x_pos += 0.5
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=5)
        ax.set_ylim(0, 1.05)

        self._batch_fc.draw()

    def _batch_update_novelty(self, results: dict):
        """Добавить конкретные числа в раздел научной новизны."""
        import numpy as np
        lines = []
        for key, data in results.items():
            succ = [ep.get("success", False) for ep in data if "error" not in ep]
            dur  = [ep.get("duration", 0)    for ep in data if "error" not in ep]
            rw   = [ep.get("reward", 0)      for ep in data if "error" not in ep]
            if not succ:
                continue
            lbl = "Flat Q-learning" if key == "flat" else "Иерархический RL"
            lines.append(f"\n{'─'*50}")
            lines.append(f"  {lbl}:")
            lines.append(f"  P(ext) = {np.mean(succ):.3f}  (n={len(succ)})")
            if dur:
                lines.append(f"  Длительность: M={np.mean(dur):.1f} мин, σ={np.std(dur):.1f}")
            if rw:
                lines.append(f"  Σ Награда: M={np.mean(rw):.1f}, σ={np.std(rw):.1f}")

        t = self._novelty_text
        t.config(state="normal")
        t.insert("end", "\n\nРЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА\n" + "═"*50)
        for line in lines:
            t.insert("end", "\n" + line)
        t.config(state="disabled")

    def _batch_export(self):
        """Экспортировать результаты массового эксперимента в JSON + PDF."""
        import threading
        import json as _json
        self._batch_status_var.set("⏳ Экспорт…")

        def _run():
            try:
                out_dir  = os.path.dirname(os.path.abspath(__file__))
                ts       = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
                json_path = os.path.join(out_dir, f"batch_results_{ts}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    _json.dump(self._batch_results, f, ensure_ascii=False, indent=2)
                self._batch_path = json_path
                self.after(0, lambda: [
                    self._batch_status_var.set(f"✅ Экспортировано: {os.path.basename(json_path)}"),
                    self._batch_open_btn.config(state="normal"),
                ])
            except Exception as e:
                self.after(0, lambda ex=e: self._batch_status_var.set(f"❌ {ex}"))

        threading.Thread(target=_run, daemon=True).start()

    # ── Tab: Настройки ────────────────────────────────────────────────────────
    def _build_settings_tab(self, parent):
        canvas = tk.Canvas(parent, bg=P["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=P["bg"])
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        def section(title):
            f = tk.LabelFrame(inner, text=f"  {title}  ", bg=P["panel"],
                              fg=P["accent"], font=("Arial", 9, "bold"),
                              bd=1, relief="groove")
            f.pack(fill="x", padx=10, pady=6, ipadx=6, ipady=4)
            return f

        def slider_row(parent, label, from_, to, init, fmt="{:.0f}", step=1):
            row = tk.Frame(parent, bg=P["panel"])
            row.pack(fill="x", padx=8, pady=3)
            tk.Label(row, text=f"{label}:", bg=P["panel"], fg=P["text"],
                     font=("Arial", 8), width=28, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=init)
            val_lbl = tk.Label(row, textvariable=var, bg=P["panel"], fg=P["hi"],
                               font=("Consolas", 8), width=8)
            val_lbl.pack(side="right")
            s = tk.Scale(row, variable=var, from_=from_, to=to, orient="horizontal",
                         resolution=step, showvalue=False,
                         bg=P["panel"], fg=P["text2"], troughcolor=P["grid"],
                         highlightthickness=0, length=200)
            s.pack(side="left", padx=6)
            var.trace_add("write", lambda *_: val_lbl.config(text=fmt.format(var.get())))
            return var

        # ── Выбор сценария ────────────────────────────────────────────────────
        f_scen = section("Сценарий моделирования")
        self._scenario_var = tk.StringVar(value=self._scenario_key)
        for key, cfg in SCENARIOS.items():
            tk.Radiobutton(
                f_scen, text=cfg["name"], variable=self._scenario_var, value=key,
                bg=P["panel"], fg=P["text"], selectcolor=P["panel2"],
                font=("Arial", 8), activebackground=P["panel"],
                activeforeground=P["hi"], wraplength=520, justify="left",
                command=self._apply_settings,
            ).pack(anchor="w", padx=10, pady=2)
        # Кнопка загрузки пользовательского сценария из JSON
        btn_row = tk.Frame(f_scen, bg=P["panel"])
        btn_row.pack(anchor="w", padx=10, pady=(4, 2))
        tk.Button(btn_row, text="📂 Загрузить сценарий из JSON…",
                  command=self._load_scenario_from_json,
                  bg=P["info"], fg="white", font=("Arial", 8),
                  relief="flat", padx=8, pady=3
                  ).pack(side="left")
        self._loaded_scen_label = tk.Label(btn_row, text="", bg=P["panel"],
                                           fg=P["text2"], font=("Arial", 8))
        self._loaded_scen_label.pack(side="left", padx=6)

        # Нормативные требования (динамически по сценарию)
        self._norms_label_var = tk.StringVar(value=self._get_norms_text())
        tk.Label(f_scen, textvariable=self._norms_label_var,
                 font=("Consolas", 8), bg=P["panel"], fg=P["hi"],
                 justify="left").pack(anchor="w", padx=14, pady=(4, 2))

        # ── Параметры пожара ──────────────────────────────────────────────────
        f_fire = section("Параметры пожара")
        self._var_init_area   = slider_row(f_fire, "Начальная площадь (м²)",  500, 3000, 1250, step=50)
        self._var_spread_rate = slider_row(f_fire, "Скорость распространения (м²/мин)", 0, 20, 3, step=0.5)
        self._var_fuel_vol    = slider_row(f_fire, "Объём РВС (м³)",  1000, 50000, 20000, step=1000)

        # ── Параметры тушения ─────────────────────────────────────────────────
        f_ext = section("Параметры тушения")
        self._var_foam_eff  = slider_row(f_ext, "Эффективность пены (%)",  0, 100, 20, step=5)
        self._var_tech_rel  = slider_row(f_ext, "Надёжность техники (%)",  30, 100, 70, step=5)
        self._var_foam_conc = slider_row(f_ext, "Запас пенообразователя (т)", 1, 30, 12, step=1)

        # ── Параметры RL-агента ───────────────────────────────────────────────
        f_rl = section("Параметры RL-агента")
        self._var_epsilon   = slider_row(f_rl, "Начальный epsilon (исследование)", 0.0, 1.0, 0.90, "{:.2f}", 0.05)
        self._var_lr        = slider_row(f_rl, "Скорость обучения (α)",            0.01,0.5, 0.15, "{:.2f}", 0.01)
        self._var_gamma     = slider_row(f_rl, "Коэффициент дисконтирования (γ)",  0.8, 1.0, 0.95, "{:.2f}", 0.01)

        # ── Режим управления ──────────────────────────────────────────────────
        f_mode = section("Режим управления")
        self._mode_var = tk.StringVar(value="rl_train")
        modes = [("Обучение с подкреплением (ε-greedy)",  "rl_train"),
                 ("Эксплуатация агента (greedy)",         "rl_greedy"),
                 ("Ручное управление",        "manual")]
        for text, val in modes:
            tk.Radiobutton(f_mode, text=text, variable=self._mode_var, value=val,
                           bg=P["panel"], fg=P["text"], selectcolor=P["panel2"],
                           font=("Arial", 9), activebackground=P["panel"],
                           activeforeground=P["hi"]).pack(anchor="w", padx=10, pady=2)

        # ── Конструктор сценариев ─────────────────────────────────────────────
        f_scen_ed = section("Конструктор пользовательских сценариев")
        tk.Label(f_scen_ed,
                 text=("Создайте свой сценарий пожара: разместите объекты на карте,\n"
                       "задайте тип горючего, ранг, кровлю — и запустите симуляцию."),
                 font=("Arial", 8), bg=P["panel"], fg=P["text2"], justify="left"
                 ).pack(anchor="w", padx=10, pady=(4, 2))
        tk.Button(f_scen_ed, text="🗺  Открыть конструктор сценариев",
                  command=self._open_scenario_editor,
                  bg=P["accent"], fg="#000", font=("Arial", 10, "bold"),
                  relief="flat", padx=14, pady=6
                  ).pack(anchor="w", padx=10, pady=6)

        # Кнопки
        f_btn = section("Действия")
        btn_row = tk.Frame(f_btn, bg=P["panel"])
        btn_row.pack(pady=6)
        for text, cmd in [
            ("Применить настройки", self._apply_settings),
            ("Сбросить симуляцию",  self._on_reset),
            ("Новый эпизод",        self._new_episode),
        ]:
            tk.Button(btn_row, text=text, command=cmd, bg=P["accent"],
                      fg="#000", font=("Arial", 9, "bold"),
                      relief="flat", padx=10, pady=4).pack(side="left", padx=4)

        # Ручное действие
        f_manual = section("Ручное действие РТП")
        self._manual_var = tk.StringVar()
        codes_list = [f"[{a[0]}] {a[2]}" for a in ACTIONS]
        cb = ttk.Combobox(f_manual, textvariable=self._manual_var, values=codes_list,
                          font=("Arial", 8), state="readonly", width=52)
        cb.set(codes_list[12])   # O4 — разведка
        cb.pack(padx=10, pady=4)
        tk.Button(f_manual, text="▶  Выполнить действие",
                  command=self._do_manual_action, bg=P["success"],
                  fg="#fff", font=("Arial", 9, "bold"),
                  relief="flat", padx=10, pady=4).pack(pady=4)

        # ── Встроенный мануал ─────────────────────────────────────────────────
        f_help = section("Справочник: настройки и режимы моделирования")
        _manual_text = (
            "САУР-ПСП — симулятор тактики тушения пожара РВС с RL-агентом.\n"
            "═══════════════════════════════════════════════════════════════\n\n"
            "ВКЛАДКА «НАСТРОЙКИ»\n"
            "────────────────────────────────────────────────────────\n"
            "• Сценарий моделирования — выбор объекта:\n"
            "    Серпухов (Ранг 2): РВС-9, топливо = нефть, S₀=1250 м², плавающая крыша 70%.\n"
            "    Туапсе (Ранг 4):   РВС-20000, топливо = мазут, S₀=2000 м², плавающая крыша 70%.\n"
            "• Параметры пожара:\n"
            "    Начальная площадь (м²)          — площадь горения в момент t=0.\n"
            "    Скорость распространения (м²/мин) — линейная скорость распространения пожара.\n"
            "    Объём РВС (м³)                  — вместимость резервуара.\n"
            "• Параметры тушения:\n"
            "    Эффективность пены (%)  — вероятность успешной атаки при номинальных условиях.\n"
            "    Надёжность техники (%)  — вероятность безотказной работы на каждом шаге.\n"
            "    Запас пенообразователя (т) — расходуется 1.8 т/атаку; при исчерпании — невозможно.\n"
            "• Параметры RL-агента:\n"
            "    Начальный ε (исследование) — доля случайных действий в начале обучения.\n"
            "    Скорость обучения α         — шаг обновления Q-таблицы.\n"
            "    Коэффициент дисконтирования γ — вес будущих наград (0.8–1.0).\n"
            "• Режим управления:\n"
            "    Обучение с подкреплением (ε-greedy)  — агент исследует и обновляет Q-таблицу.\n"
            "    Эксплуатация агента (greedy)         — агент использует только лучшие действия.\n"
            "    Ручное управление       — РТП выбирает действие через список вкладки «Настройки».\n\n"
            "ПОРЯДОК МОДЕЛИРОВАНИЯ\n"
            "────────────────────────────────────────────────────────\n"
            "1. БАЗОВАЯ СИМУЛЯЦИЯ (1 эпизод)\n"
            "   a) Выберите сценарий → нажмите «Применить настройки».\n"
            "   b) Нажмите ▶ (Play) на нижней панели управления.\n"
            "   c) Наблюдайте за картой, метриками и журналом событий.\n"
            "   d) Нажмите ⏸ для паузы, ↺ для сброса эпизода.\n\n"
            "2. ОБУЧЕНИЕ FLAT Q-LEARNING АГЕНТА\n"
            "   a) Откройте вкладку «RL-агент».\n"
            "   b) Введите число эпизодов (рекомендуется 500–2000).\n"
            "   c) Нажмите «▶ Запустить обучение»; прогресс отображается полосой.\n"
            "   d) После обучения на графиках появятся Q-значения и кривая наград.\n\n"
            "3. ОБУЧЕНИЕ ИЕРАРХИЧЕСКОГО RL (3 уровня)\n"
            "   a) Откройте вкладку «Иерархический RL».\n"
            "   b) Настройте параметры k2/k3, α, γ, ε через поля конфигурации или\n"
            "      редактируйте hrl_config.json вручную.\n"
            "   c) Установите курикулум (последовательность сценариев и число эпизодов).\n"
            "   d) Нажмите «▶ Запустить HRL»; система обучает L3→L2→L1 поочерёдно.\n"
            "   e) По завершении — сравнительная таблица Flat vs HRL появится на вкладке.\n\n"
            "4. МАССОВОЕ МОДЕЛИРОВАНИЕ (научное сравнение)\n"
            "   a) Откройте вкладку «🔬 Массовое».\n"
            "   b) Выберите число запусков N (рекомендуется 100–500), сценарии и агентов.\n"
            "   c) Нажмите «Запустить пакетную симуляцию».\n"
            "   d) Результаты выводятся в таблице с 95% ДИ и p-значениями (Mann-Whitney U).\n"
            "   e) Экспортируйте CSV для использования в LaTeX/Word.\n\n"
            "5. ГЕНЕРАЦИЯ ОТЧЁТА\n"
            "   a) Откройте вкладку «Отчёт».\n"
            "   b) После проведения симуляции (и опционально обучения агентов) нажмите\n"
            "      «Сформировать PDF».\n"
            "   c) Отчёт включает: параметры сценария, хронологию событий, графики,\n"
            "      нормативный анализ (ГОСТ), сравнение Flat vs HRL и выводы.\n"
            "   d) Кнопка «📂 Открыть» активируется после генерации.\n\n"
            "ФАЗЫ ПОЖАРА (S1–S5)\n"
            "────────────────────────────────────────────────────────\n"
            "  S1 — Обнаружение/оповещение        (Ч+0 … Ч+5 мин)\n"
            "  S2 — Разведка и сосредоточение сил  (Ч+5 … Ч+20 мин)\n"
            "  S3 — Активное горение / охлаждение  (Ч+20 … Ч+60 мин)\n"
            "  S4 — Подача пены / атака            (после выполнения условий)\n"
            "  S5 — Ликвидация / контроль          (после успешной атаки)\n\n"
            "ДЕЙСТВИЯ РТП (коды)\n"
            "────────────────────────────────────────────────────────\n"
            "  S1-S4: Стратегические (уровень Стратег.) — эвакуация, оцепление, запрос АКП-50\n"
            "  O1-O6: Оперативные (Тактич./Операт.) — стволы, ПНС, пенная атака, разведка\n"
            "  Подробный список доступен в разделе «Ручное действие РТП» этой вкладки.\n"
        )
        txt_frame = tk.Frame(f_help, bg=P["panel"])
        txt_frame.pack(fill="both", expand=True, padx=8, pady=4)
        sb = tk.Scrollbar(txt_frame, orient="vertical")
        sb.pack(side="right", fill="y")
        txt = tk.Text(txt_frame, height=14, wrap="word",
                      bg=P["canvas"], fg=P["text"], font=("Consolas", 8),
                      relief="flat", yscrollcommand=sb.set,
                      selectbackground=P["accent"], selectforeground="#000")
        txt.insert("1.0", _manual_text)
        txt.config(state="disabled")
        sb.config(command=txt.yview)
        txt.pack(side="left", fill="both", expand=True)

        # ── PDF-мануал ────────────────────────────────────────────────────────
        f_man = section("PDF-мануал программы")
        tk.Label(f_man,
                 text=("Руководство пользователя: интерфейс, сценарии, действия РТП,\n"
                       "Q-learning агент, физическая модель, нормативная база."),
                 font=("Arial", 8), bg=P["panel"], fg=P["text2"], justify="left"
                 ).pack(anchor="w", padx=10, pady=(4, 2))
        self._manual_status = tk.StringVar(value="")
        self._manual_open_btn = None
        man_btns = tk.Frame(f_man, bg=P["panel"])
        man_btns.pack(anchor="w", padx=10, pady=4)
        tk.Button(man_btns, text="📖  Создать мануал (PDF)",
                  command=self._do_generate_manual,
                  bg=P["success"], fg="#fff", font=("Arial", 9, "bold"),
                  relief="flat", padx=12, pady=5).pack(side="left", padx=(0, 6))
        self._manual_open_btn = tk.Button(man_btns, text="📂  Открыть",
                  command=lambda: self._open_file(self._manual_path),
                  bg=P["info"], fg="#fff", font=("Arial", 9),
                  relief="flat", padx=8, pady=5, state="disabled")
        self._manual_open_btn.pack(side="left")
        tk.Label(f_man, textvariable=self._manual_status,
                 font=("Consolas", 8), bg=P["panel"], fg=P["success"]
                 ).pack(anchor="w", padx=10)

    def _open_scenario_editor(self):
        """Открыть конструктор сценариев пожара."""
        try:
            from .scenario_editor import ScenarioEditorApp
        except ImportError:
            from scenario_editor import ScenarioEditorApp
        self._on_pause()
        ScenarioEditorApp(self, on_launch=self._launch_custom_scenario)

    def _launch_custom_scenario(self, cfg: dict):
        """Загрузить и запустить пользовательский сценарий из конструктора."""
        # Добавить пользовательский сценарий в реестр под ключом «custom»
        SCENARIOS["custom"] = cfg
        self._scenario_key = "custom"
        self.sim = TankFireSim(seed=42, training=True, scenario="custom")
        self._hdr_var.set(cfg["name"])
        self._norms_label_var.set(self._get_norms_text())
        self._draw_map()
        self._update_charts()
        self._update_status()

    def _get_norms_text(self) -> str:
        """Нормативные требования для текущего сценария."""
        key = getattr(self, "_scenario_key", "tuapse")
        cfg = SCENARIOS[key]
        S   = cfg["initial_fire_area"]
        I   = cfg["foam_intensity"]
        Q_req = I * S
        D   = cfg["rvs_diameter_m"]
        import math as _m
        Q_cool = 0.8 * _m.pi * D  # л/с охлаждение горящего РВС
        roof = cfg["roof_obstruction_init"]
        return (f"  {cfg['rvs_name']}  |  S_зеркала={S:.0f} м²  |  "
                f"I_пены={I:.3f} л/(с·м²)  |  Q_пены≥{Q_req:.1f} л/с  |  "
                f"Q_охл≥{Q_cool:.1f} л/с  |  "
                f"Препятствие крыши: {roof*100:.0f}%")

    def _load_scenario_from_json(self):
        """Загрузить пользовательский сценарий из JSON-файла (экспорт конструктора)."""
        path = filedialog.askopenfilename(
            filetypes=[("JSON файл", "*.json")],
            title="Загрузить сценарий из JSON",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("САУР-ПСП", f"Ошибка чтения файла:\n{e}")
            return

        # Поддержка двух форматов: прямой cfg и обёртка article_data JSON
        cfg = data if "_objects" in data or "name" in data else data.get("scenario")
        if cfg is None:
            messagebox.showerror("САУР-ПСП",
                                 "Не удалось распознать формат файла сценария.")
            return

        # Заполнить обязательные поля значениями по умолчанию
        cfg.setdefault("_custom", True)
        cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
        cfg.setdefault("short", cfg["name"][:20])
        cfg.setdefault("fuel", "бензин")
        cfg.setdefault("fire_rank_default", 2)
        cfg.setdefault("total_min", 300)
        cfg.setdefault("initial_fire_area", 168.0)
        cfg.setdefault("rvs_name", "РВС (польз.)")
        cfg.setdefault("rvs_diameter_m", 14.62)
        cfg.setdefault("roof_obstruction_init", 0.0)
        cfg.setdefault("foam_intensity", 0.065)
        cfg.setdefault("actions_by_phase", None)
        if "tl_lookup" not in cfg:
            cfg["tl_lookup"] = {}
        SCENARIOS["custom"] = cfg
        self._scenario_key = "custom"
        self._scenario_var.set("custom")
        self._on_pause()
        self.sim._cfg = cfg
        self.sim.scenario = "custom"
        self.sim.reset()
        self._draw_map()
        self._update_charts()
        self._update_status()
        self._norms_label_var.set(self._get_norms_text())
        if hasattr(self, "_loaded_scen_label"):
            self._loaded_scen_label.config(text=f"✓ {cfg['name']}")
        # Добавить радиокнопку, если "custom" ещё нет среди кнопок
        # (не перестраиваем список — достаточно текстового уведомления)

    def _apply_settings(self):
        new_scenario = self._scenario_var.get()
        if new_scenario != self._scenario_key:
            self._on_pause()
            self._scenario_key = new_scenario
            seed = 42
            self.sim = TankFireSim(seed=42, training=True, scenario=new_scenario)
            self._hdr_var.set(SCENARIOS[new_scenario]["name"])
            self._norms_label_var.set(self._get_norms_text())
            self._draw_map()
            self._update_charts()
            self._update_status()
        self.sim.agent.epsilon = float(self._var_epsilon.get())
        self.sim.agent.alpha   = float(self._var_lr.get())
        self.sim.agent.gamma   = float(self._var_gamma.get())
        mode = self._mode_var.get()
        self.sim.training = (mode == "rl_train")

    def _new_episode(self):
        self._on_pause()
        self.sim.reset()
        self.sim.agent.epsilon = float(self._var_epsilon.get())
        self._draw_map()
        self._update_charts()
        self._update_status()

    def _do_manual_action(self):
        sel = self._manual_var.get()
        if not sel:
            return
        code = sel.split("]")[0].strip("[")
        idx  = next((i for i,a in enumerate(ACTIONS) if a[0]==code), 12)
        r    = self.sim._apply(idx)
        self.sim.last_action = idx
        self._log_sim_event(self.sim.t, P["hi"],
                            f"[РУЧНОЕ] {ACTIONS[idx][0]}: {ACTIONS[idx][2]} (r={r:.2f})")
        self._update_status()
        self._draw_map()

    # ── Панель управления (нижняя) ────────────────────────────────────────────
    def _build_controls(self):
        ctrl = tk.Frame(self, bg=P["panel2"], height=52)
        ctrl.pack(fill="x", side="bottom")
        ctrl.pack_propagate(False)

        # Кнопки Play/Pause/Step/Reset
        for text, cmd, color in [
            ("▶  Пуск",   self._on_play,  P["success"]),
            ("⏸  Стоп",   self._on_pause, P["warn"]),
            ("⏭  Шаг",    self._on_step,  P["info"]),
            ("⏮  Сброс",  self._on_reset, P["danger"]),
        ]:
            tk.Button(ctrl, text=text, command=cmd, bg=color, fg="#fff",
                      font=("Arial", 9, "bold"), relief="flat",
                      padx=12, pady=6).pack(side="left", padx=4, pady=8)

        tk.Label(ctrl, text="│", bg=P["panel2"], fg=P["grid"]).pack(side="left")
        _ctrl_hints = {
            "trainer":  "  Нажмите ▶ Пуск, затем выбирайте действия РТП кнопками под картой",
            "sppр":     "  Нажмите ▶ Пуск — агент покажет рекомендацию; примите или переопределите",
            "research": "  Шаг 2: нажмите ▶ Пуск для запуска симуляции  |  Обучение агентов — в полосе выше",
        }
        tk.Label(ctrl,
                 text=_ctrl_hints.get(self._mode, ""),
                 bg=P["panel2"], fg=P["text2"], font=("Arial", 8)).pack(side="left", padx=4)

        # Скорость (в тренажёре не нужна — каждый шаг ждёт пользователя)
        if self._mode != "trainer":
            tk.Label(ctrl, text="│", bg=P["panel2"], fg=P["grid"]).pack(side="left", padx=2)
        tk.Label(ctrl, text="Скорость:", bg=P["panel2"], fg=P["text"],
                 font=("Arial", 8)).pack(side="left", padx=(8,2))
        self._speed_var = tk.StringVar(value="1×")
        for label in self.SPEEDS:
            rb = tk.Radiobutton(ctrl, text=label, variable=self._speed_var, value=label,
                                command=self._on_speed_change,
                                bg=P["panel2"], fg=P["text2"], selectcolor=P["grid"],
                                font=("Arial", 8), activebackground=P["panel2"])
            rb.pack(side="left", padx=2)

        tk.Label(ctrl, text="│", bg=P["panel2"], fg=P["grid"]).pack(side="left", padx=4)

        # Прогресс-бар
        tk.Label(ctrl, text="Прогресс:", bg=P["panel2"], fg=P["text"],
                 font=("Arial", 8)).pack(side="left")
        self._prog_var = tk.StringVar(value="0 / 4862 мин  (0%)")
        tk.Label(ctrl, textvariable=self._prog_var, bg=P["panel2"], fg=P["hi"],
                 font=("Consolas", 8), width=28).pack(side="left", padx=4)
        self._prog_bar = ttk.Progressbar(ctrl, length=200, mode="determinate",
                                         maximum=TOTAL_MIN)
        self._prog_bar.pack(side="left", padx=4)

    def _build_workflow_guide(self):
        """Горизонтальная полоса с пронумерованными шагами моделирования."""
        bar = tk.Frame(self, bg=P["panel"], bd=0)
        bar.pack(fill="x")

        # Заголовок
        tk.Label(bar, text=" Порядок работы:", bg=P["panel"], fg=P["text2"],
                 font=("Arial", 8, "bold")).pack(side="left", padx=(10, 4))

        _steps_by_mode = {
            "trainer": [
                ("1", "Настройки",  "⚙ Выберите сценарий\nи параметры пожара",   0),
                ("2", "Старт",      "▶ Нажмите «Пуск» и\nвыбирайте действия",    1),
                ("3", "Справочник", "📖 Допустимые действия\nпо фазам пожара",    3),
                ("4", "Дебрифинг",  "По итогам — разбор\nкаждого решения",       7),
            ],
            "sppр": [
                ("1", "Настройки",  "⚙ Сценарий, параметры\nагента",             0),
                ("2", "Старт",      "▶ Пуск → агент\nдаёт рекомендации",         1),
                ("3", "Справочник", "📖 Справочник действий\nРТП по фазам",       3),
                ("4", "Отчёт",      "📄 Журнал отклонений\nот рекомендаций",     7),
            ],
            "research": [
                ("1", "Настройки",  "⚙ Сценарий, параметры\nпожара и RL-агента", 0),
                ("2", "Симуляция",  "▶ Пуск → Хронология\nи Метрики",           1),
                ("3", "Flat RL",    "Обучить плоский\nQ-learning агент",          4),
                ("4", "Иерарх. RL", "Обучить 3-уровневый\nагент L3→L2→L1",       5),
                ("5", "Массовое",   "N симуляций: Flat vs\nHRL + CI/p-value",     6),
                ("6", "Отчёт",      "PDF / DOCX / JSON\nс выводами и графиками",  7),
            ],
        }
        steps = _steps_by_mode.get(self._mode, _steps_by_mode["research"])

        arrow_lbl = "→"
        for idx, (num, title, hint, tab_idx) in enumerate(steps):
            # Стрелка-разделитель
            if idx > 0:
                tk.Label(bar, text=arrow_lbl, bg=P["panel"], fg=P["grid"],
                         font=("Arial", 10)).pack(side="left")

            # Контейнер шага
            step_frame = tk.Frame(bar, bg=P["accent"], cursor="hand2",
                                  bd=0, padx=1, pady=1)
            step_frame.pack(side="left", padx=1, pady=3)

            inner = tk.Frame(step_frame, bg=P["panel2"])
            inner.pack()

            # Номер
            num_lbl = tk.Label(inner, text=f" {num} ", bg=P["hi"], fg="#fff",
                               font=("Arial", 8, "bold"))
            num_lbl.pack(side="left")

            # Текст с подсказкой
            text_lbl = tk.Label(inner, text=f" {title} ", bg=P["panel2"],
                                fg=P["text"], font=("Arial", 8))
            text_lbl.pack(side="left", padx=2)

            # Tooltip и переключение на вкладку при клике
            _tab_idx = tab_idx
            def _make_click(ti):
                def _click(e):
                    try:
                        self._nb.select(ti)
                    except Exception:
                        pass
                return _click

            for w in (step_frame, inner, num_lbl, text_lbl):
                w.bind("<Button-1>", _make_click(_tab_idx))
                # Простая всплывающая подсказка
                w.bind("<Enter>", lambda e, h=hint, w=w: self._show_tip(e, h, w))
                w.bind("<Leave>", lambda e: self._hide_tip())

        # Хранилище для tooltip
        self._tip_win = None

    def _show_tip(self, event, text, widget):
        """Показать простую всплывающую подсказку."""
        self._hide_tip()
        x = widget.winfo_rootx() + 10
        y = widget.winfo_rooty() + widget.winfo_height() + 2
        self._tip_win = tw = tk.Toplevel(self)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=text, bg="#ffffcc", fg="#333",
                 font=("Arial", 8), relief="solid", bd=1,
                 justify="left", padx=6, pady=4).pack()

    def _hide_tip(self):
        if self._tip_win:
            try:
                self._tip_win.destroy()
            except Exception:
                pass
            self._tip_win = None

    # ─────────────────────────────────────────────────────────────────────────
    # КАРТА (CANVAS)
    # ─────────────────────────────────────────────────────────────────────────
    def _update_map_info(self):
        """Обновить панель легенды и действия РТП под картой."""
        sim = self.sim
        cfg = sim._cfg
        is_tuapse = (sim.scenario == "tuapse")
        is_custom = cfg.get("_custom", False)
        # Сосед
        objects = cfg.get("_objects", []) if is_custom else []
        nbr_obj = next((o for o in objects if o["otype"] == "rvs_near"), None)
        nbr_label = (nbr_obj.get("label", "Соседний РВС")[:18] if nbr_obj
                     else ("Соседний РВС" if is_tuapse else "РВС-1 (сосед.)"))
        # Водоисточник
        water_src = next((o for o in objects if o["otype"] in ("river","hydrant")), None)
        pns_label = (f"{water_src.get('label','Водоисточник')[:12]}: {sim.n_pns} ед."
                     if is_custom
                     else (f"ПНС: {sim.n_pns} ед." if is_tuapse
                           else f"ПГ/АЦ: {sim.n_pns} ед."))
        labels = [
            cfg["rvs_name"][:20],
            nbr_label,
            f"Стволы: {sim.n_trunks_burn}+{sim.n_trunks_nbr}",
            pns_label,
        ]
        for var, txt in zip(self._legend_items, labels):
            var.set(txt)
        # Обновить действие РТП
        a = sim.last_action
        code, level, desc = ACTIONS[a]
        lc = LEVEL_C[level]
        self._map_action_var.set(f"Действие РТП: [{code}] {desc}")
        self._map_action_lbl.config(fg=lc)
        self._map_phase_var.set(
            f"Уровень: {level} | Фаза: {sim.phase} | t={self._fmt_time(sim.t)}")

    # ── Перетаскивание объектов на карте ─────────────────────────────────────
    def _map_drag_register(self, name: str, cx: int, cy: int, w: int, h: int):
        """Зарегистрировать объект как перетаскиваемый (вызывается из _draw_map)."""
        self._map_drag_objects[name] = (cx, cy, w, h)

    def _map_drag_pos(self, name: str, cx: int, cy: int) -> Tuple[int, int]:
        """Получить позицию объекта с учётом пользовательского смещения."""
        dx, dy = self._map_drag_offsets.get(name, (0, 0))
        return cx + dx, cy + dy

    def _map_drag_press(self, event):
        """Нажатие мыши: найти объект под курсором."""
        ex, ey = event.x, event.y
        best, best_dist = None, 1e9
        for name, (cx, cy, w, h) in self._map_drag_objects.items():
            dx, dy = self._map_drag_offsets.get(name, (0, 0))
            ox, oy = cx + dx, cy + dy
            if abs(ex - ox) <= w//2 + 6 and abs(ey - oy) <= h//2 + 6:
                dist = (ex - ox)**2 + (ey - oy)**2
                if dist < best_dist:
                    best, best_dist = name, dist
        if best:
            self._drag_target = best
            dx, dy = self._map_drag_offsets.get(best, (0, 0))
            cx, cy, _, _ = self._map_drag_objects[best]
            self._drag_start = (ex - (cx + dx), ey - (cy + dy))
            self.canvas.config(cursor="fleur")

    def _map_drag_motion(self, event):
        """Перемещение мыши с зажатой кнопкой: двигаем объект."""
        if not self._drag_target or not self._drag_start:
            return
        name = self._drag_target
        cx, cy, _, _ = self._map_drag_objects[name]
        sx, sy = self._drag_start
        new_dx = event.x - sx - cx
        new_dy = event.y - sy - cy
        # Ограничить пределами карты
        new_dx = max(-cx + 20, min(MAP_W - cx - 20, new_dx))
        new_dy = max(-cy + 16, min(MAP_H - cy - 20, new_dy))
        self._map_drag_offsets[name] = (new_dx, new_dy)
        self._draw_map()

    def _map_drag_release(self, event):
        """Отпускание кнопки мыши: завершить перетаскивание."""
        self._drag_target = None
        self._drag_start = None
        self.canvas.config(cursor="")

    # ── Утилиты отрисовки карты ─────────────────────────────────────────────
    @staticmethod
    def _map_label(c, x, y, text, fill, font, bg=None, padx=3, pady=1, anchor="center"):
        """Подпись с полупрозрачным фоном, чтобы не сливалась с объектами."""
        tid = c.create_text(x, y, text=text, fill=fill, font=font, anchor=anchor)
        if bg:
            bb = c.bbox(tid)
            if bb:
                rid = c.create_rectangle(bb[0]-padx, bb[1]-pady, bb[2]+padx, bb[3]+pady,
                                         fill=bg, outline="", width=0)
                c.tag_lower(rid, tid)
        return tid

    @staticmethod
    def _map_vehicle(c, x, y, w, h, fill, label, label_color="#ffffff"):
        """Пожарный автомобиль — БУПО Приложение №1, раздел 1."""
        bupo_type = LABEL_TO_BUPO.get(label, BUPO_AC)
        draw_bupo_symbol(c, x, y, bupo_type, label=label, fill=fill)

    @staticmethod
    def _map_hydrant_icon(c, x, y, name, active=False):
        """Пожарный гидрант — БУПО Приложение №1, раздел 7."""
        draw_bupo_symbol(c, x, y, BUPO_PG, label=name, active=active)

    @staticmethod
    def _map_pump_icon(c, x, y, label):
        """Насосная станция (ПНС) — БУПО Приложение №1, раздел 1."""
        draw_bupo_symbol(c, x, y, BUPO_PNS, label=label)

    def _draw_map(self):
        c = self.canvas
        c.delete("all")
        sim = self.sim
        cfg = sim._cfg
        t   = sim.t

        W, H = MAP_W, MAP_H
        self._map_drag_objects.clear()

        # ── Пользовательский сценарий — отдельная отрисовка ───────────────────
        if cfg.get("_custom"):
            self._draw_map_custom(c, W, H, sim, cfg)
            return

        # ── Параметры сценария ─────────────────────────────────────────────────
        is_tuapse = (sim.scenario == "tuapse")
        rvs_diam  = cfg["rvs_diameter_m"]
        init_area = cfg["initial_fire_area"]
        rvs_name  = cfg["rvs_name"]
        r9_base   = int(H * 0.128)
        r9        = max(20, int(r9_base * rvs_diam / 40.0))
        max_fire_ref = init_area * 1.5
        fire_intensity = min(1.0, sim.fire_area / max_fire_ref) if not sim.extinguished else 0.0

        cx9, cy9   = int(W * 0.47), int(H * 0.52)
        r17        = max(15, int(r9 * 0.70))
        cx17, cy17 = int(W * 0.47), int(H * 0.16)

        # ══════════════════════════════════════════════════════════════════════
        # 1. ФОН — территория, обвалование, дороги
        # ══════════════════════════════════════════════════════════════════════
        c.create_rectangle(0, 0, W, H, fill="#e8ece0", outline="")
        # Территория резервуарного парка
        c.create_rectangle(20, 16, W-12, H-54, fill=P["ground"], outline="#8aaf86",
                           width=1)
        # Тонкая сетка территории (координатная)
        for gx in range(100, W-12, 100):
            c.create_line(gx, 16, gx, H-54, fill="#9bbd96", width=1, dash=(1, 8))
        for gy in range(80, H-54, 80):
            c.create_line(20, gy, W-12, gy, fill="#9bbd96", width=1, dash=(1, 8))

        # ── Обвалование (дамба) горящего РВС ─────────────────────────────────
        dike_r = r9 + 42
        c.create_oval(cx9-dike_r, cy9-dike_r, cx9+dike_r, cy9+dike_r,
                      outline="#8B7355", width=3, fill="", dash=(6, 3))
        self._map_label(c, cx9+dike_r-10, cy9-dike_r+8, "обвалование",
                        "#8B7355", ("Arial", 7, "italic"), bg="#e8ece0")

        # ── Дороги (подъезды) ────────────────────────────────────────────────
        road_clr, road_w = "#c0bfb8", 6
        # Главная вертикальная дорога
        c.create_line(W//2+50, H-54, W//2+50, 16, fill=road_clr, width=road_w)
        c.create_line(W//2+50, H-54, W//2+50, 16, fill="#d8d7d0", width=road_w-2)
        # Горизонтальная дорога
        c.create_line(20, int(H*0.42), W-12, int(H*0.42), fill=road_clr, width=road_w)
        c.create_line(20, int(H*0.42), W-12, int(H*0.42), fill="#d8d7d0", width=road_w-2)
        # Подъезд к горящему РВС (от дороги к обвалованию)
        c.create_line(W//2+50, cy9, cx9+dike_r+2, cy9,
                      fill=road_clr, width=4)
        self._map_label(c, W//2+50, int(H*0.42)-10, "подъездная дорога",
                        "#888", ("Arial", 7), bg="#e8ece0")

        # ══════════════════════════════════════════════════════════════════════
        # 2. ВОДОИСТОЧНИК (нижняя полоса) — различается по сценарию
        # ══════════════════════════════════════════════════════════════════════
        wy_top = H - 52
        if is_tuapse:
            # ── Туапсе: открытый водоисточник (река) ─────────────────────────
            for i, clr in enumerate(["#a8d4ed", "#85c1e9", "#6bb3e0"]):
                c.create_rectangle(0, wy_top + i*17, W, wy_top + (i+1)*17+1,
                                   fill=clr, outline="")
            for wx in range(30, W-20, 26):
                wy = wy_top + 8 + 3*math.sin(wx*0.05 + self._anim_t*0.2)
                c.create_arc(wx, wy, wx+18, wy+10, start=0, extent=180,
                             outline="#ffffff", width=1, style="arc")
            self._map_label(c, W//2, wy_top+14, "Открытый водоисточник (р. Туапсе)",
                            "#2c3e50", ("Arial", 9, "bold"), bg="#a8d4ed")
            self._map_label(c, W//2, wy_top+32, "← забор воды ПНС-110, ПАНРК →",
                            "#34495e", ("Arial", 7, "italic"), bg="#85c1e9")
        else:
            # ── Серпухов: городская дорога с пожарными гидрантами ────────────
            # Дорожное полотно (асфальт)
            c.create_rectangle(0, wy_top, W, H, fill="#b0b0a8", outline="")
            # Бордюр
            c.create_line(0, wy_top, W, wy_top, fill="#888", width=2)
            # Разделительная полоса
            for lx in range(40, W-40, 50):
                c.create_line(lx, wy_top+26, lx+25, wy_top+26,
                              fill="#e0d84c", width=2)
            # Пожарные гидранты на дороге
            pg_positions = [
                (int(W*0.25), wy_top+16, "ПГ-106"),
                (int(W*0.55), wy_top+16, "ПГ-108"),
                (int(W*0.80), wy_top+16, "ПГ-110"),
            ]
            for px, py, pname in pg_positions:
                # Колодец гидранта (красный квадрат)
                c.create_rectangle(px-8, py-8, px+8, py+8,
                                   fill="#cc3333", outline="#fff", width=1)
                c.create_line(px-4, py, px+4, py, fill="white", width=2)
                c.create_line(px, py-4, px, py+4, fill="white", width=2)
                c.create_text(px, py+16, text=pname, fill="#cc3333",
                              font=("Arial", 7, "bold"))
            self._map_label(c, W//2, wy_top+42,
                            "ул. Ворошилова — водопроводная сеть (давление ≥ 0.15 МПа)",
                            "#444", ("Arial", 7, "italic"), bg="#b0b0a8")

        # ══════════════════════════════════════════════════════════════════════
        # 3. ПРИЧАЛ / ПАНРК (Туапсе) — перетаскиваемый
        # ══════════════════════════════════════════════════════════════════════
        if is_tuapse:
            _pw, _ph = 64, 38
            _pier_cx0, _pier_cy0 = W-48, wy_top-19
            _pier_cx, _pier_cy = self._map_drag_pos("pier", _pier_cx0, _pier_cy0)
            self._map_drag_register("pier", _pier_cx0, _pier_cy0, _pw, _ph)
            px1 = _pier_cx - _pw//2
            py1 = _pier_cy - _ph//2
            px2 = _pier_cx + _pw//2
            py2 = _pier_cy + _ph//2
            c.create_rectangle(px1, py1, px2, py2, fill="#2471a3", outline="#1a5276", width=2)
            c.create_text(_pier_cx, _pier_cy-6, text="Причал", fill="white",
                          font=("Arial", 8, "bold"))
            c.create_text(_pier_cx, _pier_cy+8, text="ПАНРК", fill="#aed6f1",
                          font=("Arial", 7))
            c.create_line(px1+10, py2, px1+10, py2+12, fill="#1a5276", width=3)
            c.create_line(px2-10, py2, px2-10, py2+12, fill="#1a5276", width=3)

        # ══════════════════════════════════════════════════════════════════════
        # 4. СОСЕДНИЙ РВС (верхняя часть)
        # ══════════════════════════════════════════════════════════════════════
        # Обвалование соседнего
        dike17 = r17 + 28
        c.create_oval(cx17-dike17, cy17-dike17, cx17+dike17, cy17+dike17,
                      outline="#8B7355", width=2, fill="", dash=(4, 4))
        # Кольцо орошения (если стволы поданы)
        if sim.n_trunks_nbr > 0:
            c.create_oval(cx17-r17-10, cy17-r17-10, cx17+r17+10, cy17+r17+10,
                          outline=P["water"], width=2, dash=(3, 3))
        # Корпус резервуара
        c.create_oval(cx17-r17, cy17-r17, cx17+r17, cy17+r17,
                      fill=P["rvs17"], outline="#3e86b5", width=2)
        # Плавающая крыша (штриховка)
        for hy in range(cy17-r17+6, cy17+r17-4, 7):
            hw = int(math.sqrt(max(0, r17**2 - (hy-cy17)**2))) - 3
            if hw > 2:
                c.create_line(cx17-hw, hy, cx17+hw, hy, fill="#5599cc", width=1)
        # Подписи
        if is_tuapse:
            nbr_label = "РВС-17  V≈20 000 м³"
        else:
            nbr_label = "РВС-1  V≈1 000 м³"
        c.create_text(cx17, cy17-4, text="Соседний РВС", fill="white",
                      font=("Arial", 9, "bold"))
        self._map_label(c, cx17, cy17+r17+16, nbr_label,
                        P["rvs17"], ("Arial", 7, "bold"), bg="#e8ece0")
        # Стрелка «орошение / защита»
        hint = "защита" if is_tuapse else "орошение"
        self._map_label(c, cx17, cy17+r17+30, f"↑ {hint} ↑",
                        P["water"], ("Arial", 7, "italic"), bg="#e8ece0")

        # Стволы охлаждения соседнего РВС
        nbr_positions = [
            (cx17+r17+16, cy17,      90),
            (cx17-r17-16, cy17,      270),
            (cx17,        cy17+r17+16, 180),
            (cx17+r17+10, cy17-r17-10, 45),
            (cx17-r17-10, cy17-r17-10, 315),
        ]
        for i, (tx, ty, _) in enumerate(nbr_positions[:sim.n_trunks_nbr]):
            c.create_line(tx, ty, cx17+0.5*(tx-cx17), cy17+0.5*(ty-cy17),
                          fill="#4488ff", width=2, arrow="last", arrowshape=(8,10,4))
            c.create_oval(tx-5, ty-5, tx+5, ty+5, fill="#4488ff", outline="white", width=1)

        # ══════════════════════════════════════════════════════════════════════
        # 5. ГОРЯЩИЙ РВС (центр карты)
        # ══════════════════════════════════════════════════════════════════════

        # --- Зона теплового воздействия (красный ореол) ---
        if fire_intensity > 0:
            heat_r = r9 + 28 + int(14 * fire_intensity)
            c.create_oval(cx9-heat_r, cy9-heat_r, cx9+heat_r, cy9+heat_r,
                          outline="#ff6633", width=1, dash=(2, 4), fill="")
            c.create_oval(cx9-heat_r-10, cy9-heat_r-10, cx9+heat_r+10, cy9+heat_r+10,
                          outline="#ff9966", width=1, dash=(1, 6), fill="")

        # --- Дымовой шлейф (анимированный, сдвигается «по ветру» вправо-вверх) ---
        if fire_intensity > 0:
            for i in range(4):
                dx = 18 + i * 22 + 4 * math.sin(self._anim_t * 0.15 + i)
                dy = -(20 + i * 18 + 3 * math.cos(self._anim_t * 0.12 + i))
                sr = 14 + i * 10
                c.create_oval(cx9+dx-sr, cy9+dy-sr, cx9+dx+sr, cy9+dy+sr,
                              fill="", outline=P["smoke"], width=1, dash=(2, 5))

        # --- Тело пожара ---
        if fire_intensity > 0:
            # Заполнение — градиент от оранжевого к красному
            fire_clr = P["fire"] if fire_intensity > 0.65 else P["fire2"]
            c.create_oval(cx9-r9, cy9-r9, cx9+r9, cy9+r9,
                          fill=fire_clr, outline="")
            # Языки пламени (12 штук, вращаются и пульсируют)
            pulse = 3.5 * math.sin(self._anim_t * 0.35)
            n_flames = 12
            for k in range(n_flames):
                ang = math.radians(k * 360/n_flames + self._anim_t * 2.5)
                dist = r9 + 8 + pulse + 3*math.sin(self._anim_t*0.5 + k*1.7)
                fx = cx9 + dist * math.cos(ang)
                fy = cy9 + dist * math.sin(ang)
                fs = 4 + int(4 * fire_intensity)
                # Каждый второй язык — ярче
                fl_clr = P["flame"] if k % 2 == 0 else "#ff6600"
                c.create_oval(fx-fs, fy-fs, fx+fs, fy+fs, fill=fl_clr, outline="")
            # Внутреннее ядро (яркое)
            core_r = int(r9 * 0.45)
            c.create_oval(cx9-core_r, cy9-core_r, cx9+core_r, cy9+core_r,
                          fill=P["flame"], outline="")
        else:
            # Потушен — серый корпус
            c.create_oval(cx9-r9, cy9-r9, cx9+r9, cy9+r9,
                          fill=P["rvs9_cool"], outline="#888866", width=2)
            # Штриховка крыши потушенного
            for hy in range(cy9-r9+5, cy9+r9-3, 6):
                hw = int(math.sqrt(max(0, r9**2 - (hy-cy9)**2))) - 2
                if hw > 2:
                    c.create_line(cx9-hw, hy, cx9+hw, hy, fill="#776655", width=1)

        # Контур резервуара (рёбра жёсткости)
        c.create_oval(cx9-r9, cy9-r9, cx9+r9, cy9+r9,
                      outline="#cc2222" if fire_intensity > 0 else "#666644",
                      width=3, fill="")
        # Каркас крыши (крестовина) — видна если нет сильного огня
        if fire_intensity < 0.5 or sim.extinguished:
            c.create_line(cx9-r9+6, cy9, cx9+r9-6, cy9, fill="#666644", width=1, dash=(3,3))
            c.create_line(cx9, cy9-r9+6, cx9, cy9+r9-6, fill="#666644", width=1, dash=(3,3))

        # Подписи горящего РВС
        c.create_text(cx9, cy9-12, text=rvs_name, fill="white",
                      font=("Arial", 10, "bold"))
        area_clr = P["hi"] if fire_intensity > 0 else P["success"]
        c.create_text(cx9, cy9+6, text=f"∅{rvs_diam:.0f} м",
                      fill="white", font=("Arial", 8))
        # Площадь пожара — вынесена под резервуар, чтобы не наслаивалась
        self._map_label(c, cx9, cy9+r9+10,
                        f"S = {sim.fire_area:.0f} м²",
                        area_clr, ("Arial", 8, "bold"), bg="#e8ece0")

        # ── Розлив ────────────────────────────────────────────────────────────
        if sim.spill:
            # Неправильная фигура розлива
            sp_pts = [cx9-r9-20, cy9+r9-8,
                      cx9-r9-35, cy9+r9+12,
                      cx9-r9-28, cy9+r9+35,
                      cx9-10,    cy9+r9+38,
                      cx9+15,    cy9+r9+20,
                      cx9+5,     cy9+r9-2]
            c.create_polygon(sp_pts, fill="#ff4400", outline="#ff6600",
                             width=2, stipple="gray50")
            self._map_label(c, cx9-r9-5, cy9+r9+44,
                            f"Розлив {sim.spill_area:.0f} м²",
                            P["danger"], ("Arial", 8, "bold"), bg="#e8ece0")

        # ══════════════════════════════════════════════════════════════════════
        # 6. ЗДАНИЯ
        # ══════════════════════════════════════════════════════════════════════
        if is_tuapse:
            # Лаборатория — перетаскиваемая
            _lw, _lh = 80, 44
            _l0x, _l0y = int(W*.74)+_lw//2, int(H*.46)+_lh//2
            _lcx, _lcy = self._map_drag_pos("lab", _l0x, _l0y)
            self._map_drag_register("lab", _l0x, _l0y, _lw, _lh)
            _lx, _ly = _lcx-_lw//2, _lcy-_lh//2
            lab_clr = P["danger"] if sim.secondary_fire else "#7f8c8d"
            c.create_rectangle(_lx, _ly, _lx+_lw, _ly+_lh,
                               fill=lab_clr, outline="#555", width=1)
            c.create_line(_lx, _ly, _lx+_lw, _ly, fill="#666", width=2)
            c.create_text(_lcx, _lcy, text="Лаборатория",
                          fill="white", font=("Arial", 8))
            # Столовая — перетаскиваемая
            _c0x, _c0y = int(W*.74)+_lw//2, int(H*.60)+_lh//2
            _ccx, _ccy = self._map_drag_pos("canteen", _c0x, _c0y)
            self._map_drag_register("canteen", _c0x, _c0y, _lw, _lh)
            _cx, _cy = _ccx-_lw//2, _ccy-_lh//2
            canteen_clr = P["danger"] if sim.secondary_fire else "#7f8c8d"
            c.create_rectangle(_cx, _cy, _cx+_lw, _cy+_lh,
                               fill=canteen_clr, outline="#555", width=1)
            c.create_line(_cx, _cy, _cx+_lw, _cy, fill="#666", width=2)
            c.create_text(_ccx, _ccy, text="Столовая",
                          fill="white", font=("Arial", 8))
            if sim.secondary_fire:
                for _bx, _by in [(_lcx, _ly-6), (_ccx, _cy-6)]:
                    c.create_oval(_bx-6, _by-6, _bx+6, _by+6,
                                  fill=P["fire"], outline=P["flame"], width=1)
                self._map_label(c, _ccx, _cy+_lh+10, "ВТОРИЧНЫЙ ОЧАГ",
                                P["danger"], ("Arial", 7, "bold"), bg="#e8ece0")
        else:
            # Серпухов — Насосная — перетаскиваемая
            _pw, _ph = 78, 38
            _p0x, _p0y = int(W*.74)+_pw//2, int(H*.46)+_ph//2
            _pcx, _pcy = self._map_drag_pos("pump_st", _p0x, _p0y)
            self._map_drag_register("pump_st", _p0x, _p0y, _pw, _ph)
            _px, _py = _pcx-_pw//2, _pcy-_ph//2
            c.create_rectangle(_px, _py, _px+_pw, _py+_ph,
                               fill="#7f8c8d", outline="#555", width=1)
            c.create_line(_px, _py, _px+_pw, _py, fill="#666", width=2)
            c.create_text(_pcx, _pcy, text="Насосная ст.",
                          fill="white", font=("Arial", 8))
            # Склад ГСМ — перетаскиваемый
            _sw, _sh = _pw-10, _ph
            _s0x, _s0y = int(W*.74)+_sw//2, int(H*.58)+_sh//2
            _scx, _scy = self._map_drag_pos("storage", _s0x, _s0y)
            self._map_drag_register("storage", _s0x, _s0y, _sw, _sh)
            _sx, _sy = _scx-_sw//2, _scy-_sh//2
            c.create_rectangle(_sx, _sy, _sx+_sw, _sy+_sh,
                               fill="#7f8c8d", outline="#555", width=1)
            c.create_line(_sx, _sy, _sx+_sw, _sy, fill="#666", width=2)
            c.create_text(_scx, _scy, text="Склад ГСМ",
                          fill="white", font=("Arial", 8))

        # ══════════════════════════════════════════════════════════════════════
        # 7. ШТАБ ПОЖАРОТУШЕНИЯ (ОШ) — БУПО разд.5, перетаскиваемый
        # ══════════════════════════════════════════════════════════════════════
        _hq_cx0, _hq_cy0 = int(W*.08) + 52, int(H*.46) + 24
        _hq_cx, _hq_cy = self._map_drag_pos("hq", _hq_cx0, _hq_cy0)
        hq_w, hq_h = 50, 50
        self._map_drag_register("hq", _hq_cx0, _hq_cy0, hq_w, hq_h)
        hq_active = sim.has_shtab
        draw_bupo_symbol(c, _hq_cx, _hq_cy, BUPO_OSH,
                         label="ОШ", scale=1.2, active=hq_active)
        if hq_active:
            self._map_label(c, _hq_cx, _hq_cy + 30,
                            "РТП, НШ, НТ, ОТ", "#c0392b", ("Arial", 7), bg="#e8ece0")
        else:
            self._map_label(c, _hq_cx, _hq_cy + 30,
                            "не развёрнут", "#999", ("Arial", 7, "italic"), bg="#e8ece0")

        # ══════════════════════════════════════════════════════════════════════
        # 8. ГИДРАНТЫ / ВОДОИСТОЧНИКИ — перетаскиваемые
        # ══════════════════════════════════════════════════════════════════════
        if is_tuapse:
            _hydrants_def = [
                (int(W*.22), int(H*.74), "ПГ-1"),
                (int(W*.35), int(H*.36), "ПГ-2"),
                (int(W*.68), int(H*.32), "ПГ-3"),
            ]
        else:
            _hydrants_def = [
                (int(W*.22), int(H*.74), "ПГ-106"),
                (int(W*.68), int(H*.32), "ПГ-108"),
            ]
        _hydrants = []
        for hi, (gx0, gy0, gname) in enumerate(_hydrants_def):
            dn = f"hyd_{hi}"
            gx, gy = self._map_drag_pos(dn, gx0, gy0)
            self._map_drag_register(dn, gx0, gy0, 24, 24)
            self._map_hydrant_icon(c, gx, gy, gname, active=(sim.n_pns > 0))
            _hydrants.append((gx, gy, gname))

        # ══════════════════════════════════════════════════════════════════════
        # 9. ЛИНИИ ВОДОСНАБЖЕНИЯ + НАСОСНЫЕ СТАНЦИИ (ПНС)
        # ══════════════════════════════════════════════════════════════════════
        if is_tuapse:
            pns_sources = [
                (W//2+50, wy_top-4, cx9, cy9+dike_r+4, "ПНС-1"),
                (W-60,    wy_top-4, cx9+dike_r+4, cy9+10, "ПНС-2"),
                (140,     wy_top-4, cx9-dike_r-4, cy9+20, "ПНС-3"),
                (80,      wy_top-4, cx9-dike_r-4, cy9-20, "ПАНРК"),
            ]
        else:
            pns_sources = [
                (_hydrants[0][0], _hydrants[0][1], cx9, cy9+dike_r+4, "АЦ→ПГ"),
                (_hydrants[1][0], _hydrants[1][1], cx9+dike_r+4, cy9, "АЦ→ПГ"),
            ]
        for i, (sx0, sy0, ex, ey, lbl) in enumerate(pns_sources[:sim.n_pns]):
            dn = f"pns_{i}"
            sx, sy = self._map_drag_pos(dn, sx0, sy0)
            self._map_drag_register(dn, sx0, sy0, 26, 26)
            # Линия подачи воды (от ПНС к РВС)
            c.create_line(sx, sy, ex, ey, fill=P["water"], width=3, dash=(10, 5))
            # Анимация потока — бегущие точки
            for dot_i in range(3):
                frac = ((self._anim_t * 0.04 + dot_i * 0.33) % 1.0)
                dx = sx + (ex-sx) * frac
                dy = sy + (ey-sy) * frac
                c.create_oval(dx-3, dy-3, dx+3, dy+3, fill="white", outline=P["water"], width=1)
            # Значок ПНС у водоисточника
            self._map_pump_icon(c, sx, sy-16, lbl)

        # ══════════════════════════════════════════════════════════════════════
        # 10. СТВОЛЫ ОХЛАЖДЕНИЯ ГОРЯЩЕГО РВС — перетаскиваемые
        # ══════════════════════════════════════════════════════════════════════
        _d = int(r9 * 0.65)
        trunk_pos = [
            (cx9,        cy9+r9+22, "Ю",  180),
            (cx9+r9+22,  cy9,       "В",   90),
            (cx9-r9-22,  cy9,       "З",  270),
            (cx9,        cy9-r9-22, "С",    0),
            (cx9+_d+6,   cy9+_d+6, "ЮВ", 135),
            (cx9-_d-6,   cy9+_d+6, "ЮЗ", 225),
            (cx9+_d+6,   cy9-_d-6, "СВ",  45),
        ]
        trunk_names = ["Антенор-1", "Антенор-2", "Антенор-3", "Антенор-4",
                       "Антенор-5", "Антенор-6", "ЛС-С330"]
        for i, (tx0, ty0, side, ang_deg) in enumerate(trunk_pos[:sim.n_trunks_burn]):
            dn = f"trunk_{i}"
            tx, ty = self._map_drag_pos(dn, tx0, ty0)
            self._map_drag_register(dn, tx0, ty0, 24, 24)
            # Угол от ствола к центру РВС (направление подачи)
            aim_ang = math.degrees(math.atan2(cy9 - ty, cx9 - tx))
            # БУПО ствол — лафетный для стационарных (Антенор), ручной для остальных
            st_type = BUPO_STVOL_LAF if i < 6 else BUPO_STVOL
            draw_bupo_symbol(c, tx, ty, st_type, angle=aim_ang, scale=1.1)
            # Подпись
            lx = tx + (16 if tx >= cx9 else -16)
            ly = ty + (16 if ty >= cy9 else -16)
            self._map_label(c, lx, ly, trunk_names[i] if i < len(trunk_names) else f"Ствол-{i+1}",
                            "#222", ("Arial", 6), bg="#e8ece0")

        # ══════════════════════════════════════════════════════════════════════
        # 11. БОЕВЫЕ УЧАСТКИ (БУ) — секторные зоны, перетаскиваемые
        # ══════════════════════════════════════════════════════════════════════
        bu_defs = [
            (cx9, cy9+dike_r+18, "#e74c3c", "БУ-1 (ЮГ)", "НБУ-1"),
            (cx9+dike_r+20, cy9, "#3498db", "БУ-2 (ВОСТОК)", "НБУ-2"),
            (cx9-dike_r-20, cy9, "#2ecc71", "БУ-3 (ЗАПАД)", "НБУ-3"),
        ]
        for i, (bx0, by0, bc, bname, bnbu) in enumerate(bu_defs[:sim.n_bu]):
            drag_name = f"bu_{i}"
            bx, by = self._map_drag_pos(drag_name, bx0, by0)
            self._map_drag_register(drag_name, bx0, by0, 100, 28)
            draw_bupo_symbol(c, bx, by, BUPO_BU, label=bname, fill=bc)
            self._map_label(c, bx, by + 20, bnbu, bc, ("Arial", 6), bg="#e8ece0")

        # ══════════════════════════════════════════════════════════════════════
        # 12. ПОЖАРНАЯ ТЕХНИКА (автомобили) — вдоль дорог
        # ══════════════════════════════════════════════════════════════════════
        # Автомобили появляются постепенно по времени
        vehicles = []
        # Базовые — всегда с начала (прибытие первых подразделений)
        if t >= 5 or not is_tuapse:
            vehicles.append((int(W*0.52)+60, int(H*0.42)-12, P["unit_ac"], "АЦ"))
        if t >= 10 or not is_tuapse:
            vehicles.append((int(W*0.52)+60, int(H*0.42)+12, P["unit_ac"], "АЦ"))
            vehicles.append((int(W*0.52)+90, int(H*0.42)-12, P["unit_apt"], "АПТ"))
        if t >= 20:
            vehicles.append((int(W*0.52)+90, int(H*0.42)+12, P["unit_ac"], "АЦ"))
        if sim.has_shtab:
            vehicles.append((_hq_cx+35, _hq_cy-8, P["unit_ash"], "АШ"))
            vehicles.append((_hq_cx+35, _hq_cy+12, "#555", "АР"))
        if is_tuapse and t >= 50:
            vehicles.append((int(W*0.52)+120, int(H*0.42)-12, P["unit_pns"], "ПНС"))
        if t >= (100 if is_tuapse else 30):
            vehicles.append((int(W*0.52)+120, int(H*0.42)+12, P["unit_ac"], "АЦ"))
        if is_tuapse and t >= 283:
            vehicles.append((W-90, wy_top-62, P["unit_panrk"], "ПАНРК"))
        if is_tuapse and t >= 548:
            vehicles.append((20, int(H*0.24), P["unit_train"], "ПП"))
        if getattr(sim, "akp50_available", False):
            vehicles.append((int(W*0.52)+150, int(H*0.42)-12, "#d35400", "АКП"))

        for vi, (vx0, vy0, vc, vlbl) in enumerate(vehicles):
            dn = f"veh_{vi}"
            vx, vy = self._map_drag_pos(dn, vx0, vy0)
            self._map_drag_register(dn, vx0, vy0, 28, 14)
            self._map_vehicle(c, vx, vy, 28, 14, vc, vlbl)

        # ══════════════════════════════════════════════════════════════════════
        # 13. КОМПАС + МАСШТАБ
        # ══════════════════════════════════════════════════════════════════════
        # Компас (верхний правый угол)
        ncx, ncy = W-40, 40
        c.create_line(ncx, ncy+14, ncx, ncy-14, fill=P["text"], width=2,
                      arrow="first", arrowshape=(8, 10, 4))
        c.create_text(ncx, ncy-22, text="С", fill=P["text"],
                      font=("Arial", 8, "bold"))
        c.create_text(ncx+12, ncy, text="В", fill=P["text2"], font=("Arial", 6))
        c.create_text(ncx-12, ncy, text="З", fill=P["text2"], font=("Arial", 6))
        c.create_text(ncx, ncy+20, text="Ю", fill=P["text2"], font=("Arial", 6))

        # Масштабная линейка (нижний левый)
        sc_x, sc_y = 40, wy_top - 14
        sc_len = 80  # пикселей
        c.create_line(sc_x, sc_y, sc_x+sc_len, sc_y, fill=P["text"], width=2)
        c.create_line(sc_x, sc_y-4, sc_x, sc_y+4, fill=P["text"], width=1)
        c.create_line(sc_x+sc_len, sc_y-4, sc_x+sc_len, sc_y+4, fill=P["text"], width=1)
        c.create_text(sc_x+sc_len//2, sc_y-8, text="~100 м",
                      fill=P["text2"], font=("Arial", 7))

        # ══════════════════════════════════════════════════════════════════════
        # 14. НАДПИСЬ СЦЕНАРИЯ / БАННЕР СТАТУСА
        # ══════════════════════════════════════════════════════════════════════
        status_text = ""
        status_color = P["text"]
        if sim.extinguished:
            status_text = "ПОЖАР ЛИКВИДИРОВАН"
            status_color = P["success"]
        elif sim.localized:
            status_text = "ПОЖАР ЛОКАЛИЗОВАН"
            status_color = P["info"]
        elif sim.foam_ready:
            status_text = "ГОТОВНОСТЬ К ПЕННОЙ АТАКЕ"
            status_color = P["warn"]

        if status_text:
            bw = 220
            c.create_rectangle(W//2-bw, 2, W//2+bw, 28,
                               fill=status_color, outline="#fff", width=1)
            c.create_text(W//2, 15, text=status_text,
                          fill="white", font=("Arial", 11, "bold"))
        else:
            self._map_label(c, W//2, 8, f"Резервуарный парк — {cfg['short']}",
                            P["text"], ("Arial", 9, "bold"), bg="#e8ece0")

        self._update_map_info()

    # ─────────────────────────────────────────────────────────────────────────
    # КАРТА: ПОЛЬЗОВАТЕЛЬСКИЙ СЦЕНАРИЙ
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_map_custom(self, c, W: int, H: int, sim, cfg: dict) -> None:
        """Отрисовка карты для сценария, созданного в конструкторе.

        Объекты берутся из cfg['_objects'] (список dict с otype/x/y/r/label).
        Координаты масштабируются из пространства редактора (580×460) → холст (W×H).
        """
        ED_W_REF, ED_H_REF = 580, 460
        sx = W / ED_W_REF
        sy = H / ED_H_REF

        objects   = cfg.get("_objects", [])
        rvs_diam  = cfg["rvs_diameter_m"]
        init_area = cfg["initial_fire_area"]
        rvs_name  = cfg.get("rvs_name", "РВС")

        fire_obj = next((o for o in objects if o["otype"] == "rvs_fire"), None)
        if fire_obj is None:
            return
        cx9 = int(fire_obj["x"] * sx)
        cy9 = int(fire_obj["y"] * sy)
        r9  = max(20, int(int(H * 0.128) * rvs_diam / 40.0))

        max_fire_ref   = init_area * 1.5
        fire_intensity = min(1.0, sim.fire_area / max_fire_ref) if not sim.extinguished else 0.0

        # ── Фон ──────────────────────────────────────────────────────────────
        c.create_rectangle(0, 0, W, H, fill="#e8ece0", outline="")
        c.create_rectangle(20, 16, W-12, H-20, fill=P["ground"], outline="#8aaf86", width=1)
        for gx in range(100, W-12, 100):
            c.create_line(gx, 16, gx, H-20, fill="#9bbd96", width=1, dash=(1, 8))
        for gy in range(80, H-20, 80):
            c.create_line(20, gy, W-12, gy, fill="#9bbd96", width=1, dash=(1, 8))

        # Обвалование
        dike_r = r9 + 38
        c.create_oval(cx9-dike_r, cy9-dike_r, cx9+dike_r, cy9+dike_r,
                      outline="#8B7355", width=3, fill="", dash=(6, 3))

        # Дороги
        road_clr = "#c0bfb8"
        c.create_line(W//2, 16, W//2, H-20, fill=road_clr, width=5)
        c.create_line(W//2, 16, W//2, H-20, fill="#d8d7d0", width=3)
        c.create_line(20, H//2, W-12, H//2, fill=road_clr, width=5)
        c.create_line(20, H//2, W-12, H//2, fill="#d8d7d0", width=3)

        # ── Водоёмы / реки ───────────────────────────────────────────────────
        for o in objects:
            if o["otype"] == "river":
                ox = int(o["x"] * sx);  oy = int(o["y"] * sy)
                r  = max(28, int(o["r"] * (sx + sy) / 2))
                # Градиентная заливка
                for i, clr in enumerate(["#a8d4ed", "#85c1e9", "#6bb3e0"]):
                    c.create_rectangle(ox-r, oy-r//3+i*r//4, ox+r, oy-r//3+(i+1)*r//4+1,
                                       fill=clr, outline="")
                c.create_rectangle(ox-r, oy-r//3, ox+r, oy+r//3,
                                   fill="", outline=P["water"], width=2)
                # Волны
                for wx in range(ox-r+8, ox+r-8, 18):
                    wy = oy - 2 + 3*math.sin(wx*0.08 + self._anim_t*0.2)
                    c.create_arc(wx, wy, wx+12, wy+6, start=0, extent=180,
                                 outline="white", width=1, style="arc")
                lbl = o.get("label", "Водоём")[:22]
                self._map_label(c, ox, oy, lbl, "#2c3e50",
                                ("Arial", 9, "italic"), bg="#a8d4ed")

        # ── Здания ───────────────────────────────────────────────────────────
        for o in objects:
            if o["otype"] == "building":
                ox = int(o["x"] * sx);  oy = int(o["y"] * sy)
                r  = max(22, int(o["r"] * (sx + sy) / 2))
                lbl = o.get("label", "Здание")[:16]
                is_hq = any(kw in lbl.upper() for kw in ("ОШ", "ШТАБ", "АШ"))
                if is_hq:
                    # БУПО разд.5 — треугольник с флагом
                    draw_bupo_symbol(c, ox, oy, BUPO_OSH,
                                     label=lbl[:6], scale=1.0, active=sim.has_shtab)
                else:
                    bcolor = P["danger"] if sim.secondary_fire else "#7f8c8d"
                    c.create_rectangle(ox-r, oy-r//2, ox+r, oy+r//2,
                                       fill=bcolor, outline="#555", width=1)
                    c.create_line(ox-r, oy-r//2, ox+r, oy-r//2, fill="#666", width=2)
                    c.create_text(ox, oy, text=lbl, fill="white", font=("Arial", 8))
                    if sim.secondary_fire:
                        c.create_oval(ox-5, oy-r//2-8, ox+5, oy-r//2+2,
                                      fill=P["fire"], outline=P["flame"], width=1)

        # Авто-штаб — БУПО треугольник
        has_hq = any(o["otype"] == "building" and
                     any(kw in o.get("label","").upper() for kw in ("ОШ","ШТАБ","АШ"))
                     for o in objects)
        if not has_hq:
            _hx_c, _hy_c = int(W*.08) + 50, int(H*.46) + 22
            draw_bupo_symbol(c, _hx_c, _hy_c, BUPO_OSH,
                             label="ОШ", scale=1.1, active=sim.has_shtab)

        # ── Гидранты ─────────────────────────────────────────────────────────
        for o in objects:
            if o["otype"] == "hydrant":
                ox = int(o["x"] * sx);  oy = int(o["y"] * sy)
                lbl = o.get("label", "ПГ")[:12]
                self._map_hydrant_icon(c, ox, oy, lbl, active=(sim.n_pns > 0))

        # ── Соседние РВС ─────────────────────────────────────────────────────
        for o in objects:
            if o["otype"] == "rvs_near":
                ox = int(o["x"] * sx);  oy = int(o["y"] * sy)
                r  = max(15, int(o["r"] * (sx + sy) / 2))
                # Обвалование соседнего
                c.create_oval(ox-r-22, oy-r-22, ox+r+22, oy+r+22,
                              outline="#8B7355", width=2, dash=(4,4), fill="")
                if sim.n_trunks_nbr > 0:
                    c.create_oval(ox-r-8, oy-r-8, ox+r+8, oy+r+8,
                                  outline=P["water"], width=2, dash=(3, 3))
                c.create_oval(ox-r, oy-r, ox+r, oy+r,
                              fill=P["rvs17"], outline="#3e86b5", width=2)
                # Штриховка
                for hy in range(oy-r+5, oy+r-3, 6):
                    hw = int(math.sqrt(max(0, r**2 - (hy-oy)**2))) - 2
                    if hw > 2:
                        c.create_line(ox-hw, hy, ox+hw, hy, fill="#5599cc", width=1)
                lbl = o.get("label", "Соседний РВС")[:18]
                c.create_text(ox, oy-4, text=lbl, fill="white", font=("Arial", 8, "bold"))
                self._map_label(c, ox, oy+r+14, "орошение",
                                P["water"], ("Arial", 7, "italic"), bg="#e8ece0")
                # Стволы
                nbr_pts = [(ox+r+16, oy), (ox-r-16, oy),
                           (ox, oy+r+16), (ox+r+10, oy-r-10), (ox-r-10, oy-r-10)]
                for tx, ty in nbr_pts[:sim.n_trunks_nbr]:
                    c.create_line(tx, ty, ox+.4*(tx-ox), oy+.4*(ty-oy),
                                  fill="#4488ff", width=2, arrow="last", arrowshape=(8,10,4))
                    c.create_oval(tx-5, ty-5, tx+5, ty+5, fill="#4488ff", outline="white", width=1)

        # ── Горящий РВС ──────────────────────────────────────────────────────
        # Зона теплового воздействия
        if fire_intensity > 0:
            heat_r = r9 + 26 + int(12 * fire_intensity)
            c.create_oval(cx9-heat_r, cy9-heat_r, cx9+heat_r, cy9+heat_r,
                          outline="#ff6633", width=1, dash=(2, 4), fill="")
            # Дымовой шлейф
            for i in range(3):
                dx = 16 + i*20 + 4*math.sin(self._anim_t*0.15 + i)
                dy = -(18 + i*16 + 3*math.cos(self._anim_t*0.12 + i))
                sr = 12 + i*9
                c.create_oval(cx9+dx-sr, cy9+dy-sr, cx9+dx+sr, cy9+dy+sr,
                              fill="", outline=P["smoke"], width=1, dash=(2, 5))

        if fire_intensity > 0:
            fire_clr = P["fire"] if fire_intensity > 0.65 else P["fire2"]
            c.create_oval(cx9-r9, cy9-r9, cx9+r9, cy9+r9,
                          fill=fire_clr, outline="")
            pulse = 3.5 * math.sin(self._anim_t * 0.35)
            for k in range(12):
                ang = math.radians(k * 30 + self._anim_t * 2.5)
                dist = r9 + 8 + pulse + 3*math.sin(self._anim_t*0.5 + k*1.7)
                fx = cx9 + dist * math.cos(ang)
                fy = cy9 + dist * math.sin(ang)
                fs = 4 + int(4 * fire_intensity)
                fl_clr = P["flame"] if k % 2 == 0 else "#ff6600"
                c.create_oval(fx-fs, fy-fs, fx+fs, fy+fs, fill=fl_clr, outline="")
            core_r = int(r9 * 0.45)
            c.create_oval(cx9-core_r, cy9-core_r, cx9+core_r, cy9+core_r,
                          fill=P["flame"], outline="")
        else:
            c.create_oval(cx9-r9, cy9-r9, cx9+r9, cy9+r9,
                          fill=P["rvs9_cool"], outline="#888866", width=2)
            for hy in range(cy9-r9+5, cy9+r9-3, 6):
                hw = int(math.sqrt(max(0, r9**2 - (hy-cy9)**2))) - 2
                if hw > 2:
                    c.create_line(cx9-hw, hy, cx9+hw, hy, fill="#776655", width=1)

        c.create_oval(cx9-r9, cy9-r9, cx9+r9, cy9+r9,
                      outline="#cc2222" if fire_intensity > 0 else "#666644",
                      width=3, fill="")
        c.create_text(cx9, cy9-10, text=rvs_name[:18], fill="white",
                      font=("Arial", 10, "bold"))
        c.create_text(cx9, cy9+6, text=f"∅{rvs_diam:.0f} м",
                      fill="white", font=("Arial", 8))
        area_clr = P["hi"] if fire_intensity > 0 else P["success"]
        self._map_label(c, cx9, cy9+r9+10,
                        f"S = {sim.fire_area:.0f} м²",
                        area_clr, ("Arial", 8, "bold"), bg="#e8ece0")

        # Розлив
        if sim.spill:
            sp_pts = [cx9-r9-18, cy9+r9-6, cx9-r9-30, cy9+r9+12,
                      cx9-r9-24, cy9+r9+30, cx9-8, cy9+r9+34,
                      cx9+12, cy9+r9+18, cx9+4, cy9+r9-2]
            c.create_polygon(sp_pts, fill="#ff4400", outline="#ff6600",
                             width=2, stipple="gray50")
            self._map_label(c, cx9-r9-5, cy9+r9+40,
                            f"Розлив {sim.spill_area:.0f} м²",
                            P["danger"], ("Arial", 8, "bold"), bg="#e8ece0")

        # ── Водоснабжение ────────────────────────────────────────────────────
        water_srcs = [o for o in objects if o["otype"] in ("river", "hydrant")]
        for i, ws in enumerate(water_srcs[:sim.n_pns]):
            wx = int(ws["x"] * sx);  wy = int(ws["y"] * sy)
            c.create_line(wx, wy, cx9, cy9+r9+5, fill=P["water"], width=3, dash=(10, 5))
            for dot_i in range(3):
                frac = ((self._anim_t * 0.04 + dot_i * 0.33) % 1.0)
                dx = wx + (cx9-wx) * frac
                dy = wy + (cy9+r9+5-wy) * frac
                c.create_oval(dx-3, dy-3, dx+3, dy+3, fill="white",
                              outline=P["water"], width=1)
            self._map_pump_icon(c, wx, wy-16, f"ПНС-{i+1}")

        # ── Стволы охлаждения ────────────────────────────────────────────────
        _d = int(r9 * 0.65)
        trunk_pos = [
            (cx9, cy9+r9+22), (cx9+r9+22, cy9), (cx9-r9-22, cy9),
            (cx9, cy9-r9-22), (cx9+_d+6, cy9+_d+6), (cx9-_d-6, cy9+_d+6),
            (cx9+_d+6, cy9-_d-6),
        ]
        for i, (tx, ty) in enumerate(trunk_pos[:sim.n_trunks_burn]):
            mx = cx9 + 0.4 * (tx - cx9)
            my = cy9 + 0.4 * (ty - cy9)
            c.create_line(tx, ty, mx, my, fill=P["water"], width=3,
                          arrow="last", arrowshape=(10, 12, 5))
            c.create_oval(tx-7, ty-7, tx+7, ty+7, fill=P["unit_pns"],
                          outline="white", width=1)
            c.create_text(tx, ty, text=f"{i+1}", fill="white",
                          font=("Arial", 7, "bold"))

        # ── Боевые участки ───────────────────────────────────────────────────
        bu_defs = [
            (cx9, cy9+dike_r+16, "#e74c3c", "БУ-1"),
            (cx9+dike_r+18, cy9, "#3498db", "БУ-2"),
            (cx9-dike_r-18, cy9, "#2ecc71", "БУ-3"),
        ]
        for i, (bx, by, bc, bn) in enumerate(bu_defs[:sim.n_bu]):
            c.create_rectangle(bx-40, by-12, bx+40, by+12,
                               fill=bc, outline="#fff", width=1)
            c.create_text(bx, by, text=bn, fill="white", font=("Arial", 8, "bold"))

        # ── Компас ───────────────────────────────────────────────────────────
        ncx, ncy = W-40, 36
        c.create_line(ncx, ncy+12, ncx, ncy-12, fill=P["text"], width=2,
                      arrow="first", arrowshape=(8, 10, 4))
        c.create_text(ncx, ncy-20, text="С", fill=P["text"], font=("Arial", 8, "bold"))

        # ── Баннер статуса ───────────────────────────────────────────────────
        status_text = ""
        status_color = P["text"]
        if sim.extinguished:
            status_text = "ПОЖАР ЛИКВИДИРОВАН"
            status_color = P["success"]
        elif sim.localized:
            status_text = "ПОЖАР ЛОКАЛИЗОВАН"
            status_color = P["info"]
        elif sim.foam_ready:
            status_text = "ГОТОВНОСТЬ К ПЕННОЙ АТАКЕ"
            status_color = P["warn"]
        if status_text:
            c.create_rectangle(W//2-200, 2, W//2+200, 26,
                               fill=status_color, outline="#fff", width=1)
            c.create_text(W//2, 14, text=status_text,
                          fill="white", font=("Arial", 11, "bold"))
        else:
            self._map_label(c, W//2, 8, f"Резервуарный парк — {cfg['short']}",
                            P["text"], ("Arial", 9, "bold"), bg="#e8ece0")

        self._update_map_info()

    # ─────────────────────────────────────────────────────────────────────────
    # ОБНОВЛЕНИЕ ГРАФИКОВ
    # ─────────────────────────────────────────────────────────────────────────
    def _update_charts(self):
        sim = self.sim
        if not sim.h_fire:
            return

        # ── Метрики: извлечь временны́е ряды из истории симуляции ────────────
        ts_f  = [x[0] for x in sim.h_fire]
        val_f = [x[1] for x in sim.h_fire]
        ts_w  = [x[0] for x in sim.h_water]
        val_w = [x[1] for x in sim.h_water]
        ts_r  = [x[0] for x in sim.h_risk]
        val_r = [x[1] for x in sim.h_risk]
        ts_t  = [x[0] for x in sim.h_trunks]
        val_t = [x[1] for x in sim.h_trunks]
        ts_ph  = [x[0] for x in sim.h_phase]
        val_ph = [x[1] for x in sim.h_phase]
        ts_fc  = [x[0] for x in sim.h_foam_conc]
        val_fc = [x[1] for x in sim.h_foam_conc]

        _xlim = max((ts_f or ts_r or [100])[-1], 100)

        def _style_ax(ax):
            ax.set_facecolor(P["canvas"])
            ax.tick_params(colors=P["text2"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(P["grid"])
            ax.grid(True, color=P["grid"], linewidth=0.4, alpha=0.7)
            ax.xaxis.label.set_color(P["text2"])
            ax.yaxis.label.set_color(P["text2"])

        for ax in [self._ax_fire, self._ax_water, self._ax_trunks,
                   self._ax_risk, self._ax_phase, self._ax_foam_conc]:
            ax.cla()
            _style_ax(ax)

        # Площадь пожара
        ax = self._ax_fire
        ax.plot(ts_f, val_f, color=P["fire"], linewidth=1.5)
        ax.fill_between(ts_f, val_f, alpha=0.2, color=P["fire"])
        # Пенные атаки — показать число на графике
        if sim.foam_attacks > 0:
            ax.annotate(f"Пен. атак: {sim.foam_attacks}",
                        xy=(0.97, 0.90), xycoords="axes fraction",
                        ha="right", fontsize=7, color=P["warn"])
        ax.set_title("Площадь пожара", color=P["text"], fontsize=8, pad=3)
        ax.set_ylabel("м²", color=P["text2"], fontsize=7)
        ax.set_xlabel("время, мин", color=P["text2"], fontsize=7)
        ax.set_xlim(0, _xlim)

        # Расход ОВ
        ax = self._ax_water
        ax.plot(ts_w, val_w, color=P["water"], linewidth=1.5)
        ax.fill_between(ts_w, val_w, alpha=0.2, color=P["water"])
        ax.set_title("Расход огнетушащего вещества", color=P["text"], fontsize=8, pad=3)
        ax.set_ylabel("л/с", color=P["text2"], fontsize=7)
        ax.set_xlabel("время, мин", color=P["text2"], fontsize=7)
        ax.set_xlim(0, _xlim)

        # Стволы
        ax = self._ax_trunks
        ax.step(ts_t, val_t, color=P["unit_pns"], linewidth=1.5, where="post")
        ax.fill_between(ts_t, val_t, alpha=0.2, color=P["unit_pns"], step="post")
        ax.axhline(7, color=P["success"], linewidth=0.8, linestyle=":", alpha=0.8,
                   label="цель: 7 ств.")
        ax.set_title("Стволов (горящий РВС)", color=P["text"], fontsize=8, pad=3)
        ax.set_ylabel("шт.", color=P["text2"], fontsize=7)
        ax.set_xlabel("время, мин", color=P["text2"], fontsize=7)
        ax.legend(fontsize=6, facecolor=P["canvas"], edgecolor=P["grid"],
                  labelcolor=P["text2"])
        ax.set_xlim(0, _xlim)

        # Риск
        ax = self._ax_risk
        ax.plot(ts_r, val_r, color=P["danger"], linewidth=1.5)
        ax.fill_between(ts_r, val_r, alpha=0.15, color=P["danger"])
        ax.axhline(0.75, color=P["danger"], linewidth=0.8, linestyle=":",
                   alpha=0.7, label="критич. 0.75")
        ax.set_ylim(0, 1.05)
        ax.set_title("Индекс риска", color=P["text"], fontsize=8, pad=3)
        ax.set_ylabel("0–1", color=P["text2"], fontsize=7)
        ax.set_xlabel("время, мин", color=P["text2"], fontsize=7)
        ax.legend(fontsize=6, facecolor=P["canvas"], edgecolor=P["grid"],
                  labelcolor=P["text2"])
        ax.set_xlim(0, _xlim)

        # Фаза пожара
        ax = self._ax_phase
        _phase_labels = {1:"S1",2:"S2",3:"S3",4:"S4",5:"S5"}
        _phase_colors = {1:P["info"],2:P["warn"],3:P["fire"],4:P["tact"],5:P["success"]}
        if ts_ph:
            for i in range(1, len(ts_ph)):
                pv = val_ph[i-1]
                ax.fill_between([ts_ph[i-1], ts_ph[i]], [pv, pv],
                                color=_phase_colors.get(pv, P["text2"]), alpha=0.55)
            ax.step(ts_ph, val_ph, color=P["text"], linewidth=1.0, where="post")
        ax.set_yticks([1,2,3,4,5])
        ax.set_yticklabels(["S1","S2","S3","S4","S5"], fontsize=7, color=P["text2"])
        ax.set_ylim(0.5, 5.5)
        ax.set_title("Фаза пожара", color=P["text"], fontsize=8, pad=3)
        ax.set_ylabel("фаза", color=P["text2"], fontsize=7)
        ax.set_xlabel("время, мин", color=P["text2"], fontsize=7)
        ax.set_xlim(0, _xlim)

        # Запас пенообразователя
        ax = self._ax_foam_conc
        ax.plot(ts_fc, val_fc, color=P["success"], linewidth=1.5)
        ax.fill_between(ts_fc, val_fc, alpha=0.3, color=P["success"])
        ax.axhline(0, color=P["danger"], linewidth=0.8, linestyle=":", alpha=0.7,
                   label="исчерпан")
        ax.set_title("Запас пенообразователя", color=P["text"], fontsize=8, pad=3)
        ax.set_ylabel("т", color=P["text2"], fontsize=7)
        ax.set_xlabel("время, мин", color=P["text2"], fontsize=7)
        ax.legend(fontsize=6, facecolor=P["canvas"], edgecolor=P["grid"],
                  labelcolor=P["text2"])
        ax.set_xlim(0, _xlim)

        self._fc_metrics.draw()

        # ── RL-агент: Q-значения, частота действий, кривая наград ────────────
        for ax in [self._ax_qval, self._ax_actcnt, self._ax_reward]:
            ax.cla()
            ax.set_facecolor(P["canvas"])
            ax.tick_params(colors=P["text2"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(P["grid"])
            ax.grid(True, color=P["grid"], linewidth=0.4, alpha=0.7)

        # Q-значения текущего состояния: высота столбца = ценность действия по Q-таблице
        ax = self._ax_qval
        # Захватить снимок Q-данных под блокировкой, чтобы избежать race condition
        with self._train_lock:
            qv   = sim.agent.q_values(sim._state())
        codes = [a[0] for a in ACTIONS]
        cols  = [LEVEL_C[a[1]] for a in ACTIONS]
        bars = ax.bar(range(N_ACT), qv, color=cols, alpha=0.85, width=0.7)
        # Выделить текущее выбранное действие золотой рамкой
        bars[sim.last_action].set_edgecolor(P["hi"])
        bars[sim.last_action].set_linewidth(2)
        ax.set_xticks(range(N_ACT))
        ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=7,
                           color=P["text2"])
        ax.set_title("Q-значения действий (текущее состояние)", color=P["text"],
                     fontsize=8, pad=3)
        # Легенда уровней
        patches = [mpatches.Patch(color=P["strat"], label="Стратег."),
                   mpatches.Patch(color=P["tact"],  label="Тактич."),
                   mpatches.Patch(color=P["oper"],  label="Оперативн.")]
        ax.legend(handles=patches, fontsize=7, facecolor=P["canvas"],
                  edgecolor=P["grid"], labelcolor=P["text2"], loc="upper left")

        # Частота выбора действий
        ax = self._ax_actcnt
        with self._train_lock:
            cnt = sim.agent.action_counts.copy()
        if cnt.sum() > 0:
            ax.bar(range(N_ACT), cnt / max(cnt.sum(), 1), color=cols, alpha=0.8, width=0.7)
        ax.set_xticks(range(N_ACT))
        ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=7, color=P["text2"])
        ax.set_title("Частота выбора действий", color=P["text"], fontsize=8, pad=3)

        # Кривая накопленных наград (последние 500 шагов) + скользящее среднее
        ax = self._ax_reward
        if sim.h_reward:
            rw = sim.h_reward[-500:]   # последние 500 шагов
            cumrew = np.cumsum(rw)
            ax.plot(cumrew, color=P["success"], linewidth=1)
            # Скользящее среднее (окно 20) — сглаживает шум отдельных шагов
            if len(rw) > 20:
                window = 20
                ma = np.convolve(rw, np.ones(window)/window, mode="valid")
                ax.plot(range(window-1, len(rw)), np.cumsum(rw[window-1:]) + (cumrew[window-1] if window > 0 else 0), color=P["warn"], linewidth=0.5, alpha=0.5)
        ax.set_title(f"Накопленная награда (ε={sim.agent.epsilon:.2f})",
                     color=P["text"], fontsize=8, pad=3)

        self._fc_rl.draw()

    # ─────────────────────────────────────────────────────────────────────────
    # ОБНОВЛЕНИЕ СТАТУСА
    # ─────────────────────────────────────────────────────────────────────────
    def _update_status(self):
        sim = self.sim
        sv  = self._status_vars

        # Основные оперативные показатели
        sv["sim_time"].set(self._fmt_time(sim.t))
        sv["phase"].set(PHASE_NAMES.get(sim.phase, sim.phase))
        sv["fire_area"].set(f"{sim.fire_area:.0f} м²")
        sv["flow"].set(f"{sim.water_flow:.0f} л/с")
        sv["trunks"].set(f"{sim.n_trunks_burn} (РВС№9)  +  {sim.n_trunks_nbr} (РВС№17)")
        sv["pns"].set(f"{sim.n_pns} из 4")
        sv["bu"].set(f"{sim.n_bu} из 3")
        # Суффикс показывает итоговый статус: ✅ ликвидирован / 🔒 локализован / ⏳ готовность
        sv["foam"].set(f"{sim.foam_attacks}" + ("  ✅" if sim.extinguished else ("  🔒" if sim.localized else "  ⏳" if sim.foam_ready else "")))

        # Индекс риска → текстовая метка уровня угрозы
        risk = sim._risk()
        risk_str = "КРИТИЧЕСКИЙ" if risk > 0.75 else \
                   "ВЫСОКИЙ"     if risk > 0.50 else \
                   "СРЕДНИЙ"     if risk > 0.25 else "НИЗКИЙ"
        sv["risk"].set(f"{risk_str} ({risk:.2f})")

        code, level, desc = ACTIONS[sim.last_action]
        sv["action"].set(f"[{code}] {desc[:30]}")

        # Физика пенной атаки (ГОСТ Р 51043): текущий расход vs. нормативный минимум
        roof_pct = sim.roof_obstruction * 100
        akp_mark = " (АКП-50 ✅)" if sim.akp50_available else ""
        sv["roof_obs"].set(f"{roof_pct:.0f}%{akp_mark}")
        q_foam = sim.foam_flow_ls
        q_req  = _RVS9_PARAMS.foam_flow_required_ls()
        if q_foam > 0:
            sv["foam_flow"].set(f"{q_foam:.0f} л/с  (норм.≥{q_req:.0f})")
        else:
            sv["foam_flow"].set(f"—  (норм.≥{q_req:.0f} л/с)")

        # Прогресс-бар: доля прошедшего модельного времени от общей длительности сценария
        total = sim._cfg["total_min"]
        self._prog_bar.config(maximum=total)
        pct = min(100, int(100 * sim.t / total))
        self._prog_var.set(f"{sim.t} / {total} мин  ({pct}%)")
        self._prog_bar["value"] = sim.t

    def _log_sim_event(self, t: int, color: str, text: str):
        """Добавить запись в журнал хронологии."""
        pass  # события уже в sim.events, отображаются при необходимости

    @staticmethod
    def _fmt_time(t: int) -> str:
        """t (мин от начала) → строку вида 'Ч+N (Nч Nмин)'"""
        h = t // 60
        m = t % 60
        if h > 0:
            return f"Ч+{t} ({h}ч {m:02d}мин)"
        return f"Ч+{t} ({m}мин)"

    # ─────────────────────────────────────────────────────────────────────────
    # АНИМАЦИОННЫЙ ЦИКЛ
    # ─────────────────────────────────────────────────────────────────────────
    def _animate(self):
        if not self._running:
            return

        # ── ТРЕНАЖЁР: только визуальная анимация, без шага симуляции ─────────
        if self._mode == "trainer":
            self._anim_t += 1
            self._draw_map()
            self._after_id = self.after(150, self._animate)
            return

        # ── СППР / ИССЛЕДОВАНИЕ: стандартный цикл ────────────────────────────
        ended = False
        for _ in range(self._speed):
            if self.sim.t >= self.sim._cfg["total_min"] or self.sim.extinguished:
                ended = True
                break
            # В режиме СППР: если пользователь выбрал override — применяем его
            if self._mode == "sppр" and self._sppр_override_pending is not None:
                override_a = self._sppр_override_pending
                self._sppр_override_pending = None
                # Применить ручное действие, продвинуть время
                self.sim._update_fire()
                self.sim._update_phase()
                r = self.sim._apply(override_a)
                self.sim.last_action = override_a
                self.sim.h_reward.append(r)
                self.sim.t += self.STEP_MIN
                self._sppр_log_step(override_a, accepted=False)
            else:
                self._snap = self.sim.step(dt=self.STEP_MIN)
                if self._mode == "sppр":
                    # Логируем: агент сам выбрал → "принято автоматически"
                    self._sppр_log_step(self.sim.last_action, accepted=True)

        if ended:
            self._on_pause()
            self._record_sim_run()
            if self._mode == "sppр":
                self._sppр_show_summary()
            return

        self._anim_t += 1
        self._draw_map()
        self._update_status()

        # Обновить панель рекомендации СППР
        if self._mode == "sppр":
            self._sppр_update_rec()

        if self._anim_t % 25 == 0:
            self._update_charts()

        self._after_id = self.after(self.TICK_MS, self._animate)

    # ── Обработчики кнопок ───────────────────────────────────────────────────
    def _on_play(self):
        if not self._running:
            self._running = True
            if self._mode == "trainer":
                self._trainer_active = True
                self._trainer_update_panel()
            self._animate()

    def _on_pause(self):
        self._running = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._mode == "trainer":
            self._trainer_active = False
            self._trainer_update_panel()
        self._update_charts()

    def _on_step(self):
        """Выполнить один шаг симуляции (пошаговый режим)."""
        if self._running:
            self._on_pause()
        if self.sim.t >= self.sim._cfg["total_min"] or self.sim.extinguished:
            return
        self._snap = self.sim.step(dt=self.STEP_MIN)
        if self._mode == "sppр":
            self._sppр_log_step(self.sim.last_action, accepted=True)
            self._sppр_update_rec()
        self._anim_t += 1
        self._draw_map()
        self._update_status()
        self._update_charts()

    def _on_reset(self):
        self._on_pause()
        self.sim.reset()
        self._trainer_score = 0
        self._trainer_steps = 0
        self._trainer_log   = []
        self._trainer_active = False
        self._sppр_deviations = 0
        self._sppр_total      = 0
        self._sppр_log        = []
        if self._mode == "trainer":
            self._trainer_update_panel()
        elif self._mode == "sppр":
            self._sppр_update_rec()
        self._anim_t = 0
        self._draw_map()
        self._update_status()
        self._update_charts()

    def _on_speed_change(self):
        self._speed = self.SPEEDS.get(self._speed_var.get(), 15)

    # =========================================================================
    # РЕЖИМ ТРЕНАЖЁРА (trainer)
    # =========================================================================

    def _build_trainer_panel(self, parent):
        """Панель тренажёра РТП: кнопки выбора действия + счёт."""
        _, accent, bg_clr = APP_MODES["trainer"]

        outer = tk.LabelFrame(parent,
                              text="  🎓 ТРЕНАЖЁР: Выберите действие РТП  ",
                              bg=bg_clr, fg=accent,
                              font=("Arial", 9, "bold"),
                              bd=2, relief="groove")
        outer.pack(fill="x", padx=4, pady=(2, 0))

        # ── Шапка: фаза, подсказка, счёт ─────────────────────────────────────
        top_row = tk.Frame(outer, bg=bg_clr)
        top_row.pack(fill="x", padx=6, pady=(4, 2))

        self._tr_phase_var = tk.StringVar(value="Нажмите ▶ Пуск для начала тренажёра")
        tk.Label(top_row, textvariable=self._tr_phase_var,
                 bg=bg_clr, fg=accent, font=("Arial", 9, "bold")).pack(side="left")

        score_frame = tk.Frame(top_row, bg=bg_clr)
        score_frame.pack(side="right")
        tk.Label(score_frame, text="Счёт:", bg=bg_clr, fg=P["text2"],
                 font=("Arial", 9)).pack(side="left")
        self._tr_score_var = tk.StringVar(value="0")
        tk.Label(score_frame, textvariable=self._tr_score_var,
                 bg=bg_clr, fg=accent, font=("Arial", 12, "bold"), width=5).pack(side="left")

        tk.Label(score_frame, text="  Шагов:", bg=bg_clr, fg=P["text2"],
                 font=("Arial", 9)).pack(side="left")
        self._tr_steps_var = tk.StringVar(value="0")
        tk.Label(score_frame, textvariable=self._tr_steps_var,
                 bg=bg_clr, fg=P["text"], font=("Arial", 9)).pack(side="left")

        # ── Фидбэк последнего шага ────────────────────────────────────────────
        self._tr_feedback_var = tk.StringVar(value="")
        self._tr_feedback_lbl = tk.Label(outer, textvariable=self._tr_feedback_var,
                                          bg=bg_clr, fg=P["success"],
                                          font=("Arial", 9, "bold"), anchor="w")
        self._tr_feedback_lbl.pack(fill="x", padx=8, pady=(0, 2))

        # ── Сетка кнопок действий ─────────────────────────────────────────────
        btn_frame = tk.Frame(outer, bg=bg_clr)
        btn_frame.pack(fill="x", padx=4, pady=(2, 6))
        self._tr_action_btns: List[tk.Button] = []

        LEVEL_BG = {
            "стратег.":    "#e8d5f5",
            "тактич.":     "#fde8cc",
            "оперативн.":  "#d6eaf8",
        }
        LEVEL_FG = {
            "стратег.":    "#6c3483",
            "тактич.":     "#935116",
            "оперативн.":  "#154360",
        }

        for i, (code, level, desc) in enumerate(ACTIONS):
            r, c = divmod(i, 5)
            btn = tk.Button(
                btn_frame,
                text=f"[{code}]\n{desc[:22]}",
                font=("Arial", 7),
                bg=LEVEL_BG[level], fg=LEVEL_FG[level],
                relief="flat", bd=1,
                wraplength=110, justify="center",
                padx=2, pady=3,
                state="disabled",
                command=lambda idx=i: self._trainer_step(idx),
            )
            btn.grid(row=r, column=c, padx=2, pady=2, sticky="nsew")
            self._tr_action_btns.append(btn)
            btn_frame.columnconfigure(c, weight=1)

        # ── Кнопка итогового дебрифинга ───────────────────────────────────────
        self._tr_debrief_btn = tk.Button(
            outer, text="📋  Итоговый разбор действий",
            command=self._show_debriefing,
            bg=accent, fg="white",
            font=("Arial", 9, "bold"),
            relief="flat", padx=10, pady=4,
            state="disabled",
        )
        self._tr_debrief_btn.pack(anchor="e", padx=8, pady=(0, 4))

    def _trainer_update_panel(self):
        """Обновить состояние кнопок тренажёра под текущую фазу."""
        if not hasattr(self, "_tr_action_btns"):
            return
        sim = self.sim
        active = self._trainer_active and not sim.extinguished

        mask = sim._mask()
        for i, btn in enumerate(self._tr_action_btns):
            if active:
                new_state = "normal" if mask[i] else "disabled"
                btn.config(state=new_state)
            else:
                btn.config(state="disabled")

        phase_label = PHASE_NAMES.get(sim.phase, sim.phase)
        if active:
            self._tr_phase_var.set(
                f"Фаза {phase_label}  |  t={self._fmt_time(sim.t)}  — выберите действие:")
        else:
            self._tr_phase_var.set(
                "Нажмите ▶ Пуск для начала тренажёра" if not sim.extinguished
                else "✅ Сценарий завершён — нажмите «Итоговый разбор»")

        self._tr_score_var.set(str(self._trainer_score))
        self._tr_steps_var.set(str(self._trainer_steps))

        # Разблокировать кнопку дебрифинга если есть шаги
        if self._trainer_steps > 0:
            self._tr_debrief_btn.config(state="normal")

    def _trainer_step(self, user_action_idx: int):
        """Пользователь выбрал действие — оцениваем и делаем шаг симуляции."""
        if not self._trainer_active:
            return
        sim = self.sim

        # Лучшее действие по Q-таблице (greedy, без ε)
        qv   = sim.agent.q_values(sim._state())
        mask = sim._mask()
        q_masked = qv.copy()
        q_masked[~mask] = -1e9
        best_a = int(np.argmax(q_masked))

        # Подсчёт баллов
        sorted_q = sorted(enumerate(q_masked), key=lambda x: -x[1])
        top3 = [idx for idx, _ in sorted_q[:3]]

        if user_action_idx == best_a:
            pts = _T_PERFECT
            verdict = f"✅ Отлично! +{pts}  [{ACTIONS[user_action_idx][0]}] = оптимальное действие"
            fg = P["success"]
        elif user_action_idx in top3:
            pts = _T_GOOD
            verdict = f"👍 Хорошо! +{pts}  [{ACTIONS[user_action_idx][0]}] в топ-3 по Q-ценности"
            fg = P["warn"]
        elif mask[user_action_idx]:
            pts = _T_VALID
            verdict = f"⚠ Допустимо, но лучше [{ACTIONS[best_a][0]}]  +{pts}"
            fg = P["accent"]
        else:
            pts = _T_WRONG
            verdict = f"❌ Ошибка! Действие недопустимо в фазе {sim.phase}  {pts}"
            fg = P["danger"]

        self._trainer_score += pts
        self._trainer_steps += 1
        self._trainer_log.append({
            "t": sim.t, "phase": sim.phase,
            "user_a": user_action_idx, "best_a": best_a,
            "pts": pts, "q_user": float(qv[user_action_idx]),
            "q_best": float(qv[best_a]),
        })

        # Показать фидбэк
        self._tr_feedback_var.set(verdict)
        self._tr_feedback_lbl.config(fg=fg)

        # Шаг симуляции с выбранным действием
        sim._update_fire()
        sim._update_phase()
        r = sim._apply(user_action_idx)
        sim.last_action = user_action_idx
        sim.h_reward.append(r)
        sim.t += self.STEP_MIN

        self._draw_map()
        self._update_status()
        self._trainer_update_panel()

        # Сценарий завершён
        if sim.extinguished or sim.t >= sim._cfg["total_min"]:
            self._trainer_active = False
            self._trainer_update_panel()
            self._autosave_trainer_log()
            self.after(800, self._show_debriefing)

    def _record_sim_run(self):
        """Записать итог прогона (research/sppр) в runs.json."""
        import datetime
        sim = self.sim
        record = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "mode": self._mode,
            "scenario": self._scenario_key,
            "duration_min": sim.t,
            "extinguished": sim.extinguished,
            "localized": sim.localized,
            "foam_attacks": sim.foam_attacks,
            "risk_max": round(max((x[1] for x in sim.h_risk), default=0.0), 3),
            "sppр_deviations": self._sppр_deviations if self._mode == "sppр" else None,
        }
        self._save_run_to_db(record)

    def _autosave_trainer_log(self):
        """Сохранить журнал тренажёра в JSON и записать результат в runs.json."""
        import datetime
        record = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "mode": "trainer",
            "scenario": self._scenario_key,
            "score": self._trainer_score,
            "steps": self._trainer_steps,
            "duration_min": self.sim.t,
            "extinguished": self.sim.extinguished,
            "log": list(self._trainer_log),
        }
        # Дописать в файл истории тренажёра
        history = []
        if os.path.exists(_TRAINER_LOG_PATH):
            try:
                with open(_TRAINER_LOG_PATH, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(record)
        try:
            with open(_TRAINER_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        self._save_run_to_db(record)

    def _save_run_to_db(self, record: dict):
        """Добавить краткий результат прогона в базу runs.json."""
        runs = []
        if os.path.exists(_RUNS_DB_PATH):
            try:
                with open(_RUNS_DB_PATH, "r", encoding="utf-8") as f:
                    runs = json.load(f)
            except Exception:
                runs = []
        summary = {k: record[k] for k in
                   ("timestamp", "mode", "scenario", "score", "steps",
                    "duration_min", "extinguished") if k in record}
        runs.append(summary)
        try:
            with open(_RUNS_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(runs, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _show_debriefing(self):
        """Окно итогового разбора тренажёра."""
        if not self._trainer_log:
            return
        win = tk.Toplevel(self)
        win.title("Разбор тренажёра — САУР-ПСП")
        win.geometry("800x580")
        win.configure(bg=P["bg"])
        win.grab_set()

        _, accent, bg_clr = APP_MODES["trainer"]

        # Заголовок
        hdr = tk.Frame(win, bg=accent, height=50)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🎓  ИТОГОВЫЙ РАЗБОР ТРЕНАЖЁРА РТП",
                 font=("Arial", 13, "bold"), bg=accent, fg="white").pack(side="left", padx=16, pady=8)
        total_possible = self._trainer_steps * _T_PERFECT
        pct = int(100 * self._trainer_score / total_possible) if total_possible > 0 else 0
        rank = ("Эксперт 🏆" if pct >= 90 else
                "Опытный ✅" if pct >= 70 else
                "Средний ⚠" if pct >= 50 else "Начинающий ❌")
        tk.Label(hdr, text=f"Итог: {self._trainer_score}/{total_possible} ({pct}%)  |  {rank}",
                 font=("Arial", 10, "bold"), bg=accent, fg="#dde8f0").pack(side="right", padx=16)

        # Сводка
        summary = tk.Frame(win, bg=bg_clr, pady=4)
        summary.pack(fill="x", padx=10, pady=(8, 0))
        stats = [
            f"Шагов: {self._trainer_steps}",
            f"Оптимальных: {sum(1 for e in self._trainer_log if e['pts']==_T_PERFECT)}",
            f"Хороших: {sum(1 for e in self._trainer_log if e['pts']==_T_GOOD)}",
            f"Допустимых: {sum(1 for e in self._trainer_log if e['pts']==_T_VALID)}",
            f"Ошибочных: {sum(1 for e in self._trainer_log if e['pts']<0)}",
        ]
        for s in stats:
            tk.Label(summary, text=s, bg=bg_clr, fg=P["text"],
                     font=("Arial", 9)).pack(side="left", padx=12)

        # Таблица шагов
        cols_frame = tk.Frame(win, bg=P["bg"])
        cols_frame.pack(fill="both", expand=True, padx=10, pady=6)
        sb = tk.Scrollbar(cols_frame, orient="vertical")
        sb.pack(side="right", fill="y")
        txt = tk.Text(cols_frame, yscrollcommand=sb.set,
                      bg=P["canvas"], fg=P["text"],
                      font=("Consolas", 8), wrap="none", padx=6, pady=4)
        sb.config(command=txt.yview)
        txt.pack(fill="both", expand=True)
        txt.tag_config("perfect", foreground=P["success"], font=("Consolas", 8, "bold"))
        txt.tag_config("good",    foreground=P["warn"])
        txt.tag_config("valid",   foreground=P["text2"])
        txt.tag_config("wrong",   foreground=P["danger"], font=("Consolas", 8, "bold"))
        txt.tag_config("header",  foreground=P["hi"], font=("Consolas", 8, "bold"))

        header = f"{'Шаг':>4}  {'Время':>8}  {'Фаза':>3}  {'Вы':>4}  {'Лучш.':>5}  {'Q-вы':>7}  {'Q-луч':>7}  {'Балл':>5}  Статус\n"
        txt.insert("end", header, "header")
        txt.insert("end", "─" * 80 + "\n", "header")

        for i, e in enumerate(self._trainer_log, 1):
            ua_code = ACTIONS[e["user_a"]][0]
            ba_code = ACTIONS[e["best_a"]][0]
            status = ("✅ Оптимально" if e["pts"] == _T_PERFECT else
                      "👍 Хорошо"    if e["pts"] == _T_GOOD    else
                      "⚠ Допустимо" if e["pts"] == _T_VALID   else
                      "❌ Ошибка")
            tag = ("perfect" if e["pts"] == _T_PERFECT else
                   "good"    if e["pts"] == _T_GOOD    else
                   "valid"   if e["pts"] == _T_VALID   else "wrong")
            line = (f"{i:>4}  {self._fmt_time(e['t']):>8}  {e['phase']:>3}  "
                    f"{ua_code:>4}  {ba_code:>5}  "
                    f"{e['q_user']:>7.2f}  {e['q_best']:>7.2f}  "
                    f"{e['pts']:>+5}  {status}\n")
            txt.insert("end", line, tag)

        txt.config(state="disabled")

        # Закрыть
        tk.Button(win, text="Закрыть", command=win.destroy,
                  bg=accent, fg="white", font=("Arial", 9, "bold"),
                  relief="flat", padx=16, pady=6).pack(pady=6)

    # =========================================================================
    # РЕЖИМ СППР (sppр)
    # =========================================================================

    def _build_sppр_panel(self, parent):
        """Панель рекомендаций СППР под картой."""
        _, accent, bg_clr = APP_MODES["sppр"]

        outer = tk.LabelFrame(parent,
                              text="  🧭 СППР: Рекомендация агента  ",
                              bg=bg_clr, fg=accent,
                              font=("Arial", 9, "bold"),
                              bd=2, relief="groove")
        outer.pack(fill="x", padx=4, pady=(2, 0))

        # ── Строка рекомендации ───────────────────────────────────────────────
        rec_row = tk.Frame(outer, bg=bg_clr)
        rec_row.pack(fill="x", padx=8, pady=(6, 2))

        self._sppр_rec_var = tk.StringVar(value="Запустите симуляцию (▶ Пуск)")
        self._sppр_rec_lbl = tk.Label(rec_row, textvariable=self._sppр_rec_var,
                                       bg=bg_clr, fg=accent,
                                       font=("Arial", 10, "bold"), anchor="w", wraplength=580)
        self._sppр_rec_lbl.pack(side="left", fill="x", expand=True)

        self._sppр_conf_var = tk.StringVar(value="")
        tk.Label(rec_row, textvariable=self._sppр_conf_var,
                 bg=bg_clr, fg=P["text2"],
                 font=("Arial", 9)).pack(side="right", padx=4)

        # ── Кнопки принятия / переопределения ────────────────────────────────
        btn_row = tk.Frame(outer, bg=bg_clr)
        btn_row.pack(fill="x", padx=8, pady=(0, 4))

        self._sppр_accept_btn = tk.Button(
            btn_row, text="✅  Принять",
            command=self._sppр_accept,
            bg="#27ae60", fg="white",
            font=("Arial", 9, "bold"),
            relief="flat", padx=12, pady=4,
            state="disabled",
        )
        self._sppр_accept_btn.pack(side="left", padx=(0, 6))

        self._sppр_override_btn = tk.Button(
            btn_row, text="✏️  Переопределить действие…",
            command=self._sppр_override,
            bg="#e67e22", fg="white",
            font=("Arial", 9),
            relief="flat", padx=10, pady=4,
            state="disabled",
        )
        self._sppр_override_btn.pack(side="left")

        # Счётчик отклонений
        dev_frame = tk.Frame(btn_row, bg=bg_clr)
        dev_frame.pack(side="right")
        self._sppр_dev_var = tk.StringVar(value="Отклонений: 0 / 0")
        tk.Label(dev_frame, textvariable=self._sppр_dev_var,
                 bg=bg_clr, fg=P["text2"], font=("Arial", 8)).pack(side="right")

        # ── Мини-барграфик Q-значений ─────────────────────────────────────────
        from matplotlib.figure import Figure as _Fig
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FCA
        self._sppр_qfig = _Fig(figsize=(6.5, 1.4), facecolor=bg_clr)
        self._sppр_qax  = self._sppр_qfig.add_subplot(111)
        self._sppр_qax.set_facecolor(bg_clr)
        self._sppр_qfc  = _FCA(self._sppр_qfig, master=outer)
        self._sppр_qfc.get_tk_widget().pack(fill="x", padx=4, pady=(0, 4))
        self._sppр_qfig.tight_layout(pad=0.3)

    def _sppр_update_rec(self):
        """Обновить панель рекомендации агента (вызывается каждый тик)."""
        if not hasattr(self, "_sppр_rec_var"):
            return
        sim  = self.sim
        qv   = sim.agent.q_values(sim._state())
        mask = sim._mask()
        q_m  = qv.copy()
        q_m[~mask] = -1e9
        best_a = int(np.argmax(q_m))
        code, level, desc = ACTIONS[best_a]

        # Уверенность = softmax-вероятность лучшего действия
        q_valid = q_m[mask]
        if len(q_valid) > 0:
            exp_q = np.exp(q_valid - q_valid.max())
            conf  = float(exp_q.max() / exp_q.sum()) * 100
        else:
            conf = 0.0

        self._sppр_rec_var.set(f"Рекомендация: [{code}] {desc}")
        self._sppр_conf_var.set(f"Уверенность: {conf:.0f}%  |  Фаза: {sim.phase}")
        self._sppр_accept_btn.config(state="normal")
        self._sppр_override_btn.config(state="normal")

        # Обновить Q-barplot
        ax = self._sppр_qax
        ax.cla()
        _, accent, bg_clr = APP_MODES["sppр"]
        ax.set_facecolor(bg_clr)
        codes = [a[0] for a in ACTIONS]
        cols  = [LEVEL_C[a[1]] for a in ACTIONS]
        bars  = ax.bar(range(N_ACT), qv, color=cols, alpha=0.75, width=0.7)
        bars[best_a].set_edgecolor(accent)
        bars[best_a].set_linewidth(2)
        ax.set_xticks(range(N_ACT))
        ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=6)
        ax.set_title(f"Q-ценности действий (ε={sim.agent.epsilon:.2f})",
                     fontsize=7, color=P["text"], pad=1)
        ax.tick_params(labelsize=6, colors=P["text2"])
        for sp in ax.spines.values():
            sp.set_visible(False)
        self._sppр_qfig.tight_layout(pad=0.3)
        self._sppр_qfc.draw()

        total = self._sppр_total
        devs  = self._sppр_deviations
        self._sppр_dev_var.set(
            f"Отклонений: {devs} / {total}  ({int(100*devs/total) if total else 0}%)")

    def _sppр_accept(self):
        """Принять текущую рекомендацию (уже была применена агентом — просто логируем)."""
        if not self._running:
            return
        # Принятие регистрируется в _animate через _sppр_log_step

    def _sppр_override(self):
        """Открыть диалог выбора альтернативного действия."""
        _, accent, bg_clr = APP_MODES["sppр"]
        win = tk.Toplevel(self)
        win.title("Переопределить действие РТП")
        win.geometry("460x520")
        win.configure(bg=bg_clr)
        win.grab_set()

        tk.Label(win, text="Выберите действие вместо рекомендованного:",
                 bg=bg_clr, fg=accent,
                 font=("Arial", 10, "bold")).pack(pady=(12, 4), padx=12)

        sim  = self.sim
        mask = sim._mask()
        qv   = sim.agent.q_values(sim._state())

        fr = tk.Frame(win, bg=bg_clr)
        fr.pack(fill="both", expand=True, padx=12)

        LEVEL_BG = {"стратег.": "#e8d5f5", "тактич.": "#fde8cc", "оперативн.": "#d6eaf8"}
        LEVEL_FG = {"стратег.": "#6c3483", "тактич.": "#935116", "оперативн.": "#154360"}

        for i, (code, level, desc) in enumerate(ACTIONS):
            row_bg = LEVEL_BG[level]
            row_fg = LEVEL_FG[level]
            state  = "normal" if mask[i] else "disabled"
            q_str  = f"Q={qv[i]:+.2f}"
            def _click(idx=i):
                self._sppр_override_pending = idx
                self._sppр_deviations += 1
                self._sppр_log.append({
                    "t": sim.t, "phase": sim.phase,
                    "rec_a": int(np.argmax(qv)), "user_a": idx,
                    "accepted": False,
                })
                win.destroy()
            tk.Button(fr, text=f"[{code}]  {desc[:45]}  ({q_str})",
                      font=("Consolas", 8),
                      bg=row_bg, fg=row_fg,
                      relief="flat", anchor="w", padx=8, pady=3,
                      state=state,
                      command=_click
                      ).pack(fill="x", pady=1)

        tk.Button(win, text="Отмена", command=win.destroy,
                  bg=P["panel2"], fg=P["text"],
                  font=("Arial", 9), relief="flat", padx=12, pady=4).pack(pady=8)

    def _sppр_log_step(self, applied_action: int, accepted: bool):
        """Зафиксировать шаг СППР (агент применил действие)."""
        self._sppр_total += 1
        if not accepted:
            self._sppр_deviations += 1

    def _sppр_show_summary(self):
        """Краткое итоговое сообщение СППР."""
        if not hasattr(self, "_sppр_dev_var"):
            return
        total = self._sppр_total
        devs  = self._sppр_deviations
        pct   = int(100 * devs / total) if total > 0 else 0
        msg = (
            f"Симуляция завершена.\n\n"
            f"Шагов всего: {total}\n"
            f"Рекомендаций принято: {total - devs}  ({100-pct}%)\n"
            f"Отклонений: {devs}  ({pct}%)\n\n"
            f"Подробный журнал отклонений доступен в вкладке «Отчёт»."
        )
        import tkinter.messagebox as _mb
        _mb.showinfo("СППР — итоги симуляции", msg, parent=self)

    # =========================================================================
    # ОБЩЕЕ
    # =========================================================================

    def _switch_mode(self):
        """Перезапустить приложение в другом режиме."""
        self._on_pause()
        self.destroy()
        _restart_with_dialog()


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════════════════════════
def _restart_with_dialog():
    """Показать стартовый диалог и запустить приложение в выбранном режиме."""
    dlg  = ModeSelectDialog()
    mode = dlg.run()
    if mode:
        app = TankFireApp(mode=mode)
        app.mainloop()

def main():
    _restart_with_dialog()


if __name__ == "__main__":
    main()
