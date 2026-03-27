"""
bupo_symbols.py
════════════════════════════════════════════════════════════════════════════════
Условные графические обозначения по Приложению №1 к БУПО
(Приказ МЧС России от 16.10.2024 №777).

Отрисовка на tkinter.Canvas — нормативные тактические знаки
для планов и карточек тушения пожаров.
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import math
import tkinter as tk
from typing import List, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ ТИПОВ СИМВОЛОВ
# ══════════════════════════════════════════════════════════════════════════════

# --- Раздел 1: Мобильные средства пожаротушения и техника ---
BUPO_AC    = "ac"       # Пожарная автоцистерна
BUPO_APT   = "apt"      # Автомобиль пенного тушения
BUPO_AKT   = "akt"      # Автомобиль комбинированного тушения
BUPO_ALP   = "alp"      # Пожарная автолестница
BUPO_AKP   = "akp"      # Пожарный автоподъёмник коленчатый
BUPO_PNS   = "pns"      # Пожарная автонасосная станция
BUPO_ASH   = "ash"      # Пожарный штабной автомобиль
BUPO_AR    = "ar"       # Пожарный рукавный автомобиль
BUPO_ASO   = "aso"      # Автомобиль связи и освещения
BUPO_APP   = "app"      # Автомобиль первой помощи
BUPO_AT    = "at"       # Пожарно-технический автомобиль
BUPO_ACA   = "aca"      # Аварийно-спасательный автомобиль
BUPO_APO   = "apo"      # Автомобиль порошкового тушения
BUPO_PANRK = "panrk"    # ПАНРК (судно пожарное / катер)
BUPO_PP    = "pp"       # Пожарный поезд
BUPO_SKMP  = "skmp"     # Скорая помощь
BUPO_MVD   = "mvd"      # Автомобиль полиции

# --- Раздел 2: Пожарно-техническое вооружение ---
BUPO_STVOL       = "stvol"        # Ствол пожарный ручной (общее)
BUPO_STVOL_LAF   = "stvol_laf"    # Ствол лафетный переносной
BUPO_STVOL_STAT  = "stvol_stat"   # Ствол лафетный стационарный
BUPO_STVOL_VOZ   = "stvol_voz"    # Ствол лафетный возимый
BUPO_GPS         = "gps"          # ГПС (генератор пены)
BUPO_RUKAV       = "rukav"        # Рукавная линия напорная
BUPO_RAZV        = "razv"         # Разветвление рукавное 3-ходовое

# --- Раздел 5: Пункты управления и средства связи ---
BUPO_OSH   = "osh"      # Оперативный штаб (ЧС, красный)
BUPO_KPP   = "kpp"      # Контрольно-пропускной пункт
BUPO_BU    = "bu"       # Боевой участок
BUPO_STP   = "stp"      # Сектор тушения пожара

# --- Раздел 6: Обстановка в зоне боевых действий ---
BUPO_FIRE_INT  = "fire_int"   # Пожар внутренний (штрих красный)
BUPO_FIRE_EXT  = "fire_ext"   # Пожар наружный (штрих красный)
BUPO_OCHAG     = "ochag"      # Место возникновения пожара (очаг)
BUPO_OBVAL     = "obval"      # Обвалование (земляная насыпь)

# --- Раздел 7: Водоснабжение ---
BUPO_PG      = "pg"        # Пожарный гидрант
BUPO_PK      = "pk"        # Пожарный кран
BUPO_VODOYOM = "vodoyom"   # Пожарный водоём
BUPO_PIRS    = "pirs"       # Пирс
BUPO_KOLODEC = "kolodec"    # Колодец

# ── Нормативные цвета ────────────────────────────────────────────────────────
C_RED    = "#cc0000"   # красный — пожарная техника, очаг, штаб
C_BLUE   = "#0055aa"   # синий — водоснабжение, гидранты
C_BLACK  = "#222222"   # чёрный — линии, рукава, стволы
C_WHITE  = "#ffffff"
C_FILL   = "#e74c3c"   # заливка техники (красный, полный)
C_FILL_L = "#ffcccc"   # заливка техники (светлая, неактивный)
C_GRAY   = "#999999"   # неактивный

# ── Каталог обозначений (label → bupo type) ──────────────────────────────────
LABEL_TO_BUPO = {
    "АЦ": BUPO_AC, "АПТ": BUPO_APT, "АКТ": BUPO_AKT,
    "АЛП": BUPO_ALP, "АКП": BUPO_AKP, "ПНС": BUPO_PNS,
    "АШ": BUPO_ASH, "АР": BUPO_AR, "АСО": BUPO_ASO,
    "АПП": BUPO_APP, "АТ": BUPO_AT, "АСА": BUPO_ACA,
    "АПО": BUPO_APO, "ПАНРК": BUPO_PANRK, "ПП": BUPO_PP,
    "+": BUPO_SKMP, "МВД": BUPO_MVD,
}

# Все типы техники (для палитры редактора)
VEHICLE_TYPES = [
    (BUPO_AC,    "АЦ  — автоцистерна"),
    (BUPO_APT,   "АПТ — пенного тушения"),
    (BUPO_AKT,   "АКТ — комбинированного тушения"),
    (BUPO_ALP,   "АЛП — автолестница"),
    (BUPO_AKP,   "АКП — автоподъёмник"),
    (BUPO_PNS,   "ПНС — насосная станция"),
    (BUPO_ASH,   "АШ  — штабной"),
    (BUPO_AR,    "АР  — рукавный"),
    (BUPO_ASO,   "АСО — связь и освещение"),
    (BUPO_APP,   "АПП — первой помощи"),
    (BUPO_AT,    "АТ  — технический"),
    (BUPO_ACA,   "АСА — аварийно-спасательный"),
    (BUPO_PANRK, "ПАНРК — насосно-рукавный комплекс"),
    (BUPO_PP,    "ПП  — пожарный поезд"),
    (BUPO_SKMP,  "СМП — скорая помощь"),
    (BUPO_MVD,   "МВД — полиция"),
]


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ ОТРИСОВКИ
# ══════════════════════════════════════════════════════════════════════════════

def draw_bupo_symbol(c: tk.Canvas, x: float, y: float,
                     symbol_type: str, label: str = "",
                     scale: float = 1.0, angle: float = 0.0,
                     active: bool = True, fill: str = "",
                     **kw) -> List[int]:
    """Нарисовать условный знак БУПО на Canvas.

    Args:
        c:           Canvas
        x, y:        центр значка
        symbol_type: одна из констант BUPO_*
        label:       подпись (например "АЦ-40", "ПГ-106")
        scale:       масштаб (1.0 = стандартный)
        angle:       угол поворота в градусах (для стволов)
        active:      активный объект (яркий) или неактивный (серый)
        fill:        переопределить цвет заливки
    Returns:
        Список ID canvas-элементов (для удаления/перемещения)
    """
    x, y = int(x), int(y)
    _map = {
        # Техника (раздел 1)
        BUPO_AC:    _draw_vehicle,
        BUPO_APT:   _draw_vehicle,
        BUPO_AKT:   _draw_vehicle,
        BUPO_ALP:   _draw_vehicle,
        BUPO_AKP:   _draw_vehicle,
        BUPO_ASH:   _draw_vehicle,
        BUPO_AR:    _draw_vehicle,
        BUPO_ASO:   _draw_vehicle,
        BUPO_APP:   _draw_vehicle,
        BUPO_AT:    _draw_vehicle,
        BUPO_ACA:   _draw_vehicle,
        BUPO_APO:   _draw_vehicle,
        BUPO_PANRK: _draw_vessel,
        BUPO_PP:    _draw_train,
        BUPO_SKMP:  _draw_vehicle_service,
        BUPO_MVD:   _draw_vehicle_service,
        BUPO_PNS:   _draw_pns,
        # Вооружение (раздел 2)
        BUPO_STVOL:      _draw_stvol,
        BUPO_STVOL_LAF:  _draw_stvol_laf,
        BUPO_STVOL_STAT: _draw_stvol_stat,
        BUPO_STVOL_VOZ:  _draw_stvol_voz,
        BUPO_GPS:        _draw_gps,
        BUPO_RUKAV:      _draw_rukav,
        BUPO_RAZV:       _draw_razv,
        # Управление (раздел 5)
        BUPO_OSH:  _draw_osh,
        BUPO_KPP:  _draw_kpp,
        BUPO_BU:   _draw_bu,
        BUPO_STP:  _draw_bu,
        # Обстановка (раздел 6)
        BUPO_FIRE_INT: _draw_fire_zone,
        BUPO_FIRE_EXT: _draw_fire_zone,
        BUPO_OCHAG:    _draw_ochag,
        BUPO_OBVAL:    _draw_obval,
        # Водоснабжение (раздел 7)
        BUPO_PG:      _draw_pg,
        BUPO_PK:      _draw_pk,
        BUPO_VODOYOM: _draw_vodoyom,
        BUPO_PIRS:    _draw_pirs,
        BUPO_KOLODEC: _draw_kolodec,
    }
    fn = _map.get(symbol_type, _draw_vehicle)
    return fn(c, x, y, symbol_type=symbol_type, label=label,
              scale=scale, angle=angle, active=active, fill=fill, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 1: МОБИЛЬНЫЕ СРЕДСТВА ПОЖАРОТУШЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════

# Стандартные аббревиатуры для подписи внутри значка
_VEHICLE_LABELS = {
    BUPO_AC: "АЦ", BUPO_APT: "АПТ", BUPO_AKT: "АКТ",
    BUPO_ALP: "АЛП", BUPO_AKP: "АКП", BUPO_ASH: "Ш",
    BUPO_AR: "АР", BUPO_ASO: "АСО", BUPO_APP: "АПП",
    BUPO_AT: "АТ", BUPO_ACA: "АСА", BUPO_APO: "АПО",
}


def _draw_vehicle(c, x, y, symbol_type="", label="", scale=1.0,
                  active=True, fill="", **kw) -> List[int]:
    """Пожарный автомобиль — прямоугольник со стрелкой вправо (БУПО разд.1).

    Общее обозначение: красный контур, буквенные обозначения внутри.
      ┌──────────┐╲
      │  АЦ      │ ▷   (стрелка вправо)
      └──────────┘╱
    """
    ids: List[int] = []
    s = scale
    w, h = int(30 * s), int(16 * s)
    tip = int(8 * s)

    fc = fill or (C_FILL if active else C_FILL_L)
    oc = C_RED if active else C_GRAY

    # Полигон: прямоугольник + стрелка
    pts = [x - w//2,       y - h//2,
           x + w//2 - 2,   y - h//2,
           x + w//2 + tip,  y,
           x + w//2 - 2,   y + h//2,
           x - w//2,       y + h//2]
    ids.append(c.create_polygon(pts, fill=fc, outline=oc, width=2))

    # Внутренняя подпись
    inner = label or _VEHICLE_LABELS.get(symbol_type, "")
    if inner:
        fs = max(6, int(7 * s))
        ids.append(c.create_text(x - int(2*s), y, text=inner,
                                 fill=C_WHITE, font=("Arial", fs, "bold")))
    return ids


def _draw_pns(c, x, y, symbol_type="", label="", scale=1.0,
              active=True, fill="", **kw) -> List[int]:
    """ПНС — квадрат с надписью «ПНС» (БУПО разд.1)."""
    ids: List[int] = []
    s = scale
    sz = int(14 * s)
    fc = fill or (C_FILL if active else C_FILL_L)
    oc = C_RED if active else C_GRAY

    ids.append(c.create_rectangle(x-sz, y-sz, x+sz, y+sz,
                                  fill=fc, outline=oc, width=2))
    inner = label or "ПНС"
    fs = max(6, int(7 * s))
    ids.append(c.create_text(x, y, text=inner, fill=C_WHITE,
                             font=("Arial", fs, "bold")))
    return ids


def _draw_vessel(c, x, y, symbol_type="", label="", scale=1.0,
                 active=True, fill="", **kw) -> List[int]:
    """Судно пожарное / катер — овал со стрелкой (БУПО разд.1)."""
    ids: List[int] = []
    s = scale
    w, h = int(32 * s), int(14 * s)
    tip = int(8 * s)
    fc = fill or (C_FILL if active else C_FILL_L)
    oc = C_RED if active else C_GRAY

    # Корпус: овал
    ids.append(c.create_oval(x - w//2, y - h//2, x + w//2, y + h//2,
                             fill=fc, outline=oc, width=2))
    # Нос (стрелка)
    ids.append(c.create_polygon(x + w//2 - 4, y - h//3,
                                x + w//2 + tip, y,
                                x + w//2 - 4, y + h//3,
                                fill=fc, outline=oc, width=1))
    inner = label or "ПАНРК"
    fs = max(5, int(6 * s))
    ids.append(c.create_text(x - int(2*s), y, text=inner,
                             fill=C_WHITE, font=("Arial", fs, "bold")))
    return ids


def _draw_train(c, x, y, symbol_type="", label="", scale=1.0,
                active=True, fill="", **kw) -> List[int]:
    """Пожарный поезд — прямоугольник с крышей (БУПО разд.1)."""
    ids: List[int] = []
    s = scale
    w, h = int(34 * s), int(14 * s)
    fc = fill or (C_FILL if active else C_FILL_L)
    oc = C_RED if active else C_GRAY

    ids.append(c.create_rectangle(x - w//2, y - h//2, x + w//2, y + h//2,
                                  fill=fc, outline=oc, width=2))
    # «Крыша» — скругленная дуга сверху
    ids.append(c.create_arc(x - w//2 + 2, y - h//2 - int(6*s),
                            x + w//2 - 2, y - h//2 + int(4*s),
                            start=0, extent=180,
                            style="arc", outline=oc, width=2))
    inner = label or "ПП"
    fs = max(6, int(7 * s))
    ids.append(c.create_text(x, y, text=inner,
                             fill=C_WHITE, font=("Arial", fs, "bold")))
    return ids


def _draw_vehicle_service(c, x, y, symbol_type="", label="", scale=1.0,
                          active=True, fill="", **kw) -> List[int]:
    """Скорая помощь / МВД — прямоугольник со стрелкой, другой цвет."""
    ids: List[int] = []
    s = scale
    w, h = int(30 * s), int(16 * s)
    tip = int(8 * s)

    if symbol_type == BUPO_SKMP:
        fc = fill or ("#ffffff" if active else "#eee")
        oc = C_RED
        inner_text = label or "+"
    else:
        fc = fill or ("#3355aa" if active else "#aabbcc")
        oc = "#223388"
        inner_text = label or "МВД"

    pts = [x - w//2, y - h//2,
           x + w//2 - 2, y - h//2,
           x + w//2 + tip, y,
           x + w//2 - 2, y + h//2,
           x - w//2, y + h//2]
    ids.append(c.create_polygon(pts, fill=fc, outline=oc, width=2))
    fs = max(6, int(7 * s))
    tc = C_RED if symbol_type == BUPO_SKMP else C_WHITE
    ids.append(c.create_text(x - int(2*s), y, text=inner_text,
                             fill=tc, font=("Arial", fs, "bold")))
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 2: ПОЖАРНО-ТЕХНИЧЕСКОЕ ВООРУЖЕНИЕ
# ══════════════════════════════════════════════════════════════════════════════

def _draw_stvol(c, x, y, symbol_type="", label="", scale=1.0,
                angle=0.0, active=True, fill="", **kw) -> List[int]:
    """Ствол пожарный ручной — линия с наконечником-стрелкой (БУПО разд.2).

    Обозначение: ├──→  (черный цвет)
    Основание (├) показывает место подключения, стрелка — направление подачи.
    """
    ids: List[int] = []
    s = scale
    L = int(22 * s)
    clr = C_BLACK if active else C_GRAY

    rad = math.radians(angle)
    dx = L * math.cos(rad)
    dy = L * math.sin(rad)

    # Основание (перпендикулярная чёрточка)
    bx, by = x - dx * 0.5, y - dy * 0.5
    perp = math.radians(angle + 90)
    bp = int(5 * s)
    ids.append(c.create_line(bx + bp*math.cos(perp), by + bp*math.sin(perp),
                             bx - bp*math.cos(perp), by - bp*math.sin(perp),
                             fill=clr, width=max(2, int(2*s))))
    # Ствол (линия со стрелкой)
    ex, ey = x + dx * 0.5, y + dy * 0.5
    ids.append(c.create_line(bx, by, ex, ey,
                             fill=clr, width=max(2, int(2*s)),
                             arrow="last", arrowshape=(int(8*s), int(10*s), int(4*s))))

    if label:
        fs = max(5, int(6 * s))
        ids.append(c.create_text(x, y + int(12*s), text=label,
                                 fill=clr, font=("Arial", fs)))
    return ids


def _draw_stvol_laf(c, x, y, **kw) -> List[int]:
    """Ствол лафетный переносной — ├──→ с кружком у основания."""
    ids = _draw_stvol(c, x, y, **kw)
    s = kw.get("scale", 1.0)
    r = int(4 * s)
    angle = math.radians(kw.get("angle", 0.0))
    L = int(22 * s)
    bx = x - L * 0.5 * math.cos(angle)
    by = y - L * 0.5 * math.sin(angle)
    clr = C_BLACK if kw.get("active", True) else C_GRAY
    ids.append(c.create_oval(bx - r, by - r, bx + r, by + r,
                             fill="", outline=clr, width=max(1, int(1.5*s))))
    return ids


def _draw_stvol_stat(c, x, y, **kw) -> List[int]:
    """Ствол лафетный стационарный — ├──→ с квадратом у основания."""
    ids = _draw_stvol(c, x, y, **kw)
    s = kw.get("scale", 1.0)
    r = int(4 * s)
    angle = math.radians(kw.get("angle", 0.0))
    L = int(22 * s)
    bx = x - L * 0.5 * math.cos(angle)
    by = y - L * 0.5 * math.sin(angle)
    clr = C_BLACK if kw.get("active", True) else C_GRAY
    ids.append(c.create_rectangle(bx - r, by - r, bx + r, by + r,
                                  fill="", outline=clr, width=max(1, int(1.5*s))))
    return ids


def _draw_stvol_voz(c, x, y, **kw) -> List[int]:
    """Ствол лафетный возимый — ├──→ с крестиком у основания."""
    ids = _draw_stvol(c, x, y, **kw)
    s = kw.get("scale", 1.0)
    r = int(4 * s)
    angle = math.radians(kw.get("angle", 0.0))
    L = int(22 * s)
    bx = x - L * 0.5 * math.cos(angle)
    by = y - L * 0.5 * math.sin(angle)
    clr = C_BLACK if kw.get("active", True) else C_GRAY
    ids.append(c.create_line(bx - r, by - r, bx + r, by + r,
                             fill=clr, width=max(1, int(1.5*s))))
    ids.append(c.create_line(bx - r, by + r, bx + r, by - r,
                             fill=clr, width=max(1, int(1.5*s))))
    return ids


def _draw_gps(c, x, y, symbol_type="", label="", scale=1.0,
              active=True, fill="", **kw) -> List[int]:
    """ГПС (генератор пенный) — круг с перечёркнутыми линиями (БУПО разд.2)."""
    ids: List[int] = []
    s = scale
    r = int(10 * s)
    clr = C_BLACK if active else C_GRAY
    ids.append(c.create_oval(x - r, y - r, x + r, y + r,
                             fill="", outline=clr, width=max(2, int(2*s))))
    # Перекрестие
    d = int(r * 0.7)
    ids.append(c.create_line(x - d, y - d, x + d, y + d, fill=clr, width=1))
    ids.append(c.create_line(x - d, y + d, x + d, y - d, fill=clr, width=1))
    if label:
        fs = max(5, int(6 * s))
        ids.append(c.create_text(x, y + r + int(8*s), text=label,
                                 fill=clr, font=("Arial", fs)))
    return ids


def _draw_rukav(c, x, y, symbol_type="", label="", scale=1.0,
                angle=0.0, active=True, fill="", **kw) -> List[int]:
    """Рукавная линия напорная — сплошная чёрная линия (БУПО разд.2)."""
    ids: List[int] = []
    s = scale
    L = int(30 * s)
    clr = C_BLACK if active else C_GRAY
    rad = math.radians(angle)
    ex = x + L * math.cos(rad)
    ey = y + L * math.sin(rad)
    ids.append(c.create_line(x, y, ex, ey, fill=clr, width=max(2, int(2*s))))
    return ids


def _draw_razv(c, x, y, symbol_type="", label="", scale=1.0,
               active=True, fill="", **kw) -> List[int]:
    """Разветвление рукавное 3-ходовое (БУПО разд.2)."""
    ids: List[int] = []
    s = scale
    clr = C_BLACK if active else C_GRAY
    r = int(5 * s)
    ids.append(c.create_oval(x - r, y - r, x + r, y + r,
                             fill=clr, outline=clr))
    L = int(12 * s)
    for ang_deg in [-45, 0, 45]:
        rad = math.radians(ang_deg)
        ids.append(c.create_line(x, y, x + L*math.cos(rad), y + L*math.sin(rad),
                                 fill=clr, width=max(2, int(2*s))))
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 5: ПУНКТЫ УПРАВЛЕНИЯ И СРЕДСТВА СВЯЗИ
# ══════════════════════════════════════════════════════════════════════════════

def _draw_osh(c, x, y, symbol_type="", label="", scale=1.0,
              active=True, fill="", **kw) -> List[int]:
    """Оперативный штаб — треугольник (вымпел) с флагом (БУПО разд.5).

    Красный треугольник (вершиной вверх), с надписью «ЧС» внутри.
    """
    ids: List[int] = []
    s = scale
    sz = int(22 * s)
    fc = fill or (C_RED if active else C_GRAY)

    # Треугольник вершиной вверх
    pts = [x, y - sz,
           x - int(sz * 0.8), y + int(sz * 0.5),
           x + int(sz * 0.8), y + int(sz * 0.5)]
    ids.append(c.create_polygon(pts, fill=fc, outline="#880000", width=2))

    # Флажок на вершине
    fx, fy = x, y - sz
    fh = int(10 * s)
    fw = int(14 * s)
    ids.append(c.create_line(fx, fy, fx, fy - fh, fill="#880000", width=2))
    ids.append(c.create_polygon(fx, fy - fh, fx + fw, fy - fh + int(3*s),
                                fx, fy - int(3*s),
                                fill=fc, outline=""))

    # Текст внутри
    inner = label or "ОШ"
    fs = max(6, int(7 * s))
    ids.append(c.create_text(x, y + int(2*s), text=inner,
                             fill=C_WHITE, font=("Arial", fs, "bold")))
    return ids


def _draw_kpp(c, x, y, symbol_type="", label="", scale=1.0,
              active=True, fill="", **kw) -> List[int]:
    """КПП — красный круг с надписью «КПП» (БУПО разд.5)."""
    ids: List[int] = []
    s = scale
    r = int(14 * s)
    fc = fill or (C_RED if active else C_GRAY)
    ids.append(c.create_oval(x - r, y - r, x + r, y + r,
                             fill=fc, outline="#880000", width=2))
    inner = label or "КПП"
    fs = max(5, int(6 * s))
    ids.append(c.create_text(x, y, text=inner,
                             fill=C_WHITE, font=("Arial", fs, "bold")))
    return ids


def _draw_bu(c, x, y, symbol_type="", label="", scale=1.0,
             active=True, fill="", **kw) -> List[int]:
    """Боевой участок — красная скобка с подписью (БУПО разд.5).

    ├──── БУ-1 (ЮГ) ────┤
    """
    ids: List[int] = []
    s = scale
    bw = int(50 * s)
    bh = int(12 * s)
    fc = fill or (C_RED if active else C_GRAY)

    # Горизонтальная линия
    ids.append(c.create_line(x - bw, y, x + bw, y,
                             fill=fc, width=max(2, int(2*s))))
    # Вертикальные засечки
    ids.append(c.create_line(x - bw, y - bh, x - bw, y + bh,
                             fill=fc, width=max(2, int(2*s))))
    ids.append(c.create_line(x + bw, y - bh, x + bw, y + bh,
                             fill=fc, width=max(2, int(2*s))))
    # Подпись
    inner = label or "БУ"
    fs = max(6, int(7 * s))
    ids.append(c.create_text(x, y - bh - int(4*s), text=inner,
                             fill=fc, font=("Arial", fs, "bold")))
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 6: ОБСТАНОВКА В ЗОНЕ БОЕВЫХ ДЕЙСТВИЙ
# ══════════════════════════════════════════════════════════════════════════════

def _draw_fire_zone(c, x, y, symbol_type="", label="", scale=1.0,
                    active=True, fill="", **kw) -> List[int]:
    """Пожар внутренний/наружный — заштрихованный прямоугольник (БУПО разд.6)."""
    ids: List[int] = []
    s = scale
    w, h = int(24 * s), int(18 * s)
    fc = C_RED
    ids.append(c.create_rectangle(x - w, y - h, x + w, y + h,
                                  fill="", outline=fc, width=2))
    # Штриховка
    step = max(4, int(6 * s))
    for hx in range(x - w + step, x + w, step):
        ids.append(c.create_line(hx, y - h + 2, hx, y + h - 2,
                                 fill=fc, width=1))
    # Наружный — двойной контур
    if symbol_type == BUPO_FIRE_EXT:
        d = int(3 * s)
        ids.append(c.create_rectangle(x - w - d, y - h - d, x + w + d, y + h + d,
                                      fill="", outline=fc, width=1))
    return ids


def _draw_ochag(c, x, y, symbol_type="", label="", scale=1.0,
                active=True, fill="", **kw) -> List[int]:
    """Очаг пожара — Г-образная метка (БУПО разд.6)."""
    ids: List[int] = []
    s = scale
    L = int(14 * s)
    fc = C_RED
    # Вертикальная линия вниз
    ids.append(c.create_line(x, y, x, y + L, fill=fc, width=max(3, int(3*s))))
    # Горизонтальная линия влево
    ids.append(c.create_line(x, y + L, x - L, y + L, fill=fc, width=max(3, int(3*s))))
    return ids


def _draw_obval(c, x, y, symbol_type="", label="", scale=1.0,
                active=True, fill="", **kw) -> List[int]:
    """Обвалование / земляная насыпь — волнистая линия (БУПО разд.7/17)."""
    ids: List[int] = []
    s = scale
    r = kw.get("radius", int(60 * s))
    clr = "#8B7355"
    # Круговое обвалование (пунктир с зубцами)
    ids.append(c.create_oval(x - r, y - r, x + r, y + r,
                             fill="", outline=clr, width=max(3, int(3*s)),
                             dash=(6, 3)))
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 7: ВОДОСНАБЖЕНИЕ
# ══════════════════════════════════════════════════════════════════════════════

def _draw_pg(c, x, y, symbol_type="", label="", scale=1.0,
             active=True, fill="", **kw) -> List[int]:
    """Пожарный гидрант — круг с Т-образной вставкой (БУПО разд.7).

    БУПО: круг синего цвета с буквой ПГ и номером, вертикальная чёрточка
    с горизонтальной перекладиной сверху (Т-образная форма штока).
    """
    ids: List[int] = []
    s = scale
    r = int(11 * s)
    clr = C_BLUE if active else C_GRAY

    ids.append(c.create_oval(x - r, y - r, x + r, y + r,
                             fill="", outline=clr, width=max(2, int(2*s))))
    # Т-образный шток внутри
    stem = int(r * 0.65)
    cap = int(r * 0.6)
    ids.append(c.create_line(x, y + stem, x, y - stem + int(2*s),
                             fill=clr, width=max(2, int(2*s))))
    ids.append(c.create_line(x - cap, y - stem + int(2*s),
                             x + cap, y - stem + int(2*s),
                             fill=clr, width=max(2, int(2*s))))
    # Подпись
    if label:
        fs = max(5, int(6 * s))
        ids.append(c.create_text(x, y + r + int(9*s), text=label,
                                 fill=clr, font=("Arial", fs, "bold")))
    return ids


def _draw_pk(c, x, y, symbol_type="", label="", scale=1.0,
             active=True, fill="", **kw) -> List[int]:
    """Пожарный кран — круг с ПК (БУПО разд.7)."""
    ids: List[int] = []
    s = scale
    r = int(9 * s)
    clr = C_BLUE if active else C_GRAY
    ids.append(c.create_oval(x - r, y - r, x + r, y + r,
                             fill="", outline=clr, width=max(2, int(2*s))))
    fs = max(5, int(6 * s))
    ids.append(c.create_text(x, y, text=label or "ПК",
                             fill=clr, font=("Arial", fs, "bold")))
    return ids


def _draw_vodoyom(c, x, y, symbol_type="", label="", scale=1.0,
                  active=True, fill="", **kw) -> List[int]:
    """Пожарный водоём — прямоугольник с объёмом (БУПО разд.7)."""
    ids: List[int] = []
    s = scale
    w, h = int(20 * s), int(16 * s)
    clr = C_BLUE if active else C_GRAY

    ids.append(c.create_rectangle(x - w, y - h, x + w, y + h,
                                  fill="", outline=clr, width=max(2, int(2*s))))
    inner = label or "50"
    fs = max(6, int(8 * s))
    ids.append(c.create_text(x, y, text=inner, fill=clr,
                             font=("Arial", fs, "bold")))
    return ids


def _draw_pirs(c, x, y, symbol_type="", label="", scale=1.0,
               active=True, fill="", **kw) -> List[int]:
    """Пирс — крест с числом (БУПО разд.7)."""
    ids: List[int] = []
    s = scale
    L = int(12 * s)
    clr = C_BLACK if active else C_GRAY
    ids.append(c.create_line(x, y - L, x, y + L, fill=clr, width=max(2, int(2*s))))
    ids.append(c.create_line(x - L, y, x + L, y, fill=clr, width=max(2, int(2*s))))
    if label:
        fs = max(5, int(6 * s))
        ids.append(c.create_text(x, y + L + int(8*s), text=label,
                                 fill=clr, font=("Arial", fs)))
    return ids


def _draw_kolodec(c, x, y, symbol_type="", label="", scale=1.0,
                  active=True, fill="", **kw) -> List[int]:
    """Колодец — шестиугольник (БУПО разд.7)."""
    ids: List[int] = []
    s = scale
    r = int(10 * s)
    clr = C_BLUE if active else C_GRAY
    pts = []
    for i in range(6):
        ang = math.radians(60 * i - 90)
        pts.extend([x + r * math.cos(ang), y + r * math.sin(ang)])
    ids.append(c.create_polygon(pts, fill="", outline=clr, width=max(2, int(2*s))))
    return ids
