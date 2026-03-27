"""
scenario_editor.py
════════════════════════════════════════════════════════════════════════════════
Интерактивный редактор сценариев пожара.

Пользователь размещает на карте объекты (РВС, здания, гидранты, водоёмы),
задаёт параметры пожара и гарнизона, после чего запускает симуляцию
с созданным сценарием — либо сохраняет его в JSON для последующего использования.

Запуск отдельно:
    python -m saur_sim.scenario_editor

Или из главного окна: кнопка «Конструктор сценариев» в настройках.
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math
import json
import os
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any


# ══════════════════════════════════════════════════════════════════════════════
# ЦВЕТОВАЯ ПАЛИТРА (согласована с tank_fire_sim.py)
# ══════════════════════════════════════════════════════════════════════════════
P = dict(
    bg="#1a1f2e", panel="#1e2535", panel2="#232c40", canvas="#0d1420",
    fire="#ff4500", fire2="#ff8c00",
    rvs_burning="#c0392b", rvs_neighbor="#2471a3",
    water="#00aaff", foam="#90ee90",
    building="#4a4a6a", ground="#1a3322",
    hydrant="#1abc9c", river="#1a5276",
    success="#27ae60", warn="#e67e22", danger="#c0392b", info="#2980b9",
    text="#ecf0f1", text2="#95a5a6", hi="#f1c40f",
    grid="#2c3e50", accent="#e67e22",
    select="#f1c40f",          # контур выбранного объекта
    ghost="#ffffff44",         # полупрозрачный «призрак» при размещении
    toolbar_btn="#232c40",
    toolbar_active="#e67e22",
)

ED_W, ED_H = 580, 460   # размер холста редактора
GRID_SIZE   = 20        # шаг сетки привязки (пиксели)

# Типы объектов
OBJ_RVS_FIRE  = "rvs_fire"      # горящий резервуар (один!)
OBJ_RVS_NEAR  = "rvs_near"      # соседний резервуар
OBJ_HYDRANT   = "hydrant"       # пожарный гидрант / подземный источник
OBJ_RIVER     = "river"         # водоём / река / ПНС-источник
OBJ_BUILDING  = "building"      # здание / сооружение
OBJ_ROAD      = "road"          # дорога (только для отображения)

# Параметры типов объектов: (метка, цвет заливки, цвет контура, форма, min_r, max_r)
OBJ_SPECS: Dict[str, dict] = {
    OBJ_RVS_FIRE: dict(
        label="РВС (горящий)", fill=P["rvs_burning"], outline="#ff4444",
        shape="circle", min_r=20, max_r=90, default_r=55,
        hint="Резервуар, на котором возник пожар. Только один.",
    ),
    OBJ_RVS_NEAR: dict(
        label="РВС (соседний)", fill=P["rvs_neighbor"], outline="#4499cc",
        shape="circle", min_r=15, max_r=80, default_r=45,
        hint="Соседний резервуар под угрозой теплового воздействия.",
    ),
    OBJ_HYDRANT: dict(
        label="Гидрант / ПГ", fill=P["hydrant"], outline="white",
        shape="circle", min_r=6, max_r=10, default_r=8,
        hint="Подземный пожарный гидрант. Даёт ~15 л/с.",
    ),
    OBJ_RIVER: dict(
        label="Водоём / река", fill=P["river"], outline=P["water"],
        shape="rect", default_r=30,
        hint="Открытый водоисточник для ПНС. Даёт ~110 л/с.",
    ),
    OBJ_BUILDING: dict(
        label="Здание", fill=P["building"], outline="#888",
        shape="rect", default_r=25,
        hint="Здание / сооружение на территории объекта.",
    ),
    OBJ_ROAD: dict(
        label="Дорога", fill="", outline=P["grid"],
        shape="line",
        hint="Проездной путь. Для отображения.",
    ),
}

FUELS = ["бензин", "нефть", "дизель", "мазут"]
ROOF_TYPES = {
    "Плавающая крыша (70%)": 0.70,
    "Плавающая крыша (50%)": 0.50,
    "Плавающая крыша (30%)": 0.30,
    "Конусная кровля (0%)":  0.00,
    "Открытый резервуар":    0.00,
}
FIRE_RANKS = ["№1", "№2", "№3", "№4"]


# ══════════════════════════════════════════════════════════════════════════════
# МОДЕЛЬ ДАННЫХ ОБЪЕКТА НА КАРТЕ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MapObject:
    """Один объект, размещённый на карте редактора."""
    otype: str          # тип объекта (OBJ_*)
    x: float            # центр X, пиксели
    y: float            # центр Y, пиксели
    r: float            # радиус / полуразмер
    label: str = ""     # подпись
    volume_m3: float = 0.0      # объём РВС, м³ (только для РВС)
    diameter_m: float = 0.0     # диаметр, м (только для РВС)
    fuel: str = "бензин"        # тип топлива (только для РВС)
    canvas_ids: List[int] = field(default_factory=list, repr=False)

    @property
    def spec(self) -> dict:
        return OBJ_SPECS[self.otype]

    def hit_test(self, mx: float, my: float) -> bool:
        """Попадает ли точка (mx,my) в объект."""
        if self.spec.get("shape") in ("circle",):
            return math.hypot(mx - self.x, my - self.y) <= self.r + 6
        else:
            return (self.x - self.r - 6 <= mx <= self.x + self.r + 6 and
                    self.y - self.r - 6 <= my <= self.y + self.r + 6)

    def to_dict(self) -> dict:
        return {
            "otype": self.otype, "x": self.x, "y": self.y, "r": self.r,
            "label": self.label, "volume_m3": self.volume_m3,
            "diameter_m": self.diameter_m, "fuel": self.fuel,
        }


# ══════════════════════════════════════════════════════════════════════════════
# АВТОГЕНЕРАЦИЯ ХРОНОЛОГИИ
# ══════════════════════════════════════════════════════════════════════════════

def _auto_timeline(fire_rank: int, total_min: int, rvs_label: str,
                   fire_area: float, fuel: str) -> List[Tuple]:
    """Автоматически сгенерировать хронологию событий по рангу пожара.

    Возвращает список (t_min, метка, описание, цвет_ключ).
    """
    try:
        from .tank_fire_sim import P as _P
    except ImportError:
        from tank_fire_sim import P as _P

    tl = []
    add = lambda t, lbl, desc, c: tl.append((t, lbl, desc, _P[c]))

    # Начало
    add(0, "Ч+0", f"Загорание {rvs_label}. S={fire_area:.0f} м², горючее: {fuel}.", "danger")

    if fire_rank >= 1:
        add(5,  "Ч+5",  "Обнаружение. Сигнал в ЕДДС. Выезд дежурного расчёта.", "warn")
        add(10, "Ч+10", "Прибытие 1-го подразделения. РТП-1 принял руководство. Разведка.", "info")
        add(12, "Ч+12", "Подача первых стволов на охлаждение горящего РВС.", "info")

    if fire_rank >= 2:
        add(15, "Ч+15", "Прибытие 2-го подразделения. Развёртывание. 2 ствола А + 2 ГПС-600.", "info")
        add(18, "Ч+18", "АЦ установлены на ПГ. Готовность к пенной атаке. Q_пены готов.", "warn")
        add(20, "Ч+20", "⚡ Пенная атака. Подача ГПС-600 на зеркало горения.", "warn")
        add(25, "Ч+25", "Штаб тушения пожара создан. НШ, НБУ-1, НТ назначены.", "info")

    if fire_rank >= 3:
        add(30, "Ч+30", "Прибытие 3-го подразделения. Установка ПНС на водоисточник (+110 л/с).", "info")
        add(45, "Ч+45", "6 стволов А на охлаждении. 3 ГПС-600 на тушении.", "info")
        add(60, "Ч+60", "Созданы 3 боевых участка. Непрерывное охлаждение соседних РВС.", "info")

    if fire_rank >= 4:
        add(90,  "Ч+90",  "Прибытие ПАНРК. Установка на глубоководный причал (+110 л/с).", "info")
        add(120, "Ч+120", "7 лафетных стволов на охлаждении. Пенная атака №2.", "warn")
        add(180, "Ч+180", "Прибытие пожарного поезда. Дополнительный запас пенообразователя.", "info")
        add(300, "Ч+300", "Пенная атака №3 с АКП-50. Подача ГПС-1000 через верхний люк.", "warn")

    # Локализация
    loc_t = {1: 30, 2: 60, 3: 120, 4: 300}.get(fire_rank, 60)
    add(loc_t, f"Ч+{loc_t}", f"🔒 Пожар локализован. S={fire_area:.0f} м².", "success")

    # Ликвидация
    ext_t = {1: 60, 2: 120, 3: 240, 4: 600}.get(fire_rank, 120)
    ext_t = min(ext_t, total_min - 10)
    add(ext_t, f"Ч+{ext_t}", f"✅ Ликвидировано горение в {rvs_label}.", "success")
    add(total_min, f"Ч+{total_min}",
        "🏁 Пожар ликвидирован. Охлаждение резервуаров продолжается.", "success")

    return tl


# ══════════════════════════════════════════════════════════════════════════════
# КОНВЕРТЕР: объекты карты → конфиг сценария для TankFireSim
# ══════════════════════════════════════════════════════════════════════════════

def build_scenario_config(objects: List[MapObject], params: dict) -> dict:
    """Собрать словарь конфигурации сценария из размещённых объектов и параметров.

    Результат совместим со структурой SCENARIOS в tank_fire_sim.py.
    """
    try:
        from .tank_fire_sim import P as _P
    except ImportError:
        from tank_fire_sim import P as _P

    # Найти горящий РВС
    fire_obj = next((o for o in objects if o.otype == OBJ_RVS_FIRE), None)
    if fire_obj is None:
        raise ValueError("На карте должен быть хотя бы один горящий РВС.")

    fuel        = params.get("fuel", "бензин")
    fire_rank   = int(params.get("fire_rank", 2))
    total_min   = int(params.get("total_min", 300))
    roof_obs    = float(params.get("roof_obstruction", 0.0))
    name        = params.get("name", "Пользовательский сценарий")

    # Площадь горения из радиуса объекта (масштаб: 1 px = 0.6 м)
    scale = float(params.get("map_scale_m_per_px", 0.6))
    diam_m = fire_obj.diameter_m if fire_obj.diameter_m > 0 else fire_obj.r * scale * 2
    fire_area = math.pi * (diam_m / 2) ** 2

    # Нормативная интенсивность пены
    foam_I = {"бензин": 0.065, "нефть": 0.060, "дизель": 0.048, "мазут": 0.060}
    foam_intensity = foam_I.get(fuel, 0.065)

    # Хронология
    rvs_label = fire_obj.label or "РВС"
    tl = _auto_timeline(fire_rank, total_min, rvs_label, fire_area, fuel)

    # Lookup по времени
    tl_lookup: Dict[int, List] = {}
    for ev in tl:
        tl_lookup.setdefault(ev[0], []).append(ev)

    # Сериализованные объекты для отрисовки карты
    obj_list = [o.to_dict() for o in objects]

    cfg = {
        "name":               name,
        "short":              name[:20],
        "total_min":          total_min,
        "initial_fire_area":  fire_area,
        "fuel":               fuel,
        "rvs_name":           rvs_label,
        "rvs_diameter_m":     diam_m,
        "fire_rank_default":  fire_rank,
        "roof_obstruction_init": roof_obs,
        "foam_intensity":     foam_intensity,
        "tl_lookup":          tl_lookup,
        "actions_by_phase":   None,   # используется глобальный ACTIONS_BY_PHASE
        # Дополнительные поля для карты
        "_custom":            True,
        "_objects":           obj_list,
        "_map_scale":         scale,
        "_params":            copy.deepcopy(params),
    }
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНОЕ ОКНО РЕДАКТОРА
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioEditorApp(tk.Toplevel):
    """Интерактивный редактор сценариев пожара.

    Позволяет размещать объекты на карте, задавать параметры пожара
    и экспортировать сценарий для запуска в симуляции.

    При передаче callback'а on_launch(cfg) — вызывает его при нажатии
    «Запустить симуляцию», передавая готовый конфиг сценария.
    """

    TOOL_SELECT  = "select"
    TOOLS = [
        (OBJ_RVS_FIRE,  "🔥 РВС (огонь)"),
        (OBJ_RVS_NEAR,  "🛢 РВС (сосед)"),
        (OBJ_HYDRANT,   "💧 Гидрант"),
        (OBJ_RIVER,     "🌊 Водоём"),
        (OBJ_BUILDING,  "🏭 Здание"),
        (TOOL_SELECT,   "↖ Выбор / Перемещение"),
    ]

    def __init__(self, parent=None, on_launch=None):
        super().__init__(parent)
        self.title("Конструктор сценария пожара — САУР ПСП")
        self.configure(bg=P["bg"])
        self.resizable(True, True)
        self.minsize(1000, 680)

        self._on_launch = on_launch      # callback при запуске симуляции
        self._objects: List[MapObject] = []
        self._selected: Optional[MapObject] = None
        self._tool = OBJ_RVS_FIRE        # текущий инструмент
        self._drag_start: Optional[Tuple[float, float]] = None
        self._ghost_ids: List[int] = []  # временные canvas-объекты «призрака»
        self._anim_t = 0

        self._build_ui()
        self._draw_grid()
        self._draw_all_objects()

        # Примерная сцена по умолчанию
        self._load_default_scene()

    # ──────────────────────────────────────────────────────────────────────────
    # ПОСТРОЕНИЕ UI
    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Верхняя строка
        hdr = tk.Frame(self, bg=P["panel"], height=44)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🗺  КОНСТРУКТОР СЦЕНАРИЯ ПОЖАРА",
                 font=("Arial", 12, "bold"), bg=P["panel"], fg=P["hi"]
                 ).pack(side="left", padx=14, pady=8)
        tk.Label(hdr, text="Размести объекты на карте → задай параметры → Запустить симуляцию",
                 font=("Arial", 9), bg=P["panel"], fg=P["text2"]
                 ).pack(side="left")

        # Основная область
        body = tk.Frame(self, bg=P["bg"])
        body.pack(fill="both", expand=True)

        # Левая панель: тулбар + холст
        left = tk.Frame(body, bg=P["panel2"], bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(4, 2), pady=4)
        self._build_toolbar(left)
        self._build_canvas(left)

        # Правая панель: параметры + свойства объекта
        right = tk.Frame(body, bg=P["bg"], width=310)
        right.pack(side="right", fill="y", padx=(2, 4), pady=4)
        right.pack_propagate(False)
        self._build_right_panel(right)

        # Нижняя строка (_status_var создаётся здесь — должна быть до _set_tool)
        self._build_bottom_bar()
        # Установить активный инструмент только после создания _status_var
        self._set_tool(OBJ_RVS_FIRE)

    def _build_toolbar(self, parent):
        tb = tk.Frame(parent, bg=P["panel"], height=40)
        tb.pack(fill="x", padx=2, pady=(2, 0))
        tb.pack_propagate(False)

        tk.Label(tb, text="Инструмент:", bg=P["panel"], fg=P["text2"],
                 font=("Arial", 8)).pack(side="left", padx=8)

        self._tool_btns: Dict[str, tk.Button] = {}
        for tool_key, tool_label in self.TOOLS:
            btn = tk.Button(
                tb, text=tool_label, bg=P["toolbar_btn"], fg=P["text"],
                font=("Arial", 8), relief="flat", padx=8, pady=4,
                command=lambda k=tool_key: self._set_tool(k),
            )
            btn.pack(side="left", padx=2, pady=4)
            self._tool_btns[tool_key] = btn
        # _set_tool вызывается позже, в _build_ui, когда _status_var уже создан

        # Кнопки управления
        tk.Label(tb, text="│", bg=P["panel"], fg=P["grid"]).pack(side="left", padx=4)
        tk.Button(tb, text="🗑 Удалить", bg=P["danger"], fg="#fff",
                  font=("Arial", 8), relief="flat", padx=8, pady=4,
                  command=self._delete_selected).pack(side="left", padx=2, pady=4)
        tk.Button(tb, text="🧹 Очистить", bg=P["grid"], fg=P["text"],
                  font=("Arial", 8), relief="flat", padx=8, pady=4,
                  command=self._clear_all).pack(side="left", padx=2, pady=4)
        tk.Button(tb, text="📐 Сетка", bg=P["grid"], fg=P["text"],
                  font=("Arial", 8), relief="flat", padx=8, pady=4,
                  command=self._toggle_grid).pack(side="left", padx=2, pady=4)
        self._show_grid = True

    def _build_canvas(self, parent):
        lbl = tk.Label(parent, text=" КАРТА ОБЪЕКТА ",
                       font=("Arial", 8, "bold"), bg=P["panel"], fg=P["text2"])
        lbl.pack(fill="x", padx=2)

        cf = tk.Frame(parent, bg=P["canvas"])
        cf.pack(fill="both", expand=True, padx=4, pady=4)

        self._canvas = tk.Canvas(
            cf, bg=P["canvas"], cursor="crosshair",
            highlightthickness=1, highlightbackground=P["grid"]
        )
        self._canvas.pack(fill="both", expand=True)

        # Событийные обработчики
        self._canvas.bind("<ButtonPress-1>",   self._on_click)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>",          self._on_motion)
        self._canvas.bind("<Double-Button-1>", self._on_double_click)
        self._canvas.bind("<Configure>",       lambda e: (self._draw_grid(), self._draw_all_objects()))

    def _build_right_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",     background=P["panel"])
        style.configure("TNotebook.Tab", background=P["panel2"], foreground=P["text2"],
                        padding=(8, 3), font=("Arial", 8))
        style.map("TNotebook.Tab",
                  background=[("selected", P["accent"])],
                  foreground=[("selected", "#000")])

        # Вкладка 1: Параметры сценария
        t1 = tk.Frame(nb, bg=P["bg"])
        nb.add(t1, text="⚙ Параметры")
        self._build_params_tab(t1)

        # Вкладка 2: Свойства объекта
        t2 = tk.Frame(nb, bg=P["bg"])
        nb.add(t2, text="📋 Объект")
        self._build_object_tab(t2)

        # Вкладка 3: Хронология
        t3 = tk.Frame(nb, bg=P["bg"])
        nb.add(t3, text="📅 Хронология")
        self._build_timeline_tab(t3)

        self._nb = nb

    def _build_params_tab(self, parent):
        """Вкладка параметров сценария."""
        canvas = tk.Canvas(parent, bg=P["bg"], highlightthickness=0)
        sb = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=P["bg"])
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        def section(title):
            f = tk.LabelFrame(inner, text=f" {title} ", bg=P["panel"],
                              fg=P["accent"], font=("Arial", 8, "bold"),
                              bd=1, relief="groove")
            f.pack(fill="x", padx=6, pady=4, ipadx=4, ipady=3)
            return f

        def entry_row(parent, label, var, width=14):
            row = tk.Frame(parent, bg=P["panel"])
            row.pack(fill="x", padx=6, pady=2)
            tk.Label(row, text=f"{label}:", bg=P["panel"], fg=P["text"],
                     font=("Arial", 8), width=20, anchor="w").pack(side="left")
            e = tk.Entry(row, textvariable=var, bg=P["panel2"], fg=P["text"],
                         insertbackground=P["text"], font=("Consolas", 8),
                         width=width, relief="flat", bd=2)
            e.pack(side="left", padx=4)
            return e

        def combo_row(parent, label, var, values, width=16):
            row = tk.Frame(parent, bg=P["panel"])
            row.pack(fill="x", padx=6, pady=2)
            tk.Label(row, text=f"{label}:", bg=P["panel"], fg=P["text"],
                     font=("Arial", 8), width=20, anchor="w").pack(side="left")
            cb = ttk.Combobox(row, textvariable=var, values=values,
                              font=("Arial", 8), state="readonly", width=width)
            cb.pack(side="left", padx=4)
            return cb

        # Общие параметры
        f1 = section("Сценарий")
        self._p_name  = tk.StringVar(value="Мой сценарий")
        self._p_total = tk.StringVar(value="300")
        self._p_scale = tk.StringVar(value="0.6")
        entry_row(f1, "Название", self._p_name, width=18)
        entry_row(f1, "Длительность (мин)", self._p_total, width=8)
        entry_row(f1, "Масштаб (м/пкс)", self._p_scale, width=8)

        # Параметры пожара
        f2 = section("Параметры пожара")
        self._p_fuel  = tk.StringVar(value="бензин")
        self._p_rank  = tk.StringVar(value="№2")
        self._p_roof  = tk.StringVar(value="Конусная кровля (0%)")
        combo_row(f2, "Горючее", self._p_fuel, FUELS)
        combo_row(f2, "Ранг пожара", self._p_rank, FIRE_RANKS)
        combo_row(f2, "Тип кровли", self._p_roof, list(ROOF_TYPES.keys()))

        # Параметры тушения
        f3 = section("Тушение и ресурсы")
        self._p_foam_conc  = tk.StringVar(value="12")
        self._p_n_stations = tk.StringVar(value="3")
        self._p_water_src  = tk.StringVar(value="гидранты")
        entry_row(f3, "Запас пенообразователя (т)", self._p_foam_conc, width=6)
        entry_row(f3, "Кол-во ПЧ по расписанию", self._p_n_stations, width=6)
        combo_row(f3, "Осн. водоисточник", self._p_water_src,
                  ["гидранты", "река / ПНС", "водоём", "комбинация"])

        # Информация о нормативах
        f4 = section("Нормативный расчёт (авто)")
        self._norms_var = tk.StringVar(value="—")
        tk.Label(f4, textvariable=self._norms_var,
                 font=("Consolas", 8), bg=P["panel"], fg=P["hi"],
                 justify="left", wraplength=260).pack(anchor="w", padx=6, pady=4)
        tk.Button(f4, text="🔄 Пересчитать", bg=P["info"], fg="#fff",
                  font=("Arial", 8), relief="flat", padx=8, pady=3,
                  command=self._recalc_norms).pack(anchor="w", padx=6, pady=3)

    def _build_object_tab(self, parent):
        """Вкладка свойств выбранного объекта."""
        self._obj_frame = tk.Frame(parent, bg=P["bg"])
        self._obj_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self._render_object_props()

    def _render_object_props(self):
        """Перерисовать панель свойств для выбранного объекта."""
        for w in self._obj_frame.winfo_children():
            w.destroy()

        if self._selected is None:
            tk.Label(self._obj_frame,
                     text="Выберите объект на карте\n(инструмент ↖ Выбор)",
                     font=("Arial", 9), bg=P["bg"], fg=P["text2"],
                     justify="center").pack(expand=True)
            return

        obj = self._selected
        spec = obj.spec

        # Заголовок
        hf = tk.Frame(self._obj_frame, bg=P["panel2"])
        hf.pack(fill="x")
        tk.Label(hf, text=f"  {spec['label']}  ", font=("Arial", 9, "bold"),
                 bg=P["panel2"], fg=P["hi"]).pack(side="left", padx=8, pady=4)

        # Поля
        fields_frame = tk.Frame(self._obj_frame, bg=P["bg"])
        fields_frame.pack(fill="x", padx=4, pady=4)

        def row(label, var, width=14):
            r = tk.Frame(fields_frame, bg=P["bg"])
            r.pack(fill="x", pady=2)
            tk.Label(r, text=f"{label}:", bg=P["bg"], fg=P["text2"],
                     font=("Arial", 8), width=18, anchor="w").pack(side="left")
            e = tk.Entry(r, textvariable=var, bg=P["panel2"], fg=P["text"],
                         insertbackground=P["text"], font=("Consolas", 8),
                         width=width, relief="flat", bd=2)
            e.pack(side="left", padx=4)
            return var

        # Общие поля
        self._ov_label = tk.StringVar(value=obj.label)
        row("Подпись", self._ov_label)

        self._ov_x = tk.StringVar(value=f"{obj.x:.0f}")
        self._ov_y = tk.StringVar(value=f"{obj.y:.0f}")
        self._ov_r = tk.StringVar(value=f"{obj.r:.0f}")
        row("X (пкс)", self._ov_x, 7)
        row("Y (пкс)", self._ov_y, 7)
        row("Радиус (пкс)", self._ov_r, 7)

        # РВС-специфичные поля
        if obj.otype in (OBJ_RVS_FIRE, OBJ_RVS_NEAR):
            self._ov_vol  = tk.StringVar(value=f"{obj.volume_m3:.0f}")
            self._ov_diam = tk.StringVar(value=f"{obj.diameter_m:.1f}")
            self._ov_fuel = tk.StringVar(value=obj.fuel)
            row("Объём РВС (м³)", self._ov_vol)
            row("Диаметр (м)",    self._ov_diam, 7)
            tk.Button(fields_frame, text="↩ Авто-диаметр",
                      bg=P["grid"], fg=P["text"], font=("Arial", 7),
                      relief="flat", padx=6, pady=2,
                      command=self._auto_diameter).pack(anchor="w", pady=2)

        # Кнопки
        btn_f = tk.Frame(self._obj_frame, bg=P["bg"])
        btn_f.pack(fill="x", padx=4, pady=6)
        tk.Button(btn_f, text="✅ Применить",
                  bg=P["success"], fg="#fff", font=("Arial", 8, "bold"),
                  relief="flat", padx=10, pady=4,
                  command=self._apply_object_props).pack(side="left", padx=4)
        tk.Button(btn_f, text="🗑 Удалить",
                  bg=P["danger"], fg="#fff", font=("Arial", 8, "bold"),
                  relief="flat", padx=10, pady=4,
                  command=self._delete_selected).pack(side="left", padx=4)

        # Подсказка
        tk.Label(self._obj_frame, text=spec["hint"],
                 font=("Arial", 7), bg=P["bg"], fg=P["text2"],
                 wraplength=260, justify="left").pack(anchor="w", padx=8, pady=4)

    def _build_timeline_tab(self, parent):
        """Вкладка предпросмотра автогенерируемой хронологии."""
        hdr = tk.Frame(parent, bg=P["panel2"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="  Предпросмотр хронологии (авто)",
                 font=("Arial", 8, "bold"), bg=P["panel2"], fg=P["hi"]
                 ).pack(side="left", padx=8, pady=4)
        tk.Button(hdr, text="🔄 Обновить",
                  bg=P["info"], fg="#fff", font=("Arial", 8),
                  relief="flat", padx=8, pady=2,
                  command=self._refresh_timeline_preview
                  ).pack(side="right", padx=8, pady=4)

        tf = tk.Frame(parent, bg=P["bg"])
        tf.pack(fill="both", expand=True, padx=4, pady=4)
        sb = tk.Scrollbar(tf)
        sb.pack(side="right", fill="y")
        self._tl_text = tk.Text(
            tf, yscrollcommand=sb.set, bg=P["canvas"],
            fg=P["text"], font=("Consolas", 8), state="disabled",
            wrap="word", bd=0, padx=6, pady=4, spacing1=2,
        )
        self._tl_text.pack(fill="both", expand=True)
        sb.config(command=self._tl_text.yview)
        for tag, col in [("warn", P["warn"]), ("danger", P["danger"]),
                          ("success", P["success"]), ("info", P["info"])]:
            self._tl_text.tag_config(tag, foreground=col)

    def _build_bottom_bar(self):
        bar = tk.Frame(self, bg=P["panel2"], height=50)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        # Статусная строка
        self._status_var = tk.StringVar(value="Выберите инструмент и кликните на карту для размещения объектов.")
        tk.Label(bar, textvariable=self._status_var,
                 bg=P["panel2"], fg=P["text2"], font=("Arial", 8)
                 ).pack(side="left", padx=12, pady=6)

        # Кнопки действий
        for text, cmd, color in [
            ("💾 Сохранить JSON", self._save_json,    P["info"]),
            ("📂 Загрузить JSON", self._load_json,    P["grid"]),
            ("▶  Запустить симуляцию", self._launch,  P["success"]),
        ]:
            tk.Button(bar, text=text, command=cmd, bg=color, fg="#fff",
                      font=("Arial", 9, "bold"), relief="flat",
                      padx=12, pady=6).pack(side="right", padx=4, pady=6)

    # ──────────────────────────────────────────────────────────────────────────
    # ИНСТРУМЕНТЫ
    # ──────────────────────────────────────────────────────────────────────────
    def _set_tool(self, key: str):
        self._tool = key
        for k, btn in self._tool_btns.items():
            btn.config(bg=P["toolbar_active"] if k == key else P["toolbar_btn"])
        hints = {
            OBJ_RVS_FIRE: "Кликните на карту — поставить горящий РВС. Только один.",
            OBJ_RVS_NEAR: "Кликните — поставить соседний РВС.",
            OBJ_HYDRANT:  "Кликните — поставить гидрант ПГ (~15 л/с).",
            OBJ_RIVER:    "Кликните — поставить водоём / реку для ПНС (~110 л/с).",
            OBJ_BUILDING: "Кликните — поставить здание / сооружение.",
            self.TOOL_SELECT: "Кликните объект — выбрать. Перетащите — переместить.",
        }
        self._status_var.set(hints.get(key, ""))

    # ──────────────────────────────────────────────────────────────────────────
    # РИСОВАНИЕ
    # ──────────────────────────────────────────────────────────────────────────
    def _draw_grid(self):
        """Нарисовать координатную сетку."""
        c = self._canvas
        c.delete("grid")
        if not getattr(self, "_show_grid", True):
            return
        W = c.winfo_width() or ED_W
        H = c.winfo_height() or ED_H
        for x in range(0, W, GRID_SIZE):
            c.create_line(x, 0, x, H, fill=P["grid"], width=1, tags="grid",
                          dash=(1, GRID_SIZE - 1))
        for y in range(0, H, GRID_SIZE):
            c.create_line(0, y, W, y, fill=P["grid"], width=1, tags="grid",
                          dash=(1, GRID_SIZE - 1))

    def _snap(self, x: float, y: float) -> Tuple[float, float]:
        """Привязать координаты к сетке."""
        return (round(x / GRID_SIZE) * GRID_SIZE,
                round(y / GRID_SIZE) * GRID_SIZE)

    def _draw_object(self, obj: MapObject):
        """Нарисовать один объект на холсте."""
        c = self._canvas
        # Удалить старые canvas-объекты этого MapObject
        for cid in obj.canvas_ids:
            c.delete(cid)
        obj.canvas_ids.clear()

        spec = obj.spec
        is_sel = (obj is self._selected)
        outline = P["select"] if is_sel else spec["outline"]
        width   = 3 if is_sel else 2

        x, y, r = obj.x, obj.y, obj.r

        if spec["shape"] == "circle":
            cid = c.create_oval(x-r, y-r, x+r, y+r,
                                fill=spec["fill"], outline=outline,
                                width=width, tags=("obj",))
            obj.canvas_ids.append(cid)

        elif spec["shape"] == "rect":
            cid = c.create_rectangle(x-r, y-r*0.6, x+r, y+r*0.6,
                                     fill=spec["fill"], outline=outline,
                                     width=width, tags=("obj",))
            obj.canvas_ids.append(cid)

        elif spec["shape"] == "line":
            cid = c.create_line(x-r, y, x+r, y,
                                fill=spec.get("outline", P["grid"]),
                                width=3, tags=("obj",))
            obj.canvas_ids.append(cid)

        # Метка
        if obj.label:
            tid = c.create_text(x, y + r + 10, text=obj.label,
                                fill=P["text2"], font=("Arial", 7),
                                tags=("obj",))
            obj.canvas_ids.append(tid)

        # Огонь на горящем РВС
        if obj.otype == OBJ_RVS_FIRE:
            fid = c.create_text(x, y, text="🔥", font=("Arial", int(r*0.6)),
                                tags=("obj",))
            obj.canvas_ids.append(fid)

        # Выделение
        if is_sel:
            sid = c.create_oval(x-r-4, y-r-4, x+r+4, y+r+4,
                                outline=P["select"], width=2,
                                dash=(4, 3), tags=("obj",))
            obj.canvas_ids.append(sid)

    def _draw_all_objects(self):
        """Перерисовать все объекты."""
        self._canvas.delete("obj")
        for obj in self._objects:
            obj.canvas_ids.clear()
        for obj in self._objects:
            self._draw_object(obj)

    def _toggle_grid(self):
        self._show_grid = not getattr(self, "_show_grid", True)
        self._draw_grid()

    # ──────────────────────────────────────────────────────────────────────────
    # СОБЫТИЯ МЫШИ
    # ──────────────────────────────────────────────────────────────────────────
    def _on_click(self, ev):
        x, y = self._snap(ev.x, ev.y)

        if self._tool == self.TOOL_SELECT:
            # Найти объект под курсором
            hit = next((o for o in reversed(self._objects) if o.hit_test(ev.x, ev.y)), None)
            self._selected = hit
            self._draw_all_objects()
            self._render_object_props()
            if hit:
                self._drag_start = (ev.x - hit.x, ev.y - hit.y)
                self._nb.select(1)  # переключить на вкладку «Объект»
            return

        # Размещение нового объекта
        if self._tool == OBJ_RVS_FIRE:
            # Только один горящий РВС
            self._objects = [o for o in self._objects if o.otype != OBJ_RVS_FIRE]

        spec = OBJ_SPECS[self._tool]
        vol = 5000.0 if self._tool in (OBJ_RVS_FIRE, OBJ_RVS_NEAR) else 0.0
        label_map = {
            OBJ_RVS_FIRE: f"РВС-{int(vol)}",
            OBJ_RVS_NEAR: f"РВС-{int(vol)}",
            OBJ_HYDRANT:  f"ПГ-{len([o for o in self._objects if o.otype==OBJ_HYDRANT])+1}",
            OBJ_RIVER:    "р. Водоём",
            OBJ_BUILDING: "Здание",
            OBJ_ROAD:     "",
        }
        obj = MapObject(
            otype=self._tool, x=x, y=y,
            r=spec.get("default_r", 30),
            label=label_map.get(self._tool, ""),
            volume_m3=vol,
        )
        self._objects.append(obj)
        self._selected = obj
        self._draw_all_objects()
        self._render_object_props()
        self._recalc_norms()
        self._status_var.set(
            f"Размещён: {spec['label']} в ({x:.0f}, {y:.0f}). "
            f"Переключитесь в ↖ Выбор для редактирования."
        )

    def _on_drag(self, ev):
        if self._tool == self.TOOL_SELECT and self._selected and self._drag_start:
            dx, dy = self._drag_start
            nx, ny = self._snap(ev.x - dx, ev.y - dy)
            self._selected.x = nx
            self._selected.y = ny
            self._draw_all_objects()

    def _on_release(self, ev):
        self._drag_start = None
        # Очистить призраки
        for gid in self._ghost_ids:
            self._canvas.delete(gid)
        self._ghost_ids.clear()

    def _on_motion(self, ev):
        """Показать «призрак» объекта перед размещением."""
        if self._tool == self.TOOL_SELECT:
            return
        for gid in self._ghost_ids:
            self._canvas.delete(gid)
        self._ghost_ids.clear()

        x, y = self._snap(ev.x, ev.y)
        spec = OBJ_SPECS.get(self._tool)
        if not spec:
            return
        r = spec.get("default_r", 25)

        if spec["shape"] == "circle":
            gid = self._canvas.create_oval(
                x-r, y-r, x+r, y+r,
                outline=P["hi"], width=2, dash=(4, 3), fill="")
        else:
            gid = self._canvas.create_rectangle(
                x-r, y-r*0.6, x+r, y+r*0.6,
                outline=P["hi"], width=2, dash=(4, 3), fill="")
        self._ghost_ids.append(gid)

    def _on_double_click(self, ev):
        """Двойной клик: перейти к редактированию свойств объекта."""
        hit = next((o for o in reversed(self._objects) if o.hit_test(ev.x, ev.y)), None)
        if hit:
            self._selected = hit
            self._draw_all_objects()
            self._render_object_props()
            self._nb.select(1)
            self._set_tool(self.TOOL_SELECT)

    # ──────────────────────────────────────────────────────────────────────────
    # СВОЙСТВА ОБЪЕКТА
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_object_props(self):
        """Применить изменения из панели свойств к выбранному объекту."""
        if not self._selected:
            return
        obj = self._selected
        try:
            obj.label = self._ov_label.get()
            obj.x = float(self._ov_x.get())
            obj.y = float(self._ov_y.get())
            obj.r = float(self._ov_r.get())
            if obj.otype in (OBJ_RVS_FIRE, OBJ_RVS_NEAR):
                obj.volume_m3  = float(self._ov_vol.get())
                obj.diameter_m = float(self._ov_diam.get())
        except ValueError as exc:
            messagebox.showerror("Ошибка ввода", str(exc), parent=self)
            return
        self._draw_all_objects()
        self._recalc_norms()
        self._status_var.set(f"Применены свойства: {obj.label} ({obj.spec['label']})")

    def _auto_diameter(self):
        """Вычислить диаметр РВС по объёму и радиусу на карте."""
        if not self._selected or self._selected.otype not in (OBJ_RVS_FIRE, OBJ_RVS_NEAR):
            return
        try:
            vol = float(self._ov_vol.get())
            # Типовое соотношение H ≈ D для РВС: V = π/4 · D² · H → D ≈ (4V/π)^(1/3)
            diam = (4 * vol / math.pi) ** (1/3)
            self._ov_diam.set(f"{diam:.1f}")
            # Обновить радиус на карте
            scale = float(self._p_scale.get() or 0.6)
            self._ov_r.set(f"{diam / scale / 2:.0f}")
        except Exception:
            pass

    def _delete_selected(self):
        if self._selected:
            self._objects.remove(self._selected)
            self._selected = None
            self._draw_all_objects()
            self._render_object_props()
            self._recalc_norms()

    def _clear_all(self):
        if messagebox.askyesno("Очистить карту",
                                "Удалить все объекты с карты?", parent=self):
            self._objects.clear()
            self._selected = None
            self._draw_all_objects()
            self._render_object_props()
            self._recalc_norms()

    # ──────────────────────────────────────────────────────────────────────────
    # НОРМАТИВНЫЙ РАСЧЁТ
    # ──────────────────────────────────────────────────────────────────────────
    def _recalc_norms(self):
        """Пересчитать нормативные требования и обновить вкладку параметров."""
        fire_obj = next((o for o in self._objects if o.otype == OBJ_RVS_FIRE), None)
        if not fire_obj:
            self._norms_var.set("⚠ На карте нет горящего РВС.")
            return
        try:
            scale = float(self._p_scale.get() or 0.6)
            fuel  = self._p_fuel.get()
            roof_key = self._p_roof.get()
            roof_obs = ROOF_TYPES.get(roof_key, 0.0)
        except Exception:
            return

        diam_m = fire_obj.diameter_m if fire_obj.diameter_m > 0 else fire_obj.r * scale * 2
        S = math.pi * (diam_m / 2) ** 2
        I = {"бензин": 0.065, "нефть": 0.060, "дизель": 0.048, "мазут": 0.060}.get(fuel, 0.065)
        Q_foam = I * S
        Q_cool = 0.8 * math.pi * diam_m
        n_gps = math.ceil(Q_foam / 5.64)  # ГПС-600: q=5.64 л/с

        lines = [
            f"D = {diam_m:.1f} м",
            f"S = {S:.0f} м²",
            f"I_пены = {I:.3f} л/(м²·с)",
            f"Q_пены ≥ {Q_foam:.1f} л/с",
            f"Q_охл ≥ {Q_cool:.1f} л/с",
            f"ГПС-600: ≥ {n_gps} шт",
            f"Препятствие крыши: {roof_obs*100:.0f}%",
        ]
        self._norms_var.set("\n".join(lines))

    # ──────────────────────────────────────────────────────────────────────────
    # ХРОНОЛОГИЯ
    # ──────────────────────────────────────────────────────────────────────────
    def _refresh_timeline_preview(self):
        """Обновить предпросмотр авто-хронологии."""
        fire_obj = next((o for o in self._objects if o.otype == OBJ_RVS_FIRE), None)
        if not fire_obj:
            self._tl_text.config(state="normal")
            self._tl_text.delete("1.0", "end")
            self._tl_text.insert("end", "⚠ Нет горящего РВС на карте.\n")
            self._tl_text.config(state="disabled")
            return
        try:
            rank = int(self._p_rank.get().replace("№", ""))
            total = int(self._p_total.get())
            scale = float(self._p_scale.get() or 0.6)
            fuel  = self._p_fuel.get()
            diam  = fire_obj.diameter_m if fire_obj.diameter_m > 0 else fire_obj.r * scale * 2
            S     = math.pi * (diam/2)**2
        except Exception:
            return

        tl = _auto_timeline(rank, total, fire_obj.label or "РВС", S, fuel)

        self._tl_text.config(state="normal")
        self._tl_text.delete("1.0", "end")
        self._tl_text.insert("end", f"{'─'*45}\n Авто-хронология | ранг №{rank} | {total} мин\n{'─'*45}\n\n")
        color_map = {}
        try:
            from .tank_fire_sim import P as _P
        except ImportError:
            from tank_fire_sim import P as _P
        _TAG = {_P["danger"]: "danger", _P["warn"]: "warn",
                _P["success"]: "success", _P["info"]: "info"}
        for t, lbl, desc, color in tl:
            tag = _TAG.get(color, "info")
            self._tl_text.insert("end", f"  [{lbl:>6}]  ", "")
            self._tl_text.insert("end", f"{desc}\n", tag)
        self._tl_text.config(state="disabled")

    # ──────────────────────────────────────────────────────────────────────────
    # СОХРАНЕНИЕ / ЗАГРУЗКА JSON
    # ──────────────────────────────────────────────────────────────────────────
    def _get_params(self) -> dict:
        """Собрать словарь текущих параметров из формы."""
        return {
            "name":             self._p_name.get(),
            "total_min":        int(self._p_total.get() or 300),
            "map_scale_m_per_px": float(self._p_scale.get() or 0.6),
            "fuel":             self._p_fuel.get(),
            "fire_rank":        int(self._p_rank.get().replace("№", "")),
            "roof_obstruction": ROOF_TYPES.get(self._p_roof.get(), 0.0),
            "foam_conc":        float(self._p_foam_conc.get() or 12),
            "n_stations":       int(self._p_n_stations.get() or 3),
            "water_src":        self._p_water_src.get(),
        }

    def _save_json(self):
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Сохранить сценарий",
            defaultextension=".json",
            filetypes=[("JSON сценарий", "*.json"), ("Все файлы", "*.*")],
            initialfile=f"{self._p_name.get()}.json",
        )
        if not path:
            return
        data = {
            "params":  self._get_params(),
            "objects": [o.to_dict() for o in self._objects],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._status_var.set(f"✅ Сценарий сохранён: {os.path.basename(path)}")

    def _load_json(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Загрузить сценарий",
            filetypes=[("JSON сценарий", "*.json"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            params  = data.get("params", {})
            obj_dicts = data.get("objects", [])

            # Применить параметры
            self._p_name.set(params.get("name", "Загруженный"))
            self._p_total.set(str(params.get("total_min", 300)))
            self._p_scale.set(str(params.get("map_scale_m_per_px", 0.6)))
            self._p_fuel.set(params.get("fuel", "бензин"))
            self._p_rank.set(f"№{params.get('fire_rank', 2)}")
            self._p_foam_conc.set(str(params.get("foam_conc", 12)))
            self._p_n_stations.set(str(params.get("n_stations", 3)))

            # Восстановить объекты
            self._objects.clear()
            for d in obj_dicts:
                obj = MapObject(
                    otype=d["otype"], x=d["x"], y=d["y"], r=d["r"],
                    label=d.get("label", ""),
                    volume_m3=d.get("volume_m3", 0),
                    diameter_m=d.get("diameter_m", 0),
                    fuel=d.get("fuel", "бензин"),
                )
                self._objects.append(obj)

            self._selected = None
            self._draw_all_objects()
            self._render_object_props()
            self._recalc_norms()
            self._status_var.set(f"✅ Загружен: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки", str(exc), parent=self)

    # ──────────────────────────────────────────────────────────────────────────
    # ЗАПУСК СИМУЛЯЦИИ
    # ──────────────────────────────────────────────────────────────────────────
    def _launch(self):
        """Собрать конфиг сценария и запустить симуляцию."""
        fire_obj = next((o for o in self._objects if o.otype == OBJ_RVS_FIRE), None)
        if fire_obj is None:
            messagebox.showwarning(
                "Нет горящего РВС",
                "Разместите на карте хотя бы один объект «🔥 РВС (горящий)».",
                parent=self
            )
            return

        params = self._get_params()
        try:
            cfg = build_scenario_config(self._objects, params)
        except Exception as exc:
            messagebox.showerror("Ошибка конфигурации", str(exc), parent=self)
            return

        if self._on_launch:
            self._on_launch(cfg)
            self.destroy()
        else:
            messagebox.showinfo(
                "Сценарий готов",
                f"Сценарий «{cfg['name']}» построен.\n"
                f"S = {cfg['initial_fire_area']:.0f} м², "
                f"Q_пены ≥ {cfg['foam_intensity']*cfg['initial_fire_area']:.1f} л/с.\n\n"
                f"Откройте главное окно и запустите симуляцию.",
                parent=self
            )

    # ──────────────────────────────────────────────────────────────────────────
    # СЦЕНА ПО УМОЛЧАНИЮ
    # ──────────────────────────────────────────────────────────────────────────
    def _load_default_scene(self):
        """Загрузить пример-подсказку: небольшой нефтебазовый парк."""
        self.after(50, self._place_defaults)

    def _place_defaults(self):
        W = self._canvas.winfo_width() or ED_W
        H = self._canvas.winfo_height() or ED_H
        cx, cy = W // 2, H // 2

        # Горящий РВС по центру
        fire = MapObject(OBJ_RVS_FIRE, cx, cy, r=55, label="РВС-5000",
                         volume_m3=5000)
        # Соседний РВС
        near = MapObject(OBJ_RVS_NEAR, cx, cy - 120, r=40, label="РВС-2000",
                         volume_m3=2000)
        # Гидранты
        h1 = MapObject(OBJ_HYDRANT, cx - 140, cy + 80, r=8, label="ПГ-1")
        h2 = MapObject(OBJ_HYDRANT, cx + 140, cy + 80, r=8, label="ПГ-2")
        # Водоём
        river = MapObject(OBJ_RIVER, cx, H - 60, r=60, label="Водоём / ПНС")
        # Здание
        bld = MapObject(OBJ_BUILDING, cx + 150, cy, r=30, label="Насосная")

        self._objects = [fire, near, h1, h2, river, bld]
        self._draw_grid()
        self._draw_all_objects()
        self._recalc_norms()
        self._status_var.set(
            "Пример сцены загружен. Выбери объект (↖) для редактирования или добавь новые."
        )


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК КАК САМОСТОЯТЕЛЬНОГО ПРИЛОЖЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    root.withdraw()   # скрыть пустое главное окно
    app = ScenarioEditorApp(root)
    app.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
