"""
tank_fire_sim.py
════════════════════════════════════════════════════════════════════════════════
Интерактивная симуляция управления тушением пожара резервуарного парка.
Реальный сценарий: РВС №9, ООО «РН-Морской терминал Туапсе», г. Туапсе.
14–17 марта 2025 года | Ранг пожара: №4 | Продолжительность: 81 ч 2 мин.

Запуск:  python -m saur_sim.tank_fire_sim
         python saur_sim/tank_fire_sim.py
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import math, random, os, sys
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════════════
# ЦВЕТОВАЯ ПАЛИТРА И КОНСТАНТЫ
# ══════════════════════════════════════════════════════════════════════════════
P = dict(
    bg="#1a1f2e", panel="#1e2535", panel2="#232c40", canvas="#0d1420",
    fire="#ff4500", fire2="#ff8c00", flame="#ffcc00",
    rvs9="#c0392b", rvs17="#2471a3", rvs9_cool="#884422",
    water="#00aaff", foam="#90ee90", smoke="#556677",
    building="#4a4a6a", ground="#1a3322", road="#333355",
    unit_ac="#e74c3c", unit_apt="#e67e22", unit_pns="#3498db",
    unit_panrk="#8e44ad", unit_train="#16a085", unit_ash="#f1c40f",
    hydrant="#1abc9c", river="#1a5276",
    success="#27ae60", warn="#e67e22", danger="#c0392b", info="#2980b9",
    text="#ecf0f1", text2="#95a5a6", hi="#f1c40f",
    strat="#e74c3c", tact="#e67e22", oper="#3498db",
    grid="#2c3e50", accent="#e67e22",
    phase_s1="#e74c3c", phase_s2="#e67e22", phase_s3="#f39c12",
    phase_s4="#27ae60", phase_s5="#3498db",
)

MAP_W, MAP_H = 540, 410
TOTAL_MIN    = 4862   # 81 ч 2 мин

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
    ("S2", "стратег.",  "Защита соседнего РВС №17 от воспламенения"),
    ("S3", "стратег.",  "Локализация горения в контуре РВС №9"),
    ("S4", "стратег.",  "Ликвидация горения — пенная атака"),
    ("S5", "стратег.",  "Предотвращение вскипания / выброса нефти"),
    ("T1", "тактич.",   "Создать боевые участки (БУ) по секторам"),
    ("T2", "тактич.",   "Перегруппировать силы и средства"),
    ("T3", "тактич.",   "Вызов доп. С и С (повышение ранга пожара)"),
    ("T4", "тактич.",   "Установить ПНС/ПАНРК на водоисточник"),
    ("O1", "оперативн.", "Подача Антенор-1500 на охлаждение РВС №9"),
    ("O2", "оперативн.", "Охлаждение РВС №17 (орошение + стволы)"),
    ("O3", "оперативн.", "Пенная атака (Акрон Аполло/Муссон/ЛС-С330)"),
    ("O4", "оперативн.", "Разведка пожара — уточнение обстановки"),
    ("O5", "оперативн.", "Ликвидация розлива горящего топлива"),
    ("O6", "оперативн.", "Сигнал отхода — экстренный вывод ЛС"),
]
N_ACT = len(ACTIONS)
LEVEL_C = {"стратег.": P["strat"], "тактич.": P["tact"], "оперативн.": P["oper"]}

# ── Допустимые действия по фазам ──────────────────────────────────────────────
PHASE_VALID: Dict[str, List[int]] = {
    "S1": [0, 2, 7, 8, 12],
    "S2": [1, 2, 5, 8, 9, 10, 12],
    "S3": [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13],
    "S4": [0, 3, 4, 6, 11, 12, 13, 14],
    "S5": [10, 12, 13, 14],
}

# ── Реальная хронология событий (мин от 03:20, метка времени, описание, цвет) ─
TIMELINE: List[Tuple[int, str, str, str]] = [
    (0,    "03:20", "Сообщение о загорании РВС №9 (V=20 000 м³, прямогонный бензин)", P["warn"]),
    (1,    "03:21", "Направлены подразделения по рангу пожара №4 (АЦ-16, АПТ-4, АР-1, ПНС-1, ППП-1)", P["info"]),
    (2,    "03:22", "Направлены скорая помощь, ДПС; объявлен сбор ЛС 6 ПСО", P["info"]),
    (5,    "03:25", "РТП-1 прибыл: горение РВС №9 по всей площади зеркала, S=1250 м²; ранг №4 подтверждён", P["danger"]),
    (10,   "03:30", "Прибытие 2 АЦ, 1 АШ (РТП-1 — Довгаль В.Б.), 3 АЦ + 2 АПТ + ППП ПЧ-23", P["info"]),
    (11,   "03:31", "Поданы 3 ствола Антенор-1500: охлаждение РВС №9 с ЮГА, ВОСТОКА, ЗАПАДА", P["info"]),
    (12,   "03:32", "Создан оперативный штаб (ОШ); НШ — Владимиров Д.В.; назначены НБУ-1,2, НТ, ОТ", P["info"]),
    (15,   "03:35", "РТП-1: 3 ствола поданы; охлаждение РВС №17 — кольца орошения", P["info"]),
    (20,   "03:40", "РТП-1 запросил: ПНС-110, АР-2, АКП-50 10 ПСЧ 6 ПСО", P["warn"]),
    (50,   "04:10", "Прибыли: ПНС-110 ПЧ-12, ПНС-110 10ПСЧ 6ПСО, АНР-130 ПЧ-23", P["info"]),
    (55,   "04:15", "ПНС-110 ПЧ-12 → р. Туапсе (с помощью АР-2); 4-й ствол Антенор на охлаждение (ЮГ)", P["info"]),
    (110,  "05:10", "ПНС-110 10ПСЧ → р. Туапсе (под мостом); 4-ходовое разветвление; 5-й ствол (ВОСТОК)", P["info"]),
    (125,  "05:25", "АНР-130 → р. Туапсе (под мостом); 6-й ствол Антенор (ЗАПАД)", P["info"]),
    (159,  "05:59", "Прибытие дежурной смены СПТ на АШ (РТП-2 — Потахов А.В.)", P["info"]),
    (160,  "06:00", "РТП-2 принял руководство; ранг №4 подтверждён; штаб переназначен", P["info"]),
    (231,  "07:11", "РТП-3 (ВрИО нач. ГУ МЧС — Платонихин Н.В.) принял руководство", P["info"]),
    (283,  "08:03", "Прибыл ПАНРК СПСЧ; руководитель — Голубенко В.В.", P["info"]),
    (330,  "08:50", "НБУ-2 докладывает: готовность к пенной атаке", P["warn"]),
    (340,  "09:00", "⚡ ПЕННАЯ АТАКА №1: 2 Акрон Аполло с ППП ПЧ-23 и ПЧ-12", P["warn"]),
    (360,  "09:20", "❌ Атака №1 ПРЕКРАЩЕНА: выход из строя ППП ПЧ-12", P["danger"]),
    (361,  "09:21", "Замена ППП ПЧ-12 → ППП ПЧ-18", P["info"]),
    (370,  "09:30", "ПНС контейнерного типа → водоём очистных сооружений; подвоз воды АЦ", P["info"]),
    (400,  "10:00", "ПАНРК СПСЧ → глубоководный причал порта; перекачка через АР-2", P["info"]),
    (440,  "10:40", "НТ докладывает: бесперебойная подача ОВ обеспечена от ПНС, ПАНРК, АНР", P["success"]),
    (470,  "11:10", "Прибыл ПАНРК 39 ПСЧ 15 ПСО", P["info"]),
    (495,  "11:35", "Переназначены НБУ-1 (ЮГ+лаб.), НБУ-2 (ВОСТОК), НБУ-3 (ЗАПАД); 3 БУ", P["info"]),
    (502,  "11:42", "НБУ-2: установлен ЛС-С330 (расход 330 л/с) около РВС №3", P["info"]),
    (505,  "11:45", "6 лафетных стволов на РВС №9, 2+1 на РВС №17; 6 ед. на открытые водоисточники", P["info"]),
    (548,  "12:28", "Прибыл пожарный поезд (г. Туапсе) → сливо-наливная эстакада", P["info"]),
    (557,  "12:37", "🚨 СВИЩ! Розлив горящего бензина 300 м² с южной стороны; S=1550 м²", P["danger"]),
    (558,  "12:38", "2 ствола Антенор-1500 → тушение розлива", P["warn"]),
    (580,  "13:00", "✅ Розлив ликвидирован (300 м²); общая S вернулась к 1250 м²", P["success"]),
    (758,  "15:00", "⚡ ПЕННАЯ АТАКА №2: ЛС-С330 + Акрон Аполло с ППП ПЧ-23 + Муссон-125 с ППП ПЧ-18", P["warn"]),
    (775,  "15:15", "❌ Атака №2 ПРЕКРАЩЕНА: каркас крыши внутри РВС №9, разрушение пены", P["danger"]),
    (924,  "17:04", "Прибыл пожарный поезд (г. Белореченск) → сливо-наливная эстакада", P["info"]),
    (1242, "20:42", "🔥 Возгорание заводской столовой 50 м² (тепловое воздействие от РВС №9)", P["danger"]),
    (1243, "20:43", "Звено ГДЗС от АЦ-1 33 ПСЧ → тушение столовой (ствол Дельта-500)", P["warn"]),
    (1247, "20:47", "✅ Пожар в столовой ликвидирован (50 м²)", P["success"]),
    (1260, "21:00", "7 стволов на РВС №9, 5 на РВС №17; 3 БУ; 10 магистральных линий", P["info"]),
    (1820, "18:00", "⚡ ПЕННАЯ АТАКА №3 (15.03): Акрон Аполло + Муссон-125 с ППП", P["warn"]),
    (1848, "18:28", "❌ Атака №3 ПРЕКРАЩЕНА: каркасы, карманы, высокая интенсивность горения", P["danger"]),
    (3375, "16:15", "⚡ ПЕННАЯ АТАКА №4 (16.03): Акрон Аполло + Муссон-125", P["warn"]),
    (3395, "16:35", "❌ Атака №4 ПРЕКРАЩЕНА: аналогично — каркасы и карманы", P["danger"]),
    (3480, "18:00", "⚡ ПЕННАЯ АТАКА №5 (16.03): Акрон Аполло + Муссон-125", P["warn"]),
    (3508, "18:28", "❌ Атака №5 ПРЕКРАЩЕНА", P["danger"]),
    (3510, "23:30", "🔒 ПОЖАР ЛОКАЛИЗОВАН на площади 1250 м²", P["success"]),
    (4740, "09:00", "⚡ ПЕННАЯ АТАКА №6 (17.03): Антенор-1500 + Муссон-125 + 2×ГПС-1000 с АКП-50", P["warn"]),
    (4760, "09:40", "✅ Видимое горение в РВС №9 ОТСУТСТВУЕТ! Подача ОВ продолжается", P["success"]),
    (4862, "12:22", "🏁 ПОЖАР ЛИКВИДИРОВАН. Продолжается охлаждение резервуаров.", P["success"]),
]
_TL_LOOKUP: Dict[int, List] = {}
for _ev in TIMELINE:
    _TL_LOOKUP.setdefault(_ev[0], []).append(_ev)

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
        ("O1", "Подать Антенор-1500 на охлаждение РВС №9", "3 ствола: с ЮГА, ВОСТОКА, ЗАПАДА одновременно"),
        ("O2", "Охлаждение соседнего РВС №17", "Стационарные кольца орошения + 1 ствол с ВОСТОКА"),
        ("T4", "Установить ПА на водоисточники", "АЦ-1 → ПГ-106; АПТ-1 → ПГ-108; АПТ-2 → ПГ-107"),
        ("T2", "Определить границы боевых участков", "БУ-1 (ЮГ+лаб), БУ-2 (ВОСТОК+РВС №17), задача НБУ"),
    ],
    "S3": [
        ("T4", "Установить ПНС на р. Туапсе", "ПНС-110 под мост → 4-ходовое разветвление → Антенор-1500"),
        ("O1", "Наращивать стволы до 6–7 лафетных", "Задействовать ПАНРК СПСЧ с глубоководного причала"),
        ("S2", "Усилить защиту РВС №17", "3 ствола + кольца орошения; НБУ-2 контролирует температуру"),
        ("T3", "Запросить ПАНРК, пожарные поезда", "ПАНРК → причал порта; поезда → сливо-наливная эстакада"),
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
    """Простой Q-learning агент для управления тушением пожара."""

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
        ph  = {"S1": 0,"S2": 1,"S3": 2,"S4": 3,"S5": 4}.get(s.get("phase","S1"), 0)
        tr  = min(3, s.get("n_trunks", 0) // 2)          # 0–3
        pns = min(3, s.get("n_pns", 0))                   # 0–3
        fr  = int(s.get("foam_ready", False))              # 0–1
        sp  = int(s.get("spill", False))                   # 0–1
        fa  = min(3, s.get("foam_attacks", 0))             # 0–3
        bu  = min(3, s.get("n_bu", 0))                     # 0–3
        return int((ph*64 + tr*16 + pns*4 + fr*2 + sp + fa*8 + bu*2) % 128)

    def select_action(self, s: dict, mask: Optional[np.ndarray] = None,
                      training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            valid = np.where(mask)[0] if mask is not None else np.arange(self.n_actions)
            return int(self.rng.choice(valid)) if len(valid) > 0 else 0
        q = self.Q[self.state_to_idx(s)].copy()
        if mask is not None:
            q[~mask] = -1e9
        return int(np.argmax(q))

    def update(self, s: dict, a: int, r: float, s_next: dict, done: bool):
        i, j = self.state_to_idx(s), self.state_to_idx(s_next)
        td = r + (0.0 if done else self.gamma * np.max(self.Q[j]))
        self.Q[i, a] += self.alpha * (td - self.Q[i, a])
        self._ep_reward += r
        self.action_counts[a] += 1

    def end_episode(self):
        self.episode_rewards.append(self._ep_reward)
        self._ep_reward = 0.0
        self.epsilon = max(0.05, self.epsilon * 0.99)

    def q_values(self, s: dict) -> np.ndarray:
        return self.Q[self.state_to_idx(s)].copy()


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


class TankFireSim:
    """Дискретно-событийная симуляция тушения пожара РВС №9."""

    def __init__(self, seed: int = 42, training: bool = True):
        self.rng      = random.Random(seed)
        self.np_rng   = np.random.RandomState(seed)
        self.agent    = QLAgent(seed=seed)
        self.training = training
        self.reset()

    # ── Инициализация ──────────────────────────────────────────────────────────
    def reset(self):
        self.t              = 0
        self.phase          = "S1"
        self.fire_area      = 1250.0
        self.n_trunks_burn  = 0
        self.n_trunks_nbr   = 0
        self.n_pns          = 0          # ПНС на водоисточниках
        self.n_bu           = 0
        self.has_shtab      = False
        self.foam_attacks   = 0
        self.foam_ready     = False
        self.foam_conc      = 12.0       # запас пенообразователя (т)
        self.spill          = False
        self.spill_area     = 0.0
        self.secondary_fire = False
        self.localized      = False
        self.extinguished   = False
        self.water_flow     = 0.0
        self.last_action    = 12         # O4: разведка по умолчанию

        # История для графиков
        self.h_fire:   List[Tuple[int,float]] = [(0, 1250.0)]
        self.h_water:  List[Tuple[int,float]] = [(0, 0.0)]
        self.h_risk:   List[Tuple[int,float]] = [(0, 0.3)]
        self.h_trunks: List[Tuple[int,int]]   = [(0, 0)]
        self.h_reward: List[float]            = []

        # Журнал событий (t, цвет, текст)
        self.events: List[Tuple[int,str,str]] = [
            (0, P["warn"], "Поступило сообщение о загорании РВС №9")
        ]
        self._last_state: Optional[dict] = None
        self._scripted_triggered = set()

    # ── Состояние для агента ───────────────────────────────────────────────────
    def _state(self) -> dict:
        return dict(
            phase=self.phase,
            n_trunks=self.n_trunks_burn,
            n_pns=self.n_pns,
            foam_ready=self.foam_ready,
            spill=self.spill,
            foam_attacks=self.foam_attacks,
            n_bu=self.n_bu,
        )

    def _mask(self) -> np.ndarray:
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
        m[14] = self._risk() > 0.85 # O6 — отход только при критическом риске
        if not m.any():
            m[12] = True            # запасной: разведка
        return m

    def _risk(self) -> float:
        base = {"S1":0.25,"S2":0.4,"S3":0.65,"S4":0.5,"S5":0.2}.get(self.phase,0.3)
        if self.spill:          base += 0.20
        if self.secondary_fire: base += 0.10
        if self.foam_attacks >= 3 and not self.localized: base += 0.10
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
                self._log(P["info"], f"O1: подан ствол Антенор-1500 → итого {self.n_trunks_burn} на РВС №9")

        elif code == "O2":  # Охлаждение соседнего
            if self.n_trunks_nbr < 5:
                self.n_trunks_nbr += 1
                self.water_flow += 30.0
                r = 0.3
                self._log(P["info"], f"O2: охлаждение РВС №17 — {self.n_trunks_nbr} ствола")

        elif code in ("O3","S4"):  # Пенная атака
            if self.foam_ready and self.foam_conc > 0:
                self.foam_attacks += 1
                self.foam_conc -= 1.8
                ok = self._foam_success()
                if ok:
                    self.extinguished = True
                    r = 12.0
                    self._log(P["success"], f"⚡ ПЕННАЯ АТАКА №{self.foam_attacks}: УСПЕШНА — горение ликвидировано!")
                else:
                    r = -0.4
                    reasons = ["каркас крыши блокирует подачу пены",
                               "образование карманов + высокая интенсивность",
                               "разрушение пены при подаче",
                               "недостаточный суммарный расход пены"]
                    self._log(P["danger"], f"❌ Атака №{self.foam_attacks} не дала результата: {self.rng.choice(reasons)}")
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
                self.fire_area = max(1250.0, self.fire_area - 300)
                r = 1.5
                self._log(P["success"], "✅ O5: розлив горящего топлива ликвидирован (−300 м²)")

        elif code == "O6":  # Отход
            r = -1.0
            self._log(P["danger"], "O6: сигнал отхода — экстренный вывод ЛС из опасной зоны")

        elif code == "T1":  # Создать БУ
            if self.n_bu < 3:
                self.n_bu += 1
                r = 0.5
                names = ["БУ-1 (ЮГ + лаборатория)", "БУ-2 (ВОСТОК + РВС №17)", "БУ-3 (ЗАПАД)"]
                self._log(P["info"], f"T1: создан {names[self.n_bu-1]}")

        elif code == "T2":  # Перегруппировка
            r = 0.2
            self._log(P["info"], "T2: перегруппировка С и С по секторам; оптимизация позиций стволов")

        elif code == "T3":  # Вызов подкрепления
            r = 0.4
            self._log(P["warn"], "T3: запрос доп. С и С — ПНС, ПАНРК, пожарный поезд")

        elif code == "T4":  # Наладить водоснабжение
            if self.n_pns < 4:
                self.n_pns += 1
                self.water_flow += 110.0
                r = 0.7
                srcs = ["р. Туапсе (ПГС под мостом)", "р. Туапсе (ПАНРК)", "порт (ПАНРК СПСЧ)", "водоём очистных сооружений"]
                self._log(P["info"], f"T4: ПНС/ПАНРК №{self.n_pns} → {srcs[self.n_pns-1]}")

        elif code == "S1":  # Спасение людей
            r = 0.8
            self._log(P["danger"], "S1: РН — спасение людей; АСА и АЛ направлены к очагу угрозы")

        elif code == "S2":  # Защита соседнего
            r = 0.5
            if not self.has_shtab:
                self.has_shtab = True
            self._log(P["info"], "S2: РН — защита РВС №17; усиление охлаждения с ВОСТОЧНОЙ стороны")

        elif code == "S3":  # Локализация
            r = 0.3
            self._log(P["info"], "S3: РН — локализация; удерживать огонь в контуре РВС №9")

        elif code == "S5":  # Предотвращение вскипания
            r = 0.6
            self._log(P["warn"], "S5: РН — предотвращение вскипания; максимальное охлаждение стенок")

        # Обновить готовность к пенной атаке
        self.foam_ready = (
            self.foam_conc > 0
            and self.n_pns >= 1
            and self.n_trunks_burn >= 3
        )

        # Постоянный штраф за площадь пожара
        r -= 0.005 * self.fire_area / 1000
        return r

    def _foam_success(self) -> bool:
        """Вероятность успеха пенной атаки."""
        p = 0.03
        if self.water_flow > 500:     p += 0.08
        if self.foam_attacks >= 5:    p += 0.35  # учёт опыта (реальный: 6-я атака)
        if self.n_pns >= 3:           p += 0.15  # достаточно насосных станций
        if self.n_trunks_burn >= 6:   p += 0.05  # хорошее охлаждение
        # Случайная составляющая
        return self.rng.random() < min(0.9, p)

    def _log(self, color: str, text: str):
        self.events.append((self.t, color, text))

    # ── Основной шаг симуляции ─────────────────────────────────────────────────
    def step(self, dt: int = 5) -> SimSnapshot:
        self.t += dt

        # Применить скриптованные события из хронологии
        for step_t in range(self.t - dt + 1, self.t + 1):
            for ev in _TL_LOOKUP.get(step_t, []):
                if step_t not in self._scripted_triggered:
                    self._scripted_triggered.add(step_t)
                    self._apply_scripted(step_t, ev[2])
                    self._log(ev[3], f"[{ev[1]}] {ev[2]}")

        # Обновить физику пожара
        self._update_fire()

        # Обновить фазу
        self._update_phase()

        # RL-решение
        state = self._state()
        mask  = self._mask()
        a     = self.agent.select_action(state, mask, training=self.training)
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
        )

    def _apply_scripted(self, t: int, desc: str):
        """Применить автоматические изменения из хронологии."""
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
        if "пнс" in lo and ("туапсе" in lo or "реке" in lo or "мост" in lo):
            self.n_pns = min(4, self.n_pns + 1)
            self.water_flow += 110.0
        if "свищ" in lo or ("розлив" in lo and "горящего" in lo and "бензина" in lo):
            self.spill      = True
            self.spill_area = 300.0
            self.fire_area  = 1550.0
        if "розлив ликвидирован" in lo or ("розлив" in lo and "ликвидир" in lo):
            self.spill      = False
            self.fire_area  = 1250.0
        if "возгорание столовой" in lo:
            self.secondary_fire = True
        if "столовая потушена" in lo or "столовой ликвидирован" in lo:
            self.secondary_fire = False
        if "локализован" in lo and "пожар" in lo:
            self.localized = True
        if "ликвидирован" in lo and ("пожар" in lo or "горение" in lo) and "розлив" not in lo:
            self.extinguished = True
            self.fire_area    = 0.0
        if "готовность к пенной" in lo:
            self.foam_ready = (self.foam_conc > 0 and self.n_pns >= 1)
            self.foam_conc  = max(self.foam_conc, 4.0)

    def _update_fire(self):
        if self.extinguished:
            self.fire_area = 0.0
            return
        if self.localized:
            # Медленное уменьшение при локализации
            self.fire_area = max(800.0, self.fire_area - self.rng.uniform(0, 1))
        elif self.n_trunks_burn < 4:
            # Рост при недостаточном охлаждении
            self.fire_area = min(2500.0, self.fire_area + self.rng.uniform(0, 3))

    def _update_phase(self):
        if   self.extinguished:          self.phase = "S5"
        elif self.t >= 4740:             self.phase = "S4"
        elif self.t >= 160:              self.phase = "S3"
        elif self.t >= 10:               self.phase = "S2"
        else:                            self.phase = "S1"


# ══════════════════════════════════════════════════════════════════════════════
# ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ══════════════════════════════════════════════════════════════════════════════
class TankFireApp(tk.Tk):

    SPEEDS = {"1×": 1, "5×": 5, "15×": 15, "60×": 60, "300×": 300}
    STEP_MIN = 5   # минут на шаг симуляции
    TICK_MS  = 80  # мс между шагами GUI

    def __init__(self):
        super().__init__()
        self.title("Симуляция тушения пожара РВС — г. Туапсе, 14–17.03.2025")
        self.configure(bg=P["bg"])
        self.resizable(True, True)
        self.minsize(1100, 760)

        self.sim       = TankFireSim(seed=42, training=True)
        self._running  = False
        self._after_id: Optional[str] = None
        self._speed    = 15
        self._anim_t   = 0          # animation counter for fire flicker
        self._snap: Optional[SimSnapshot] = None

        self._build_ui()
        self._draw_map()
        self.after(200, self._update_charts)

    # ─────────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Заголовок ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=P["panel"], height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🔥  СИМУЛЯЦИЯ УПРАВЛЕНИЯ ТУШЕНИЕМ ПОЖАРА РВС",
                 font=("Arial", 15, "bold"), bg=P["panel"], fg=P["hi"]).pack(side="left", padx=16, pady=8)
        tk.Label(hdr,
                 text="ООО «РН-Морской терминал Туапсе» | РВС №9 (V=20 000 м³) | 14–17.03.2025 | Ранг №4",
                 font=("Arial", 9), bg=P["panel"], fg=P["text2"]).pack(side="left")

        # ── Главная панель ────────────────────────────────────────────────────
        paned = tk.PanedWindow(self, orient="horizontal", bg=P["bg"],
                               sashwidth=5, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        left = self._build_left(paned)
        right = self._build_right(paned)
        paned.add(left,  minsize=350)
        paned.add(right, minsize=620)
        paned.paneconfig(left,  width=MAP_W + 20)
        paned.paneconfig(right, width=640)

        # ── Панель управления ─────────────────────────────────────────────────
        self._build_controls()

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
        self.canvas.pack(padx=4, pady=4)

        # Статусная таблица
        sf = tk.Frame(frm, bg=P["panel2"])
        sf.pack(fill="x", padx=4, pady=2)
        self._status_vars = {}
        rows = [
            ("Время симуляции",  "sim_time",   "03:20"),
            ("Фаза пожара",      "phase",      "S1 — Обнаружение"),
            ("Площадь пожара",   "fire_area",  "1250 м²"),
            ("Расход ОВ",        "flow",       "0 л/с"),
            ("Стволов на РВС",   "trunks",     "0 / 0"),
            ("ПНС на воде",      "pns",        "0"),
            ("Боевых участков",  "bu",         "0 из 3"),
            ("Пенных атак",      "foam",       "0"),
            ("Риск",             "risk",       "НИЗКИЙ"),
            ("Действие RL",      "action",     "O4 — Разведка"),
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

        # Tab 1 — Хронология
        t1 = tk.Frame(nb, bg=P["bg"])
        nb.add(t1, text="📋  Хронология")
        self._build_timeline_tab(t1)

        # Tab 2 — Метрики
        t2 = tk.Frame(nb, bg=P["bg"])
        nb.add(t2, text="📊  Метрики")
        self._build_metrics_tab(t2)

        # Tab 3 — RL-агент
        t3 = tk.Frame(nb, bg=P["bg"])
        nb.add(t3, text="🤖  RL-агент")
        self._build_rl_tab(t3)

        # Tab 4 — Справочник
        t4 = tk.Frame(nb, bg=P["bg"])
        nb.add(t4, text="📖  Справочник действий")
        self._build_reference_tab(t4)

        # Tab 5 — Настройки
        t5 = tk.Frame(nb, bg=P["bg"])
        nb.add(t5, text="⚙️  Настройки")
        self._build_settings_tab(t5)

        self._nb = nb
        return frm

    # ── Tab: Хронология ──────────────────────────────────────────────────────
    def _build_timeline_tab(self, parent):
        hdr = tk.Frame(parent, bg=P["panel2"])
        hdr.pack(fill="x", padx=4, pady=(4,0))
        tk.Label(hdr, text="Журнал событий | 14–17 марта 2025",
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
        self._log_text.insert("end", " ХРОНОЛОГИЯ РЕАЛЬНОГО ПОЖАРА (РВС №9, Туапсе)\n", "hi")
        self._log_text.insert("end", "═"*68 + "\n\n", "neutral")
        for t_ev, time_lbl, desc, color in TIMELINE:
            day = 14 + t_ev // 1440
            tag = "warn" if color == P["warn"] else \
                  "danger" if color == P["danger"] else \
                  "success" if color == P["success"] else "info"
            date_str = f"[{day:02d}.03 {time_lbl}]"
            self._log_text.insert("end", f"{date_str:18s} ", "neutral")
            self._log_text.insert("end", f"{desc}\n", tag)
        self._log_text.config(state="disabled")
        self._log_text.see("end")

    # ── Tab: Метрики ──────────────────────────────────────────────────────────
    def _build_metrics_tab(self, parent):
        self._fig_metrics = Figure(figsize=(8, 5.5), facecolor=P["bg"])
        self._fig_metrics.subplots_adjust(left=0.09, right=0.97, top=0.93,
                                          bottom=0.09, hspace=0.45, wspace=0.38)
        gs = gridspec.GridSpec(2, 2, figure=self._fig_metrics)

        ax_kw = dict(facecolor=P["canvas"])
        self._ax_fire   = self._fig_metrics.add_subplot(gs[0, 0], **ax_kw)
        self._ax_water  = self._fig_metrics.add_subplot(gs[0, 1], **ax_kw)
        self._ax_trunks = self._fig_metrics.add_subplot(gs[1, 0], **ax_kw)
        self._ax_risk   = self._fig_metrics.add_subplot(gs[1, 1], **ax_kw)

        for ax, title in [
            (self._ax_fire,   "Площадь пожара, м²"),
            (self._ax_water,  "Расход ОВ, л/с"),
            (self._ax_trunks, "Стволов на охлаждении РВС №9"),
            (self._ax_risk,   "Индекс риска"),
        ]:
            ax.set_title(title, color=P["text"], fontsize=8, pad=3)
            ax.tick_params(colors=P["text2"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(P["grid"])
            ax.set_facecolor(P["canvas"])
            ax.grid(True, color=P["grid"], linewidth=0.4, alpha=0.7)

        self._fc_metrics = FigureCanvasTkAgg(self._fig_metrics, master=parent)
        self._fc_metrics.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    # ── Tab: RL-агент ─────────────────────────────────────────────────────────
    def _build_rl_tab(self, parent):
        self._fig_rl = Figure(figsize=(8, 5.5), facecolor=P["bg"])
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
        acts = ACTIONS_BY_PHASE.get(ph, [])
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
        modes = [("RL обучение (ε-greedy)",  "rl_train"),
                 ("RL эксплуатация (greedy)", "rl_greedy"),
                 ("Ручное управление",        "manual")]
        for text, val in modes:
            tk.Radiobutton(f_mode, text=text, variable=self._mode_var, value=val,
                           bg=P["panel"], fg=P["text"], selectcolor=P["panel2"],
                           font=("Arial", 9), activebackground=P["panel"],
                           activeforeground=P["hi"]).pack(anchor="w", padx=10, pady=2)

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

    def _apply_settings(self):
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

        # Кнопки Play/Pause/Reset
        for text, cmd, color in [
            ("▶  Пуск",   self._on_play,  P["success"]),
            ("⏸  Стоп",   self._on_pause, P["warn"]),
            ("⏮  Сброс",  self._on_reset, P["danger"]),
        ]:
            tk.Button(ctrl, text=text, command=cmd, bg=color, fg="#fff",
                      font=("Arial", 9, "bold"), relief="flat",
                      padx=12, pady=6).pack(side="left", padx=4, pady=8)

        tk.Label(ctrl, text="│", bg=P["panel2"], fg=P["grid"]).pack(side="left")

        # Скорость
        tk.Label(ctrl, text="Скорость:", bg=P["panel2"], fg=P["text"],
                 font=("Arial", 8)).pack(side="left", padx=(8,2))
        self._speed_var = tk.StringVar(value="15×")
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

    # ─────────────────────────────────────────────────────────────────────────
    # КАРТА (CANVAS)
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_map(self):
        c = self.canvas
        c.delete("all")
        sim = self.sim
        t   = sim.t

        W, H = MAP_W, MAP_H

        # ── Фон — территория терминала ────────────────────────────────────────
        c.create_rectangle(0, 0, W, H, fill=P["canvas"], outline="")
        # Земля
        c.create_rectangle(30, 20, W-10, H-50, fill="#0a1f0a", outline=P["grid"],
                           width=1, dash=(4,4))
        # Надпись объекта
        c.create_text(W//2, 12, text="ООО «РН-Морской терминал Туапсе»",
                      fill=P["text2"], font=("Arial", 7))

        # ── Река Туапсе (внизу) ───────────────────────────────────────────────
        c.create_rectangle(0, H-48, W, H, fill="#0a2a4a", outline="")
        c.create_text(W//2, H-36, text="р. Туапсе", fill=P["water"], font=("Arial", 8,"italic"))
        c.create_text(W//2, H-22, text="↑ водоисточник для ПНС-110", fill=P["text2"], font=("Arial", 6))
        # Волны
        for wx in range(40, W-40, 30):
            c.create_arc(wx, H-44, wx+20, H-32, start=0, extent=180,
                         outline=P["water"], width=1, style="arc")

        # ── Порт (глубоководный причал) ───────────────────────────────────────
        c.create_rectangle(W-70, H-80, W-20, H-52, fill="#0d2244", outline=P["water"], width=1)
        c.create_text(W-45, H-66, text="Причал\n(ПАНРК)", fill=P["water"],
                      font=("Arial", 6), justify="center")

        # ── Дороги ────────────────────────────────────────────────────────────
        c.create_line(W//2, H-48, W//2, 20, fill=P["road"], width=2, dash=(6,4))
        c.create_line(30, H//2, W-10, H//2, fill=P["road"], width=2, dash=(6,4))

        # ── РВС №17 (соседний, вверху) ───────────────────────────────────────
        cx17, cy17, r17 = 265, 105, 52
        # Охлаждающий ореол
        if sim.n_trunks_nbr > 0:
            c.create_oval(cx17-r17-8, cy17-r17-8, cx17+r17+8, cy17+r17+8,
                          outline=P["water"], width=2, dash=(4,3))
        c.create_oval(cx17-r17, cy17-r17, cx17+r17, cy17+r17,
                      fill=P["rvs17"], outline="#5599cc", width=2)
        c.create_text(cx17, cy17-8, text="РВС №17", fill="white", font=("Arial", 8,"bold"))
        c.create_text(cx17, cy17+8, text="V≈20 000 м³", fill="#aaccee", font=("Arial", 6))
        c.create_text(cx17, cy17+20, text="защита", fill=P["water"], font=("Arial", 6,"italic"))

        # ── РВС №9 (горящий, по центру) ──────────────────────────────────────
        cx9, cy9, r9 = 265, 230, 65
        fire_intensity = min(1.0, sim.fire_area / 2000) if not sim.extinguished else 0.0

        if fire_intensity > 0:
            # Дым (внешний ореол)
            smoke_r = r9 + 20 + int(10 * fire_intensity)
            smoke_alpha = int(80 + 40 * fire_intensity)
            for i in range(3):
                sr = smoke_r + i*8
                c.create_oval(cx9-sr, cy9-sr, cx9+sr, cy9+sr,
                              outline=P["smoke"], width=1, dash=(2,6))
            # Тело пожара
            fire_color = P["fire"] if fire_intensity > 0.7 else P["fire2"]
            c.create_oval(cx9-r9, cy9-r9, cx9+r9, cy9+r9,
                          fill=fire_color, outline="#ff0000", width=3)
            # Пульсирующее пламя
            pulse = 3 * math.sin(self._anim_t * 0.4)
            for angle_deg in range(0, 360, 45):
                angle = math.radians(angle_deg + self._anim_t * 3)
                fx = cx9 + (r9 + 10 + pulse) * math.cos(angle)
                fy = cy9 + (r9 + 10 + pulse) * math.sin(angle)
                fs = 5 + int(4 * fire_intensity)
                c.create_oval(fx-fs, fy-fs, fx+fs, fy+fs,
                              fill=P["flame"], outline="")
        else:
            # Потушен
            c.create_oval(cx9-r9, cy9-r9, cx9+r9, cy9+r9,
                          fill=P["rvs9_cool"], outline="#888866", width=2)

        # Контур резервуара
        c.create_oval(cx9-r9, cy9-r9, cx9+r9, cy9+r9,
                      outline="#cc2222" if fire_intensity > 0 else "#666644",
                      width=3, fill="")
        c.create_text(cx9, cy9-6, text="РВС №9", fill="white", font=("Arial", 9,"bold"))
        c.create_text(cx9, cy9+10, text=f"S={sim.fire_area:.0f} м²",
                      fill=P["hi"] if fire_intensity > 0 else P["success"],
                      font=("Arial", 8,"bold"))

        # ── Розлив ────────────────────────────────────────────────────────────
        if sim.spill:
            c.create_oval(cx9-r9-30, cy9+r9-10, cx9+10, cy9+r9+40,
                          fill="#ff4400", outline="#ff6600", width=2, stipple="gray50")
            c.create_text(cx9-20, cy9+r9+20, text="Розлив 300 м²", fill=P["danger"],
                          font=("Arial", 7,"bold"))

        # ── Здания ────────────────────────────────────────────────────────────
        # Лаборатория
        lab_color = P["danger"] if sim.secondary_fire else P["building"]
        c.create_rectangle(380, 200, 450, 240, fill=lab_color, outline="#888", width=1)
        c.create_text(415, 220, text="Лаб.", fill="white", font=("Arial", 7))
        # Столовая
        canteen_color = P["danger"] if sim.secondary_fire else P["building"]
        c.create_rectangle(380, 255, 450, 295, fill=canteen_color, outline="#888", width=1)
        c.create_text(415, 275, text="Столов.", fill="white", font=("Arial", 7))
        if sim.secondary_fire:
            c.create_text(415, 295+8, text="🔥 горит", fill=P["danger"], font=("Arial", 6,"bold"))

        # Штаб (АШ)
        c.create_rectangle(50, 195, 120, 235, fill="#2d2d5e", outline=P["hi"], width=2)
        c.create_text(85, 215, text="ОШ (АШ)", fill=P["hi"], font=("Arial", 7,"bold"))

        # ── Гидранты ─────────────────────────────────────────────────────────
        for gx, gy, name in [(140, 330, "ПГ-106"), (195, 180, "ПГ-107"), (355, 140, "ПГ-108")]:
            c.create_oval(gx-7, gy-7, gx+7, gy+7, fill=P["hydrant"], outline="white", width=1)
            c.create_text(gx, gy+14, text=name, fill=P["hydrant"], font=("Arial", 6))

        # ── Водоснабжение (линии от ПНС к РВС) ───────────────────────────────
        if sim.n_pns >= 1:
            c.create_line(W//2, H-48, W//2, cy9+r9+5,
                          fill=P["water"], width=2, dash=(8,4))
        if sim.n_pns >= 2:
            c.create_line(W-45, H-60, cx9+r9+5, cy9+10,
                          fill=P["water"], width=2, dash=(8,4))
        if sim.n_pns >= 3:
            c.create_line(140, H-48, 140, H//2, cx9-r9-5, cy9+20,
                          fill=P["water"], width=2, dash=(8,4))

        # ── Стволы охлаждения РВС №9 ─────────────────────────────────────────
        trunk_positions = [
            (cx9,    cy9+r9+15, "↑Ю"),    # Юг
            (cx9+r9+15, cy9,    "→В"),    # Восток
            (cx9-r9-15, cy9,    "←З"),    # Запад
            (cx9,    cy9-r9-15, "↓С"),    # Север
            (cx9+45, cy9+55,    "ЮВ"),
            (cx9-45, cy9+55,    "ЮЗ"),
            (cx9+55, cy9-45,    "СВ"),
        ]
        for i, (tx, ty, side) in enumerate(trunk_positions[:sim.n_trunks_burn]):
            c.create_line(tx, ty, cx9 + 0.5*(tx-cx9),
                          cy9 + 0.5*(ty-cy9),
                          fill=P["water"], width=3, arrow="last")
            c.create_oval(tx-5, ty-5, tx+5, ty+5, fill=P["unit_pns"], outline="")
            c.create_text(tx, ty+10, text=f"А{i+1}", fill=P["water"], font=("Arial", 5))

        # Стволы на РВС №17
        nbr_positions = [
            (cx17+r17+12, cy17, "В"), (cx17-r17-12, cy17, "З"),
            (cx17, cy17+r17+12, "Ю"), (cx17+35, cy17-35, "СВ"),
            (cx17-35, cy17-35, "СЗ"),
        ]
        for i, (tx, ty, _) in enumerate(nbr_positions[:sim.n_trunks_nbr]):
            c.create_line(tx, ty, cx17 + 0.5*(tx-cx17), cy17 + 0.5*(ty-cy17),
                          fill="#4488ff", width=2, arrow="last")

        # ── Боевые участки (секторы) ──────────────────────────────────────────
        bu_configs = [
            (cx9, cy9+r9+55, "#e74c3c", "БУ-1 (ЮГ)"),
            (cx9+r9+65, cy9, "#3498db", "БУ-2 (ВОСТОК)"),
            (cx9-r9-65, cy9, "#2ecc71", "БУ-3 (ЗАПАД)"),
        ]
        for i, (bx, by, bc, bname) in enumerate(bu_configs[:sim.n_bu]):
            c.create_rectangle(bx-30, by-12, bx+30, by+12,
                               fill=bc, outline="", stipple="gray50")
            c.create_text(bx, by, text=bname, fill="white", font=("Arial", 6,"bold"))

        # ── Легенда ───────────────────────────────────────────────────────────
        legend_y = 34
        legend_x = W - 10
        items = [
            (P["rvs9"],     "РВС №9 (пожар)"),
            (P["rvs17"],    "РВС №17 (защита)"),
            (P["water"],    f"Стволы ({sim.n_trunks_burn}+{sim.n_trunks_nbr})"),
            (P["unit_pns"], f"ПНС ({sim.n_pns} ед.)"),
        ]
        for color, label in items:
            c.create_oval(legend_x-110, legend_y-5, legend_x-100, legend_y+5,
                          fill=color, outline="")
            c.create_text(legend_x-95, legend_y, text=label, anchor="w",
                          fill=P["text2"], font=("Arial", 6))
            legend_y += 14

        # ── Текущее действие ──────────────────────────────────────────────────
        a = sim.last_action
        code, level, desc = ACTIONS[a]
        lc = LEVEL_C[level]
        c.create_rectangle(30, H-110, W-30, H-75, fill="#0a0a1a", outline=lc, width=1)
        c.create_text(W//2, H-100, text=f"Действие РТП: [{code}] {desc}",
                      fill=lc, font=("Arial", 7,"bold"))
        c.create_text(W//2, H-85, text=f"Уровень: {level} | Фаза: {sim.phase} | t={self._fmt_time(sim.t)}",
                      fill=P["text2"], font=("Arial", 6))

        # ── Статус пожара ─────────────────────────────────────────────────────
        status_text = ""
        status_color = P["text"]
        if sim.extinguished:
            status_text = "✅ ПОЖАР ЛИКВИДИРОВАН"
            status_color = P["success"]
        elif sim.localized:
            status_text = "🔒 ПОЖАР ЛОКАЛИЗОВАН"
            status_color = P["info"]
        elif sim.foam_ready:
            status_text = "⚡ ГОТОВНОСТЬ К ПЕННОЙ АТАКЕ"
            status_color = P["warn"]
        if status_text:
            c.create_text(W//2, H-120, text=status_text,
                          fill=status_color, font=("Arial", 9,"bold"))

    # ─────────────────────────────────────────────────────────────────────────
    # ОБНОВЛЕНИЕ ГРАФИКОВ
    # ─────────────────────────────────────────────────────────────────────────
    def _update_charts(self):
        sim = self.sim
        if not sim.h_fire:
            return

        # ── Метрики ───────────────────────────────────────────────────────────
        ts_f  = [x[0] for x in sim.h_fire]
        val_f = [x[1] for x in sim.h_fire]
        ts_w  = [x[0] for x in sim.h_water]
        val_w = [x[1] for x in sim.h_water]
        ts_r  = [x[0] for x in sim.h_risk]
        val_r = [x[1] for x in sim.h_risk]
        ts_t  = [x[0] for x in sim.h_trunks]
        val_t = [x[1] for x in sim.h_trunks]

        # Отметки пенных атак
        foam_ts = [t for t, _ in sim.foam_attacks] if sim.foam_attacks else []
        foam_ok = [ok for _, ok in sim.foam_attacks] if sim.foam_attacks else []

        for ax in [self._ax_fire, self._ax_water, self._ax_trunks, self._ax_risk]:
            ax.cla()
            ax.set_facecolor(P["canvas"])
            ax.tick_params(colors=P["text2"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(P["grid"])
            ax.grid(True, color=P["grid"], linewidth=0.4, alpha=0.7)

        # Площадь пожара
        ax = self._ax_fire
        ax.plot(ts_f, val_f, color=P["fire"], linewidth=1.5)
        ax.fill_between(ts_f, val_f, alpha=0.2, color=P["fire"])
        for i, (ft, ok) in enumerate(zip(foam_ts, foam_ok)):
            ax.axvline(ft, color=P["success"] if ok else P["danger"],
                       linewidth=1, linestyle="--", alpha=0.8)
        ax.set_title("Площадь пожара, м²", color=P["text"], fontsize=8, pad=3)
        ax.set_ylabel("м²", color=P["text2"], fontsize=7)
        if ts_f:
            ax.set_xlim(0, max(ts_f[-1], 100))
        ax.yaxis.label.set_color(P["text2"])

        # Расход ОВ
        ax = self._ax_water
        ax.plot(ts_w, val_w, color=P["water"], linewidth=1.5)
        ax.fill_between(ts_w, val_w, alpha=0.2, color=P["water"])
        ax.set_title("Расход ОВ, л/с", color=P["text"], fontsize=8, pad=3)
        ax.set_ylabel("л/с", color=P["text2"], fontsize=7)
        if ts_w:
            ax.set_xlim(0, max(ts_w[-1], 100))

        # Стволы
        ax = self._ax_trunks
        ax.step(ts_t, val_t, color=P["unit_pns"], linewidth=1.5, where="post")
        ax.fill_between(ts_t, val_t, alpha=0.2, color=P["unit_pns"], step="post")
        ax.axhline(7, color=P["success"], linewidth=0.8, linestyle=":", alpha=0.8,
                   label="цель: 7 стволов")
        ax.set_title("Стволов на охлаждении РВС №9", color=P["text"], fontsize=8, pad=3)
        ax.legend(fontsize=6, facecolor=P["canvas"], edgecolor=P["grid"],
                  labelcolor=P["text2"])
        if ts_t:
            ax.set_xlim(0, max(ts_t[-1], 100))

        # Риск
        ax = self._ax_risk
        ax.plot(ts_r, val_r, color=P["danger"], linewidth=1.5)
        ax.fill_between(ts_r, val_r, alpha=0.15, color=P["danger"])
        ax.axhline(0.75, color=P["danger"], linewidth=0.8, linestyle=":",
                   alpha=0.7, label="критич.")
        ax.set_ylim(0, 1.05)
        ax.set_title("Индекс риска", color=P["text"], fontsize=8, pad=3)
        ax.legend(fontsize=6, facecolor=P["canvas"], edgecolor=P["grid"],
                  labelcolor=P["text2"])
        if ts_r:
            ax.set_xlim(0, max(ts_r[-1], 100))

        self._fc_metrics.draw()

        # ── RL-агент ──────────────────────────────────────────────────────────
        for ax in [self._ax_qval, self._ax_actcnt, self._ax_reward]:
            ax.cla()
            ax.set_facecolor(P["canvas"])
            ax.tick_params(colors=P["text2"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(P["grid"])
            ax.grid(True, color=P["grid"], linewidth=0.4, alpha=0.7)

        # Q-значения текущего состояния
        ax = self._ax_qval
        qv   = sim.agent.q_values(sim._state())
        codes = [a[0] for a in ACTIONS]
        cols  = [LEVEL_C[a[1]] for a in ACTIONS]
        bars = ax.bar(range(N_ACT), qv, color=cols, alpha=0.85, width=0.7)
        # Выделить текущее действие
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
        cnt = sim.agent.action_counts
        if cnt.sum() > 0:
            ax.bar(range(N_ACT), cnt / max(cnt.sum(), 1), color=cols, alpha=0.8, width=0.7)
        ax.set_xticks(range(N_ACT))
        ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=7, color=P["text2"])
        ax.set_title("Частота выбора действий", color=P["text"], fontsize=8, pad=3)

        # Кривая наград
        ax = self._ax_reward
        if sim.h_reward:
            rw = sim.h_reward[-500:]   # последние 500 шагов
            cumrew = np.cumsum(rw)
            ax.plot(cumrew, color=P["success"], linewidth=1)
            # Скользящее среднее
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

        sv["sim_time"].set(self._fmt_time(sim.t))
        sv["phase"].set(PHASE_NAMES.get(sim.phase, sim.phase))
        sv["fire_area"].set(f"{sim.fire_area:.0f} м²")
        sv["flow"].set(f"{sim.water_flow:.0f} л/с")
        sv["trunks"].set(f"{sim.n_trunks_burn} (РВС№9)  +  {sim.n_trunks_nbr} (РВС№17)")
        sv["pns"].set(f"{sim.n_pns} из 4")
        sv["bu"].set(f"{sim.n_bu} из 3")
        sv["foam"].set(f"{sim.foam_attacks}" + ("  ✅" if sim.extinguished else ("  🔒" if sim.localized else "  ⏳" if sim.foam_ready else "")))

        risk = sim._risk()
        risk_str = "КРИТИЧЕСКИЙ" if risk > 0.75 else \
                   "ВЫСОКИЙ"     if risk > 0.50 else \
                   "СРЕДНИЙ"     if risk > 0.25 else "НИЗКИЙ"
        sv["risk"].set(f"{risk_str} ({risk:.2f})")

        code, level, desc = ACTIONS[sim.last_action]
        sv["action"].set(f"[{code}] {desc[:30]}")

        # Прогресс
        pct = min(100, int(100 * sim.t / TOTAL_MIN))
        self._prog_var.set(f"{sim.t} / {TOTAL_MIN} мин  ({pct}%)")
        self._prog_bar["value"] = sim.t

    def _log_sim_event(self, t: int, color: str, text: str):
        """Добавить запись в журнал хронологии."""
        pass  # события уже в sim.events, отображаются при необходимости

    @staticmethod
    def _fmt_time(t: int) -> str:
        """t (мин от 03:20 14.03) → строку вида '14.03 03:20'"""
        base_h, base_m = 3, 20
        total_m = base_m + t
        h = (base_h + total_m // 60) % 24
        m = total_m % 60
        day = 14 + (base_h * 60 + base_m + t) // (24 * 60)
        return f"{day:02d}.03  {h:02d}:{m:02d}"

    # ─────────────────────────────────────────────────────────────────────────
    # АНИМАЦИОННЫЙ ЦИКЛ
    # ─────────────────────────────────────────────────────────────────────────
    def _animate(self):
        if not self._running:
            return
        for _ in range(self._speed):
            if self.sim.t >= TOTAL_MIN or self.sim.extinguished:
                self._on_pause()
                break
            self._snap = self.sim.step(dt=self.STEP_MIN)

        self._anim_t += 1
        self._draw_map()
        self._update_status()

        # Обновлять графики каждые ~2 секунды
        if self._anim_t % 25 == 0:
            self._update_charts()

        self._after_id = self.after(self.TICK_MS, self._animate)

    # ── Обработчики кнопок ───────────────────────────────────────────────────
    def _on_play(self):
        if not self._running:
            self._running = True
            self._animate()

    def _on_pause(self):
        self._running = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._update_charts()

    def _on_reset(self):
        self._on_pause()
        self.sim.reset()
        self._anim_t = 0
        self._draw_map()
        self._update_status()
        self._update_charts()

    def _on_speed_change(self):
        self._speed = self.SPEEDS.get(self._speed_var.get(), 15)


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════════════════════════
def main():
    app = TankFireApp()
    app.mainloop()


if __name__ == "__main__":
    main()
