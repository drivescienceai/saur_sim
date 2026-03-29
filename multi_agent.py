"""
multi_agent.py — Мультиагентное обучение с подкреплением (МАОП).
═══════════════════════════════════════════════════════════════════════════════
Мультиагентная система управления тушением пожара РВС.

ОТЛИЧИЕ ОТ ИЕРАРХИЧЕСКОГО ОП (моноагентного):
═══════════════════════════════════════════════════════════════════════════════
  Иерархическое ОП (hrl_agents.py):    Мультиагентное ОП (этот модуль):
  ─────────────────────────────────     ──────────────────────────────────
  1 агент L3 (режим)                   1 агент РТП (стратегия)
  1 агент L2 (цель)      ← ОДИН       3 агента НБУ (секторы)  ← ТРИ+
  1 агент L1 (действие)                1 агент НТ (тыл)
                                       ─────────────────────────
  Единое наблюдение                    Локальные наблюдения (свой сектор)
  Единая Q-таблица L2                  Отдельная Q-таблица на каждого НБУ
  Нет конкуренции за ресурсы           Общий пул воды/пены → конфликт
  α = скалярная величина               α_i для каждого агента НБУ
═══════════════════════════════════════════════════════════════════════════════

Агенты:
  РТП  — руководитель тушения пожара (стратегический уровень)
         Наблюдает: всю обстановку. Действует: ставит задачи НБУ.
  НБУ-k — начальник боевого участка k ∈ {1, 2, 3}
         Наблюдает: свой сектор + команду РТП. Действует: 8 оперативных действий.
  НТ   — начальник тыла
         Наблюдает: ресурсы. Действует: распределение воды/пены между НБУ.

Взаимодействие:
  - Общий ресурс: водоснабжение = Σ расход_НБУk ≤ Q_ПНС
  - Общий ресурс: запас пенообразователя = ΣΣ пена_НБУk ≤ запас
  - Конфликт: два НБУ запрашивают пенную атаку одновременно
  - Координация: РТП назначает приоритет секторам
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import IntEnum

import numpy as np

try:
    from .hrl_agents import TabularQAgent, HGoal, GOAL_ACTION_MAP
    from .rl_agent import N_ACTIONS, ACTION_COST
except ImportError:
    from hrl_agents import TabularQAgent, HGoal, GOAL_ACTION_MAP
    from rl_agent import N_ACTIONS, ACTION_COST


# ═══════════════════════════════════════════════════════════════════════════
# СЕКТОРЫ БОЕВЫХ УЧАСТКОВ
# ═══════════════════════════════════════════════════════════════════════════
class Sector(IntEnum):
    """Секторы вокруг РВС (по сторонам света)."""
    SOUTH = 0       # НБУ-1: юг
    EAST = 1        # НБУ-2: восток
    WEST = 2        # НБУ-3: запад


SECTOR_NAMES = {
    Sector.SOUTH: "НБУ-1 (юг)",
    Sector.EAST:  "НБУ-2 (восток)",
    Sector.WEST:  "НБУ-3 (запад)",
}

# Действия, доступные НБУ (оперативные + часть тактических)
# НБУ не может менять ранг пожара или создавать штаб — это прерогатива РТП
NBU_ACTIONS = [
    (9,  "О1", "Подать ствол на охлаждение"),
    (10, "О2", "Охлаждение соседнего РВС"),
    (11, "О3", "Пенная атака"),
    (12, "О4", "Разведка сектора"),
    (13, "О5", "Ликвидация розлива"),
    (14, "О6", "Сигнал отхода"),
    (6,  "Т2", "Перегруппировка в секторе"),
    (8,  "Т4", "Установить ПНС"),
]
N_NBU_ACTIONS = len(NBU_ACTIONS)
NBU_ACTION_IDX = [a[0] for a in NBU_ACTIONS]  # глобальные индексы

# Действия НТ (начальника тыла)
NT_ACTIONS = [
    (0, "Приоритет НБУ-1", "Направить ресурсы на южный участок"),
    (1, "Приоритет НБУ-2", "Направить ресурсы на восточный участок"),
    (2, "Приоритет НБУ-3", "Направить ресурсы на западный участок"),
    (3, "Равномерное",      "Равномерное распределение ресурсов"),
    (4, "Запрос подвоза",   "Запросить подвоз воды/пены"),
]
N_NT_ACTIONS = len(NT_ACTIONS)


# ═══════════════════════════════════════════════════════════════════════════
# ЛОКАЛЬНОЕ НАБЛЮДЕНИЕ СЕКТОРА
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SectorObservation:
    """Наблюдение НБУ — только свой сектор."""
    sector: Sector
    phase: str                  # фаза пожара (общая)
    n_trunks_local: int         # стволов в своём секторе
    water_flow_local: float     # расход ОВ в секторе (л/с)
    fire_intensity_local: float # интенсивность горения (0..1)
    foam_available: float       # доля пены, выделенная этому НБУ
    has_spill: bool             # розлив в секторе
    rtp_goal: int               # цель от РТП (HGoal)
    risk_local: float           # локальный риск (0..1)

    def to_index(self, n_states: int = 128) -> int:
        """Кодирование в индекс состояния."""
        phase_map = {"S1": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4}
        p = phase_map.get(self.phase, 2)
        trunks_q = min(3, self.n_trunks_local // 2)
        flow_q = 0 if self.water_flow_local < 50 else (1 if self.water_flow_local < 150 else 2)
        risk_q = 0 if self.risk_local < 0.3 else (1 if self.risk_local < 0.6 else 2)
        foam_q = 0 if self.foam_available < 0.2 else 1
        goal_q = min(4, self.rtp_goal)

        idx = (p * 24 + trunks_q * 6 + flow_q * 2 + risk_q) % n_states
        return idx


@dataclass
class GlobalObservation:
    """Наблюдение РТП — вся обстановка."""
    phase: str
    fire_area: float
    total_water_flow: float
    total_trunks: int
    n_pns: int
    foam_reserve: float
    risk: float
    foam_attacks: int
    roof_obstruction: float
    n_bu: int
    sector_risks: Dict[Sector, float] = field(default_factory=dict)

    def to_index(self, n_states: int = 64) -> int:
        phase_map = {"S1": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4}
        p = phase_map.get(self.phase, 2)
        risk_q = 0 if self.risk < 0.3 else (1 if self.risk < 0.6 else 2)
        trunks_q = min(3, self.total_trunks // 3)
        pns_q = min(2, self.n_pns)
        return (p * 12 + risk_q * 4 + trunks_q) % n_states


# ═══════════════════════════════════════════════════════════════════════════
# АГЕНТЫ
# ═══════════════════════════════════════════════════════════════════════════
class RTPAgent:
    """Агент РТП — стратегический уровень (один на всю операцию).

    Действия: назначение целей секторам + стратегические решения.
    Пространство действий: 5 целей (HGoal) × 3 сектора не комбинируем,
    а выбираем ОДНУ общую цель (как в HRL L2) + приоритетный сектор.
    Итого: 5 целей × 3 приоритета + 2 стратегических = 17 действий.
    """

    N_STATES = 64
    N_ACTIONS = 5  # HGoal (цель для всех НБУ)

    def __init__(self, alpha=0.10, gamma=0.92, epsilon=0.90, seed=42):
        self.agent = TabularQAgent(
            n_states=self.N_STATES, n_actions=self.N_ACTIONS,
            alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
        self.current_goal = HGoal.DEFENSE
        self.sector_priority = Sector.SOUTH

    def select_goal(self, obs: GlobalObservation,
                    training: bool = True) -> int:
        """Выбрать тактическую цель."""
        s = obs.to_index(self.N_STATES)
        goal = self.agent.select_action(s, training=training)
        self.current_goal = goal

        # Приоритетный сектор — тот, где риск максимален
        if obs.sector_risks:
            self.sector_priority = max(obs.sector_risks,
                                       key=obs.sector_risks.get)
        return goal

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        self.agent.update(s, a, r, s_next, done)

    def end_episode(self):
        self.agent.end_episode()


class NBUAgent:
    """Агент НБУ — начальник боевого участка (один на сектор).

    Видит: свой сектор + цель от РТП.
    Действует: 8 оперативно-тактических действий.
    Внутренняя мотивация: бонус за соответствие цели РТП.
    """

    N_STATES = 128

    def __init__(self, sector: Sector,
                 alpha=0.15, gamma=0.95, epsilon=0.90,
                 lambda_intrinsic=0.30, seed=42):
        self.sector = sector
        self.name = SECTOR_NAMES[sector]
        self.lambda_intrinsic = lambda_intrinsic

        self.agent = TabularQAgent(
            n_states=self.N_STATES, n_actions=N_NBU_ACTIONS,
            alpha=alpha, gamma=gamma, epsilon=epsilon,
            seed=seed + sector.value)

        # Статистика автономности
        self.steps_total = 0
        self.steps_deviated = 0  # отклонение от цели РТП

    def select_action(self, obs: SectorObservation,
                      training: bool = True) -> int:
        """Выбрать действие в своём секторе.

        Возвращает: ЛОКАЛЬНЫЙ индекс действия (0..7)
        """
        s = obs.to_index(self.N_STATES)
        local_a = self.agent.select_action(s, training=training)
        return local_a

    def local_to_global(self, local_action: int) -> int:
        """Конвертировать локальный индекс → глобальный (0..14)."""
        if 0 <= local_action < N_NBU_ACTIONS:
            return NBU_ACTION_IDX[local_action]
        return 12  # О4 по умолчанию

    def intrinsic_reward(self, global_action: int, rtp_goal: int) -> float:
        """Внутренняя мотивация: соответствие цели РТП."""
        goal_actions = GOAL_ACTION_MAP.get(rtp_goal, [])
        if global_action in goal_actions:
            return 0.5
        return -0.2

    def total_reward(self, env_r: float, global_action: int,
                     rtp_goal: int) -> float:
        """Полная награда = среда + λ·внутренняя."""
        r_intr = self.intrinsic_reward(global_action, rtp_goal)
        return env_r + self.lambda_intrinsic * r_intr

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        self.agent.update(s, a, r, s_next, done)

    def end_episode(self):
        self.agent.end_episode()

    @property
    def alpha_autonomy(self) -> float:
        """Коэффициент автономности: доля отклонений от цели РТП."""
        if self.steps_total == 0:
            return 0.0
        return self.steps_deviated / self.steps_total

    def track_autonomy(self, global_action: int, rtp_goal: int):
        """Учёт автономности для α-анализа."""
        self.steps_total += 1
        goal_actions = GOAL_ACTION_MAP.get(rtp_goal, [])
        if global_action not in goal_actions:
            self.steps_deviated += 1


class NTAgent:
    """Агент НТ — начальник тыла (один на всю операцию).

    Распределяет ресурсы (воду, пену) между секторами.
    """

    N_STATES = 32
    N_ACTIONS = N_NT_ACTIONS  # 5

    def __init__(self, alpha=0.12, gamma=0.90, epsilon=0.90, seed=42):
        self.agent = TabularQAgent(
            n_states=self.N_STATES, n_actions=self.N_ACTIONS,
            alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed + 100)
        self.allocation: Dict[Sector, float] = {
            Sector.SOUTH: 0.33, Sector.EAST: 0.33, Sector.WEST: 0.34,
        }

    def select_allocation(self, total_water: float, total_foam: float,
                          sector_risks: Dict[Sector, float],
                          training: bool = True) -> Dict[Sector, float]:
        """Распределить ресурсы между секторами.

        Возвращает: {Sector: доля_ресурса}
        """
        risk_sum = sum(sector_risks.values()) or 1.0
        s_idx = int(sum(v * 10 for v in sector_risks.values())) % self.N_STATES
        action = self.agent.select_action(s_idx, training=training)

        if action == 0:    # приоритет НБУ-1
            self.allocation = {Sector.SOUTH: 0.50, Sector.EAST: 0.25, Sector.WEST: 0.25}
        elif action == 1:  # приоритет НБУ-2
            self.allocation = {Sector.SOUTH: 0.25, Sector.EAST: 0.50, Sector.WEST: 0.25}
        elif action == 2:  # приоритет НБУ-3
            self.allocation = {Sector.SOUTH: 0.25, Sector.EAST: 0.25, Sector.WEST: 0.50}
        elif action == 3:  # равномерное
            self.allocation = {Sector.SOUTH: 0.33, Sector.EAST: 0.33, Sector.WEST: 0.34}
        else:              # запрос подвоза
            self.allocation = {s: sector_risks.get(s, 0.33) / risk_sum
                               for s in Sector}

        return self.allocation

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        self.agent.update(s, a, r, s_next, done)

    def end_episode(self):
        self.agent.end_episode()


# ═══════════════════════════════════════════════════════════════════════════
# МУЛЬТИАГЕНТНАЯ СИСТЕМА
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class MAConfig:
    """Конфигурация мультиагентной системы."""
    n_sectors: int = 3
    alpha_rtp: float = 0.10
    alpha_nbu: float = 0.15
    alpha_nt: float = 0.12
    gamma_rtp: float = 0.92
    gamma_nbu: float = 0.95
    gamma_nt: float = 0.90
    epsilon_start: float = 0.90
    epsilon_decay: float = 0.993
    lambda_intrinsic: float = 0.30
    rtp_freq: int = 6          # РТП принимает решение каждые 6 шагов (30 мин)
    nt_freq: int = 3           # НТ распределяет ресурсы каждые 3 шага (15 мин)
    seed: int = 42


class MultiAgentSystem:
    """Мультиагентная система управления тушением пожара.

    Координация: РТП (1) → НБУ (3) + НТ (1) = 5 агентов.

    Цикл на каждом шаге:
      1. РТП (каждые rtp_freq шагов): выбирает цель → рассылает НБУ
      2. НТ (каждые nt_freq шагов): распределяет ресурсы → НБУ
      3. НБУ-1, НБУ-2, НБУ-3 (каждый шаг): выбирают действие в секторе
      4. Действия агрегируются → единое действие в среде TankFireSim
      5. Награда распределяется: env_reward / n_sectors + intrinsic
    """

    def __init__(self, cfg: MAConfig = None):
        if cfg is None:
            cfg = MAConfig()
        self.cfg = cfg

        self.rtp = RTPAgent(alpha=cfg.alpha_rtp, gamma=cfg.gamma_rtp,
                            epsilon=cfg.epsilon_start, seed=cfg.seed)
        self.nbu = [
            NBUAgent(sector=Sector(i), alpha=cfg.alpha_nbu,
                     gamma=cfg.gamma_nbu, epsilon=cfg.epsilon_start,
                     lambda_intrinsic=cfg.lambda_intrinsic,
                     seed=cfg.seed + i)
            for i in range(cfg.n_sectors)
        ]
        self.nt = NTAgent(alpha=cfg.alpha_nt, gamma=cfg.gamma_nt,
                          epsilon=cfg.epsilon_start, seed=cfg.seed)

        self._step_count = 0
        self._episode_count = 0

        # Журнал решений всех агентов
        self.decision_log: List[Dict] = []

    def step(self, global_obs: GlobalObservation,
             sector_obs: List[SectorObservation],
             training: bool = True) -> Tuple[int, Dict]:
        """Один шаг мультиагентной системы.

        Возвращает: (global_action_idx, info_dict)
        """
        self._step_count += 1
        info = {"agents": {}}

        # ── 1. РТП: выбор цели (каждые rtp_freq шагов) ───────────────────
        if self._step_count % self.cfg.rtp_freq == 1 or self._step_count == 1:
            goal = self.rtp.select_goal(global_obs, training)
            info["rtp_goal"] = goal
            info["rtp_goal_name"] = ["Штурм", "Оборона", "Наращивание",
                                     "Разведка", "Эвакуация"][min(goal, 4)]

        current_goal = self.rtp.current_goal

        # ── 2. НТ: распределение ресурсов (каждые nt_freq шагов) ──────────
        if self._step_count % self.cfg.nt_freq == 1 or self._step_count == 1:
            alloc = self.nt.select_allocation(
                global_obs.total_water_flow,
                global_obs.foam_reserve,
                global_obs.sector_risks,
                training)
            info["nt_allocation"] = {SECTOR_NAMES[s]: round(v, 2)
                                     for s, v in alloc.items()}

        # ── 3. НБУ: локальные действия (каждый шаг) ──────────────────────
        nbu_actions = []
        for i, (nbu_agent, obs) in enumerate(zip(self.nbu, sector_obs)):
            obs.rtp_goal = current_goal
            local_a = nbu_agent.select_action(obs, training)
            global_a = nbu_agent.local_to_global(local_a)
            nbu_agent.track_autonomy(global_a, current_goal)

            nbu_actions.append(global_a)
            info["agents"][nbu_agent.name] = {
                "local_action": local_a,
                "global_action": global_a,
                "action_name": NBU_ACTIONS[local_a][1],
                "alpha": round(nbu_agent.alpha_autonomy, 3),
            }

        # ── 4. Агрегация: выбрать действие приоритетного сектора ──────────
        # Действие приоритетного НБУ (по оценке РТП) идёт в среду
        prio = self.rtp.sector_priority.value
        if 0 <= prio < len(nbu_actions):
            chosen_action = nbu_actions[prio]
        else:
            chosen_action = nbu_actions[0]
        info["chosen_action"] = chosen_action
        info["priority_sector"] = SECTOR_NAMES.get(
            Sector(prio), "НБУ-1")

        # Журнал
        self.decision_log.append({
            "step": self._step_count,
            "rtp_goal": current_goal,
            "nbu_actions": list(nbu_actions),
            "chosen": chosen_action,
            "priority": prio,
        })

        return chosen_action, info

    def distribute_reward(self, env_reward: float):
        """Распределить награду среды между агентами."""
        # РТП получает полную награду (отвечает за общий результат)
        # НБУ — долю + внутреннюю мотивацию
        # НТ — долю, зависящую от стабильности водоснабжения
        share = env_reward / max(self.cfg.n_sectors, 1)
        return {
            "rtp": env_reward,
            "nbu_shares": [share] * self.cfg.n_sectors,
            "nt": env_reward * 0.5,
        }

    def end_episode(self):
        """Завершить эпизод: decay ε, обнулить счётчики, сохранить в БД."""
        self.rtp.end_episode()
        for nbu in self.nbu:
            nbu.end_episode()
        self.nt.end_episode()

        # Автосохранение в централизованную БД
        try:
            from results_db import get_db
            get_db().log_multiagent(
                n_agents=1 + self.cfg.n_sectors + 1,
                autonomy=self.autonomy_report(),
                decision_log_size=len(self.decision_log))
        except Exception:
            pass

        self._step_count = 0
        self._episode_count += 1

    def autonomy_report(self) -> Dict[str, float]:
        """Коэффициенты автономности всех агентов."""
        report = {"РТП": 0.0}  # РТП автономен по определению
        for nbu in self.nbu:
            report[nbu.name] = nbu.alpha_autonomy
        report["НТ"] = 0.0
        return report

    def stats(self) -> Dict:
        """Статистика мультиагентной системы."""
        return {
            "n_agents": 1 + self.cfg.n_sectors + 1,
            "agents": ["РТП"] + [nbu.name for nbu in self.nbu] + ["НТ"],
            "episodes": self._episode_count,
            "rtp_goal": self.rtp.current_goal,
            "sector_priority": SECTOR_NAMES.get(
                self.rtp.sector_priority, "—"),
            "autonomy": self.autonomy_report(),
            "nt_allocation": {SECTOR_NAMES[s]: round(v, 2)
                              for s, v in self.nt.allocation.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')

    mas = MultiAgentSystem()
    rng = np.random.RandomState(42)

    phases = ["S1"] * 3 + ["S2"] * 5 + ["S3"] * 20 + ["S4"] * 8 + ["S5"] * 4

    for i, phase in enumerate(phases):
        risk = 0.3 + 0.4 * (phase == "S3") + rng.normal(0, 0.05)
        risk = max(0.0, min(1.0, risk))

        g_obs = GlobalObservation(
            phase=phase, fire_area=1250.0, total_water_flow=300.0,
            total_trunks=6, n_pns=3, foam_reserve=12.0, risk=risk,
            foam_attacks=0, roof_obstruction=0.70, n_bu=3,
            sector_risks={Sector.SOUTH: risk * 0.4,
                          Sector.EAST: risk * 0.35,
                          Sector.WEST: risk * 0.25},
        )

        s_obs = [
            SectorObservation(
                sector=Sector(k), phase=phase,
                n_trunks_local=2, water_flow_local=100.0,
                fire_intensity_local=risk * (0.4 if k == 0 else 0.3),
                foam_available=0.33, has_spill=False,
                rtp_goal=0, risk_local=risk * [0.4, 0.35, 0.25][k])
            for k in range(3)
        ]

        action, info = mas.step(g_obs, s_obs)

    print("Мультиагентная система: 5 агентов")
    print(f"\nСтатистика:")
    for k, v in mas.stats().items():
        print(f"  {k}: {v}")

    print(f"\nАвтономность агентов:")
    for name, alpha in mas.autonomy_report().items():
        print(f"  {name}: alpha = {alpha:.3f}")

    print(f"\nПоследние 5 решений:")
    for d in mas.decision_log[-5:]:
        print(f"  Шаг {d['step']}: цель={d['rtp_goal']}, "
              f"НБУ={d['nbu_actions']}, выбрано={d['chosen']}")
