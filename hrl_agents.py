"""
hrl_agents.py
════════════════════════════════════════════════════════════════════════════════
3-уровневые иерархические RL-агенты для симуляции тушения пожара РВС.

Иерархия командования:
  L3 — НГ / ГУ МЧС   : стратегический режим (горизонт ~30 мин, каждые K3 шагов)
  L2 — РТП / НШ       : тактическая цель    (горизонт ~10 мин, каждые K2 шагов)
  L1 — НБТП / командир: примитивное действие (каждый шаг, dt=5 мин)

Алгоритм на каждом уровне: табличный Q-learning с ε-жадной стратегией.
Информация передаётся сверху вниз: L3-режим → L2-цель → L1-действие.
Вознаграждение L1 включает интринзическую составляющую за следование цели L2.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# МАКРО-ЦЕЛИ L2 (РТП / НШ)
# ══════════════════════════════════════════════════════════════════════════════

class HGoal:
    """Тактические цели L2-менеджера (РТП / НШ).

    Каждая цель определяет стратегический режим работы на 1–2 шага (10 мин).
    Рабочий (L1) получает интринзическое вознаграждение за выбор действий,
    соответствующих текущей цели.
    """
    ASSAULT  = 0  # НАСТУПЛЕНИЕ — создать условия и провести пенную атаку
    DEFENSE  = 1  # ОБОРОНА     — охлаждение периметра, защита соседних РВС
    BUILD_UP = 2  # НАКОПЛЕНИЕ  — развёртывание, водоснабжение, боевые участки
    MONITOR  = 3  # КОНТРОЛЬ    — разведка, поддержание обстановки
    EVACUATE = 4  # ЭВАКУАЦИЯ   — угроза личному составу → отход
    N = 5

    NAMES: Dict[int, str] = {
        0: "Наступление",
        1: "Оборона",
        2: "Накопление сил",
        3: "Контроль",
        4: "Эвакуация",
    }
    # Цвета для визуализации timeline целей в GUI
    COLORS: Dict[int, str] = {
        0: "#e74c3c",   # красный — атака
        1: "#3498db",   # синий   — оборона
        2: "#f39c12",   # оранж.  — наращивание
        3: "#95a5a6",   # серый   — контроль
        4: "#8e44ad",   # фиолет. — эвакуация
    }


# ══════════════════════════════════════════════════════════════════════════════
# СТРАТЕГИЧЕСКИЕ РЕЖИМЫ L3 (НГ / ГУ МЧС)
# ══════════════════════════════════════════════════════════════════════════════

class L3Mode:
    """Стратегические режимы L3-агента (НГ / ГУ МЧС).

    Режим задаёт разрешённые тактические цели для L2-агента.
    Менеджер L3 действует с наибольшим горизонтом (каждые K3 шагов ≈ 30 мин).
    """
    OWN_FORCES = 0  # Работать силами гарнизона
    REGION     = 1  # Привлечь межрайонную группировку
    EMERGENCY  = 2  # Ввести режим ЧС, максимальная группировка
    N = 3

    NAMES: Dict[int, str] = {
        0: "Силы гарнизона",
        1: "Межрайонная группировка",
        2: "Режим ЧС",
    }
    COLORS: Dict[int, str] = {
        0: "#27ae60",   # зелёный
        1: "#e67e22",   # оранжевый
        2: "#c0392b",   # тёмно-красный
    }


# Разрешённые цели L2 при каждом режиме L3
L3_GOAL_ALLOW: Dict[int, List[int]] = {
    L3Mode.OWN_FORCES: [HGoal.DEFENSE, HGoal.BUILD_UP, HGoal.MONITOR, HGoal.ASSAULT],
    L3Mode.REGION:     [HGoal.ASSAULT, HGoal.DEFENSE, HGoal.BUILD_UP, HGoal.MONITOR],
    L3Mode.EMERGENCY:  [HGoal.EVACUATE, HGoal.DEFENSE],
}

# Соответствие цели L2 → примитивные действия L1 (индексы из ACTIONS)
# S1=0,S2=1,S3=2,S4=3,S5=4, T1=5,T2=6,T3=7,T4=8, O1=9,O2=10,O3=11,O4=12,O5=13,O6=14
GOAL_ACTION_MAP: Dict[int, List[int]] = {
    HGoal.ASSAULT:  [3, 11],           # S4, O3 — пенная атака
    HGoal.DEFENSE:  [1, 4, 9, 10],     # S2, S5, O1, O2
    HGoal.BUILD_UP: [5, 6, 7, 8],      # T1, T2, T3, T4
    HGoal.MONITOR:  [12, 13],          # O4, O5
    HGoal.EVACUATE: [0, 14],           # S1, O6
}


# ══════════════════════════════════════════════════════════════════════════════
# БАЗОВЫЙ ТАБЛИЧНЫЙ Q-АГЕНТ
# ══════════════════════════════════════════════════════════════════════════════

class TabularQAgent:
    """Универсальный табличный Q-агент с ε-жадной стратегией.

    Используется на всех трёх уровнях иерархии — L3, L2, L1.
    Размер таблицы и количество действий задаются конкретным уровнем.

    Обновление: Q(s,a) ← Q(s,a) + α·[r + γ·max_a' Q(s',a') − Q(s,a)]
    Исследование: с вероятностью ε — случайное допустимое действие.
    Эксплуатация: argmax Q(s,·) среди допустимых действий.
    """

    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.15, gamma: float = 0.95,
                 epsilon: float = 0.90, epsilon_decay: float = 0.993,
                 epsilon_min: float = 0.05, seed: int = 42):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        self.rng    = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Q-таблица, инициализированная нулями
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

        # Статистика для анализа обучения
        self.episode_rewards:      List[float] = []
        self._ep_reward:           float = 0.0
        self.action_counts:        np.ndarray = np.zeros(n_actions, dtype=np.int32)
        self.state_visit_counts:   np.ndarray = np.zeros(n_states,  dtype=np.int32)

    # ── Выбор действия ────────────────────────────────────────────────────────
    def select_action(self, state_idx: int,
                      mask: Optional[np.ndarray] = None,
                      training: bool = True) -> int:
        """ε-жадный выбор действия.

        training=True : исследование с вероятностью ε
        training=False: чисто жадная политика (оценочный режим)
        mask           : булев массив допустимых действий (True=допустимо)
        """
        if training and self.rng.random() < self.epsilon:
            valid = np.where(mask)[0] if mask is not None else np.arange(self.n_actions)
            return int(self.rng.choice(valid)) if len(valid) > 0 else 0
        q = self.Q[state_idx].copy()
        if mask is not None:
            q[~mask] = -1e9
        return int(np.argmax(q))

    # ── TD-обновление ─────────────────────────────────────────────────────────
    def update(self, s_idx: int, a: int, r: float,
               s_next_idx: int, done: bool):
        """Обновление Q-значения по правилу TD(0).

        При done=True целевое значение равно просто r (нет будущих наград).
        """
        td_target = r + (0.0 if done else self.gamma * float(np.max(self.Q[s_next_idx])))
        self.Q[s_idx, a] += self.alpha * (td_target - self.Q[s_idx, a])
        self._ep_reward += r
        self.action_counts[a] += 1
        self.state_visit_counts[s_idx] += 1

    # ── Конец эпизода ─────────────────────────────────────────────────────────
    def end_episode(self):
        """Сохранить награду эпизода, убавить ε."""
        self.episode_rewards.append(self._ep_reward)
        self._ep_reward = 0.0
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Вспомогательные методы ───────────────────────────────────────────────
    def q_values(self, state_idx: int) -> np.ndarray:
        return self.Q[state_idx].copy()

    def coverage(self) -> float:
        """Доля ячеек Q-таблицы, посещённых хотя бы 1 раз (0..1)."""
        visited = int(np.sum(self.state_visit_counts > 0))
        return visited / self.n_states

    def reset_episode_stats(self):
        """Сброс счётчиков эпизода — не сбрасывает Q-таблицу."""
        self.episode_rewards.clear()
        self._ep_reward = 0.0
        self.action_counts[:] = 0

    def set_epsilon(self, eps: float):
        self.epsilon = max(self.epsilon_min, min(1.0, eps))


# ══════════════════════════════════════════════════════════════════════════════
# L3 — НГ / ГУ МЧС  (стратегический уровень)
# ══════════════════════════════════════════════════════════════════════════════

class L3Agent(TabularQAgent):
    """Стратегический агент НГ / ГУ МЧС.

    Выбирает режим управления группировкой каждые K3 шагов (~30 мин).

    Кодирование состояния (32 ячейки):
      rank_q   — квантованный ранг пожара (0–3)
      phase_g  — группа фаз: 0=S1-S2, 1=S3, 2=S4-S5
      hours_g  — прошедшие часы: 0=<4ч, 1=4-24ч, 2=24-48ч, 3=48+ч
      res_ok   — достаточность ресурсов (n_pns≥2 И n_bu≥2): 0 или 1
    """

    def __init__(self, alpha: float = 0.10, gamma: float = 0.90,
                 epsilon: float = 0.90, epsilon_decay: float = 0.993,
                 epsilon_min: float = 0.05, seed: int = 42):
        super().__init__(n_states=32, n_actions=L3Mode.N,
                         alpha=alpha, gamma=gamma, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                         seed=seed)

    def encode(self, s: dict) -> int:
        """Кодирование состояния НГ в индекс 0..31."""
        rank_q  = min(3, max(0, s.get("fire_rank", 2) - 1))              # 0–3
        phase_g = {"S1": 0, "S2": 0, "S3": 1, "S4": 2, "S5": 2}.get(
                      s.get("phase", "S3"), 1)                             # 0–2
        hours_g = min(3, s.get("elapsed_hours", 0) // 4)                  # 0–3
        res_ok  = int(s.get("n_pns", 0) >= 2 and s.get("n_bu", 0) >= 2)  # 0–1
        return int((rank_q * 8 + phase_g * 2 + hours_g + res_ok * 16) % 32)

    def mode_mask(self, s: dict) -> np.ndarray:
        """Ограничения выбора режима по реальной обстановке."""
        m = np.ones(L3Mode.N, dtype=bool)
        # Режим ЧС — только при ранге 4 или критической угрозе
        if s.get("fire_rank", 2) < 4 and s.get("threat", 0.3) < 0.85:
            m[L3Mode.EMERGENCY] = False
        # Межрайонная группировка — только при ранге ≥ 3
        if s.get("fire_rank", 2) < 3:
            m[L3Mode.REGION] = False
        if not m.any():
            m[L3Mode.OWN_FORCES] = True
        return m


# ══════════════════════════════════════════════════════════════════════════════
# L2 — РТП / НШ  (оперативно-тактический уровень)
# ══════════════════════════════════════════════════════════════════════════════

class L2Agent(TabularQAgent):
    """Оперативный агент РТП / НШ.

    Выбирает тактическую цель каждые K2 шагов (~10 мин) с учётом режима L3.

    Кодирование состояния (64 ячейки):
      rank_q   — ранг пожара квантованный (0–3)
      phase_g  — группа фаз (0–2)
      res_lvl  — уровень ресурсов: сумма n_pns + n_bu, квантованная (0–3)
      threat_q — квантованная угроза (0–3, шаг 0.25)
      l3_mode  — текущий режим L3 (0–2)
    """

    def __init__(self, alpha: float = 0.12, gamma: float = 0.92,
                 epsilon: float = 0.90, epsilon_decay: float = 0.993,
                 epsilon_min: float = 0.05, seed: int = 42):
        super().__init__(n_states=64, n_actions=HGoal.N,
                         alpha=alpha, gamma=gamma, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                         seed=seed)

    def encode(self, s: dict) -> int:
        """Кодирование состояния РТП в индекс 0..63."""
        rank_q  = min(3, max(0, s.get("fire_rank", 2) - 1))
        phase_g = {"S1": 0, "S2": 0, "S3": 1, "S4": 2, "S5": 2}.get(
                      s.get("phase", "S3"), 1)
        res_lvl = min(3, s.get("n_pns", 0) + s.get("n_bu", 0))
        threat_q = min(3, int(s.get("threat", 0.3) / 0.25))
        l3 = min(2, s.get("l3_mode", 0))
        return int((rank_q * 16 + phase_g * 4 + res_lvl + threat_q * 8 + l3 * 3) % 64)

    def goal_mask(self, s: dict, l3_mode: int) -> np.ndarray:
        """Маска допустимых целей: фаза + ресурсы + разрешение L3."""
        m = np.zeros(HGoal.N, dtype=bool)
        # Разрешения от L3
        for g in L3_GOAL_ALLOW.get(l3_mode, list(range(HGoal.N))):
            m[g] = True
        # Атака недоступна без пены
        if not s.get("foam_ready", False):
            m[HGoal.ASSAULT] = False
        # Эвакуация — только при критической угрозе
        if s.get("threat", 0.3) < 0.70:
            m[HGoal.EVACUATE] = False
        if not m.any():
            m[HGoal.MONITOR] = True  # запасной режим
        return m

    def set_goal_prior(self, prior: Dict[int, float]):
        """Инициализировать Q-таблицу из калибровочных весов (prior).

        Вызывается когда доступны данные из актов пожаров.
        prior: {goal_idx: вес 0..1}, веса не обязаны суммироваться в 1.
        Мягкая инициализация — обучение постепенно откорректирует значения.
        """
        for g, w in prior.items():
            if 0 <= g < HGoal.N:
                self.Q[:, g] += float(w) * 0.5


# ══════════════════════════════════════════════════════════════════════════════
# L1 — НБТП / командир отделения  (исполнительный уровень)
# ══════════════════════════════════════════════════════════════════════════════

class L1Agent(TabularQAgent):
    """Исполнительный агент НБТП / командира отделения.

    Выбирает примитивное действие каждый шаг с учётом целей L2 и L3.

    Кодирование состояния (256 ячеек):
      Базовые признаки среды (phase, n_trunks, n_pns, foam_ready, spill,
                               foam_attacks, n_bu, roof_low)
      + l2_goal  (0–4) — текущая цель от L2
      + l3_mode  (0–2) — текущий режим от L3

    Вознаграждение:
      r_total = r_env + λ · r_intrinsic(action, l2_goal)
      r_intrinsic = +0.5 если действие соответствует цели, −0.2 иначе
    """

    def __init__(self, alpha: float = 0.15, gamma: float = 0.95,
                 epsilon: float = 0.90, epsilon_decay: float = 0.993,
                 epsilon_min: float = 0.05,
                 lambda_intrinsic: float = 0.30,
                 seed: int = 42):
        super().__init__(n_states=256, n_actions=15,
                         alpha=alpha, gamma=gamma, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                         seed=seed)
        self.lambda_intrinsic = lambda_intrinsic

    def encode(self, s: dict) -> int:
        """Кодирование состояния НБТП в индекс 0..255."""
        ph   = {"S1": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4}.get(
                   s.get("phase", "S1"), 0)
        tr   = min(3, s.get("n_trunks", 0) // 2)
        pns  = min(3, s.get("n_pns", 0))
        fr   = int(s.get("foam_ready", False))
        sp   = int(s.get("spill", False))
        fa   = min(3, s.get("foam_attacks", 0))
        bu   = min(3, s.get("n_bu", 0))
        rl   = int(s.get("roof_low", False))
        goal = min(4, s.get("l2_goal", HGoal.MONITOR))
        l3   = min(2, s.get("l3_mode", L3Mode.OWN_FORCES))
        # Хэш-свёртка с равномерным распределением по 256 ячейкам
        return int((ph * 64 + tr * 16 + pns * 4 + fr * 2 + sp
                    + fa * 8 + bu * 2 + rl * 32 + goal * 50 + l3 * 80) % 256)

    def action_mask(self, base_mask: np.ndarray, goal: int,
                    soft: bool = True) -> np.ndarray:
        """Маскирование действий по цели L2.

        soft=True  — мягкое (реалистичное): базовый маск по фазе сохраняется,
                     несоответствие цели учитывается только через интринзик.
        soft=False — жёсткое (быстрее сходится): разрешены только действия
                     из GOAL_ACTION_MAP для текущей цели.
        """
        if soft:
            return base_mask
        goal_acts = GOAL_ACTION_MAP.get(goal, [])
        goal_mask = np.zeros(self.n_actions, dtype=bool)
        for a in goal_acts:
            if a < self.n_actions:
                goal_mask[a] = True
        combined = base_mask & goal_mask
        return combined if combined.any() else base_mask

    def intrinsic_reward(self, action: int, goal: int) -> float:
        """Интринзическое вознаграждение за соответствие действия цели L2."""
        return 0.5 if action in GOAL_ACTION_MAP.get(goal, []) else -0.2

    def total_reward(self, env_r: float, action: int, goal: int) -> float:
        """Полное вознаграждение: среда + λ·интринзик."""
        return env_r + self.lambda_intrinsic * self.intrinsic_reward(action, goal)


# ══════════════════════════════════════════════════════════════════════════════
# УТИЛИТА: создать 3 агента из конфига
# ══════════════════════════════════════════════════════════════════════════════

def create_agents(alpha_l3: float = 0.10, alpha_l2: float = 0.12,
                  alpha_l1: float = 0.15, gamma_l3: float = 0.90,
                  gamma_l2: float = 0.92, gamma_l1: float = 0.95,
                  epsilon: float = 0.90, epsilon_decay: float = 0.993,
                  epsilon_min: float = 0.05, lambda_intrinsic: float = 0.30,
                  seed: int = 42,
                  goal_prior: Optional[Dict[int, float]] = None
                  ) -> Tuple[L3Agent, L2Agent, L1Agent]:
    """Создать все три агента с заданными параметрами."""
    l3 = L3Agent(alpha=alpha_l3, gamma=gamma_l3, epsilon=epsilon,
                 epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                 seed=seed)
    l2 = L2Agent(alpha=alpha_l2, gamma=gamma_l2, epsilon=epsilon,
                 epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                 seed=seed + 1)
    l1 = L1Agent(alpha=alpha_l1, gamma=gamma_l1, epsilon=epsilon,
                 epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                 lambda_intrinsic=lambda_intrinsic,
                 seed=seed + 2)
    if goal_prior:
        l2.set_goal_prior(goal_prior)
    return l3, l2, l1
