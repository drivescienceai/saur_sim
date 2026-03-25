"""
hrl_sim.py
════════════════════════════════════════════════════════════════════════════════
HierarchicalTankFireSim — симуляция тушения пожара с 3-уровневым иерарх. RL.

Уровни управления:
  L3 (НГ/ГУ МЧС)  — стратегический режим,   горизонт K3 шагов (≈30 мин)
  L2 (РТП/НШ)     — тактическая цель,        горизонт K2 шагов (≈10 мин)
  L1 (НБТП/команд)— примитивное действие,    каждый шаг (dt=5 мин)

Curriculum learning:
  Фаза 1 (200 эп.)  — Серпухов ранг 2         → базовая тактика
  Фаза 2 (300 эп.)  — Серпухов + Туапсе 50/50 → обобщение
  Фаза 3 (500 эп.)  — Туапсе ранг 4           → сложный сценарий

Интеграция со средой:
  Использует TankFireSim.step(action=…) — внешний выбор действия.
  Физика пожара, хронология событий, нормативные расчёты — без изменений.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from .tank_fire_sim import TankFireSim, SCENARIOS, SimSnapshot
    from .hrl_agents import (L3Agent, L2Agent, L1Agent, HGoal, L3Mode,
                              create_agents, GOAL_ACTION_MAP)
except ImportError:
    from tank_fire_sim import TankFireSim, SCENARIOS, SimSnapshot
    from hrl_agents import (L3Agent, L2Agent, L1Agent, HGoal, L3Mode,
                             create_agents, GOAL_ACTION_MAP)


# ══════════════════════════════════════════════════════════════════════════════
# CURRICULUM: фазы и расписание обучения
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CurriculumPhase:
    """Одна фаза curriculum learning."""
    episodes:  int         # число эпизодов в фазе
    scenarios: List[str]   # сценарии (ключи SCENARIOS) — выбираются случайно
    label:     str         # человекочитаемая метка для GUI и отчётов

    def sample_scenario(self, rng: random.Random) -> str:
        return rng.choice(self.scenarios)


# Расписание по умолчанию — изменяется через HRLConfig
CURRICULUM_DEFAULT: List[CurriculumPhase] = [
    CurriculumPhase(200, ["serp"],           "Базовая тактика (ранг 2)"),
    CurriculumPhase(300, ["serp", "tuapse"], "Обобщение (ранг 2+4)"),
    CurriculumPhase(500, ["tuapse"],         "Сложный сценарий (ранг 4)"),
]


# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ ИЕРАРХИЧЕСКОГО RL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HRLConfig:
    """Все настраиваемые параметры иерархического RL.

    Значения по умолчанию — теоретически обоснованные.
    Пользователь может изменить через GUI или hrl_config.json.
    """
    # ── Частоты действий агентов ──────────────────────────────────────────
    k3: int   = 6    # L3: каждые 6 шагов (30 мин при dt=5)
    k2: int   = 2    # L2: каждые 2 шага  (10 мин при dt=5)

    # ── Параметры Q-learning по уровням ──────────────────────────────────
    alpha_l3: float = 0.10
    alpha_l2: float = 0.12
    alpha_l1: float = 0.15
    gamma_l3: float = 0.90
    gamma_l2: float = 0.92
    gamma_l1: float = 0.95

    # ── ε-greedy (общие для всех уровней, убывают независимо) ─────────────
    epsilon_start: float = 0.90
    epsilon_decay: float = 0.993   # к эпизоду ~700: ε ≈ 0.05
    epsilon_min:   float = 0.05

    # ── Интринзическое вознаграждение ─────────────────────────────────────
    lambda_intrinsic: float = 0.30   # вес интринзик-награды в r_total
    soft_masking:     bool  = True   # True=мягкое, False=жёсткое маскирование

    # ── Curriculum ────────────────────────────────────────────────────────
    curriculum: List[CurriculumPhase] = field(
        default_factory=lambda: list(CURRICULUM_DEFAULT))

    # ── Калибровочные веса целей (из актов пожаров; None = равномерно) ────
    goal_prior: Optional[Dict[int, float]] = None

    # ── Оценочный прогон ──────────────────────────────────────────────────
    n_eval_episodes:  int   = 100
    bootstrap_n:      int   = 10000
    significance_lvl: float = 0.05

    @property
    def total_train_episodes(self) -> int:
        return sum(p.episodes for p in self.curriculum)

    @classmethod
    def from_json(cls, path: str) -> "HRLConfig":
        """Загрузить конфигурацию из JSON-файла."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        cfg = cls()
        scalar_keys = {k for k in d if k not in ("curriculum", "goal_prior")}
        for k in scalar_keys:
            if hasattr(cfg, k):
                setattr(cfg, k, d[k])
        if "curriculum" in d:
            cfg.curriculum = [
                CurriculumPhase(
                    episodes=p["episodes"],
                    scenarios=p["scenarios"],
                    label=p.get("label", "")
                ) for p in d["curriculum"]
            ]
        if "goal_prior" in d and d["goal_prior"] is not None:
            cfg.goal_prior = {int(k): float(v) for k, v in d["goal_prior"].items()}
        return cfg

    def to_json(self, path: str):
        """Сохранить конфигурацию в JSON-файл."""
        d: dict = {}
        for k in ("k3", "k2", "alpha_l3", "alpha_l2", "alpha_l1",
                  "gamma_l3", "gamma_l2", "gamma_l1", "epsilon_start",
                  "epsilon_decay", "epsilon_min", "lambda_intrinsic",
                  "soft_masking", "n_eval_episodes", "bootstrap_n",
                  "significance_lvl"):
            d[k] = getattr(self, k)
        d["curriculum"] = [
            {"episodes": p.episodes, "scenarios": p.scenarios, "label": p.label}
            for p in self.curriculum
        ]
        d["goal_prior"] = (
            {str(k): v for k, v in self.goal_prior.items()}
            if self.goal_prior else None
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# ИЕРАРХИЧЕСКАЯ СИМУЛЯЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

class HierarchicalTankFireSim:
    """Симуляция тушения пожара с 3-уровневым иерархическим Q-learning.

    Структура шага:
      1. Каждые k3 шагов — L3 выбирает режим (L3Mode), обновляет свою Q-таблицу
      2. Каждые k2 шагов — L2 выбирает цель (HGoal) в рамках режима L3
      3. Каждый шаг     — L1 выбирает примитивное действие, среда обновляется

    Накопление наград:
      L1 обновляется на каждом шаге: r_total = r_env + λ·r_intrinsic
      L2 обновляется раз в k2 шагов: r = Σ r_env за k2 шагов
      L3 обновляется раз в k3 шагов: r = Σ r_env за k3 шагов
    """

    MAX_STEPS = 2000   # защита от бесконечных эпизодов

    def __init__(self, cfg: Optional[HRLConfig] = None, seed: int = 42,
                 scenario: str = "tuapse"):
        self.cfg      = cfg or HRLConfig()
        self.seed     = seed
        self._rng     = random.Random(seed)
        self.scenario = scenario

        # Три иерархических агента
        self.l3, self.l2, self.l1 = create_agents(
            alpha_l3=self.cfg.alpha_l3, alpha_l2=self.cfg.alpha_l2,
            alpha_l1=self.cfg.alpha_l1, gamma_l3=self.cfg.gamma_l3,
            gamma_l2=self.cfg.gamma_l2, gamma_l1=self.cfg.gamma_l1,
            epsilon=self.cfg.epsilon_start,
            epsilon_decay=self.cfg.epsilon_decay,
            epsilon_min=self.cfg.epsilon_min,
            lambda_intrinsic=self.cfg.lambda_intrinsic,
            seed=seed,
            goal_prior=self.cfg.goal_prior,
        )

        # Среда (физика, хронология, нормы ГОСТ)
        self._env: TankFireSim = TankFireSim(
            seed=seed, training=True, scenario=scenario)

        # История целей/режимов для визуализации
        self.goal_history: List[Tuple[int, int]] = []   # (t, l2_goal)
        self.mode_history: List[Tuple[int, int]] = []   # (t, l3_mode)
        self.episode_count: int = 0
        self.curriculum_phase_label: str = ""

        self._reset_hierarchy()

    # ── Сброс иерархических переменных (не агентов!) ──────────────────────────
    def _reset_hierarchy(self):
        self.current_l3_mode = L3Mode.OWN_FORCES
        self.current_l2_goal = HGoal.MONITOR
        self.step_count      = 0

        # Предыдущие индексы состояний и действий для TD-обновления
        self._l3_s_prev:    Optional[int] = None
        self._l3_a_prev:    Optional[int] = None
        self._l3_r_accum:   float = 0.0

        self._l2_s_prev:    Optional[int] = None
        self._l2_a_prev:    Optional[int] = None
        self._l2_r_accum:   float = 0.0

        self._l1_s_prev:    Optional[int] = None
        self._l1_a_prev:    Optional[int] = None

        self.goal_history.clear()
        self.mode_history.clear()

    def reset(self, scenario: Optional[str] = None):
        """Сброс среды + иерархии. Агенты (Q-таблицы) не сбрасываются."""
        if scenario:
            self.scenario = scenario
        env_seed = self._rng.randint(0, 99999)
        self._env = TankFireSim(seed=env_seed, training=True,
                                scenario=self.scenario)
        self._reset_hierarchy()

    # ── Вспомогательные состояния для каждого уровня ─────────────────────────

    def _env_state(self) -> dict:
        """Полный вектор признаков среды (совместим с L1.encode)."""
        env = self._env
        return dict(
            phase        = env.phase,
            n_trunks     = env.n_trunks_burn,
            n_pns        = env.n_pns,
            foam_ready   = env.foam_ready,
            spill        = env.spill,
            foam_attacks = env.foam_attacks,
            n_bu         = env.n_bu,
            roof_low     = env.roof_obstruction < 0.40,
            l2_goal      = self.current_l2_goal,
            l3_mode      = self.current_l3_mode,
        )

    def _l3_state_dict(self) -> dict:
        env = self._env
        return dict(
            fire_rank     = self._current_rank(),
            phase         = env.phase,
            elapsed_hours = env.t // 60,
            n_pns         = env.n_pns,
            n_bu          = env.n_bu,
            threat        = env._risk(),
        )

    def _l2_state_dict(self) -> dict:
        env = self._env
        return dict(
            fire_rank  = self._current_rank(),
            phase      = env.phase,
            n_pns      = env.n_pns,
            n_bu       = env.n_bu,
            threat     = env._risk(),
            foam_ready = env.foam_ready,
            l3_mode    = self.current_l3_mode,
        )

    def _current_rank(self) -> int:
        cfg  = self._env._cfg
        area = self._env.fire_area
        base = cfg["fire_rank_default"]
        if area < cfg["initial_fire_area"] * 0.5: return max(1, base - 1)
        if area < cfg["initial_fire_area"] * 1.2: return base
        return min(4, base + 1)

    # ── Основной шаг симуляции ────────────────────────────────────────────────

    def step(self, dt: int = 5, training: bool = True) -> SimSnapshot:
        """Один шаг симуляции с 3-уровневым иерархическим управлением."""
        env  = self._env
        done = env.extinguished or env.t >= env._cfg["total_min"]
        if done:
            return self._make_snapshot(0, 0.0)

        # ── L3: НГ действует каждые k3 шагов ─────────────────────────────
        if self.step_count % self.cfg.k3 == 0:
            s3      = self._l3_state_dict()
            s3_idx  = self.l3.encode(s3)
            mask3   = self.l3.mode_mask(s3)
            new_mode = self.l3.select_action(s3_idx, mask3, training=training)

            # TD-обновление за прошедшие k3 шагов
            if self._l3_s_prev is not None:
                self.l3.update(self._l3_s_prev, self._l3_a_prev,
                               self._l3_r_accum, s3_idx, done)

            self.current_l3_mode = new_mode
            self._l3_s_prev  = s3_idx
            self._l3_a_prev  = new_mode
            self._l3_r_accum = 0.0
            self.mode_history.append((env.t, new_mode))

        # ── L2: РТП действует каждые k2 шагов ────────────────────────────
        if self.step_count % self.cfg.k2 == 0:
            s2      = self._l2_state_dict()
            s2_idx  = self.l2.encode(s2)
            mask2   = self.l2.goal_mask(s2, self.current_l3_mode)
            new_goal = self.l2.select_action(s2_idx, mask2, training=training)

            if self._l2_s_prev is not None:
                self.l2.update(self._l2_s_prev, self._l2_a_prev,
                               self._l2_r_accum, s2_idx, done)

            self.current_l2_goal = new_goal
            self._l2_s_prev  = s2_idx
            self._l2_a_prev  = new_goal
            self._l2_r_accum = 0.0
            self.goal_history.append((env.t, new_goal))

        # ── L1: НБТП выбирает примитивное действие каждый шаг ───────────
        s1      = self._env_state()
        s1_idx  = self.l1.encode(s1)
        base_mask = env._mask()
        act_mask  = self.l1.action_mask(
            base_mask, self.current_l2_goal, soft=self.cfg.soft_masking)

        action  = self.l1.select_action(s1_idx, act_mask, training=training)
        env_r   = env._apply(action)
        total_r = self.l1.total_reward(env_r, action, self.current_l2_goal)

        # Шаг физики среды (время, события, физика, фаза)
        snap = env.step(dt=dt, action=action)

        # Обновление L1
        s1_next     = self._env_state()
        s1_next_idx = self.l1.encode(s1_next)
        done_now    = env.extinguished or env.t >= env._cfg["total_min"]
        if training:
            self.l1.update(s1_idx, action, total_r, s1_next_idx, done_now)

        # Накопление награды для L2 и L3
        self._l2_r_accum += env_r
        self._l3_r_accum += env_r

        if done_now and training:
            self._finalize_episode()

        self.step_count += 1
        return snap

    def _finalize_episode(self):
        """Финальное TD-обновление L2 и L3 по завершении эпизода."""
        s3_idx = self.l3.encode(self._l3_state_dict())
        s2_idx = self.l2.encode(self._l2_state_dict())

        if self._l3_s_prev is not None and self._l3_a_prev is not None:
            self.l3.update(self._l3_s_prev, self._l3_a_prev,
                           self._l3_r_accum, s3_idx, True)
        if self._l2_s_prev is not None and self._l2_a_prev is not None:
            self.l2.update(self._l2_s_prev, self._l2_a_prev,
                           self._l2_r_accum, s2_idx, True)

        self.l1.end_episode()
        self.l2.end_episode()
        self.l3.end_episode()
        self.episode_count += 1

    def _make_snapshot(self, action: int, reward: float) -> SimSnapshot:
        env = self._env
        return SimSnapshot(
            t=env.t, phase=env.phase, fire_area=env.fire_area,
            water_flow=env.water_flow, n_trunks_burn=env.n_trunks_burn,
            n_trunks_neighbor=env.n_trunks_nbr, n_pns=env.n_pns,
            n_bu=env.n_bu, has_shtab=env.has_shtab,
            foam_attacks=env.foam_attacks, foam_ready=env.foam_ready,
            spill=env.spill, secondary_fire=env.secondary_fire,
            localized=env.localized, extinguished=env.extinguished,
            risk=env._risk(), last_action=action, reward=reward,
            roof_obstruction=env.roof_obstruction, foam_flow_ls=env.foam_flow_ls,
        )

    # ── Полный прогон одного эпизода ─────────────────────────────────────────

    def run_episode(self, dt: int = 5, training: bool = True) -> dict:
        """Прогон одного эпизода до конца. Возвращает итоговую статистику."""
        self.reset()
        env = self._env
        steps = 0
        while (not env.extinguished
               and env.t < env._cfg["total_min"]
               and steps < self.MAX_STEPS):
            self.step(dt=dt, training=training)
            steps += 1

        init_area = env._cfg["initial_fire_area"]
        ep_reward = self.l1.episode_rewards[-1] if self.l1.episode_rewards else 0.0
        attacks   = env.foam_attacks

        return {
            "extinguished":       env.extinguished,
            "localized":          env.localized,
            "total_steps":        steps,
            "final_fire_area":    env.fire_area,
            "foam_attacks":       attacks,
            "total_reward":       ep_reward,
            "fire_area_reduction":max(0.0, (init_area - env.fire_area) / init_area),
            "foam_efficiency":    (1.0 / max(1, attacks)) if env.extinguished else 0.0,
            "goal_switches":      len(self.goal_history),
            "dominant_goal":      self._dominant_goal(),
            "scenario":           self.scenario,
        }

    def _dominant_goal(self) -> int:
        """Наиболее часто выбиравшаяся цель L2 за эпизод."""
        if not self.goal_history:
            return HGoal.MONITOR
        from collections import Counter
        return Counter(g for _, g in self.goal_history).most_common(1)[0][0]

    # ── Curriculum learning ───────────────────────────────────────────────────

    def train_curriculum(self,
                         progress_cb: Optional[Callable[[int, int, str, dict], None]] = None,
                         stop_flag:   Optional[list] = None) -> dict:
        """Обучение по всем фазам curriculum.

        progress_cb(ep, total, phase_label, result) — вызывается после каждого эпизода.
        stop_flag — список [False]; установить [True] для досрочной остановки.
        """
        total_episodes = self.cfg.total_train_episodes
        episode = 0

        for phase in self.cfg.curriculum:
            self.curriculum_phase_label = phase.label
            for _ in range(phase.episodes):
                if stop_flag and stop_flag[0]:
                    break
                scen   = phase.sample_scenario(self._rng)
                self.reset(scenario=scen)
                result = self.run_episode(training=True)
                episode += 1
                if progress_cb:
                    progress_cb(episode, total_episodes, phase.label, result)
            if stop_flag and stop_flag[0]:
                break

        return {
            "total_episodes": episode,
            "l1_coverage":    self.l1.coverage(),
            "l2_coverage":    self.l2.coverage(),
            "l3_coverage":    self.l3.coverage(),
            "l1_eps":         self.l1.epsilon,
            "l2_eps":         self.l2.epsilon,
            "l3_eps":         self.l3.epsilon,
        }

    def set_eval_mode(self):
        """Отключить исследование на всех уровнях — для оценочного прогона."""
        self.l1.epsilon = 0.0
        self.l2.epsilon = 0.0
        self.l3.epsilon = 0.0

    # ── Сериализация Q-таблиц ─────────────────────────────────────────────────

    def save_qtables(self, path: str):
        """Сохранить Q-таблицы всех агентов в .npz файл."""
        np.savez(path, l1=self.l1.Q, l2=self.l2.Q, l3=self.l3.Q)

    def load_qtables(self, path: str):
        """Загрузить Q-таблицы из .npz файла."""
        d = np.load(path)
        if "l1" in d: self.l1.Q = d["l1"]
        if "l2" in d: self.l2.Q = d["l2"]
        if "l3" in d: self.l3.Q = d["l3"]

    # ── Свойства для GUI ──────────────────────────────────────────────────────

    @property
    def env(self) -> TankFireSim:
        return self._env

    @property
    def current_goal_name(self) -> str:
        return HGoal.NAMES.get(self.current_l2_goal, "?")

    @property
    def current_mode_name(self) -> str:
        return L3Mode.NAMES.get(self.current_l3_mode, "?")
