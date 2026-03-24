"""L3: Operational Level — HQ coordination + RL-based adaptive allocation.

Functions: f_COP, f_plan, f_allocate, f_predict, f_coordinate, f_document
Autonomy: alpha3 in [0.5, 0.8]
Metric: mu3 = regroup latency, forecast RMSE
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .state_space import (FirePhase, SituationState, ResourceUnit,
                          ResourceSpace, AdaptationResult, AdaptationMode)
from .rl_agent import (QLearningAgent, RLState, compute_reward,
                       ACTION_NAMES, ACTION_SPACE, ActionLevel,
                       get_action_mask, _CODE_TO_IDX)
from .metrics import compute_risk_score, compute_delta_s, adaptation_trigger, compute_fire_rank


@dataclass
class OperationalPlan:
    """Оперативный план РТП — текущая диспозиция сил и средств."""
    timestamp: float
    phase: FirePhase
    allocated_units: List[str]         # задействованные единицы
    sectors: Dict[str, str]            # unit_id -> боевой участок (БУ)
    tactics: Dict[str, str]            # unit_id -> боевая задача
    reserve_requested: int = 0
    priority: str = "НОРМАЛЬНЫЙ"
    action_code: str = "O4"            # код действия RL, породившего план
    action_level: str = "оперативный"
    fire_rank: int = 1                 # номер пожара (ранг вызова)
    has_shtab: bool = False            # создан оперативный штаб (ОШ)
    bu_count: int = 1                  # число боевых участков (БУ)
    stp_count: int = 0                 # число секторов тушения пожара (СТП)


class L3OperationalHQ:
    """L3 operational HQ with RL-based adaptive resource allocation.

    The RL agent learns the optimal resource allocation policy over
    repeated simulation episodes.
    Autonomy alpha3 in [0.5, 0.8]: significant autonomous decision capability.
    """

    def __init__(self, rl_agent: Optional[QLearningAgent] = None,
                 alpha: float = 0.65, seed: Optional[int] = None):
        self.alpha = alpha
        self.rl_agent = rl_agent or QLearningAgent(seed=seed)
        self._current_plan: Optional[OperationalPlan] = None
        self._L7_target = 0.90
        self._last_rl_state: Optional[RLState] = None
        self._last_action: int = _CODE_TO_IDX["O4"]   # default: recon
        self.plan_history: List[OperationalPlan] = []
        self._regroup_start: Optional[float] = None
        self._regroup_latency: float = 0.0
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_rl_state(self, situation: SituationState,
                       resources: ResourceSpace, L7: float) -> RLState:
        total    = max(len(resources.vehicles), 1)
        avail    = len(resources.available_units)
        area_norm = min(1.0, situation.fire_area_m2 / 10000.0)
        return RLState.from_metrics(
            phase_idx=situation.phase.value,
            avail_frac=avail / total,
            fire_area_norm=area_norm,
            L7=L7,
        )

    def _build_mask(self, situation: SituationState,
                    resources: ResourceSpace) -> np.ndarray:
        """Build action mask from current situation."""
        total = max(len(resources.vehicles), 1)
        avail = len(resources.available_units)
        res_level = 0 if avail / total < 0.4 else (1 if avail / total < 0.7 else 2)
        has_people = situation.casualties > 0
        # Foam available if resource level is not empty
        has_foam = res_level >= 1
        return get_action_mask(
            phase_idx=situation.phase.value,
            resource_level=res_level,
            has_people=has_people,
            has_foam=has_foam,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def f_COP(self, cop_from_l1, reports_from_l2: List[dict]) -> SituationState:
        """Maintain common operating picture from L1 + L2 reports."""
        situation = cop_from_l1.situation
        if reports_from_l2:
            areas = [r.get("fire_area_m2", 0.0) for r in reports_from_l2]
            if areas:
                situation.fire_area_m2 = max(situation.fire_area_m2, max(areas))
        return situation

    def f_allocate(self, situation: SituationState, resources: ResourceSpace,
                   L7: float, L1: float, t: float,
                   training: bool = True) -> Tuple[int, AdaptationResult]:
        """Решение РТП по распределению С и С на основе RL с маскировкой действий.

        Возвращает (индекс_действия 0..14, AdaptationResult с delta_pi).
        """
        rl_state = self._make_rl_state(situation, resources, L7)
        mask     = self._build_mask(situation, resources)
        action   = self.rl_agent.select_action(rl_state, training=training,
                                               mask=mask)

        # Q-learning update from previous step
        if self._last_rl_state is not None:
            reward = compute_reward(
                L1_norm=min(1.0, L1 / 60.0),
                casualties=situation.casualties,
                action=self._last_action,
                L7=L7,
            )
            done = (situation.phase == FirePhase.RESOLVED)
            self.rl_agent.update(
                self._last_rl_state, self._last_action,
                reward, rl_state, done)
            self._cumulative_reward += reward

        self._last_rl_state = rl_state
        self._last_action   = action
        self._step_count   += 1

        result = self._action_to_delta_pi(action, situation, resources, t)
        return action, result

    def _action_to_delta_pi(self, action_idx: int, situation: SituationState,
                            resources: ResourceSpace, t: float) -> AdaptationResult:
        """Преобразовать индекс действия RL в конкретный оперативный план delta_pi."""
        rt = ACTION_SPACE[action_idx]
        avail  = resources.available_units
        active = resources.active_units
        code   = rt.code

        # ── Стратегические действия (РН по БУПО) ─────────────────────────
        if code == "S1":
            return AdaptationResult(
                mode=AdaptationMode.MOBILIZATION,
                actions=["РН: спасение людей — реальная угроза жизни",
                         "Направить АСА и АЛ к очагу угрозы",
                         "Запросить скорую помощь; организовать эвакуацию"],
                priority_level="КРИТИЧЕСКИЙ")

        if code == "S2":
            return AdaptationResult(
                mode=AdaptationMode.TACTICAL,
                actions=["РН: защита соседних объектов от распространения огня",
                         f"Выставить {max(1, len(active)//3)} ствола на периметр"],
                priority_level="ПРЕДУПРЕЖДЕНИЕ")

        if code == "S3":
            return AdaptationResult(
                mode=AdaptationMode.OPERATIONAL,
                actions=["РН: локализация — ограничение площади горения",
                         "Охватить периметр рукавными линиями",
                         "Сосредоточить С и С на решающем направлении"],
                priority_level="ПРЕДУПРЕЖДЕНИЕ")

        if code == "S4":
            n_trunks = max(1, len(avail))
            return AdaptationResult(
                mode=AdaptationMode.OPERATIONAL,
                actions=[f"РН: ликвидация горения — ввести {n_trunks} ствол(а) на РН",
                         "Максимальная интенсивность подачи огнетушащего вещества (ОВ)"],
                priority_level="СРОЧНЫЙ")

        if code == "S5":
            return AdaptationResult(
                mode=AdaptationMode.MOBILIZATION,
                actions=["РН: предотвращение вскипания/выброса нефтепродуктов (ОФП)",
                         "Охлаждение стенок резервуара",
                         "Готовность к экстренному отходу — сигнал ГДЗС"],
                resources_requested=0,
                priority_level="КРИТИЧЕСКИЙ")

        # ── Тактические действия (НУТ/НС) ────────────────────────────────
        if code == "T1":
            self._regroup_start = t
            return AdaptationResult(
                mode=AdaptationMode.TACTICAL,
                actions=["Создать боевой участок (БУ) / сектор тушения пожара (СТП)",
                         f"Назначить НУТ, распределить {len(avail)} ед. С и С"],
                priority_level="ИНФОРМАЦИОННЫЙ")

        if code == "T2":
            self._regroup_start = t
            return AdaptationResult(
                mode=AdaptationMode.TACTICAL,
                actions=[f"Перераспределить {len(active)} ед. С и С по БУ/секторам",
                         "Оптимизировать позиции стволов и рукавных линий"],
                priority_level="ИНФОРМАЦИОННЫЙ")

        if code == "T3":
            n_req = max(1, len(active) // 2)
            return AdaptationResult(
                mode=AdaptationMode.MOBILIZATION,
                actions=[f"Повышение ранга пожара: вызов {n_req} ед. от Л4 (ПСГ)",
                         "Запрос специальной техники (АЛ, АГДЗС, АШ)",
                         "Создать оперативный штаб (ОШ) при необходимости"],
                resources_requested=n_req,
                priority_level="ПРЕДУПРЕЖДЕНИЕ")

        if code == "T4":
            return AdaptationResult(
                mode=AdaptationMode.NORMAL,
                actions=["Изменить схему развёртывания насосно-рукавной системы (НРС)",
                         "Скорректировать способ подачи огнетушащего вещества (ОВ)"],
                priority_level="НОРМАЛЬНЫЙ")

        # ── Оперативные действия (непосредственное тушение) ───────────────
        if code == "O1":
            return AdaptationResult(
                mode=AdaptationMode.OPERATIONAL,
                actions=["Подать ствол на позицию (РН)",
                         "Установить пожарный автомобиль (ПА) на водоисточник"],
                priority_level="НОРМАЛЬНЫЙ")

        if code == "O2":
            return AdaptationResult(
                mode=AdaptationMode.OPERATIONAL,
                actions=["Охлаждение строительных конструкций/резервуаров",
                         "Поддерживать расход ОВ 3–5 л/с на ствол"],
                priority_level="НОРМАЛЬНЫЙ")

        if code == "O3":
            return AdaptationResult(
                mode=AdaptationMode.OPERATIONAL,
                actions=["Пенная атака на резервуар с нефтепродуктами",
                         "Подать ГПС-600 по периметру резервуара"],
                priority_level="СРОЧНЫЙ")

        if code == "O4":
            return AdaptationResult(
                mode=AdaptationMode.NORMAL,
                actions=["Разведка пожара: определить границы зоны горения",
                         "Установить наличие людей (ОТО) и угрозу распространения ОФП"],
                priority_level="НОРМАЛЬНЫЙ")

        if code == "O5":
            return AdaptationResult(
                mode=AdaptationMode.MOBILIZATION,
                actions=["Эвакуация: провести людей по безопасному маршруту",
                         "АЛ-30 к точкам спасения; организовать пункт сбора"],
                priority_level="СРОЧНЫЙ")

        if code == "O6":
            return AdaptationResult(
                mode=AdaptationMode.DEGRADED,
                actions=["Сигнал отхода — немедленный вывод личного состава (ЛС)",
                         "Определить зону безопасного отхода; доклад РТП"],
                priority_level="КРИТИЧЕСКИЙ")

        # запасной вариант
        return AdaptationResult(mode=AdaptationMode.NORMAL, actions=[])

    def f_predict(self, situation: SituationState, t: float) -> dict:
        """Fire spread forecast (linear extrapolation, 15-min horizon)."""
        dt      = 15.0
        spread  = situation.fire_spread_rate
        current = situation.fire_area_m2
        return {
            "t_horizon_min":   dt,
            "predicted_area_m2": current + spread * dt * 0.5,
            "delta_area_m2":   spread * dt * 0.5,
        }

    def finish_episode(self) -> None:
        """Call at end of episode to decay epsilon and log reward."""
        self.rl_agent.decay_epsilon()
        self.rl_agent.episode_rewards.append(self._cumulative_reward)
        self._cumulative_reward = 0.0
        self._last_rl_state     = None
