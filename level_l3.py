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
from .rl_agent import QLearningAgent, RLState, compute_reward, ACTION_NAMES
from .metrics import compute_risk_score, compute_delta_s, adaptation_trigger


@dataclass
class OperationalPlan:
    timestamp: float
    phase: FirePhase
    allocated_units: List[str]
    sectors: Dict[str, str]       # unit_id -> sector
    tactics: Dict[str, str]       # unit_id -> tactic
    reserve_requested: int = 0
    priority: str = "NORMAL"


class L3OperationalHQ:
    """L3 operational HQ with RL-based adaptive resource allocation.

    The RL agent (QLearningAgent) learns the optimal resource allocation
    policy over repeated simulation episodes.
    Autonomy alpha3 in [0.5, 0.8]: significant autonomous decision capability.
    """

    def __init__(self, rl_agent: Optional[QLearningAgent] = None,
                 alpha: float = 0.65, seed: Optional[int] = None):
        self.alpha = alpha
        self.rl_agent = rl_agent or QLearningAgent(seed=seed)
        self._current_plan: Optional[OperationalPlan] = None
        self._L7_target = 0.90
        self._last_rl_state: Optional[RLState] = None
        self._last_action: int = 0
        self.plan_history: List[OperationalPlan] = []
        self._regroup_start: Optional[float] = None
        self._regroup_latency: float = 0.0
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0

    def _make_rl_state(self, situation: SituationState,
                       resources: ResourceSpace, L7: float) -> RLState:
        total = max(len(resources.vehicles), 1)
        avail = len(resources.available_units)
        area_norm = min(1.0, situation.fire_area_m2 / 10000.0)
        return RLState.from_metrics(
            phase_idx=situation.phase.value,
            avail_frac=avail / total,
            fire_area_norm=area_norm,
            L7=L7,
        )

    def f_COP(self, cop_from_l1, reports_from_l2: List[dict]) -> SituationState:
        """Maintain common operating picture from L1 + L2 reports."""
        situation = cop_from_l1.situation
        # Merge L2 reports (e.g. additional fire observations)
        if reports_from_l2:
            areas = [r.get("fire_area_m2", 0.0) for r in reports_from_l2]
            if areas:
                situation.fire_area_m2 = max(situation.fire_area_m2, max(areas))
        return situation

    def f_allocate(self, situation: SituationState, resources: ResourceSpace,
                   L7: float, L1: float, t: float,
                   training: bool = True) -> Tuple[int, AdaptationResult]:
        """RL-based allocation decision.

        Returns (action_index, AdaptationResult with delta_pi).
        """
        rl_state = self._make_rl_state(situation, resources, L7)
        action = self.rl_agent.select_action(rl_state, training=training)

        # RL update if we have a previous state
        if self._last_rl_state is not None:
            reward = compute_reward(
                L1_norm=min(1.0, L1 / 60.0),
                casualties=situation.casualties,
                action=self._last_action,
                L7=L7,
            )
            done = (situation.phase == FirePhase.RESOLVED)
            td = self.rl_agent.update(
                self._last_rl_state, self._last_action,
                reward, rl_state, done)
            self._cumulative_reward += reward

        self._last_rl_state = rl_state
        self._last_action = action
        self._step_count += 1

        # Map action to operational response
        action_name = ACTION_NAMES[action]
        result = self._action_to_delta_pi(action_name, situation, resources, t)
        return action, result

    def _action_to_delta_pi(self, action_name: str, situation: SituationState,
                            resources: ResourceSpace, t: float) -> AdaptationResult:
        """Convert RL action to concrete operational plan delta_pi."""
        avail = resources.available_units
        active = resources.active_units

        if action_name == "HOLD":
            return AdaptationResult(mode=AdaptationMode.NORMAL,
                                    actions=["Продолжить текущую расстановку"],
                                    priority_level="NORMAL")

        if action_name == "REGROUP":
            self._regroup_start = t
            acts = [f"Перегруппировать {len(active)} ед. по секторам",
                    "Оптимизировать подачу стволов"]
            return AdaptationResult(mode=AdaptationMode.TACTICAL,
                                    actions=acts, priority_level="ADVISORY",
                                    resources_requested=0)

        if action_name == "REQUEST_RESERVES":
            n_req = max(1, len(active) // 2)
            return AdaptationResult(mode=AdaptationMode.OPERATIONAL,
                                    actions=[f"Запрос {n_req} ед. резерва от L4",
                                             "Подготовить позиции для входа"],
                                    resources_requested=n_req,
                                    priority_level="WARNING")

        if action_name == "FULL_MOBILIZE":
            return AdaptationResult(mode=AdaptationMode.MOBILIZATION,
                                    actions=["Полная мобилизация гарнизона",
                                             "Объявить повышенный номер вызова",
                                             "Запрос межгарнизонного взаимодействия"],
                                    resources_requested=999,
                                    priority_level="URGENT")

        if action_name == "SCALE_DOWN":
            return AdaptationResult(mode=AdaptationMode.NORMAL,
                                    actions=["Вывод части сил",
                                             "Перевод на дотушивание"],
                                    priority_level="INFO")

        return AdaptationResult(mode=AdaptationMode.NORMAL, actions=[])

    def f_predict(self, situation: SituationState, t: float) -> dict:
        """Fire spread forecast (linear extrapolation)."""
        dt = 15.0  # 15-minute horizon
        spread = situation.fire_spread_rate  # m/min
        current_area = situation.fire_area_m2
        predicted_area = current_area + spread * dt * 0.5
        return {
            "t_horizon_min": dt,
            "predicted_area_m2": predicted_area,
            "delta_area_m2": predicted_area - current_area,
        }

    def finish_episode(self) -> None:
        """Call at end of simulation episode to decay epsilon."""
        self.rl_agent.decay_epsilon()
        self.rl_agent.episode_rewards.append(self._cumulative_reward)
        self._cumulative_reward = 0.0
        self._last_rl_state = None
