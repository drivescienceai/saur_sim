"""SAUR DES simulation integrating all five levels."""
from __future__ import annotations
import heapq
import copy
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Tuple

from .state_space import (FirePhase, SituationState, ResourceUnit,
                          ResourceSpace, AdaptationResult)
from .semi_markov import SemiMarkovChain
from .metrics import (SAURMetrics, compute_risk_score, compute_delta_s,
                      adaptation_trigger, EPS_THRESHOLD, compute_fire_rank)
from .database import OIL_BASE_GARRISON, UNIT_DATABASE
from .level_l1 import L1SensorLayer, COP
from .level_l2 import L2TacticalAgent
from .level_l3 import L3OperationalHQ
from .level_l4 import L4SystemicCenter
from .level_l5 import L5StrategicLayer, OperationRecord
from .rl_agent import QLearningAgent


@dataclass(order=True)
class Event:
    time: float
    etype: str = field(compare=False)
    data: Any = field(default=None, compare=False)


class E:
    TICK       = "tick"          # periodic stats + level updates
    FIRE_START = "fire_start"    # fire incident begins
    PHASE_END  = "phase_end"     # phase sojourn expired -> transition
    REGROUP    = "regroup"       # tactical regroup order
    RESERVE_IN = "reserve_in"    # reserve units arrive on scene
    RESOLVE    = "resolve"       # incident resolved


@dataclass
class SimParams:
    sim_time: float = 600.0       # total simulation minutes
    tick_interval: float = 1.0    # stats collection interval
    fire_start_time: float = 60.0 # when fire begins
    fire_x_m: float = 375.0
    fire_y_m: float = 300.0
    initial_fire_area_m2: float = 200.0
    fire_spread_rate: float = 5.0  # m^2/min growth
    seed: int = 42
    n_replicas: int = 1
    training: bool = True


class SAURSimulation:
    """Full SAUR simulation: semi-Markov phases + RL allocation + 5 levels."""

    def __init__(self, params: Optional[SimParams] = None,
                 rl_agent: Optional[QLearningAgent] = None):
        self.p = params or SimParams()
        self._rng = np.random.RandomState(self.p.seed)

        # Build resource space from garrison database
        self._resources = self._build_resources()

        # Situation state (ground truth)
        self._true_situation = SituationState(timestamp=0.0)

        # Semi-Markov chain
        self._chain = SemiMarkovChain(seed=self.p.seed)

        # Five levels
        self._l1 = L1SensorLayer(seed=self.p.seed)
        self._l2_agents = [
            L2TacticalAgent(u) for u in self._resources.vehicles[:4]
        ]
        self._l3 = L3OperationalHQ(rl_agent=rl_agent, seed=self.p.seed)
        self._l4 = L4SystemicCenter(self._resources)
        self._l5 = L5StrategicLayer()

        # Event queue
        self._eq: List[Event] = []
        self._t: float = 0.0

        # History
        self.metrics_history: List[SAURMetrics] = []
        self._cop_history: List[COP] = []
        self._adaptation_log: List[Tuple[float, AdaptationResult]] = []
        self._L7_samples: List[float] = []
        self._L1_samples: List[float] = []
        self._fire_active = False

    def _build_resources(self) -> ResourceSpace:
        units = []
        for g in OIL_BASE_GARRISON:
            utype = UNIT_DATABASE.get(g["type"])
            if utype is None:
                continue
            u = ResourceUnit(
                unit_id=g["id"],
                unit_type=g["type"],
                crew_size=utype.crew_full,
                x_m=float(g["x_m"]),
                y_m=float(g["y_m"]),
                station_id=g["station"],
                response_time_min=utype.response_time_min,
            )
            units.append(u)
        return ResourceSpace(vehicles=units)

    def _schedule(self, dt: float, etype: str, data: Any = None) -> None:
        heapq.heappush(self._eq, Event(self._t + dt, etype, data))

    def _update_true_situation(self) -> None:
        """Обновить оперативную обстановку (ОТО) из цепи Маркова."""
        from .state_space import ReshayusheeNapravlenie
        phase, _, transitioned = self._chain.step(self._t)
        self._true_situation.phase = phase
        self._true_situation.timestamp = self._t

        if self._fire_active:
            self._true_situation.fire_area_m2 += (
                self.p.fire_spread_rate * self.p.tick_interval)
            self._true_situation.fire_area_m2 = min(
                self._true_situation.fire_area_m2, 20000.0)

        if transitioned and phase == FirePhase.RESOLVED:
            self._fire_active = False
            self._true_situation.fire_area_m2 = 0.0

        # Обновить номер пожара и решающее направление по фазе
        self._true_situation.fire_rank = compute_fire_rank(
            self._true_situation, self._resources)
        rn_by_phase = {
            FirePhase.S1: ReshayusheeNapravlenie.SPASENIE_LYUDEY,
            FirePhase.S2: ReshayusheeNapravlenie.LOKALIZATSIYA,
            FirePhase.S3: ReshayusheeNapravlenie.ZASHCHITA_SOSEDNIKH,
            FirePhase.S4: ReshayusheeNapravlenie.LIKVIDATSIYA,
            FirePhase.S5: ReshayusheeNapravlenie.LIKVIDATSIYA,
        }
        self._true_situation.reshayushee_napravlenie = rn_by_phase.get(
            phase, ReshayusheeNapravlenie.LOKALIZATSIYA)

    def _compute_snapshot_metrics(self, cop: COP, action: int,
                                   result: AdaptationResult) -> SAURMetrics:
        """Compute full metrics for current tick."""
        sit = self._true_situation

        # L7 — fraction of resources with readiness > 0.7 as proxy
        total = max(len(self._resources.vehicles), 1)
        ready = sum(1 for u in self._resources.vehicles if u.readiness > 0.7)
        L7 = ready / total

        # L1 — information staleness proxy (phase deviation)
        phase_dev = abs(sit.phase.value - cop.situation.phase.value)
        L1 = float(phase_dev * 10.0)  # in minutes (proxy)

        self._L7_samples.append(L7)
        self._L1_samples.append(L1)

        risk = compute_risk_score(sit, self._resources, L7, L1)
        delta_s = compute_delta_s(sit, FirePhase.S4, L7)

        n_active = len(self._resources.active_units)
        n_avail  = len(self._resources.available_units)

        if risk >= 0.75:
            risk_level = "КРИТИЧЕСКИЙ"
        elif risk >= 0.50:
            risk_level = "ВЫСОКИЙ"
        elif risk >= 0.25:
            risk_level = "СРЕДНИЙ"
        else:
            risk_level = "НИЗКИЙ"

        fire_rank = compute_fire_rank(sit, self._resources)
        has_shtab = fire_rank >= 2
        bu_count  = max(1, n_active // 3)
        stp_count = bu_count // 3

        return SAURMetrics(
            cop_accuracy=cop.confidence,
            response_time_min=5.0,
            L1_mean_tau=L1,
            L7_reliability=L7,
            garrison_readiness=ready / total,
            risk_score=risk,
            risk_level=risk_level,
            delta_s=delta_s,
            adaptation_needed=adaptation_trigger(delta_s),
            phase=sit.phase,
            n_active_units=n_active,
            n_available_units=n_avail,
            fire_area_m2=sit.fire_area_m2,
            casualties=sit.casualties,
            fire_rank=fire_rank,
            has_shtab=has_shtab,
            bu_count=bu_count,
            stp_count=stp_count,
            reshayushee_napravlenie=sit.reshayushee_napravlenie.name,
        )

    def _handle_tick(self, ev: Event) -> None:
        """Main periodic tick: update all levels."""
        self._update_true_situation()

        # L1: sense + fuse
        cop = self._l1.update(self._t, self._true_situation)
        self._cop_history.append(cop)

        # L3 COP + allocation (RL decision)
        l2_reports = []
        l3_sit = self._l3.f_COP(cop, l2_reports)

        # Compute current L7 + L1 for RL
        L7 = self._L7_samples[-1] if self._L7_samples else 1.0
        L1 = self._L1_samples[-1] if self._L1_samples else 0.0

        action, result = self._l3.f_allocate(
            l3_sit, self._resources, L7, L1, self._t,
            training=self.p.training,
        )

        if result.resources_requested > 0:
            l4_resp = self._l4.receive_l3_request(result)
            self._adaptation_log.append((self._t, l4_resp))
        else:
            self._adaptation_log.append((self._t, result))

        # L2: each agent steps
        for agent in self._l2_agents:
            agent.step(l3_sit)

        # Compute metrics snapshot
        m = self._compute_snapshot_metrics(cop, action, result)
        self.metrics_history.append(m)

        # Schedule next tick
        if self._t + self.p.tick_interval <= self.p.sim_time:
            self._schedule(self.p.tick_interval, E.TICK)

    def _handle_fire_start(self, ev: Event) -> None:
        self._fire_active = True
        self._true_situation.fire_area_m2 = self.p.initial_fire_area_m2
        self._true_situation.fire_x_m = self.p.fire_x_m
        self._true_situation.fire_y_m = self.p.fire_y_m
        # Force chain to S1
        self._chain.force_transition(self._t, FirePhase.S1)
        self._true_situation.phase = FirePhase.S1

    def _handle_resolve(self, ev: Event) -> None:
        self._fire_active = False
        self._l3.finish_episode()
        # Record to L5
        record = OperationRecord(
            duration_min=self._t - self.p.fire_start_time,
            max_phase="S3",
            casualties=self._true_situation.casualties,
            total_units_deployed=len(self._resources.active_units),
            L7_final=float(np.mean(self._L7_samples)) if self._L7_samples else 0.0,
            J_final=0.75,
            peak_fire_area_m2=self._true_situation.fire_area_m2,
            outcome="controlled",
        )
        self._l5.record_operation(record)

    _HANDLERS = {
        E.TICK:       _handle_tick,
        E.FIRE_START: _handle_fire_start,
        E.RESOLVE:    _handle_resolve,
    }

    def run(self) -> Dict[str, object]:
        """Execute simulation. Returns summary metrics dict."""
        # Reset
        self._eq = []
        self._t = 0.0
        self.metrics_history = []
        self._cop_history = []
        self._adaptation_log = []
        self._L7_samples = []
        self._L1_samples = []
        self._fire_active = False
        self._true_situation = SituationState(timestamp=0.0)
        self._chain = SemiMarkovChain(seed=self.p.seed)

        # Schedule initial events
        self._schedule(0, E.TICK)
        self._schedule(self.p.fire_start_time, E.FIRE_START)
        self._schedule(self.p.sim_time * 0.85, E.RESOLVE)

        # Main loop
        while self._eq:
            ev = heapq.heappop(self._eq)
            if ev.time > self.p.sim_time:
                break
            self._t = ev.time
            handler = self._HANDLERS.get(ev.etype)
            if handler:
                handler(self, ev)

        return self._summarize()

    def _summarize(self) -> Dict[str, object]:
        if not self.metrics_history:
            return {}
        L7s = [m.L7_reliability for m in self.metrics_history]
        L1s = [m.L1_mean_tau for m in self.metrics_history]
        risks = [m.risk_score for m in self.metrics_history]
        fire_m = [m for m in self.metrics_history if m.phase not in (
            FirePhase.NORMAL, FirePhase.RESOLVED)]
        return {
            "mean_L7": float(np.mean(L7s)),
            "min_L7": float(np.min(L7s)),
            "mean_L1_tau": float(np.mean(L1s)),
            "mean_risk": float(np.mean(risks)),
            "max_risk": float(np.max(risks)),
            "mean_L7_fire": float(np.mean([m.L7_reliability for m in fire_m])) if fire_m else 0.0,
            "n_adaptations": sum(1 for m in self.metrics_history if m.adaptation_needed),
            "rl_epsilon": self._l3.rl_agent.epsilon,
            "chain_summary": self._chain.summary(),
            "l5_kpis": self._l5.strategic_kpis(),
        }

    @staticmethod
    def run_training(n_episodes: int = 100, seed: int = 42,
                     sim_time: float = 600.0) -> QLearningAgent:
        """Train RL agent over n_episodes and return trained agent."""
        agent = QLearningAgent(seed=seed)
        for ep in range(n_episodes):
            ep_seed = seed + ep * 17
            p = SimParams(sim_time=sim_time, seed=ep_seed, training=True)
            p.fire_start_time = 30.0 + (ep % 20) * 5.0  # vary fire timing
            sim = SAURSimulation(params=p, rl_agent=agent)
            sim.run()
        return agent
