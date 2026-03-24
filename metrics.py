"""KPI computation: mu1-mu5, risk score, adaptation trigger."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from .state_space import FirePhase, SituationState, ResourceSpace


@dataclass
class SAURMetrics:
    """All KPI values for one simulation snapshot."""
    # L1 metrics
    cop_accuracy: float = 1.0       # mu1: P_COP — situational picture accuracy
    sensor_coverage: float = 1.0    # fraction of area with sensor coverage

    # L2 metrics
    response_time_min: float = 0.0  # time from alarm to first unit on scene
    personnel_safety: float = 1.0   # 0 = casualty, 1 = safe

    # L3 metrics
    L1_mean_tau: float = 0.0        # mean information staleness (min)
    L3_coverage: float = 1.0        # fraction of objects with fresh data
    L7_reliability: float = 0.0     # P_k — joint coverage probability
    swarm_J: float = 0.0            # J(t) swarm efficiency (if UAVs integrated)
    regroup_latency_min: float = 0.0 # time from trigger to regroup completion
    forecast_rmse: float = 0.0      # RMSE of fire spread forecast

    # L4 metrics
    garrison_readiness: float = 1.0  # fraction of units ready
    territory_coverage_p: float = 0.95  # P(arrival <= 10 min)

    # L5 metrics
    mortality_trend: float = 0.0    # delta(deaths/incidents) — negative = improvement

    # Derived
    risk_score: float = 0.0
    risk_level: str = "LOW"
    delta_s: float = 0.0            # |s(tau) - s*(tau)|
    adaptation_needed: bool = False

    # Phase and resource info
    phase: FirePhase = FirePhase.NORMAL
    n_active_units: int = 0
    n_available_units: int = 0
    fire_area_m2: float = 0.0
    casualties: int = 0


def compute_risk_score(situation: SituationState,
                       resources: ResourceSpace,
                       L7: float, L1: float) -> float:
    """Compute composite risk score in [0, 1]."""
    # Staleness component
    delta_t_ref = 60.0  # reference deltaT (minutes)
    staleness = min(1.0, L1 / max(delta_t_ref, 1.0))

    # Fire spread component
    area_norm = min(1.0, situation.fire_area_m2 / 10000.0)  # normalize to 1 ha
    phase_weight = {
        FirePhase.NORMAL: 0.0, FirePhase.S1: 0.2, FirePhase.S2: 0.4,
        FirePhase.S3: 0.9,     FirePhase.S4: 0.5, FirePhase.S5: 0.2,
        FirePhase.RESOLVED: 0.0,
    }
    fire_factor = area_norm * phase_weight.get(situation.phase, 0.0)

    # Resource deficit component
    total = len(resources.vehicles)
    avail = len(resources.available_units)
    deficit = max(0.0, 1.0 - avail / max(total, 1))

    score = 0.40 * staleness + 0.35 * fire_factor + 0.25 * deficit
    return float(np.clip(score, 0.0, 1.0))


def compute_delta_s(situation: SituationState, target_phase: FirePhase,
                    L7: float, L7_target: float = 0.90) -> float:
    """Compute state deviation delta_s = |s(tau) - s*(tau)|."""
    # Phase deviation (0 if at target or better)
    phase_val = situation.phase.value
    target_val = target_phase.value
    phase_dev = abs(phase_val - target_val) / 6.0  # normalize to [0,1]

    # Reliability deviation
    rel_dev = max(0.0, L7_target - L7)

    # Combined
    return float(0.6 * phase_dev + 0.4 * rel_dev)


EPS_THRESHOLD = 0.20  # delta_s > epsilon triggers adaptation


def adaptation_trigger(delta_s: float) -> bool:
    return delta_s > EPS_THRESHOLD
