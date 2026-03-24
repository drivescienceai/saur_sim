"""L4: Systemic Level — garrison/TSУKS management."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .state_space import ResourceSpace, ResourceUnit, AdaptationResult, AdaptationMode
import copy


@dataclass
class GarrisonStatus:
    total_units: int
    ready_units: int
    deployed_units: int
    readiness_fraction: float
    active_incidents: int
    inter_garrison_support: bool = False


class L4SystemicCenter:
    """L4 garrison command center (TSУKS/ТУС).

    Functions: f_monitor_readiness, f_reserve, f_coordinate_inter,
               f_stat, f_risk_assess
    Autonomy: alpha4 in [0.7, 0.9]
    Metric: mu4 = P(arrival <= 10 min) >= 0.95
    """

    def __init__(self, resources: ResourceSpace, alpha: float = 0.80):
        self.resources = resources
        self.alpha = alpha
        self._incident_log: List[dict] = []
        self._inter_garrison_threshold = 0.40  # readiness below this -> request inter-garrison

    def monitor_readiness(self) -> GarrisonStatus:
        total = len(self.resources.vehicles)
        ready = sum(1 for u in self.resources.vehicles if u.readiness > 0.5)
        deployed = sum(1 for u in self.resources.vehicles if u.task != "standby")
        frac = ready / max(total, 1)
        return GarrisonStatus(
            total_units=total, ready_units=ready, deployed_units=deployed,
            readiness_fraction=frac, active_incidents=1 if deployed > 0 else 0,
        )

    def f_reserve(self, n_requested: int, incident_priority: str = "NORMAL") -> List[ResourceUnit]:
        """Allocate reserve units to incident."""
        available = [u for u in self.resources.vehicles if u.is_available]
        # Sort by response time
        available.sort(key=lambda u: u.response_time_min)
        allocated = available[:n_requested]
        for u in allocated:
            u.task = "dispatched_reserve"
        return allocated

    def f_coordinate_inter(self, status: GarrisonStatus) -> Optional[str]:
        """Request inter-garrison support if readiness too low."""
        if status.readiness_fraction < self._inter_garrison_threshold:
            return (f"Запрос межгарнизонного взаимодействия "
                    f"(готовность {status.readiness_fraction:.0%})")
        return None

    def receive_l3_request(self, request: AdaptationResult) -> AdaptationResult:
        """Process resource request from L3."""
        status = self.monitor_readiness()
        n = request.resources_requested
        if n == 0:
            return AdaptationResult(mode=AdaptationMode.NORMAL,
                                    actions=["Запрос не требует ресурсов L4"])
        allocated = self.f_reserve(n)
        inter = self.f_coordinate_inter(status)
        actions = [f"Направлено {len(allocated)} ед. из гарнизонного резерва"]
        if inter:
            actions.append(inter)
        return AdaptationResult(
            mode=AdaptationMode.OPERATIONAL,
            actions=actions,
            resources_requested=len(allocated),
            priority_level=request.priority_level,
        )
