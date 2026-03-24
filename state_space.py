from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Tuple


class FirePhase(Enum):
    NORMAL = 0
    S1 = 1   # Detection/dispatch
    S2 = 2   # First attack
    S3 = 3   # Active fire
    S4 = 4   # Knockdown
    S5 = 5   # Overhaul
    RESOLVED = 6


@dataclass
class SituationState:  # s = <O, E, H, W, tau>
    object_type: str = "oil_base"        # O - object characteristics
    fire_area_m2: float = 0.0            # E - emergency params
    fire_spread_rate: float = 0.0        # E
    casualties: int = 0                   # H - human threat
    wind_speed_ms: float = 3.0           # W - environment
    visibility_m: float = 1000.0         # W
    phase: FirePhase = FirePhase.NORMAL
    timestamp: float = 0.0               # tau
    fire_x_m: float = 0.0
    fire_y_m: float = 0.0


@dataclass
class ResourceUnit:  # r = <rho, epsilon, lambda, delta>
    unit_id: str
    unit_type: str   # "fire_truck" | "ladder" | "tanker" | "rescue" | "command"
    crew_size: int
    readiness: float = 1.0       # rho in [0,1]
    resource_level: float = 1.0  # epsilon (water/foam level 0..1)
    x_m: float = 0.0             # lambda - location
    y_m: float = 0.0
    task: str = "standby"        # delta - current task
    station_id: str = ""
    response_time_min: float = 5.0

    @property
    def is_available(self) -> bool:
        return self.readiness > 0.3 and self.task == "standby"


@dataclass
class ResourceSpace:  # R = Rv union Rp union Re union Ri
    vehicles: List[ResourceUnit] = field(default_factory=list)   # Rv
    personnel: int = 0                                            # Rp (total count)
    equipment: Dict[str, int] = field(default_factory=dict)      # Re
    info_channels: List[str] = field(default_factory=list)       # Ri

    @property
    def available_units(self) -> List[ResourceUnit]:
        return [u for u in self.vehicles if u.is_available]

    @property
    def active_units(self) -> List[ResourceUnit]:
        return [u for u in self.vehicles if u.task != "standby"]


class AdaptationMode(Enum):
    NORMAL = "normal"               # delta_s <= epsilon
    TACTICAL = "tactical"           # delta_s > epsilon at L2
    OPERATIONAL = "operational"     # delta_s > epsilon at L3
    MOBILIZATION = "mobilization"   # multiple incidents
    DEGRADED = "degraded"           # loss of comms


@dataclass
class AdaptationResult:  # Delta_pi - corrective plan increment
    mode: AdaptationMode
    actions: List[str] = field(default_factory=list)
    resources_requested: int = 0
    priority_level: str = "NORMAL"
    delta_s: float = 0.0           # deviation from target state
