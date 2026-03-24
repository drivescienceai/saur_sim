"""Fire-rescue unit database for SAUR simulation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import copy


@dataclass(frozen=True)
class UnitType:
    name: str
    crew_min: int             # minimum crew
    crew_full: int            # full crew
    water_capacity_l: int     # 0 for ladder/rescue
    foam_capacity_l: int
    flow_rate_l_min: int      # water/foam delivery rate
    speed_kmh: float
    response_time_min: float  # typical first response time
    cost_rel: float           # relative cost index (0..1)


UNIT_DATABASE: Dict[str, UnitType] = {
    "АЦ-40": UnitType("АЦ-40",        crew_min=3, crew_full=5,
                      water_capacity_l=4000,  foam_capacity_l=200,
                      flow_rate_l_min=40, speed_kmh=90,
                      response_time_min=5.0, cost_rel=0.5),
    "АЦ-40/60": UnitType("АЦ-40/60",  crew_min=3, crew_full=5,
                      water_capacity_l=6000, foam_capacity_l=360,
                      flow_rate_l_min=60, speed_kmh=85,
                      response_time_min=5.5, cost_rel=0.65),
    "АЛ-30": UnitType("АЛ-30",        crew_min=2, crew_full=3,
                      water_capacity_l=0, foam_capacity_l=0,
                      flow_rate_l_min=0, speed_kmh=80,
                      response_time_min=6.0, cost_rel=0.75),
    "АСА":   UnitType("АСА",          crew_min=5, crew_full=6,
                      water_capacity_l=0, foam_capacity_l=0,
                      flow_rate_l_min=0, speed_kmh=90,
                      response_time_min=4.5, cost_rel=0.4),
    "АШ":    UnitType("АШ",           crew_min=3, crew_full=5,
                      water_capacity_l=0, foam_capacity_l=0,
                      flow_rate_l_min=0, speed_kmh=90,
                      response_time_min=5.0, cost_rel=0.3),
    "ПНС-110": UnitType("ПНС-110",    crew_min=2, crew_full=3,
                      water_capacity_l=0, foam_capacity_l=0,
                      flow_rate_l_min=110, speed_kmh=75,
                      response_time_min=8.0, cost_rel=0.45),
}

# Typical garrison configuration for an oil base response
OIL_BASE_GARRISON: List[dict] = [
    # Station 1 — main, on-site
    {"id": "ПА-01", "type": "АЦ-40/60", "station": "ПЧ-1", "x_m":  100, "y_m":   0},
    {"id": "ПА-02", "type": "АЦ-40/60", "station": "ПЧ-1", "x_m":  100, "y_m":   0},
    {"id": "ПА-03", "type": "АЛ-30",    "station": "ПЧ-1", "x_m":  100, "y_m":   0},
    {"id": "ПА-04", "type": "АСА",      "station": "ПЧ-1", "x_m":  100, "y_m":   0},
    {"id": "ПА-05", "type": "АШ",       "station": "ПЧ-1", "x_m":  100, "y_m":   0},
    {"id": "ПА-06", "type": "ПНС-110",  "station": "ПЧ-1", "x_m":  100, "y_m":   0},
    # Station 2 — backup, 3 km
    {"id": "ПА-07", "type": "АЦ-40",   "station": "ПЧ-2", "x_m": 3000, "y_m": 1000},
    {"id": "ПА-08", "type": "АЦ-40",   "station": "ПЧ-2", "x_m": 3000, "y_m": 1000},
    {"id": "ПА-09", "type": "АЛ-30",   "station": "ПЧ-2", "x_m": 3000, "y_m": 1000},
    # Reserve — garrison HQ, 8 km
    {"id": "РА-01", "type": "АЦ-40/60","station": "ЦУКС",  "x_m": 8000, "y_m": 2000},
    {"id": "РА-02", "type": "АЦ-40/60","station": "ЦУКС",  "x_m": 8000, "y_m": 2000},
    {"id": "РА-03", "type": "ПНС-110", "station": "ЦУКС",  "x_m": 8000, "y_m": 2000},
]
