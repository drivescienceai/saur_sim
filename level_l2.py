"""L2: Tactical Level — unit commander decisions.

Functions: f_assess, f_decide, f_deploy, f_control_safety
Autonomy: alpha2 in [0.3, 0.6]
Metric: mu2 = speed of localization, personnel casualties = 0
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .state_space import (FirePhase, SituationState, ResourceUnit,
                          AdaptationMode, AdaptationResult)


TACTIC_MAP = {
    FirePhase.NORMAL:   "patrol",
    FirePhase.S1:       "recon_and_entry",
    FirePhase.S2:       "attack_perimeter",
    FirePhase.S3:       "defensive_protection",
    FirePhase.S4:       "surround_and_drown",
    FirePhase.S5:       "overhaul_and_check",
    FirePhase.RESOLVED: "withdrawal",
}


@dataclass
class TacticalOrder:
    unit_id: str
    tactic: str
    sector: str
    priority: int = 1
    safety_check: bool = True


class L2TacticalAgent:
    """L2 tactical commander agent.

    Implements situation-action rules for local tactic selection
    and personnel safety monitoring.
    Autonomy alpha2 in [0.3, 0.6]: can act independently within sector.
    """

    def __init__(self, unit: ResourceUnit, alpha: float = 0.45):
        self.unit = unit
        self.alpha = alpha    # autonomy level
        self.current_tactic = "standby"
        self.safety_ok = True
        self.orders_log: List[TacticalOrder] = []

    def assess(self, situation: SituationState) -> str:
        """f_assess: evaluate local tactical situation."""
        phase = situation.phase
        if situation.fire_area_m2 > 3000 and phase == FirePhase.S3:
            return "CRITICAL"
        if phase in (FirePhase.S2, FirePhase.S3, FirePhase.S4):
            return "ACTIVE"
        if phase == FirePhase.S1:
            return "ESCALATING"
        return "STABLE"

    def decide(self, situation: SituationState,
               directive: Optional[str] = None) -> str:
        """f_decide: select tactic. Directive from L3 overrides if given."""
        if directive:
            return directive
        assessment = self.assess(situation)
        tactic = TACTIC_MAP.get(situation.phase, "standby")
        if assessment == "CRITICAL" and self.alpha >= 0.4:
            tactic = "emergency_withdrawal"  # autonomous safety decision
        return tactic

    def deploy(self, tactic: str, sector: str = "A") -> TacticalOrder:
        """f_deploy: issue tactical order to unit."""
        self.current_tactic = tactic
        order = TacticalOrder(
            unit_id=self.unit.unit_id,
            tactic=tactic,
            sector=sector,
            priority=1,
        )
        self.orders_log.append(order)
        self.unit.task = tactic
        return order

    def control_safety(self, situation: SituationState) -> bool:
        """f_control_safety: check personnel safety conditions."""
        # Check evacuation conditions
        if (situation.phase == FirePhase.S3
                and situation.fire_area_m2 > 5000
                and situation.wind_speed_ms > 10):
            self.safety_ok = False
            return False
        self.safety_ok = True
        return True

    def step(self, situation: SituationState,
             directive: Optional[str] = None) -> TacticalOrder:
        """Full L2 decision cycle."""
        safe = self.control_safety(situation)
        if not safe:
            tactic = "emergency_withdrawal"
        else:
            tactic = self.decide(situation, directive)
        sector = getattr(situation, "_sector", "A")
        return self.deploy(tactic, sector)

    def request_resources(self, reason: str) -> Dict[str, object]:
        """Send resource request to L3."""
        return {
            "from": self.unit.unit_id,
            "reason": reason,
            "priority": "HIGH" if not self.safety_ok else "NORMAL",
        }
