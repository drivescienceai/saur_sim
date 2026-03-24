"""L5: Strategic Level — institutional learning."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class OperationRecord:
    """Stored result of a completed operation for learning."""
    duration_min: float
    max_phase: str
    casualties: int
    total_units_deployed: int
    L7_final: float
    J_final: float
    peak_fire_area_m2: float
    outcome: str  # "controlled" | "uncontrolled" | "evacuated"


class L5StrategicLayer:
    """L5 strategic + institutional learning layer.

    Functions: f_analyze, f_norm, f_plan_infra, f_knowledge, f_risk_forecast
    Autonomy: alpha5 = 1.0 (fully autonomous)
    Metric: mu5 = mortality trend, KPI trend
    """

    def __init__(self):
        self.operation_records: List[OperationRecord] = []
        self._kpi_weights: Dict[str, float] = {
            "L7": 0.30, "response_time": 0.25,
            "L1": 0.20, "casualties": 0.25,
        }
        self._lessons_learned: List[str] = []

    def record_operation(self, record: OperationRecord) -> None:
        self.operation_records.append(record)
        self._extract_lessons(record)

    def _extract_lessons(self, record: OperationRecord) -> None:
        if record.casualties > 0:
            self._lessons_learned.append(
                f"Потери л/с ({record.casualties}) — пересмотреть пороги отхода на фазе {record.max_phase}")
        if record.L7_final < 0.70:
            self._lessons_learned.append(
                "Критически низкая надёжность мониторинга — увеличить группировку БПЛА")
        if record.duration_min > 240:
            self._lessons_learned.append(
                "Операция >4 ч — проверить стратегию ресурсного резерва")

    def mortality_trend(self) -> float:
        """Compute delta(casualties/operations) — negative = improvement."""
        if len(self.operation_records) < 2:
            return 0.0
        recent = self.operation_records[-10:]
        n = len(recent)
        half = n // 2
        if half == 0:
            return 0.0
        early = sum(r.casualties for r in recent[:half]) / half
        late  = sum(r.casualties for r in recent[half:]) / max(n - half, 1)
        return float(late - early)

    def strategic_kpis(self) -> Dict[str, float]:
        if not self.operation_records:
            return {}
        records = self.operation_records
        return {
            "mean_duration_min": np.mean([r.duration_min for r in records]),
            "mean_L7": np.mean([r.L7_final for r in records]),
            "casualty_rate": sum(r.casualties for r in records) / len(records),
            "control_rate": sum(1 for r in records if r.outcome == "controlled") / len(records),
            "mortality_trend": self.mortality_trend(),
        }

    def lessons(self) -> List[str]:
        return list(set(self._lessons_learned))
