"""L2: Тактический уровень — решения начальника участка тушения (НУТ/НС).

Функции: f_assess (оценка ОТО), f_decide (выбор БЗ), f_deploy (расстановка
сил), f_control_safety (контроль безопасности ЛС).
Самостоятельность alpha2 in [0.3, 0.6].
Метрика mu2: время локализации, потери ЛС = 0.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .state_space import (FirePhase, SituationState, ResourceUnit,
                          AdaptationMode, AdaptationResult,
                          ReshayusheeNapravlenie)


# Боевые задачи НУТ по фазам пожара (БУПО: раздел «Действия на пожаре»)
TACTIC_MAP: Dict[FirePhase, str] = {
    FirePhase.NORMAL:   "несение дежурства",
    FirePhase.S1:       "разведка пожара",          # выявить границы, угрозу людям
    FirePhase.S2:       "боевое развёртывание",     # ввод сил на РН
    FirePhase.S3:       "локализация пожара",        # ограничение площади горения
    FirePhase.S4:       "ликвидация горения",        # подача ОВ на РН
    FirePhase.S5:       "ликвидация последствий",   # проверка, докрут
    FirePhase.RESOLVED: "сбор и возвращение",       # свёртывание сил
}


@dataclass
class TacticalOrder:
    """Боевой приказ НУТ/НС подчинённому подразделению."""
    unit_id: str
    boyevaya_zadacha: str                  # Боевая задача (тактический манёвр)
    uchastok_tusheniya: str                # Боевой участок (БУ) / сектор (СТП)
    priority: int = 1
    safety_check: bool = True
    reshayushee_napravlenie: ReshayusheeNapravlenie = (
        ReshayusheeNapravlenie.LOKALIZATSIYA)  # РН, на котором задействована единица


class L2TacticalAgent:
    """L2 — начальник участка тушения (НУТ) / начальник сектора (НС).

    Реализует ситуационно-ролевые правила выбора боевой задачи и контроля
    безопасности личного состава.
    Самостоятельность alpha2 in [0.3, 0.6]: действует в пределах БУ/сектора.
    """

    def __init__(self, unit: ResourceUnit, alpha: float = 0.45):
        self.unit = unit
        self.alpha = alpha              # уровень самостоятельности
        self.current_tactic = "несение дежурства"
        self.safety_ok = True
        self.orders_log: List[TacticalOrder] = []

    # ------------------------------------------------------------------
    # f_assess — оценка тактической обстановки на БУ
    # ------------------------------------------------------------------

    def assess(self, situation: SituationState) -> str:
        """f_assess: оценить локальную обстановку на боевом участке."""
        phase = situation.phase
        if situation.fire_area_m2 > 3000 and phase == FirePhase.S3:
            return "КРИТИЧЕСКАЯ"
        if phase in (FirePhase.S2, FirePhase.S3, FirePhase.S4):
            return "АКТИВНАЯ"
        if phase == FirePhase.S1:
            return "НАРАСТАЮЩАЯ"
        return "ШТАТНАЯ"

    # ------------------------------------------------------------------
    # f_decide — выбор боевой задачи
    # ------------------------------------------------------------------

    def decide(self, situation: SituationState,
               directive: Optional[str] = None) -> str:
        """f_decide: выбрать БЗ. Директива РТП (L3) имеет приоритет."""
        if directive:
            return directive
        assessment = self.assess(situation)
        tactic = TACTIC_MAP.get(situation.phase, "несение дежурства")
        if assessment == "КРИТИЧЕСКАЯ" and self.alpha >= 0.4:
            tactic = "экстренный отход"    # самостоятельное решение по безопасности
        return tactic

    # ------------------------------------------------------------------
    # f_deploy — расстановка сил и средств
    # ------------------------------------------------------------------

    def deploy(self, tactic: str,
               uchastok: str = "БУ-1",
               rn: ReshayusheeNapravlenie = ReshayusheeNapravlenie.LOKALIZATSIYA,
               ) -> TacticalOrder:
        """f_deploy: выдать боевой приказ подразделению."""
        self.current_tactic = tactic
        order = TacticalOrder(
            unit_id=self.unit.unit_id,
            boyevaya_zadacha=tactic,
            uchastok_tusheniya=uchastok,
            priority=1,
            reshayushee_napravlenie=rn,
        )
        self.orders_log.append(order)
        self.unit.task = tactic
        return order

    # ------------------------------------------------------------------
    # f_control_safety — контроль безопасности ЛС (ГДЗС и т.п.)
    # ------------------------------------------------------------------

    def control_safety(self, situation: SituationState) -> bool:
        """f_control_safety: проверить условия безопасной работы на БУ.

        Критерий отхода: фаза S3, площадь > 5000 м², ветер > 10 м/с
        (ОФП выходят за пределы допустимых — сигнал отхода по БУПО).
        """
        if (situation.phase == FirePhase.S3
                and situation.fire_area_m2 > 5000
                and situation.wind_speed_ms > 10):
            self.safety_ok = False
            return False
        self.safety_ok = True
        return True

    def step(self, situation: SituationState,
             directive: Optional[str] = None) -> TacticalOrder:
        """Полный цикл принятия решений L2 (оценка → решение → приказ)."""
        safe = self.control_safety(situation)
        if not safe:
            tactic = "экстренный отход"
        else:
            tactic = self.decide(situation, directive)
        uchastok = getattr(situation, "_uchastok", "БУ-1")
        # Определить РН по текущей фазе
        rn_map = {
            FirePhase.S1: ReshayusheeNapravlenie.SPASENIE_LYUDEY,
            FirePhase.S2: ReshayusheeNapravlenie.LOKALIZATSIYA,
            FirePhase.S3: ReshayusheeNapravlenie.ZASHCHITA_SOSEDNIKH,
            FirePhase.S4: ReshayusheeNapravlenie.LIKVIDATSIYA,
            FirePhase.S5: ReshayusheeNapravlenie.LIKVIDATSIYA,
        }
        rn = rn_map.get(situation.phase, ReshayusheeNapravlenie.LOKALIZATSIYA)
        return self.deploy(tactic, uchastok, rn)

    def request_resources(self, reason: str) -> Dict[str, object]:
        """Запрос дополнительных сил и средств у РТП (L3)."""
        return {
            "от": self.unit.unit_id,
            "причина": reason,
            "приоритет": "ВЫСОКИЙ" if not self.safety_ok else "НОРМАЛЬНЫЙ",
        }
