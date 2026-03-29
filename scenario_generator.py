"""
scenario_generator.py — Генерация синтетических сценариев пожаров.
═══════════════════════════════════════════════════════════════════════════════
Создаёт правдоподобные сценарии на основе статистических моделей,
обученных на реальных данных (базе прецедентов).

Методы:
  1. Параметрическая генерация (из распределений калиброванных параметров)
  2. Пертурбация прецедентов (модификация реального сценария)
  3. Комбинаторная (комбинации объём × топливо × кровля × ранг)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    from .cbr_engine import CaseBase, FireCase, FUEL_CODES, ROOF_CODES
except ImportError:
    from cbr_engine import CaseBase, FireCase, FUEL_CODES, ROOF_CODES

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


FUEL_LIST = ["бензин", "нефть", "дизель", "мазут", "керосин"]
ROOF_LIST = ["конусная", "плавающая", "понтонная", "стационарная"]
VOLUME_LIST = [700, 1000, 2000, 3000, 5000, 10000, 15000, 20000, 30000, 50000]

# Нормативная интенсивность пены по типу топлива
FOAM_NORMS = {"бензин": 0.065, "нефть": 0.060, "дизель": 0.048,
              "мазут": 0.060, "керосин": 0.050}

# Препятствие крыши
ROOF_OBSTRUCTION = {"конусная": 0.0, "плавающая": 0.70,
                    "понтонная": 0.50, "стационарная": 0.30}


@dataclass
class GeneratedScenario:
    """Синтетический сценарий пожара."""
    scenario_id: str
    method: str             # "parametric", "perturbation", "combinatorial"
    rvs_volume: float
    rvs_diameter: float
    fire_area: float
    fire_rank: int
    fuel_type: str
    roof_type: str
    roof_obstruction: float
    duration_min: int
    foam_intensity: float
    timeline_events: int
    source_case: str = ""   # ID прецедента-основы (для пертурбации)

    def to_simulator_format(self) -> dict:
        return {
            "name": f"Синтетич. {self.scenario_id} (V={self.rvs_volume:.0f} м³, "
                    f"{self.fuel_type}, ранг №{self.fire_rank})",
            "short": f"Синт-{self.scenario_id}",
            "total_min": self.duration_min,
            "initial_fire_area": self.fire_area,
            "fuel": self.fuel_type,
            "rvs_name": f"РВС (V={self.rvs_volume:.0f} м³)",
            "rvs_diameter_m": self.rvs_diameter,
            "fire_rank_default": self.fire_rank,
            "roof_obstruction_init": self.roof_obstruction,
            "foam_intensity": self.foam_intensity,
            "tl_lookup": {},
            "timeline": [],
            "scripted_effects": {},
            "actions_by_phase": None,
            "source_file": f"synthetic_{self.scenario_id}",
        }


class ScenarioGenerator:
    """Генератор синтетических сценариев пожаров."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._case_base: Optional[CaseBase] = None

    def set_case_base(self, cb: CaseBase):
        """Задать базу прецедентов для обучения генератора."""
        self._case_base = cb

    # ── 1. Параметрическая генерация ──────────────────────────────────────
    def generate_parametric(self, n: int = 50) -> List[GeneratedScenario]:
        """Генерация из распределений параметров.

        Если база прецедентов задана — параметры из неё.
        Иначе — из нормативных диапазонов.
        """
        scenarios = []
        for i in range(n):
            volume = self.rng.choice(VOLUME_LIST)
            fuel = self.rng.choice(FUEL_LIST)
            roof = self.rng.choice(ROOF_LIST)
            rank = self._rank_from_volume(volume)
            diameter = 2 * math.sqrt(volume / (math.pi * 12))
            area = math.pi * (diameter / 2) ** 2 * self.rng.uniform(0.3, 1.0)
            obstruction = ROOF_OBSTRUCTION.get(roof, 0.0)
            duration = self._duration_from_rank(rank)
            foam = FOAM_NORMS.get(fuel, 0.05)

            scenarios.append(GeneratedScenario(
                scenario_id=f"P{i + 1:04d}",
                method="parametric",
                rvs_volume=volume,
                rvs_diameter=round(diameter, 1),
                fire_area=round(area, 0),
                fire_rank=rank,
                fuel_type=fuel,
                roof_type=roof,
                roof_obstruction=obstruction,
                duration_min=duration,
                foam_intensity=foam,
                timeline_events=max(5, rank * 8 + self.rng.randint(-3, 5)),
            ))
        return scenarios

    # ── 2. Пертурбация прецедентов ────────────────────────────────────────
    def generate_perturbation(self, n: int = 50,
                              noise_level: float = 0.2
                              ) -> List[GeneratedScenario]:
        """Создать вариации существующих прецедентов.

        Модифицирует параметры реального случая на ±noise_level.
        """
        if not self._case_base or len(self._case_base) == 0:
            return self.generate_parametric(n)

        scenarios = []
        cases = self._case_base.cases
        for i in range(n):
            base = self.rng.choice(cases)
            noise = lambda x: x * (1 + self.rng.uniform(-noise_level, noise_level))

            volume = max(500, noise(base.rvs_volume))
            rank = max(1, min(5, base.fire_rank + self.rng.choice([-1, 0, 0, 1])))
            fuel = base.fuel_type if self.rng.random() > 0.2 else self.rng.choice(FUEL_LIST)
            roof = base.roof_type if self.rng.random() > 0.2 else self.rng.choice(ROOF_LIST)
            diameter = 2 * math.sqrt(volume / (math.pi * 12))
            area = max(50, noise(base.features[2])) if base.features[2] > 0 else math.pi * (diameter / 2) ** 2 * 0.5
            duration = max(30, int(noise(base.duration_min)))
            obstruction = ROOF_OBSTRUCTION.get(roof, 0.0)

            scenarios.append(GeneratedScenario(
                scenario_id=f"R{i + 1:04d}",
                method="perturbation",
                rvs_volume=round(volume, 0),
                rvs_diameter=round(diameter, 1),
                fire_area=round(area, 0),
                fire_rank=rank,
                fuel_type=fuel,
                roof_type=roof,
                roof_obstruction=obstruction,
                duration_min=duration,
                foam_intensity=FOAM_NORMS.get(fuel, 0.05),
                timeline_events=max(5, rank * 6),
                source_case=base.case_id,
            ))
        return scenarios

    # ── 3. Комбинаторная генерация ────────────────────────────────────────
    def generate_combinatorial(self,
                               volumes: Optional[List] = None,
                               fuels: Optional[List] = None,
                               roofs: Optional[List] = None,
                               ) -> List[GeneratedScenario]:
        """Полное перечисление комбинаций параметров.

        По умолчанию: 10 объёмов × 5 топлив × 4 кровли = 200 сценариев.
        """
        if volumes is None:
            volumes = VOLUME_LIST
        if fuels is None:
            fuels = FUEL_LIST
        if roofs is None:
            roofs = ROOF_LIST

        scenarios = []
        idx = 0
        for volume in volumes:
            for fuel in fuels:
                for roof in roofs:
                    idx += 1
                    rank = self._rank_from_volume(volume)
                    diameter = 2 * math.sqrt(volume / (math.pi * 12))
                    area = math.pi * (diameter / 2) ** 2 * 0.7
                    obstruction = ROOF_OBSTRUCTION.get(roof, 0.0)
                    duration = self._duration_from_rank(rank)

                    scenarios.append(GeneratedScenario(
                        scenario_id=f"C{idx:04d}",
                        method="combinatorial",
                        rvs_volume=volume,
                        rvs_diameter=round(diameter, 1),
                        fire_area=round(area, 0),
                        fire_rank=rank,
                        fuel_type=fuel,
                        roof_type=roof,
                        roof_obstruction=obstruction,
                        duration_min=duration,
                        foam_intensity=FOAM_NORMS.get(fuel, 0.05),
                        timeline_events=rank * 8,
                    ))
        return scenarios

    def _rank_from_volume(self, volume: float) -> int:
        if volume >= 20000:
            return min(5, 4 + self.rng.choice([0, 0, 1]))
        elif volume >= 5000:
            return 3 + self.rng.choice([0, 0, 1])
        elif volume >= 1000:
            return 2
        return 1

    def _duration_from_rank(self, rank: int) -> int:
        base = {1: 90, 2: 200, 3: 400, 4: 1200, 5: 3000}
        return int(base.get(rank, 300) * self.rng.uniform(0.6, 1.5))

    # ── Сохранение ────────────────────────────────────────────────────────
    def save_scenarios(self, scenarios: List[GeneratedScenario],
                       output_dir: str = "") -> List[str]:
        """Сохранить сценарии в JSON."""
        if not output_dir:
            output_dir = os.path.join(_DATA_DIR, "scenarios_synthetic")
        os.makedirs(output_dir, exist_ok=True)

        paths = []
        for s in scenarios:
            data = s.to_simulator_format()
            path = os.path.join(output_dir, f"{s.scenario_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            paths.append(path)
        return paths

    def stats(self, scenarios: List[GeneratedScenario]) -> Dict:
        """Статистика набора сценариев."""
        if not scenarios:
            return {"n": 0}
        volumes = [s.rvs_volume for s in scenarios]
        ranks = [s.fire_rank for s in scenarios]
        fuels = {}
        roofs = {}
        for s in scenarios:
            fuels[s.fuel_type] = fuels.get(s.fuel_type, 0) + 1
            roofs[s.roof_type] = roofs.get(s.roof_type, 0) + 1
        return {
            "n": len(scenarios),
            "volume_min": min(volumes),
            "volume_max": max(volumes),
            "volume_mean": round(np.mean(volumes), 0),
            "rank_distribution": {r: ranks.count(r) for r in sorted(set(ranks))},
            "fuel_distribution": fuels,
            "roof_distribution": roofs,
            "methods": {s.method for s in scenarios},
        }
