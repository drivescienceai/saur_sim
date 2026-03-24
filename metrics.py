"""Вычисление КПЭ: mu1-mu5, оценка риска, триггер адаптации.

Терминология соответствует БУПО и методологии САУР.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .state_space import FirePhase, SituationState, ResourceSpace


@dataclass
class SAURMetrics:
    """Все значения КПЭ для одного снимка моделирования."""

    # L1 — метрики оценки тактической обстановки (ОТО)
    cop_accuracy: float = 1.0       # mu1: P_ОТО — точность оценки обстановки
    sensor_coverage: float = 1.0    # доля площади под наблюдением

    # L2 — метрики тактического уровня (НУТ/НС)
    response_time_min: float = 0.0  # время следования до места пожара (мин)
    personnel_safety: float = 1.0   # 0 = потери ЛС, 1 = ЛС в безопасности

    # L3 — метрики оперативного уровня (РТП)
    L1_mean_tau: float = 0.0         # средняя задержка информации (мин)
    L3_coverage: float = 1.0         # доля объектов с актуальными данными
    L7_reliability: float = 0.0      # P_k — вероятность выполнения задачи
    swarm_J: float = 0.0             # J(t) — эффективность группировки (БПЛА)
    regroup_latency_min: float = 0.0 # время перегруппировки (мин)
    forecast_rmse: float = 0.0       # RMSE прогноза распространения огня

    # L4 — метрики гарнизонного уровня (ПСГ)
    garrison_readiness: float = 1.0      # доля боеготовых единиц гарнизона
    territory_coverage_p: float = 0.95  # P(прибытие <= 10 мин)

    # L5 — метрики стратегического уровня
    mortality_trend: float = 0.0    # delta(погибшие/происшествия) — < 0 улучшение

    # Производные
    risk_score: float = 0.0
    risk_level: str = "НИЗКИЙ"
    delta_s: float = 0.0            # |s(tau) - s*(tau)| — отклонение от целевого состояния
    adaptation_needed: bool = False

    # Фаза и ресурсы
    phase: FirePhase = FirePhase.NORMAL
    n_active_units: int = 0
    n_available_units: int = 0
    fire_area_m2: float = 0.0
    casualties: int = 0

    # Оперативные реквизиты БУПО
    fire_rank: int = 0              # Номер пожара (ранг вызова: 1–5+)
    has_shtab: bool = False         # Создан оперативный штаб (ОШ)
    bu_count: int = 0               # Число боевых участков (БУ)
    stp_count: int = 0              # Число секторов тушения пожара (СТП)
    reshayushee_napravlenie: str = "локализация"  # Текущее РН


def compute_risk_score(situation: SituationState,
                       resources: ResourceSpace,
                       L7: float, L1: float) -> float:
    """Вычислить комплексный показатель риска в [0, 1]."""
    # Компонента устаревания ОТО
    delta_t_ref = 60.0  # эталонная задержка (мин)
    staleness = min(1.0, L1 / max(delta_t_ref, 1.0))

    # Компонента распространения огня
    area_norm = min(1.0, situation.fire_area_m2 / 10000.0)  # норм. к 1 га
    phase_weight = {
        FirePhase.NORMAL:   0.0,
        FirePhase.S1:       0.2,
        FirePhase.S2:       0.4,
        FirePhase.S3:       0.9,
        FirePhase.S4:       0.5,
        FirePhase.S5:       0.2,
        FirePhase.RESOLVED: 0.0,
    }
    fire_factor = area_norm * phase_weight.get(situation.phase, 0.0)

    # Компонента дефицита сил и средств
    total = len(resources.vehicles)
    avail = len(resources.available_units)
    deficit = max(0.0, 1.0 - avail / max(total, 1))

    score = 0.40 * staleness + 0.35 * fire_factor + 0.25 * deficit
    return float(np.clip(score, 0.0, 1.0))


def compute_delta_s(situation: SituationState, target_phase: FirePhase,
                    L7: float, L7_target: float = 0.90) -> float:
    """Вычислить отклонение от целевого состояния delta_s = |s(tau) - s*(tau)|."""
    phase_val  = situation.phase.value
    target_val = target_phase.value
    phase_dev  = abs(phase_val - target_val) / 6.0  # норм. к [0,1]

    rel_dev = max(0.0, L7_target - L7)

    return float(0.6 * phase_dev + 0.4 * rel_dev)


EPS_THRESHOLD = 0.20  # delta_s > epsilon — необходима адаптация


def adaptation_trigger(delta_s: float) -> bool:
    return delta_s > EPS_THRESHOLD


def compute_fire_rank(situation: SituationState,
                      resources: ResourceSpace) -> int:
    """Определить номер пожара (ранг вызова) по БУПО.

    1 — первое подразделение;
    2 — привлечено ≥ 2 отделений, создан ОШ;
    3 — привлечено ≥ 6 отделений;
    4 — привлечено ≥ 10 отделений;
    5+ — крупный пожар.
    """
    n_active = len(resources.active_units)
    if n_active >= 10:
        return 5
    elif n_active >= 6:
        return 3
    elif n_active >= 4:
        return 2
    elif n_active >= 1:
        return 1
    return 0
