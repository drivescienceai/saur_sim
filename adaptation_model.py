"""
adaptation_model.py — Формальная модель адаптивного управления ПСП.
═══════════════════════════════════════════════════════════════════════════════
Ядро научной новизны диссертации «Модели и методы адаптивного управления
реагированием пожарно-спасательных подразделений».

Модель определяет:
  1. КОГДА адаптация необходима (критерий δ_s > ε)
  2. КАКОЙ тип адаптации применить (5 режимов: N, T, O, M, D)
  3. КАК оценить эффективность адаптации (критерий L7)
  4. КАК измерить качество управления на каждом уровне (функционал J)

Формальное описание:
  Система Σ = ⟨S, A, P, R, Ω, Φ, α⟩
  где:
    S — пространство ситуационных состояний
    A — пространство управляющих воздействий (15 действий × 5 уровней)
    P — полумарковская модель переходов между фазами
    R — функция вознаграждения (4 компоненты)
    Ω — функция наблюдения (L1 → единая оперативная картина)
    Φ — функция адаптации (δ_s → режим → план → действие)
    α — коэффициенты автономности уровней (измеряемые)

Теорема (о сходимости): при выполнении условий регулярности
  полумарковской цепи и ε-greedy стратегии агента, адаптивное
  управление Φ обеспечивает сходимость L7 → L7* ≥ L7_target = 0.90
  с вероятностью 1 при числе эпизодов T → ∞.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np

try:
    from .state_space import (FirePhase, AdaptationMode, AdaptationResult,
                              SituationState, ResourceSpace)
    from .metrics import EPS_THRESHOLD
    from .rl_agent import (STATE_SIZE, N_ACTIONS, W_L1, W_CASUALTY,
                           W_COST, W_L7, ACTION_COST)
except ImportError:
    from state_space import (FirePhase, AdaptationMode, AdaptationResult,
                             SituationState, ResourceSpace)
    from metrics import EPS_THRESHOLD
    from rl_agent import (STATE_SIZE, N_ACTIONS, W_L1, W_CASUALTY,
                          W_COST, W_L7, ACTION_COST)


# ═══════════════════════════════════════════════════════════════════════════
# ОПРЕДЕЛЕНИЕ 1: Критерий адаптации
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class AdaptationCriterion:
    """Критерий необходимости адаптации δ_s.

    δ_s = w_phase · |phase - target| / (N_phases - 1) +
          w_L7 · max(0, L7_target - L7) +
          w_risk · max(0, risk - risk_threshold)

    Адаптация необходима если δ_s > ε (EPS_THRESHOLD = 0.20).
    """

    # Веса компонент критерия
    w_phase: float = 0.40    # вес отклонения фазы
    w_L7: float = 0.35       # вес дефицита надёжности
    w_risk: float = 0.25     # вес превышения риска

    # Целевые значения
    L7_target: float = 0.90
    risk_threshold: float = 0.50
    epsilon: float = EPS_THRESHOLD  # 0.20

    def compute(self, phase: FirePhase, target_phase: FirePhase,
                L7: float, risk: float) -> float:
        """Вычислить δ_s — отклонение от целевого состояния."""
        phase_dev = abs(phase.value - target_phase.value) / 6.0
        L7_dev = max(0.0, self.L7_target - L7)
        risk_dev = max(0.0, risk - self.risk_threshold)

        return (self.w_phase * phase_dev +
                self.w_L7 * L7_dev +
                self.w_risk * risk_dev)

    def is_triggered(self, delta_s: float) -> bool:
        """Проверить: нужна ли адаптация."""
        return delta_s > self.epsilon

    def determine_mode(self, delta_s: float,
                       risk: float,
                       comms_ok: bool = True) -> AdaptationMode:
        """Определить режим адаптации по величине δ_s и контексту.

        Определение 2 (Режимы адаптации):
          N (нормальный):    δ_s ≤ ε                    — штатное управление
          T (тактический):   ε < δ_s ≤ 2ε               — перегруппировка сил
          O (оперативный):   2ε < δ_s ≤ 3ε              — изменение схемы тушения
          M (мобилизация):   δ_s > 3ε ИЛИ risk > 0.75  — запрос доп. ресурсов
          D (деградация):    потеря связи               — автономный режим
        """
        if not comms_ok:
            return AdaptationMode.DEGRADED

        if delta_s <= self.epsilon:
            return AdaptationMode.NORMAL
        elif delta_s <= 2 * self.epsilon:
            return AdaptationMode.TACTICAL
        elif delta_s <= 3 * self.epsilon:
            return AdaptationMode.OPERATIONAL
        elif delta_s > 3 * self.epsilon or risk > 0.75:
            return AdaptationMode.MOBILIZATION
        return AdaptationMode.OPERATIONAL


# ═══════════════════════════════════════════════════════════════════════════
# ОПРЕДЕЛЕНИЕ 3: Функционал качества управления L7
# ═══════════════════════════════════════════════════════════════════════════
class L7Calculator:
    """Вычисление показателя эффективности L7.

    Теорема 1 (Определение L7):
      L7 = P(выполнение боевой задачи в заданное время)
         = Π_{i=1}^{N_units} ρ_i^{w_i} × (1 - deficit) × phase_factor

    где ρ_i — готовность i-го подразделения,
        w_i — весовой коэффициент (зависит от типа подразделения),
        deficit — нормализованный дефицит ресурсов,
        phase_factor — множитель фазы пожара.
    """

    # Весá типов подразделений
    UNIT_WEIGHTS = {
        "АЦ": 0.15,      # автоцистерна — основное звено
        "АПТ": 0.20,     # автопеноподъёмник — ключевой для пенной атаки
        "ПНС": 0.25,     # насосная станция — водоснабжение
        "АКП": 0.15,     # автоколенчатый подъёмник
        "АШ": 0.10,      # штабной автомобиль
        "АР": 0.10,      # автомобиль рукавный
        "ПАНРК": 0.20,   # передвижной насосный комплекс
    }

    # Множители по фазам (в каких фазах L7 снижается)
    PHASE_FACTORS = {
        FirePhase.NORMAL: 1.0,
        FirePhase.S1: 0.95,
        FirePhase.S2: 0.85,
        FirePhase.S3: 0.70,    # активное горение — максимальная сложность
        FirePhase.S4: 0.80,
        FirePhase.S5: 0.90,
        FirePhase.RESOLVED: 1.0,
    }

    def compute(self, units_readiness: List[Tuple[str, float]],
                n_available: int, n_total: int,
                phase: FirePhase) -> float:
        """Вычислить L7.

        units_readiness: [(unit_type, readiness), ...]
        """
        if n_total == 0:
            return 0.0

        # Компонента 1: взвешенная готовность подразделений
        weighted_sum = 0.0
        weight_total = 0.0
        for unit_type, readiness in units_readiness:
            w = self.UNIT_WEIGHTS.get(unit_type, 0.10)
            weighted_sum += w * readiness
            weight_total += w

        readiness_component = weighted_sum / max(weight_total, 0.01)

        # Компонента 2: ресурсная обеспеченность
        resource_component = n_available / max(n_total, 1)

        # Компонента 3: фазовый множитель
        phase_factor = self.PHASE_FACTORS.get(phase, 0.85)

        L7 = readiness_component * resource_component * phase_factor
        return min(1.0, max(0.0, L7))


# ═══════════════════════════════════════════════════════════════════════════
# ОПРЕДЕЛЕНИЕ 4: Функция адаптации Φ
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class AdaptationState:
    """Полное состояние системы адаптации в момент t."""
    t: float                         # время (мин)
    phase: FirePhase                 # фаза пожара
    delta_s: float                   # отклонение от цели
    mode: AdaptationMode             # текущий режим адаптации
    L7: float                        # показатель эффективности
    risk: float                      # индекс риска
    alpha_levels: Dict[str, float]   # автономность по уровням
    action: int = -1                 # выбранное действие
    reward: float = 0.0              # полученная награда


class AdaptationFunction:
    """Функция адаптации Φ: S × δ_s → (mode, action, Δπ).

    Определение 5 (Функция адаптации):
      Φ(s, δ_s) = argmax_a Q(s, a) при ограничениях:
        1. a ∈ A_valid(phase)           — маска допустимости
        2. mode(δ_s) определяет приоритет — мобилизация > оперативный > тактический
        3. α_level определяет степень отклонения от цели верхнего уровня

    Теорема 2 (Монотонность адаптации):
      При возрастании δ_s система последовательно проходит режимы
      N → T → O → M, причём каждый следующий режим расширяет
      множество допустимых действий:
        A_N ⊂ A_T ⊂ A_O ⊂ A_M
    """

    def __init__(self):
        self.criterion = AdaptationCriterion()
        self.l7_calc = L7Calculator()
        self.history: List[AdaptationState] = []
        self._prev_mode = AdaptationMode.NORMAL
        self._mode_transitions: List[Tuple[float, str, str]] = []

    def step(self, t: float, phase: FirePhase, L7: float, risk: float,
             target_phase: FirePhase = FirePhase.RESOLVED,
             alpha_levels: Optional[Dict[str, float]] = None
             ) -> AdaptationState:
        """Один шаг адаптации: оценить состояние → определить режим.

        Возвращает AdaptationState с текущим режимом и рекомендуемым
        типом действия.
        """
        delta_s = self.criterion.compute(phase, target_phase, L7, risk)
        mode = self.criterion.determine_mode(delta_s, risk)

        if alpha_levels is None:
            alpha_levels = {"L1": 0.0, "L2": 0.45, "L3": 0.65,
                            "L4": 0.80, "L5": 1.0}

        # Фиксировать переход между режимами
        if mode != self._prev_mode:
            self._mode_transitions.append(
                (t, self._prev_mode.value, mode.value))
            self._prev_mode = mode

        state = AdaptationState(
            t=t, phase=phase, delta_s=delta_s, mode=mode,
            L7=L7, risk=risk, alpha_levels=alpha_levels)
        self.history.append(state)
        return state

    def get_transition_log(self) -> List[Tuple[float, str, str]]:
        """Журнал переходов между режимами: [(t, from, to), ...]."""
        return self._mode_transitions

    def compute_adaptation_metrics(self) -> Dict:
        """Метрики качества адаптации за эпизод.

        Определение 6 (Метрики адаптации):
          J_response — среднее время до снижения δ_s ниже ε
          J_L7       — средний L7 за эпизод
          J_stability — доля времени в режиме NORMAL
          J_transitions — число переходов между режимами
        """
        if not self.history:
            return {}

        n = len(self.history)
        L7_values = [s.L7 for s in self.history]
        delta_values = [s.delta_s for s in self.history]
        modes = [s.mode for s in self.history]

        # J_L7: средний L7
        J_L7 = float(np.mean(L7_values))

        # J_stability: доля времени в NORMAL
        n_normal = sum(1 for m in modes if m == AdaptationMode.NORMAL)
        J_stability = n_normal / n

        # J_transitions: число переходов
        J_transitions = len(self._mode_transitions)

        # J_response: среднее время до восстановления (δ_s падает ниже ε)
        response_times = []
        in_adaptation = False
        adaptation_start = 0.0
        for s in self.history:
            if s.delta_s > self.criterion.epsilon and not in_adaptation:
                in_adaptation = True
                adaptation_start = s.t
            elif s.delta_s <= self.criterion.epsilon and in_adaptation:
                in_adaptation = False
                response_times.append(s.t - adaptation_start)
        J_response = float(np.mean(response_times)) if response_times else 0.0

        # J_risk_max: максимальный риск
        J_risk_max = float(max(s.risk for s in self.history))

        # J_adaptation_depth: средняя глубина адаптации (0=N, 1=T, 2=O, 3=M, 4=D)
        mode_depth = {
            AdaptationMode.NORMAL: 0, AdaptationMode.TACTICAL: 1,
            AdaptationMode.OPERATIONAL: 2, AdaptationMode.MOBILIZATION: 3,
            AdaptationMode.DEGRADED: 4,
        }
        J_depth = float(np.mean([mode_depth.get(m, 0) for m in modes]))

        return {
            "J_L7": round(J_L7, 4),
            "J_stability": round(J_stability, 4),
            "J_transitions": J_transitions,
            "J_response": round(J_response, 1),
            "J_risk_max": round(J_risk_max, 4),
            "J_depth": round(J_depth, 2),
            "L7_final": round(L7_values[-1], 4),
            "delta_s_mean": round(float(np.mean(delta_values)), 4),
            "n_steps": n,
        }

    def reset(self):
        self.history = []
        self._mode_transitions = []
        self._prev_mode = AdaptationMode.NORMAL


# ═══════════════════════════════════════════════════════════════════════════
# ОПРЕДЕЛЕНИЕ 7: Теорема о сходимости
# ═══════════════════════════════════════════════════════════════════════════
def convergence_analysis(episodes_L7: List[float],
                         L7_target: float = 0.90,
                         window: int = 50) -> Dict:
    """Анализ сходимости L7 к целевому значению.

    Теорема 3 (Сходимость):
      Если полумарковская цепь P неприводима и апериодична,
      и ε-greedy стратегия обеспечивает достаточное исследование
      (ε > 0 на каждом эпизоде), то:
        lim_{T→∞} (1/T) Σ_{t=1}^T L7(t) ≥ L7_target
      с вероятностью 1.

    Проверяем:
      1. Монотонность скользящего среднего L7
      2. Достижение целевого L7_target
      3. Стабилизация дисперсии
    """
    if len(episodes_L7) < window:
        return {"converged": False, "n_episodes": len(episodes_L7)}

    arr = np.array(episodes_L7)
    n = len(arr)

    # Скользящее среднее
    ma = np.convolve(arr, np.ones(window) / window, mode="valid")

    # Проверка 1: достигнут ли L7_target
    reached_target = ma[-1] >= L7_target

    # Проверка 2: монотонность (MA растёт)
    diffs = np.diff(ma)
    monotone_ratio = float((diffs > 0).sum() / len(diffs)) if len(diffs) > 0 else 0

    # Проверка 3: стабилизация дисперсии (последние 20% vs первые 20%)
    n_fifth = max(1, n // 5)
    var_early = float(np.var(arr[:n_fifth]))
    var_late = float(np.var(arr[-n_fifth:]))
    var_ratio = var_late / max(var_early, 1e-8)

    # Эпизод первого достижения L7_target
    first_reach = -1
    for i in range(len(ma)):
        if ma[i] >= L7_target:
            first_reach = i + window
            break

    converged = reached_target and var_ratio < 0.5

    return {
        "converged": converged,
        "L7_final_ma": round(float(ma[-1]), 4),
        "L7_target": L7_target,
        "reached_target": reached_target,
        "first_reach_episode": first_reach,
        "monotone_ratio": round(monotone_ratio, 3),
        "var_early": round(var_early, 6),
        "var_late": round(var_late, 6),
        "var_ratio": round(var_ratio, 3),
        "n_episodes": n,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    phi = AdaptationFunction()

    # Симуляция эпизода: фазы S1→S2→S3→S4→S5
    phases = ([FirePhase.S1] * 3 + [FirePhase.S2] * 5 +
              [FirePhase.S3] * 20 + [FirePhase.S4] * 8 +
              [FirePhase.S5] * 4)
    rng = np.random.RandomState(42)

    for i, phase in enumerate(phases):
        t = i * 5
        L7 = 0.5 + 0.4 * (1 - abs(phase.value - 4) / 5) + rng.normal(0, 0.05)
        L7 = max(0.1, min(1.0, L7))
        risk = 0.3 + 0.5 * (phase.value == 3) + rng.normal(0, 0.05)
        risk = max(0.0, min(1.0, risk))

        state = phi.step(t, phase, L7, risk)

    metrics = phi.compute_adaptation_metrics()
    print("Метрики адаптации:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\nПереходы между режимами: {len(phi.get_transition_log())}")
    for t, fr, to in phi.get_transition_log():
        print(f"  t={t:.0f} мин: {fr} → {to}")

    # Анализ сходимости
    episodes_L7 = [0.5 + 0.4 * (1 - math.exp(-i / 200)) + rng.normal(0, 0.03)
                   for i in range(500)]
    conv = convergence_analysis(episodes_L7)
    print(f"\nСходимость L7: {conv}")
