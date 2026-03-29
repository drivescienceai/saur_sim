"""
irl_agent.py — Обратное обучение с подкреплением (Inverse RL).
═══════════════════════════════════════════════════════════════════════════════
ЗАДАЧА: восстановить функцию вознаграждения из экспертных траекторий.

В rl_agent.py функция вознаграждения задана ВРУЧНУЮ:
    R(t) = −0.35·L1_norm − 0.25·casualty − 0.10·cost + 0.30·L7

Обратное обучение отвечает на вопрос: «Какие веса (0.35, 0.25, 0.10, 0.30)
на самом деле использует опытный РТП?» — восстанавливает их из записей
решений эксперта.

ВХОД:  траектории [(state_idx, action_idx), ...] из Тренажёра/СППР
ВЫХОД: восстановленные веса w = [w_площадь, w_потери, w_стоимость, w_качество]

ОТЛИЧИЕ ОТ ИМИТАЦИИ:
  Имитация копирует ДЕЙСТВИЯ эксперта     → «что делать в фазе S2»
  Обратное ОП извлекает ПРИОРИТЕТЫ        → «что важнее: площадь или потери?»
  Эти приоритеты можно подставить в Q-learning → агент сам найдёт лучшие
  действия, оптимизируя то, что реально ценит эксперт.

Алгоритм: Maximum Entropy IRL (Ziebart et al., 2008).
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
from typing import List, Dict, Tuple

import numpy as np

try:
    from .rl_agent import (STATE_SIZE, N_ACTIONS, ACTION_COST,
                           W_L1, W_CASUALTY, W_COST, W_L7)
except ImportError:
    from rl_agent import (STATE_SIZE, N_ACTIONS, ACTION_COST,
                          W_L1, W_CASUALTY, W_COST, W_L7)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_IRL_WEIGHTS_PATH = os.path.join(_DATA_DIR, "checkpoints", "irl_weights.npz")


# ═══════════════════════════════════════════════════════════════════════════
# ПРИЗНАКИ φ(s, a) — те же 4 компоненты, что в ручной reward
# ═══════════════════════════════════════════════════════════════════════════
N_FEATURES = 4

# Названия — точно соответствуют весам из rl_agent.py
FEATURE_NAMES = [
    "Тяжесть пожара (L1_norm)",
    "Потери личного состава",
    "Стоимость действия",
    "Качество реагирования (L7)",
]

# Ручные веса из rl_agent.py (для сравнения)
MANUAL_WEIGHTS = np.array([-W_L1, -W_CASUALTY, -W_COST, +W_L7], dtype=float)
#                          -0.35,   -0.25,      -0.10,   +0.30


def _decode_state(state_idx: int) -> Tuple[int, int, int, int]:
    """Декодировать state_idx → (phase, resource, severity, quality).

    Обратная операция к RLState.to_index():
      s = phase×45 + resource×15 + severity×3 + quality
    """
    N_Q, N_S, N_R = 3, 5, 3
    remainder = state_idx % STATE_SIZE
    phase   = remainder // (N_R * N_S * N_Q)
    remainder %= (N_R * N_S * N_Q)
    resource = remainder // (N_S * N_Q)
    remainder %= (N_S * N_Q)
    severity = remainder // N_Q
    quality  = remainder % N_Q
    return phase, resource, severity, quality


def extract_features(state_idx: int, action: int) -> np.ndarray:
    """Извлечь вектор φ(s, a) — 4 компоненты, идентичные ручной reward.

    φ₀ = severity / 4.0        ≈ L1_norm (тяжесть пожара, 0..1)
    φ₁ = (2 − resource) / 2.0  ≈ casualty_rate (дефицит → потери, 0..1)
    φ₂ = ACTION_COST[action]   = cost(a) (стоимость действия, 0..0.4)
    φ₃ = quality / 2.0         ≈ L7 (качество реагирования, 0..1)

    Ручная reward: R = −0.35·φ₀ − 0.25·φ₁ − 0.10·φ₂ + 0.30·φ₃
    IRL:           R = w₀·φ₀   + w₁·φ₁   + w₂·φ₂   + w₃·φ₃
    """
    _, resource, severity, quality = _decode_state(state_idx)
    cost = ACTION_COST[action] if 0 <= action < N_ACTIONS else 0.15

    return np.array([
        severity / 4.0,             # φ₀: тяжесть пожара
        (2 - resource) / 2.0,       # φ₁: дефицит ресурсов → потери
        cost,                        # φ₂: стоимость действия
        quality / 2.0,              # φ₃: качество реагирования
    ], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# АЛГОРИТМ MaxEnt IRL
# ═══════════════════════════════════════════════════════════════════════════
class MaxEntIRL:
    """Обратное обучение с подкреплением — восстановление весов reward.

    Алгоритм Maximum Entropy IRL:
    1. Вычислить средний вектор признаков экспертных траекторий: μ_expert
    2. Итеративно:
       a) R(s,a) = w^T · φ(s,a) — текущая модель reward
       b) π_w(a|s) = softmax(R(s,a)) — политика из текущих весов
       c) μ_π = E_π[φ(s,a)] — средние признаки под этой политикой
       d) w ← w + lr · (μ_expert − μ_π) — градиентный шаг
    3. Результат: w* — восстановленные веса

    Практический смысл:
    - Если w₀ < −0.35 → эксперт больше ручной формулы ценит снижение площади
    - Если w₃ > +0.30 → эксперт больше ценит качество реагирования
    - Если w₂ ≈ 0.0  → эксперт не учитывает стоимость действий
    """

    def __init__(self, lr: float = 0.05, gamma: float = 0.95,
                 n_iters: int = 100):
        self.lr = lr
        self.gamma = gamma
        self.n_iters = n_iters
        self.weights = np.zeros(N_FEATURES, dtype=float)
        self._trained = False
        self.history: List[Dict] = []

    def _expert_feature_expectations(
            self, trajectories: List[List[Tuple[int, int]]]) -> np.ndarray:
        """μ_expert = (1/N) Σ_traj Σ_t γ^t · φ(s_t, a_t)."""
        mu = np.zeros(N_FEATURES, dtype=float)
        for traj in trajectories:
            discount = 1.0
            for s, a in traj:
                mu += discount * extract_features(s, a)
                discount *= self.gamma
        if trajectories:
            mu /= len(trajectories)
        return mu

    def _compute_reward_table(self) -> np.ndarray:
        """R(s,a) = w^T · φ(s,a) для всех (s,a)."""
        R = np.zeros((STATE_SIZE, N_ACTIONS), dtype=float)
        for s in range(STATE_SIZE):
            for a in range(N_ACTIONS):
                R[s, a] = self.weights @ extract_features(s, a)
        return R

    def _soft_policy(self, R: np.ndarray) -> np.ndarray:
        """π(a|s) = exp(R(s,a)) / Σ_a' exp(R(s,a')) — Больцмановская политика."""
        # Стабильный softmax
        R_shifted = R - R.max(axis=1, keepdims=True)
        exp_R = np.exp(R_shifted)
        policy = exp_R / (exp_R.sum(axis=1, keepdims=True) + 1e-12)
        return policy

    def _policy_feature_expectations(self, policy: np.ndarray) -> np.ndarray:
        """μ_π = (1/|S|) Σ_s Σ_a π(a|s) · φ(s,a)."""
        mu = np.zeros(N_FEATURES, dtype=float)
        for s in range(STATE_SIZE):
            for a in range(N_ACTIONS):
                mu += policy[s, a] * extract_features(s, a)
        mu /= STATE_SIZE
        return mu

    def train(self, trajectories: List[List[Tuple[int, int]]]) -> Dict:
        """Обучить IRL на экспертных траекториях.

        Вход: trajectories = [[(s₀, a₀), (s₁, a₁), ...], ...]
        Выход: словарь с восстановленными весами и сравнением с ручными.

        Пример выхода:
        {
            "weights": [-0.42, -0.18, -0.03, +0.37],
            "manual_weights": [-0.35, -0.25, -0.10, +0.30],
            "interpretation": {
                "Тяжесть пожара": "Эксперт ценит БОЛЬШЕ ручной формулы (−0.42 vs −0.35)",
                "Потери ЛС":     "Эксперт ценит МЕНЬШЕ (−0.18 vs −0.25)",
                "Стоимость":     "Эксперт почти не учитывает (−0.03 vs −0.10)",
                "Качество L7":   "Эксперт ценит БОЛЬШЕ (+0.37 vs +0.30)",
            }
        }
        """
        if not trajectories:
            return {"weights": self.weights.tolist(),
                    "manual_weights": MANUAL_WEIGHTS.tolist(),
                    "n_iters": 0, "interpretation": {}}

        mu_expert = self._expert_feature_expectations(trajectories)
        self.history = []

        for i in range(self.n_iters):
            R = self._compute_reward_table()
            policy = self._soft_policy(R)
            mu_pi = self._policy_feature_expectations(policy)

            grad = mu_expert - mu_pi
            self.weights += self.lr * grad
            grad_norm = float(np.linalg.norm(grad))

            self.history.append({
                "iter": i,
                "grad_norm": grad_norm,
                "weights": self.weights.copy().tolist(),
            })

            if grad_norm < 1e-4:
                break

        self._trained = True
        result = self._make_result()

        # Автосохранение в централизованную БД
        try:
            from results_db import get_db
            get_db().log_irl(
                weights=result["weights"],
                feature_names=result["feature_names"],
                interpretation=result["interpretation"])
        except Exception:
            pass

        return result

    def _make_result(self) -> Dict:
        """Сформировать результат с интерпретацией."""
        interpretation = {}
        for i, name in enumerate(FEATURE_NAMES):
            w_irl = self.weights[i]
            w_man = MANUAL_WEIGHTS[i]
            diff = abs(w_irl) - abs(w_man)
            if abs(diff) < 0.03:
                verdict = f"Совпадает с ручной формулой ({w_irl:+.2f} ≈ {w_man:+.2f})"
            elif abs(w_irl) > abs(w_man):
                verdict = f"Эксперт ценит БОЛЬШЕ ({w_irl:+.2f} vs {w_man:+.2f})"
            else:
                verdict = f"Эксперт ценит МЕНЬШЕ ({w_irl:+.2f} vs {w_man:+.2f})"
            if abs(w_irl) < 0.03:
                verdict = f"Эксперт НЕ УЧИТЫВАЕТ ({w_irl:+.2f} vs {w_man:+.2f})"
            interpretation[name] = verdict

        return {
            "weights": self.weights.tolist(),
            "manual_weights": MANUAL_WEIGHTS.tolist(),
            "feature_names": FEATURE_NAMES,
            "n_iters": len(self.history),
            "grad_norm_final": self.history[-1]["grad_norm"] if self.history else 0.0,
            "interpretation": interpretation,
        }

    def reward(self, state_idx: int, action: int) -> float:
        """R(s,a) = w^T · φ(s,a) с восстановленными весами."""
        return float(self.weights @ extract_features(state_idx, action))

    def retrain_agent_with_recovered_weights(self):
        """Создать новую reward-функцию на основе восстановленных весов.

        Возвращает callable: reward_fn(L1_norm, casualties, action, L7) → float
        которую можно подставить вместо rl_agent.compute_reward().

        Это КЛЮЧЕВОЕ отличие от имитации: имитация копирует действия,
        а обратное ОП даёт ФУНКЦИЮ, которую можно использовать для обучения
        нового, потенциально лучшего агента.
        """
        w = self.weights.copy()

        def recovered_reward(L1_norm: float, casualties: int,
                             action: int, L7: float) -> float:
            cost = ACTION_COST[action] if 0 <= action < N_ACTIONS else 0.0
            phi = np.array([
                L1_norm,
                min(1.0, casualties / 10.0),
                cost,
                L7,
            ])
            return float(w @ phi)

        return recovered_reward

    def save(self, path: str = _IRL_WEIGHTS_PATH) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, weights=self.weights)
        return path

    def load(self, path: str = _IRL_WEIGHTS_PATH) -> bool:
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.weights = data["weights"]
        self._trained = True
        return True

    @property
    def trained(self) -> bool:
        return self._trained
