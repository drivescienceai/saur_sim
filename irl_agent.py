"""
irl_agent.py — Обратное обучение с подкреплением (Inverse RL).
═══════════════════════════════════════════════════════════════════════════════
Восстановление функции вознаграждения из экспертных траекторий.
Алгоритм: Maximum Entropy IRL (Ziebart et al., 2008).

Входные данные: траектории [(s, a), ...] из режимов «Тренажёр», «СППР»,
реальных планов тушения пожаров (ПТП).

Выход: вектор весов w = [w_area, w_casualties, w_cost, w_L7, ...],
определяющий линейную функцию вознаграждения R(s,a) = w^T · φ(s,a).
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    from .rl_agent import STATE_SIZE, N_ACTIONS, ACTION_COST
except ImportError:
    from rl_agent import STATE_SIZE, N_ACTIONS, ACTION_COST

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_IRL_WEIGHTS_PATH = os.path.join(_DATA_DIR, "checkpoints", "irl_weights.npz")

# ── Признаки состояния-действия φ(s, a) ─────────────────────────────────────
N_FEATURES = 6   # площадь, потери, стоимость, качество, фаза_совпадение, допустимость

def _extract_features(state_idx: int, action: int) -> np.ndarray:
    """Извлечь вектор признаков φ(s, a) из индекса состояния и действия.

    Признаки (нормализованные 0..1):
      0: нормализованная тяжесть пожара (из state encoding)
      1: уровень ресурсов (инвертирован: 0=хорошо, 1=дефицит)
      2: стоимость действия (ACTION_COST)
      3: качество реагирования (из state encoding)
      4: действие стратегическое (0/1)
      5: действие тактическое (0/1)
    """
    # Декодирование state_idx → компоненты (обратное to_index)
    N_Q, N_S, N_R = 3, 5, 3
    remainder = state_idx
    phase_idx = remainder // (N_R * N_S * N_Q)
    remainder %= (N_R * N_S * N_Q)
    res_lvl = remainder // (N_S * N_Q)
    remainder %= (N_S * N_Q)
    severity = remainder // N_Q
    quality = remainder % N_Q

    cost = ACTION_COST[action] if 0 <= action < len(ACTION_COST) else 0.15
    is_strategic = 1.0 if action < 5 else 0.0
    is_tactical = 1.0 if 5 <= action < 9 else 0.0

    return np.array([
        severity / 4.0,           # тяжесть пожара (0..1)
        1.0 - res_lvl / 2.0,     # дефицит ресурсов (0=ОК, 1=критич.)
        cost,                     # стоимость действия
        quality / 2.0,            # качество реагирования
        is_strategic,             # стратегическое действие
        is_tactical,              # тактическое действие
    ], dtype=float)


class MaxEntIRL:
    """Обратное обучение с подкреплением (Maximum Entropy IRL).

    Алгоритм:
    1. Вычислить средний вектор признаков экспертных траекторий: μ_expert
    2. Итеративно:
       a) Вычислить policy π_w из текущих весов w (soft value iteration)
       b) Вычислить ожидаемый вектор признаков μ_π под текущей политикой
       c) Обновить w += lr · (μ_expert − μ_π)
    3. Результат: восстановленные веса w
    """

    # Названия признаков (для визуализации)
    FEATURE_NAMES = [
        "Тяжесть пожара",
        "Дефицит ресурсов",
        "Стоимость действия",
        "Качество реагирования",
        "Стратегическое действие",
        "Тактическое действие",
    ]

    def __init__(self, lr: float = 0.05, gamma: float = 0.95,
                 n_iters: int = 50):
        self.lr = lr
        self.gamma = gamma
        self.n_iters = n_iters
        self.weights = np.zeros(N_FEATURES, dtype=float)
        self._trained = False
        self.history: List[Dict] = []  # [{iter, grad_norm, weights}]

    def _expert_feature_expectations(
            self, trajectories: List[List[Tuple[int, int]]]) -> np.ndarray:
        """Средний вектор признаков из экспертных траекторий."""
        mu = np.zeros(N_FEATURES, dtype=float)
        total = 0
        for traj in trajectories:
            discount = 1.0
            for s, a in traj:
                mu += discount * _extract_features(s, a)
                discount *= self.gamma
                total += 1
        if total > 0:
            mu /= len(trajectories)
        return mu

    def _compute_reward_table(self) -> np.ndarray:
        """R(s,a) = w^T · φ(s,a) для всех (s,a)."""
        R = np.zeros((STATE_SIZE, N_ACTIONS), dtype=float)
        for s in range(STATE_SIZE):
            for a in range(N_ACTIONS):
                R[s, a] = self.weights @ _extract_features(s, a)
        return R

    def _soft_value_iteration(self, R: np.ndarray,
                               n_iters: int = 30) -> np.ndarray:
        """Soft Value Iteration → таблица soft-Q → политика π(a|s)."""
        V = np.zeros(STATE_SIZE, dtype=float)
        for _ in range(n_iters):
            # Q(s,a) = R(s,a) + γ·V(s')  (упрощение: s' ≈ uniform transition)
            Q = R + self.gamma * V.mean()
            V = np.log(np.exp(Q).sum(axis=1) + 1e-12)

        # π(a|s) = exp(Q(s,a)) / Σ_a' exp(Q(s,a'))
        Q_final = R + self.gamma * V.mean()
        exp_Q = np.exp(Q_final - Q_final.max(axis=1, keepdims=True))
        policy = exp_Q / (exp_Q.sum(axis=1, keepdims=True) + 1e-12)
        return policy

    def _policy_feature_expectations(self, policy: np.ndarray) -> np.ndarray:
        """Ожидаемый вектор признаков под политикой π."""
        mu = np.zeros(N_FEATURES, dtype=float)
        for s in range(STATE_SIZE):
            for a in range(N_ACTIONS):
                mu += policy[s, a] * _extract_features(s, a)
        mu /= STATE_SIZE
        return mu

    def train(self, trajectories: List[List[Tuple[int, int]]]) -> Dict:
        """Обучить IRL на экспертных траекториях.

        trajectories: список траекторий, каждая = [(state_idx, action_idx), ...]

        Возвращает: {weights, feature_names, grad_norm_final, n_iters}
        """
        if not trajectories:
            return {"weights": self.weights.tolist(),
                    "feature_names": self.FEATURE_NAMES,
                    "grad_norm_final": 0.0, "n_iters": 0}

        mu_expert = self._expert_feature_expectations(trajectories)
        self.history = []

        for i in range(self.n_iters):
            R = self._compute_reward_table()
            policy = self._soft_value_iteration(R)
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
        return {
            "weights": self.weights.tolist(),
            "feature_names": self.FEATURE_NAMES,
            "grad_norm_final": self.history[-1]["grad_norm"] if self.history else 0.0,
            "n_iters": len(self.history),
        }

    def reward(self, state_idx: int, action: int) -> float:
        """R(s,a) = w^T · φ(s,a)."""
        return float(self.weights @ _extract_features(state_idx, action))

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

    def compare_with_manual(self) -> Dict[str, Tuple[float, float]]:
        """Сравнить восстановленные веса с ручными из rl_agent.py.

        Ручные: W_L1=0.35 (площадь), W_CASUALTY=0.25 (потери),
                W_COST=0.10 (стоимость), W_L7=0.30 (качество).
        """
        manual = {
            "Тяжесть пожара":         0.35,
            "Дефицит ресурсов":       0.25,
            "Стоимость действия":     0.10,
            "Качество реагирования":  0.30,
            "Стратегическое действие": 0.0,
            "Тактическое действие":   0.0,
        }
        result = {}
        for i, name in enumerate(self.FEATURE_NAMES):
            w_irl = float(self.weights[i]) if i < len(self.weights) else 0.0
            w_man = manual.get(name, 0.0)
            result[name] = (w_irl, w_man)
        return result
