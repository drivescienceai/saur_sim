"""
offline_rl.py — Пакетное обучение с подкреплением (Offline / Batch RL).
═══════════════════════════════════════════════════════════════════════════════
Обучение агента на фиксированном датасете переходов (s, a, r, s', done)
без взаимодействия с симулятором.

Алгоритм: Fitted Q-Iteration (FQI) — итеративное обновление Q-таблицы
на основе мини-батчей из датасета.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    from .rl_agent import STATE_SIZE, N_ACTIONS
except ImportError:
    from rl_agent import STATE_SIZE, N_ACTIONS

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_TRANSITIONS_PATH = os.path.join(_DATA_DIR, "offline_transitions.json")
_OFFLINE_Q_PATH   = os.path.join(_DATA_DIR, "checkpoints", "offline_qtable.npz")


class TransitionBuffer:
    """Буфер переходов (s, a, r, s', done) для пакетного обучения."""

    def __init__(self):
        self.transitions: List[Dict] = []

    def add(self, s: int, a: int, r: float, s_next: int, done: bool):
        self.transitions.append({
            "s": int(s), "a": int(a), "r": float(r),
            "s_next": int(s_next), "done": bool(done),
        })

    def __len__(self):
        return len(self.transitions)

    def save(self, path: str = _TRANSITIONS_PATH) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.transitions, f, ensure_ascii=False)
        return path

    def load(self, path: str = _TRANSITIONS_PATH) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            self.transitions = json.load(f)
        return True

    def merge(self, other: "TransitionBuffer"):
        self.transitions.extend(other.transitions)

    def as_arrays(self) -> Tuple[np.ndarray, ...]:
        """(states, actions, rewards, next_states, dones) как numpy."""
        n = len(self.transitions)
        S  = np.array([t["s"]      for t in self.transitions], dtype=int)
        A  = np.array([t["a"]      for t in self.transitions], dtype=int)
        R  = np.array([t["r"]      for t in self.transitions], dtype=float)
        Sn = np.array([t["s_next"] for t in self.transitions], dtype=int)
        D  = np.array([t["done"]   for t in self.transitions], dtype=bool)
        return S, A, R, Sn, D

    def stats(self) -> Dict:
        if not self.transitions:
            return {"n": 0}
        rewards = [t["r"] for t in self.transitions]
        dones = sum(1 for t in self.transitions if t["done"])
        return {
            "n": len(self.transitions),
            "episodes": dones,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }


class FittedQIteration:
    """Пакетное обучение: Fitted Q-Iteration.

    Итеративно обновляет Q-таблицу на полных проходах по датасету.
    Не требует взаимодействия с симулятором.
    """

    def __init__(self, gamma: float = 0.95, n_iterations: int = 50,
                 alpha: float = 0.1):
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.Q = np.zeros((STATE_SIZE, N_ACTIONS), dtype=float)
        self._trained = False
        self.history: List[Dict] = []

    def train(self, buffer: TransitionBuffer) -> Dict:
        """Обучить на буфере переходов.

        Возвращает: {n_iterations, td_error_final, q_mean, q_std}
        """
        if len(buffer) == 0:
            return {"n_iterations": 0, "td_error_final": 0.0,
                    "q_mean": 0.0, "q_std": 0.0}

        S, A, R, Sn, D = buffer.as_arrays()
        self.history = []

        for i in range(self.n_iterations):
            # Целевые значения: y = r + γ·max_a' Q(s', a')·(1 - done)
            q_next_max = self.Q[Sn].max(axis=1)
            targets = R + self.gamma * q_next_max * (~D).astype(float)

            # TD-ошибки
            current_q = self.Q[S, A]
            td_errors = targets - current_q

            # Обновление Q-таблицы
            # Для каждого перехода: Q(s,a) += α · (target - Q(s,a))
            for j in range(len(S)):
                s, a = S[j], A[j]
                self.Q[s, a] += self.alpha * td_errors[j]

            mean_td = float(np.mean(np.abs(td_errors)))
            self.history.append({
                "iter": i,
                "td_error_mean": mean_td,
                "q_mean": float(self.Q.mean()),
            })

            # Ранняя остановка при сходимости
            if mean_td < 1e-4:
                break

        self._trained = True
        return {
            "n_iterations": len(self.history),
            "td_error_final": self.history[-1]["td_error_mean"] if self.history else 0.0,
            "q_mean": float(self.Q.mean()),
            "q_std": float(self.Q.std()),
            "n_transitions": len(buffer),
        }

    def select_action(self, state_idx: int,
                      mask: Optional[np.ndarray] = None) -> int:
        """Greedy-выбор из обученной Q-таблицы."""
        q = self.Q[state_idx].copy()
        if mask is not None:
            q[~mask] = -np.inf
        return int(np.argmax(q))

    def save(self, path: str = _OFFLINE_Q_PATH) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, Q=self.Q)
        return path

    def load(self, path: str = _OFFLINE_Q_PATH) -> bool:
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.Q = data["Q"]
        self._trained = True
        return True

    @property
    def trained(self) -> bool:
        return self._trained
