"""
pref_rl.py — Обучение по предпочтениям эксперта (Preference-based RL).
═══════════════════════════════════════════════════════════════════════════════
Эксперт сравнивает пары траекторий и указывает, какая лучше.
Из предпочтений восстанавливается reward-модель (Bradley-Terry),
затем агент дообучается на этой модели.

Источник предпочтений: отклонения в режиме СППР (каждое отклонение =
неявное предпочтение выбора оператора над рекомендацией агента).
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
_PREF_DATASET_PATH = os.path.join(_DATA_DIR, "pref_dataset.json")
_PREF_REWARD_PATH  = os.path.join(_DATA_DIR, "checkpoints", "pref_reward.npz")


class PreferenceDataset:
    """Датасет предпочтений: пары (s, a_preferred, a_rejected)."""

    def __init__(self):
        self.comparisons: List[Dict] = []

    def add(self, state_idx: int, a_preferred: int, a_rejected: int,
            phase: str = "", source: str = "sppр"):
        """Добавить предпочтение: в состоянии s действие a_preferred лучше a_rejected."""
        self.comparisons.append({
            "s": int(state_idx),
            "a_pref": int(a_preferred),
            "a_rej": int(a_rejected),
            "phase": phase,
            "source": source,
        })

    def add_from_sppр_log(self, sppр_log: List[Dict],
                           sim_states: Optional[List[int]] = None):
        """Извлечь предпочтения из журнала СППР.

        Каждое отклонение = предпочтение: user_action > agent_action.
        Каждое принятие = предпочтение: agent_action подтверждено.
        """
        for i, entry in enumerate(sppр_log):
            if not entry.get("accepted", True):
                s = sim_states[i] if sim_states and i < len(sim_states) else 0
                self.comparisons.append({
                    "s": s,
                    "a_pref": entry.get("user_a", 0),
                    "a_rej": entry.get("rec_a", 0),
                    "phase": entry.get("phase", ""),
                    "source": "sppр",
                })

    def __len__(self):
        return len(self.comparisons)

    def save(self, path: str = _PREF_DATASET_PATH) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.comparisons, f, ensure_ascii=False, indent=1)
        return path

    def load(self, path: str = _PREF_DATASET_PATH) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            self.comparisons = json.load(f)
        return True

    def stats(self) -> Dict:
        if not self.comparisons:
            return {"n": 0}
        sources = {}
        for c in self.comparisons:
            src = c.get("source", "?")
            sources[src] = sources.get(src, 0) + 1
        return {"n": len(self.comparisons), "sources": sources}


class BradleyTerryReward:
    """Модель вознаграждения на основе модели Bradley-Terry.

    P(a_pref > a_rej | s) = σ(R(s, a_pref) - R(s, a_rej))

    R(s, a) — таблица (STATE_SIZE × N_ACTIONS), обучаемая через
    максимизацию логарифмического правдоподобия.
    """

    def __init__(self, lr: float = 0.01, n_epochs: int = 100):
        self.lr = lr
        self.n_epochs = n_epochs
        self.R = np.zeros((STATE_SIZE, N_ACTIONS), dtype=float)
        self._trained = False
        self.history: List[Dict] = []

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def train(self, dataset: PreferenceDataset) -> Dict:
        """Обучить reward-модель на датасете предпочтений.

        Максимизация log-likelihood:
        L = Σ log σ(R(s, a_pref) - R(s, a_rej))

        Градиент:
        ∂L/∂R(s, a_pref) = 1 - σ(R(s, a_pref) - R(s, a_rej))
        ∂L/∂R(s, a_rej)  = -(1 - σ(R(s, a_pref) - R(s, a_rej)))
        """
        if len(dataset) == 0:
            return {"n_epochs": 0, "log_likelihood": 0.0, "accuracy": 0.0}

        self.history = []
        comps = dataset.comparisons

        for epoch in range(self.n_epochs):
            total_ll = 0.0
            correct = 0

            for c in comps:
                s = c["s"]
                a_p = c["a_pref"]
                a_r = c["a_rej"]
                if not (0 <= s < STATE_SIZE and
                        0 <= a_p < N_ACTIONS and
                        0 <= a_r < N_ACTIONS):
                    continue

                diff = self.R[s, a_p] - self.R[s, a_r]
                prob = self._sigmoid(np.array([diff]))[0]
                total_ll += np.log(prob + 1e-12)

                if prob > 0.5:
                    correct += 1

                # Градиентное обновление
                grad = 1.0 - prob
                self.R[s, a_p] += self.lr * grad
                self.R[s, a_r] -= self.lr * grad

            n_valid = max(len(comps), 1)
            acc = correct / n_valid
            self.history.append({
                "epoch": epoch,
                "log_likelihood": float(total_ll),
                "accuracy": acc,
            })

            # Ранняя остановка при высокой точности
            if acc > 0.99 and epoch > 10:
                break

        self._trained = True
        last = self.history[-1] if self.history else {"log_likelihood": 0, "accuracy": 0}
        return {
            "n_epochs": len(self.history),
            "log_likelihood": last["log_likelihood"],
            "accuracy": last["accuracy"],
            "n_comparisons": len(comps),
        }

    def reward(self, state_idx: int, action: int) -> float:
        """R(s, a) из обученной модели."""
        if 0 <= state_idx < STATE_SIZE and 0 <= action < N_ACTIONS:
            return float(self.R[state_idx, action])
        return 0.0

    def save(self, path: str = _PREF_REWARD_PATH) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, R=self.R)
        return path

    def load(self, path: str = _PREF_REWARD_PATH) -> bool:
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.R = data["R"]
        self._trained = True
        return True

    @property
    def trained(self) -> bool:
        return self._trained

    def agreement_with_q(self, Q: np.ndarray) -> float:
        """Совпадение ранжирования: для скольких состояний argmax R == argmax Q."""
        if Q.shape != self.R.shape:
            return 0.0
        agree = (self.R.argmax(axis=1) == Q.argmax(axis=1)).sum()
        return float(agree / STATE_SIZE)
