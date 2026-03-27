"""
bc_agent.py — Клонирование поведения (Behavior Cloning).
═══════════════════════════════════════════════════════════════════════════════
Обучение агента на записях эксперта без функции вознаграждения.
Метод: supervised learning — минимизация кросс-энтропии P(a|s) vs эксперт.

Датасет собирается из режимов «Тренажёр» и «СППР»:
  [{"s": state_index, "a": action_index, "phase": "S1"}, ...]

Агент хранит таблицу вероятностей π(a|s) размером (STATE_SIZE × N_ACTIONS).
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    from .rl_agent import STATE_SIZE, N_ACTIONS, RLState, ACTION_NAMES
except ImportError:
    from rl_agent import STATE_SIZE, N_ACTIONS, RLState, ACTION_NAMES


# ── Путь для сохранения датасета и модели ────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_BC_DATASET_PATH = os.path.join(_DATA_DIR, "bc_dataset.json")
_BC_MODEL_PATH   = os.path.join(_DATA_DIR, "checkpoints", "bc_policy.npz")


class BCDataset:
    """Накопитель датасета (состояние, действие) из записей экспертов."""

    def __init__(self):
        self.samples: List[Dict] = []

    def add(self, state_idx: int, action_idx: int, phase: str = ""):
        self.samples.append({"s": int(state_idx), "a": int(action_idx),
                             "phase": phase})

    def __len__(self):
        return len(self.samples)

    def save(self, path: str = _BC_DATASET_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.samples, f, ensure_ascii=False, indent=1)
        return path

    def load(self, path: str = _BC_DATASET_PATH) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        return True

    def merge(self, other: "BCDataset"):
        self.samples.extend(other.samples)

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Вернуть (states, actions) как numpy-массивы."""
        states  = np.array([s["s"] for s in self.samples], dtype=int)
        actions = np.array([s["a"] for s in self.samples], dtype=int)
        return states, actions

    def stats(self) -> Dict:
        """Статистика датасета: размер, распределение по фазам и действиям."""
        if not self.samples:
            return {"n": 0}
        phases = {}
        acts = {}
        for s in self.samples:
            p = s.get("phase", "?")
            a = s["a"]
            phases[p] = phases.get(p, 0) + 1
            acts[a] = acts.get(a, 0) + 1
        return {"n": len(self.samples), "phases": phases, "actions": acts}


class BehaviorCloningAgent:
    """Табличный агент клонирования поведения.

    Таблица π(a|s) — вероятности действий для каждого состояния.
    Обучение: подсчёт частот action|state, Лапласовское сглаживание.
    """

    def __init__(self, laplace_alpha: float = 1.0):
        self.counts = np.zeros((STATE_SIZE, N_ACTIONS), dtype=float)
        self.policy = np.ones((STATE_SIZE, N_ACTIONS), dtype=float) / N_ACTIONS
        self.laplace_alpha = laplace_alpha
        self._trained = False

    def train(self, dataset: BCDataset) -> Dict:
        """Обучить политику на датасете.

        Возвращает словарь метрик:
          - accuracy: доля правильных предсказаний (argmax π vs эксперт)
          - n_states_seen: количество уникальных посещённых состояний
          - entropy_mean: средняя энтропия политики
        """
        states, actions = dataset.as_arrays()
        if len(states) == 0:
            return {"accuracy": 0.0, "n_states_seen": 0, "entropy_mean": 0.0}

        # Подсчёт частот
        self.counts[:] = 0
        for s, a in zip(states, actions):
            if 0 <= s < STATE_SIZE and 0 <= a < N_ACTIONS:
                self.counts[s, a] += 1

        # Нормализация с Лапласовским сглаживанием
        smoothed = self.counts + self.laplace_alpha
        row_sums = smoothed.sum(axis=1, keepdims=True)
        self.policy = smoothed / row_sums
        self._trained = True

        # Метрики
        predicted = self.policy[states].argmax(axis=1)
        accuracy = float((predicted == actions).mean())
        n_states_seen = int((self.counts.sum(axis=1) > 0).sum())
        entropy = -np.sum(self.policy * np.log(self.policy + 1e-12), axis=1)
        entropy_mean = float(entropy[self.counts.sum(axis=1) > 0].mean()) \
            if n_states_seen > 0 else 0.0

        return {
            "accuracy": accuracy,
            "n_states_seen": n_states_seen,
            "entropy_mean": entropy_mean,
            "n_samples": len(states),
        }

    def select_action(self, state: RLState,
                      mask: Optional[np.ndarray] = None) -> int:
        """Выбрать действие из обученной политики (greedy)."""
        s_idx = state.to_index()
        probs = self.policy[s_idx].copy()
        if mask is not None:
            probs[~mask] = 0.0
            s = probs.sum()
            if s > 0:
                probs /= s
        return int(np.argmax(probs))

    def action_probs(self, state: RLState,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Вернуть вероятности всех действий для состояния."""
        s_idx = state.to_index()
        probs = self.policy[s_idx].copy()
        if mask is not None:
            probs[~mask] = 0.0
            s = probs.sum()
            if s > 0:
                probs /= s
        return probs

    def save(self, path: str = _BC_MODEL_PATH) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, counts=self.counts, policy=self.policy)
        return path

    def load(self, path: str = _BC_MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.counts = data["counts"]
        self.policy = data["policy"]
        self._trained = True
        return True

    @property
    def trained(self) -> bool:
        return self._trained

    def accuracy_by_phase(self, dataset: BCDataset) -> Dict[str, float]:
        """Точность предсказания по фазам пожара."""
        phase_correct: Dict[str, int] = {}
        phase_total:   Dict[str, int] = {}
        for sample in dataset.samples:
            s, a, ph = sample["s"], sample["a"], sample.get("phase", "?")
            if 0 <= s < STATE_SIZE:
                pred = int(np.argmax(self.policy[s]))
                phase_total[ph] = phase_total.get(ph, 0) + 1
                if pred == a:
                    phase_correct[ph] = phase_correct.get(ph, 0) + 1
        return {ph: phase_correct.get(ph, 0) / max(phase_total[ph], 1)
                for ph in sorted(phase_total)}
