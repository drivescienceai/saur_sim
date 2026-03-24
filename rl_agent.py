"""Q-learning RL agent for L3 operational decision making.

State space (discretized):
  - phase_idx:       0..6 (FirePhase ordinal)
  - resource_level:  0=LOW(<40%), 1=MEDIUM(40-70%), 2=HIGH(>70%)
  - fire_severity:   0=NONE, 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL
  - response_quality:0=POOR(<0.6), 1=ADEQUATE(0.6-0.85), 2=GOOD(>0.85)

State space size: 7 x 3 x 5 x 3 = 315

Actions:
  0: HOLD         - maintain current allocation
  1: REGROUP      - redistribute available units locally
  2: REQUEST_RES  - request reserves from L4
  3: MOBILIZE     - full mobilization (highest cost)
  4: SCALE_DOWN   - reduce allocation (recovery phase)

Reward function:
  R(t) = -alpha*L1_norm - beta*casualty_flag - gamma*cost_action + delta*L7
"""
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

N_PHASES    = 7
N_RESOURCES = 3
N_SEVERITY  = 5
N_QUALITY   = 3
N_ACTIONS   = 5

STATE_SIZE  = N_PHASES * N_RESOURCES * N_SEVERITY * N_QUALITY

ACTION_NAMES = ["HOLD", "REGROUP", "REQUEST_RESERVES", "FULL_MOBILIZE", "SCALE_DOWN"]
ACTION_COST  = [0.0,    0.05,      0.15,               0.40,            0.02]

# Reward weights
W_L1      = 0.35   # penalize information staleness
W_CASUALTY = 0.25  # penalize casualties
W_COST    = 0.10   # penalize resource cost
W_L7      = 0.30   # reward monitoring reliability


@dataclass
class RLState:
    phase_idx: int        # 0..6
    resource_level: int   # 0..2
    fire_severity: int    # 0..4
    response_quality: int # 0..2

    def to_index(self) -> int:
        return (self.phase_idx * N_RESOURCES * N_SEVERITY * N_QUALITY
                + self.resource_level * N_SEVERITY * N_QUALITY
                + self.fire_severity * N_QUALITY
                + self.response_quality)

    @staticmethod
    def from_metrics(phase_idx: int, avail_frac: float,
                     fire_area_norm: float, L7: float) -> "RLState":
        res_lvl = 0 if avail_frac < 0.4 else (1 if avail_frac < 0.7 else 2)
        severity = min(4, int(fire_area_norm * 5))
        quality  = 0 if L7 < 0.6 else (1 if L7 < 0.85 else 2)
        return RLState(phase_idx, res_lvl, severity, quality)


class QLearningAgent:
    """Tabular Q-learning agent for L3 adaptive resource allocation.

    Parameters
    ----------
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon_start : float
        Initial exploration rate.
    epsilon_min : float
        Minimum exploration rate.
    epsilon_decay : float
        Multiplicative decay per episode.
    seed : int or None
        RNG seed.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995, seed: Optional[int] = None):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.RandomState(seed)

        # Q-table: shape (STATE_SIZE, N_ACTIONS)
        self.Q = np.zeros((STATE_SIZE, N_ACTIONS), dtype=float)

        # Training history
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int]   = []
        self._step_count = 0

    def select_action(self, state: RLState, training: bool = True) -> int:
        """epsilon-greedy action selection."""
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.randint(0, N_ACTIONS))
        return int(np.argmax(self.Q[state.to_index()]))

    def update(self, state: RLState, action: int, reward: float,
               next_state: RLState, done: bool) -> float:
        """Q-learning update. Returns TD error."""
        s  = state.to_index()
        s_ = next_state.to_index()
        target = reward + (0 if done else self.gamma * np.max(self.Q[s_]))
        td_error = target - self.Q[s, action]
        self.Q[s, action] += self.alpha * td_error
        self._step_count += 1
        return float(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def policy(self) -> Dict[int, int]:
        """Return greedy policy: state_index -> best_action."""
        return {s: int(np.argmax(self.Q[s])) for s in range(STATE_SIZE)}

    def value_function(self) -> np.ndarray:
        """Return V(s) = max_a Q(s,a)."""
        return np.max(self.Q, axis=1)

    def save(self, path: str) -> None:
        data = {"Q": self.Q.tolist(), "epsilon": self.epsilon,
                "step_count": self._step_count}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str, **kwargs) -> "QLearningAgent":
        with open(path) as f:
            data = json.load(f)
        agent = cls(**kwargs)
        agent.Q = np.array(data["Q"])
        agent.epsilon = data["epsilon"]
        agent._step_count = data["step_count"]
        return agent


def compute_reward(L1_norm: float, casualties: int, action: int, L7: float) -> float:
    """Compute per-step reward for the RL agent."""
    r_L1       = -W_L1 * L1_norm
    r_casualty = -W_CASUALTY * min(1.0, casualties / 10.0)
    r_cost     = -W_COST * ACTION_COST[action]
    r_L7       = W_L7 * L7
    return r_L1 + r_casualty + r_cost + r_L7
