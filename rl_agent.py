"""Hierarchical Q-learning RL agent for САУР ПСП.

Action space (15 actions, 3 levels) based on:
  Григорьев А.Н. (2012), Данилов М.М. (2015), Денисов А.Н. (2001),
  Вытовтов А.В. (2018).

Hierarchical structure
----------------------
  Strategic (S1-S5): global fire-fighting direction, freq ~10 ticks
  Tactical  (T1-T4): resource organisation,          freq ~5 ticks
  Operational(O1-O6): direct unit commands,           freq every tick

State space (discretized, shared):
  phase_idx       : 0..6 (FirePhase ordinal)
  resource_level  : 0=LOW(<40%), 1=MEDIUM(40-70%), 2=HIGH(>70%)
  fire_severity   : 0=NONE..4=CRITICAL
  response_quality: 0=POOR(<0.6), 1=ADEQUATE(0.6-0.85), 2=GOOD(>0.85)
  -> 7 x 3 x 5 x 3 = 315 states

Reward:
  R(t) = -W_L1*L1_norm - W_CASUALTY*casualty_rate
         - W_COST*cost(action) + W_L7*L7
"""
from __future__ import annotations
import numpy as np
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, List


# ---------------------------------------------------------------------------
# State space constants
# ---------------------------------------------------------------------------
N_PHASES    = 7
N_RESOURCES = 3
N_SEVERITY  = 5
N_QUALITY   = 3
STATE_SIZE  = N_PHASES * N_RESOURCES * N_SEVERITY * N_QUALITY   # 315

# ---------------------------------------------------------------------------
# Action hierarchy
# ---------------------------------------------------------------------------

class ActionLevel(Enum):
    STRATEGIC   = "strategic"    # S1-S5 : выбор решающего направления
    TACTICAL    = "tactical"     # T1-T4 : организация тушения
    OPERATIONAL = "operational"  # O1-O6 : конкретные команды


@dataclass(frozen=True)
class RTAction:
    """Single РТП action descriptor (from PDF hierarchical classification)."""
    code:    str          # S1, T2, O3 …
    level:   ActionLevel
    name_ru: str          # Russian name as in the source documents
    params:  tuple        # parameter names (for logging / GUI)
    cost:    float        # action execution cost ∈ [0,1] for reward penalty


# ---------------------------------------------------------------------------
# Full 15-action space  (order is the action index 0..14)
# ---------------------------------------------------------------------------
ACTION_SPACE: List[RTAction] = [
    # ── Strategic ────────────────────────────────────────────────────────────
    RTAction("S1", ActionLevel.STRATEGIC,   "Спасение людей",          ("приоритет",),                0.30),
    RTAction("S2", ActionLevel.STRATEGIC,   "Защита соседних объектов",("объект_id",),                0.20),
    RTAction("S3", ActionLevel.STRATEGIC,   "Локализация пожара",      ("направление",),              0.15),
    RTAction("S4", ActionLevel.STRATEGIC,   "Ликвидация горения",      ("интенсивность",),            0.25),
    RTAction("S5", ActionLevel.STRATEGIC,   "Предотвращение ЧС",       ("тип_угрозы",),               0.35),
    # ── Tactical ─────────────────────────────────────────────────────────────
    RTAction("T1", ActionLevel.TACTICAL,    "Создать УТ/сектор",       ("позиция", "ресурсы"),        0.20),
    RTAction("T2", ActionLevel.TACTICAL,    "Перераспределить силы",   ("откуда", "куда", "кол-во"),  0.10),
    RTAction("T3", ActionLevel.TACTICAL,    "Вызвать подкрепление",    ("тип", "количество"),         0.40),
    RTAction("T4", ActionLevel.TACTICAL,    "Изменить схему тушения",  ("новая_схема",),              0.05),
    # ── Operational ──────────────────────────────────────────────────────────
    RTAction("O1", ActionLevel.OPERATIONAL, "Подать ствол",            ("тип", "позиция"),            0.05),
    RTAction("O2", ActionLevel.OPERATIONAL, "Охлаждать объект",        ("объект_id", "интенсивность"),0.08),
    RTAction("O3", ActionLevel.OPERATIONAL, "Пенная атака",            ("резервуар_id",),             0.15),
    RTAction("O4", ActionLevel.OPERATIONAL, "Провести разведку",       ("зона", "тип"),               0.03),
    RTAction("O5", ActionLevel.OPERATIONAL, "Эвакуация",               ("маршрут", "приоритет"),      0.20),
    RTAction("O6", ActionLevel.OPERATIONAL, "Сигнал отхода",           ("зона", "радиус"),            0.01),
]

N_ACTIONS   = len(ACTION_SPACE)                          # 15
ACTION_NAMES = [a.name_ru for a in ACTION_SPACE]
ACTION_COST  = [a.cost    for a in ACTION_SPACE]

# Convenience index maps
_CODE_TO_IDX: Dict[str, int] = {a.code: i for i, a in enumerate(ACTION_SPACE)}
STRATEGIC_IDX  = [i for i, a in enumerate(ACTION_SPACE)
                  if a.level == ActionLevel.STRATEGIC]   # [0,1,2,3,4]
TACTICAL_IDX   = [i for i, a in enumerate(ACTION_SPACE)
                  if a.level == ActionLevel.TACTICAL]    # [5,6,7,8]
OPERATIONAL_IDX = [i for i, a in enumerate(ACTION_SPACE)
                   if a.level == ActionLevel.OPERATIONAL]# [9,10,11,12,13,14]

# ---------------------------------------------------------------------------
# Action masking
# ---------------------------------------------------------------------------
#   Mask matrix shape: (N_PHASES, N_ACTIONS) — base availability per phase
#   Additional masking applied for resource_level and situational flags.
#
#   Phase ordinals: NORMAL=0, S1=1, S2=2, S3=3, S4=4, S5=5, RESOLVED=6
#
#         S1 S2 S3 S4 S5  T1 T2 T3 T4  O1 O2 O3 O4 O5 O6
_PHASE_MASK = np.array([
    # NORMAL (0): only recon + adjust scheme
    [0, 0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 1, 0, 0],
    # S1 detection (1): save people, localize, recon, deploy, evac
    [1, 1, 1, 0, 0,  1, 0, 1, 1,  1, 0, 0, 1, 1, 0],
    # S2 first attack (2): most actions
    [1, 1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 1, 1, 1, 0],
    # S3 active fire (3): all actions
    [1, 1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, 1, 1],
    # S4 knockdown (4): no new УТ, no evacuation
    [0, 1, 1, 1, 1,  0, 1, 1, 1,  1, 1, 1, 1, 0, 1],
    # S5 overhaul (5): protect, cool, recon only
    [0, 1, 1, 1, 0,  0, 1, 0, 1,  1, 1, 0, 1, 0, 0],
    # RESOLVED (6): adjust scheme + recon only
    [0, 0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 1, 0, 0],
], dtype=bool)


def get_action_mask(phase_idx: int,
                    resource_level: int,
                    has_people: bool = False,
                    has_foam: bool = True) -> np.ndarray:
    """Return boolean mask of shape (N_ACTIONS,): True = action available.

    Parameters
    ----------
    phase_idx       : fire phase ordinal 0..6
    resource_level  : 0=LOW, 1=MEDIUM, 2=HIGH
    has_people      : whether people are present / at risk
    has_foam        : whether foam resources are available (for O3)
    """
    mask = _PHASE_MASK[phase_idx].copy()

    # S1 (save people) and O5 (evacuation) only when people at risk
    if not has_people:
        mask[_CODE_TO_IDX["S1"]] = False
        mask[_CODE_TO_IDX["O5"]] = False

    # O3 (пенная атака) requires foam
    if not has_foam:
        mask[_CODE_TO_IDX["O3"]] = False

    # T3 (call reinforcement) only when resources are LOW or MEDIUM
    if resource_level == 2:    # already HIGH — no need for more
        mask[_CODE_TO_IDX["T3"]] = False

    # S5 (prevent secondary effects) only when severity is HIGH/CRITICAL
    # expressed here as: only during S3/S4 phases (already handled by phase mask)

    # Ensure at least one action is available (fallback: O4 recon)
    if not mask.any():
        mask[_CODE_TO_IDX["O4"]] = True

    return mask


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------
W_L1       = 0.35
W_CASUALTY = 0.25
W_COST     = 0.10
W_L7       = 0.30


def compute_reward(L1_norm: float, casualties: int,
                   action: int, L7: float) -> float:
    """Per-step reward.  Signature unchanged for backwards compatibility."""
    cost = ACTION_COST[action] if 0 <= action < N_ACTIONS else 0.0
    return (
        -W_L1      * L1_norm
        -W_CASUALTY * min(1.0, casualties / 10.0)
        -W_COST    * cost
        +W_L7      * L7
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class RLState:
    phase_idx:       int   # 0..6
    resource_level:  int   # 0..2
    fire_severity:   int   # 0..4
    response_quality:int   # 0..2

    def to_index(self) -> int:
        return (self.phase_idx      * N_RESOURCES * N_SEVERITY * N_QUALITY
                + self.resource_level * N_SEVERITY * N_QUALITY
                + self.fire_severity  * N_QUALITY
                + self.response_quality)

    @staticmethod
    def from_metrics(phase_idx: int, avail_frac: float,
                     fire_area_norm: float, L7: float) -> "RLState":
        res_lvl  = 0 if avail_frac < 0.4 else (1 if avail_frac < 0.7 else 2)
        severity = min(4, int(fire_area_norm * 5))
        quality  = 0 if L7 < 0.6 else (1 if L7 < 0.85 else 2)
        return RLState(phase_idx, res_lvl, severity, quality)


# ---------------------------------------------------------------------------
# Flat Q-learning agent (extended to 15 actions)
# ---------------------------------------------------------------------------

class QLearningAgent:
    """Tabular Q-learning agent.  Q-table shape: (STATE_SIZE, N_ACTIONS)."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995, seed: Optional[int] = None):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng          = np.random.RandomState(seed)
        self.Q             = np.zeros((STATE_SIZE, N_ACTIONS), dtype=float)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int]   = []
        self._step_count = 0

    def select_action(self, state: RLState, training: bool = True,
                      mask: Optional[np.ndarray] = None) -> int:
        """ε-greedy selection with optional action mask."""
        valid = np.arange(N_ACTIONS)
        if mask is not None:
            valid = np.where(mask)[0]
            if len(valid) == 0:
                valid = np.arange(N_ACTIONS)

        if training and self._rng.random() < self.epsilon:
            return int(self._rng.choice(valid))

        q_row = self.Q[state.to_index()].copy()
        if mask is not None:
            q_masked = np.full(N_ACTIONS, -np.inf)
            q_masked[valid] = q_row[valid]
        else:
            q_masked = q_row
        return int(np.argmax(q_masked))

    def update(self, state: RLState, action: int, reward: float,
               next_state: RLState, done: bool) -> float:
        s, s_ = state.to_index(), next_state.to_index()
        target   = reward + (0.0 if done else self.gamma * np.max(self.Q[s_]))
        td_error = target - self.Q[s, action]
        self.Q[s, action] += self.alpha * td_error
        self._step_count += 1
        return float(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def policy(self) -> Dict[int, int]:
        return {s: int(np.argmax(self.Q[s])) for s in range(STATE_SIZE)}

    def value_function(self) -> np.ndarray:
        return np.max(self.Q, axis=1)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"Q": self.Q.tolist(), "epsilon": self.epsilon,
                       "step_count": self._step_count}, f)

    @classmethod
    def load(cls, path: str, **kwargs) -> "QLearningAgent":
        with open(path) as f:
            data = json.load(f)
        agent = cls(**kwargs)
        agent.Q            = np.array(data["Q"])
        agent.epsilon      = data["epsilon"]
        agent._step_count  = data["step_count"]
        return agent


# ---------------------------------------------------------------------------
# Hierarchical RL agent  (HRL — separate Q-tables per level)
# ---------------------------------------------------------------------------

class HierarchicalRLAgent:
    """Three-level hierarchical Q-learning agent.

    Architecture
    ------------
    - Meta-controller  : Q_meta[state] -> level (0=strategic,1=tactical,2=operational)
    - Strategic agent  : Q_s[state]    -> action ∈ {0..4}  (S1-S5)
    - Tactical agent   : Q_t[state]    -> action ∈ {0..3}  (T1-T4)
    - Operational agent: Q_o[state]    -> action ∈ {0..5}  (O1-O6)

    Temporal abstraction
    --------------------
    - Strategic decisions fire every `freq_strategic` ticks
    - Tactical  decisions fire every `freq_tactical`  ticks
    - Operational every tick (freq=1)
    """

    N_LEVELS    = 3
    N_STR       = len(STRATEGIC_IDX)    # 5
    N_TAC       = len(TACTICAL_IDX)     # 4
    N_OPE       = len(OPERATIONAL_IDX)  # 6

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995,
                 freq_strategic: int = 10,
                 freq_tactical:  int = 5,
                 seed: Optional[int] = None):
        kw = dict(alpha=alpha, gamma=gamma,
                  epsilon_start=epsilon_start,
                  epsilon_min=epsilon_min,
                  epsilon_decay=epsilon_decay)
        rng = np.random.RandomState(seed)
        seeds = rng.randint(0, 99999, size=4).tolist()

        # Meta-controller selects which level to activate
        self._meta = QLearningAgent(seed=seeds[0], **kw)
        self._meta.Q = np.zeros((STATE_SIZE, self.N_LEVELS))

        # Level sub-agents (their Q-tables cover global action indices)
        self._str_agent = QLearningAgent(seed=seeds[1], **kw)
        self._str_agent.Q = np.zeros((STATE_SIZE, self.N_STR))
        self._tac_agent = QLearningAgent(seed=seeds[2], **kw)
        self._tac_agent.Q = np.zeros((STATE_SIZE, self.N_TAC))
        self._ope_agent = QLearningAgent(seed=seeds[3], **kw)
        self._ope_agent.Q = np.zeros((STATE_SIZE, self.N_OPE))

        self.freq_strategic = freq_strategic
        self.freq_tactical  = freq_tactical
        self._tick = 1   # start at 1 so tick % freq is meaningful from the first step

        self._last_level:  Optional[int] = None
        self._last_action: Optional[int] = None
        self._last_state:  Optional[RLState] = None
        self.episode_rewards: List[float] = []
        self._cumulative_reward: float = 0.0

    @property
    def epsilon(self) -> float:
        return self._ope_agent.epsilon

    def select_action(self, state: RLState, training: bool = True,
                      mask: Optional[np.ndarray] = None) -> int:
        """Select global action index (0..14) using temporal hierarchy."""
        tick = self._tick
        self._tick += 1

        # Determine active level based on temporal schedule
        if tick % self.freq_strategic == 0:
            level = 0  # strategic tick
        elif tick % self.freq_tactical == 0:
            level = 1  # tactical tick
        else:
            level = 2  # operational (every tick)

        # Sub-agent picks within-level action
        if level == 0:
            sub_idx = self._select_within(self._str_agent, state,
                                          STRATEGIC_IDX, mask, training)
            global_idx = STRATEGIC_IDX[sub_idx]
        elif level == 1:
            sub_idx = self._select_within(self._tac_agent, state,
                                          TACTICAL_IDX, mask, training)
            global_idx = TACTICAL_IDX[sub_idx]
        else:
            sub_idx = self._select_within(self._ope_agent, state,
                                          OPERATIONAL_IDX, mask, training)
            global_idx = OPERATIONAL_IDX[sub_idx]

        self._last_level  = level
        self._last_action = global_idx
        self._last_state  = state
        return global_idx

    def _select_within(self, sub_agent: QLearningAgent, state: RLState,
                       global_indices: List[int],
                       global_mask: Optional[np.ndarray],
                       training: bool) -> int:
        """Select within-level action with mask projected to sub-space."""
        n_local = len(global_indices)
        if global_mask is not None:
            local_mask = np.array([global_mask[gi] for gi in global_indices], dtype=bool)
        else:
            local_mask = np.ones(n_local, dtype=bool)

        s = state.to_index()
        valid = np.where(local_mask)[0]
        if len(valid) == 0:
            valid = np.arange(n_local)

        if training and sub_agent._rng.random() < sub_agent.epsilon:
            return int(sub_agent._rng.choice(valid))

        q_row = sub_agent.Q[s].copy()
        q_masked = np.full(n_local, -np.inf)
        q_masked[valid] = q_row[valid]
        return int(np.argmax(q_masked))

    def update(self, state: RLState, action: int, reward: float,
               next_state: RLState, done: bool) -> float:
        """Update the sub-agent that made the last decision."""
        if self._last_level is None:
            return 0.0

        level = self._last_level
        if level == 0:
            sub_idx = STRATEGIC_IDX.index(action)
            agent   = self._str_agent
            n_local = self.N_STR
        elif level == 1:
            sub_idx = TACTICAL_IDX.index(action)
            agent   = self._tac_agent
            n_local = self.N_TAC
        else:
            sub_idx = OPERATIONAL_IDX.index(action)
            agent   = self._ope_agent
            n_local = self.N_OPE

        s, s_ = state.to_index(), next_state.to_index()
        target   = reward + (0.0 if done else agent.gamma * np.max(agent.Q[s_]))
        td_error = target - agent.Q[s, sub_idx]
        agent.Q[s, sub_idx] += agent.alpha * td_error
        agent._step_count += 1
        self._cumulative_reward += reward
        return float(td_error)

    def decay_epsilon(self) -> None:
        for ag in (self._str_agent, self._tac_agent, self._ope_agent):
            ag.decay_epsilon()

    def save(self, path: str) -> None:
        data = {
            "str_Q":  self._str_agent.Q.tolist(),
            "tac_Q":  self._tac_agent.Q.tolist(),
            "ope_Q":  self._ope_agent.Q.tolist(),
            "epsilon": self.epsilon,
            "tick":    self._tick,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str, **kwargs) -> "HierarchicalRLAgent":
        with open(path) as f:
            data = json.load(f)
        agent = cls(**kwargs)
        agent._str_agent.Q = np.array(data["str_Q"])
        agent._tac_agent.Q = np.array(data["tac_Q"])
        agent._ope_agent.Q = np.array(data["ope_Q"])
        agent._tick = data.get("tick", 1)
        return agent
