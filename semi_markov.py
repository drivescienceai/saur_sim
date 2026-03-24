"""Semi-Markov chain for fire incident phase transitions.

The chain drives the temporal evolution of fire phases:
NORMAL -> S1 -> S2 -> S3 -> S4 -> S5 -> RESOLVED -> NORMAL (loop)

Each state has:
  - A row in the transition matrix P defining probabilities to next states
  - A sojourn time distribution H_i(t) (Weibull) for time spent in state i

The semi-Markov kernel:
  Q(i, j, t) = P(i,j) * F_i(t)
where F_i(t) is the CDF of sojourn time in state i.
"""
import numpy as np
from scipy.stats import weibull_min
from scipy.special import gamma as gamma_func
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from .state_space import FirePhase


@dataclass
class SojournDistribution:
    """Weibull sojourn time distribution for a phase."""
    phase: FirePhase
    shape: float      # Weibull shape parameter k
    scale: float      # Weibull scale parameter lambda (minutes)

    def sample(self, rng: np.random.RandomState) -> float:
        """Sample sojourn time in minutes."""
        return float(weibull_min.rvs(self.shape, scale=self.scale, random_state=rng))

    def mean(self) -> float:
        return self.scale * gamma_func(1 + 1 / self.shape)

    def pdf(self, t: np.ndarray) -> np.ndarray:
        return weibull_min.pdf(t, self.shape, scale=self.scale)


# Sojourn time parameters (shape, scale in minutes) per phase
# Based on typical fire-rescue operation statistics
SOJOURN_PARAMS = {
    FirePhase.NORMAL:   SojournDistribution(FirePhase.NORMAL,   shape=2.0, scale=480.0),  # ~7 hours between incidents
    FirePhase.S1:       SojournDistribution(FirePhase.S1,       shape=1.5, scale=12.0),   # 8-15 min detection
    FirePhase.S2:       SojournDistribution(FirePhase.S2,       shape=2.0, scale=25.0),   # 15-40 min first attack
    FirePhase.S3:       SojournDistribution(FirePhase.S3,       shape=2.5, scale=75.0),   # 45-120 min active
    FirePhase.S4:       SojournDistribution(FirePhase.S4,       shape=2.0, scale=50.0),   # 30-80 min knockdown
    FirePhase.S5:       SojournDistribution(FirePhase.S5,       shape=1.5, scale=90.0),   # 45-180 min overhaul
    FirePhase.RESOLVED: SojournDistribution(FirePhase.RESOLVED, shape=2.0, scale=30.0),   # 20-40 min before reset
}

# Transition probability matrix P[from][to]
# Rows = current state, cols = next state
# Order: NORMAL, S1, S2, S3, S4, S5, RESOLVED
_P = np.array([
    # NOR   S1    S2    S3    S4    S5    RES
    [0.00, 0.03, 0.00, 0.00, 0.00, 0.00, 0.97],  # NORMAL -> S1 (incident) or RESOLVED (no incident)
    [0.00, 0.00, 0.70, 0.15, 0.00, 0.00, 0.15],  # S1 -> S2 (escalate) or S3 (fast spread) or resolve
    [0.00, 0.00, 0.00, 0.75, 0.10, 0.00, 0.15],  # S2 -> S3 or S4 (contain early) or resolve
    [0.00, 0.00, 0.00, 0.00, 0.80, 0.05, 0.15],  # S3 -> S4 or S5 (direct) or resolve
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.85, 0.15],  # S4 -> S5
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # S5 -> RESOLVED
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # RESOLVED -> NORMAL
], dtype=float)

PHASES = list(FirePhase)  # ordered list


class SemiMarkovChain:
    """Semi-Markov chain for fire incident phase evolution.

    Attributes
    ----------
    P : ndarray (7x7)
        Transition probability matrix.
    sojourn : dict
        Sojourn distribution per phase.
    current_phase : FirePhase
        Current state.
    t_entered : float
        Simulation time when current phase was entered.
    history : list of (t_enter, t_exit, phase)
        Full phase history for analysis.
    """

    def __init__(self, seed: Optional[int] = None,
                 P: Optional[np.ndarray] = None,
                 sojourn: Optional[dict] = None):
        self.P = P if P is not None else _P.copy()
        self.sojourn = sojourn if sojourn is not None else dict(SOJOURN_PARAMS)
        self._rng = np.random.RandomState(seed)
        self.current_phase = FirePhase.NORMAL
        self.t_entered = 0.0
        self.t_exit = self._sample_sojourn(FirePhase.NORMAL)
        self.history: List[Tuple[float, float, FirePhase]] = []
        self._cumulative_time: float = 0.0

    def _sample_sojourn(self, phase: FirePhase) -> float:
        dist = self.sojourn.get(phase)
        if dist is None:
            return 0.0
        return max(0.1, dist.sample(self._rng))

    def _next_phase(self, phase: FirePhase) -> FirePhase:
        idx = PHASES.index(phase)
        row = self.P[idx]
        return PHASES[self._rng.choice(len(PHASES), p=row)]

    def step(self, t_now: float) -> Tuple[FirePhase, float, bool]:
        """Advance chain if sojourn time has expired.

        Returns
        -------
        (current_phase, time_remaining_in_phase, transitioned)
        """
        if t_now < self.t_exit:
            return self.current_phase, self.t_exit - t_now, False

        # sojourn expired -> transition
        self.history.append((self.t_entered, self.t_exit, self.current_phase))
        next_phase = self._next_phase(self.current_phase)
        self.t_entered = self.t_exit
        self.current_phase = next_phase
        self.t_exit = self.t_entered + self._sample_sojourn(next_phase)
        return self.current_phase, self.t_exit - t_now, True

    def force_transition(self, t_now: float, target: FirePhase) -> None:
        """Force immediate transition to target phase (e.g. large fire detected)."""
        self.history.append((self.t_entered, t_now, self.current_phase))
        self.t_entered = t_now
        self.current_phase = target
        self.t_exit = t_now + self._sample_sojourn(target)

    @property
    def stationary_distribution(self) -> Dict[str, float]:
        """Compute stationary distribution pi of the embedded Markov chain."""
        # Solve pi P = pi, pi*1 = 1
        n = len(PHASES)
        A = (self.P.T - np.eye(n))
        A[-1] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b)
            pi = np.clip(pi, 0, None)
            pi /= pi.sum()
        except np.linalg.LinAlgError:
            pi = np.ones(n) / n
        return {PHASES[i].name: float(pi[i]) for i in range(n)}

    @property
    def mean_sojourn_times(self) -> Dict[str, float]:
        return {ph.name: self.sojourn[ph].mean() for ph in PHASES}

    def phase_occupancy(self) -> Dict[str, float]:
        """Fraction of time spent in each phase (semi-Markov stationary)."""
        pi = self.stationary_distribution
        means = self.mean_sojourn_times
        # pi_i * E[T_i] / sum(pi_j * E[T_j])
        num = {ph.name: pi[ph.name] * means[ph.name] for ph in PHASES}
        total = sum(num.values())
        if total <= 0:
            return {ph.name: 1 / len(PHASES) for ph in PHASES}
        return {k: v / total for k, v in num.items()}

    def summary(self) -> str:
        occ = self.phase_occupancy()
        lines = ["=== Semi-Markov Chain Summary ==="]
        lines.append(f"Current phase: {self.current_phase.name}")
        lines.append(f"Time in phase: {self.t_exit - self.t_entered:.1f} min (remaining)")
        lines.append("\nPhase occupancy (steady-state):")
        for ph, frac in occ.items():
            mean_t = self.mean_sojourn_times[ph]
            lines.append(f"  {ph:12s}: {frac * 100:5.1f}%  (mean sojourn {mean_t:.1f} min)")
        return "\n".join(lines)
