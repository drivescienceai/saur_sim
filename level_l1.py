"""L1: Sensory-Situational Layer.

Functions: f_sense, f_fuse, f_detect, f_validate
Inputs: sensors (IR, smoke, gas, CCTV, GLONASS, weather)
Outputs: aggregated state vector s(tau) -> L2/L3; alert events
Autonomy: alpha1 = 0 (fully dependent on sensors)
Metric: mu1 = P_COP >= 0.90
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .state_space import SituationState, FirePhase


@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: str      # "ir" | "smoke" | "gas" | "video" | "gps" | "weather"
    x_m: float
    y_m: float
    value: float          # normalized 0..1
    confidence: float     # source reliability weight
    timestamp: float


@dataclass
class COP:
    """Common Operating Picture — the fused situational picture."""
    timestamp: float
    situation: SituationState
    confidence: float       # P_COP
    alerts: List[str] = field(default_factory=list)
    sensor_count: int = 0


class L1SensorLayer:
    """L1 sensor fusion and COP generation.

    Implements multi-sensor fusion with dynamic confidence weighting.
    Detects anomalies (fire onset) and switches to alert mode.
    """

    _THRESHOLDS = {
        "ir":    0.6,   # IR temperature anomaly threshold
        "smoke": 0.5,
        "gas":   0.7,
        "video": 0.55,
    }

    def __init__(self, seed: Optional[int] = None, noise_std: float = 0.05):
        self._rng = np.random.RandomState(seed)
        self.noise_std = noise_std
        self._sensor_weights: dict = {}  # dynamic weight per sensor type
        self._reading_history: List[SensorReading] = []
        self.cop: Optional[COP] = None
        self._alpha = 0.0  # autonomy coefficient (fixed at 0 for L1)

    def _weight(self, sensor_type: str) -> float:
        return self._sensor_weights.get(sensor_type, 1.0)

    def sense(self, t: float, true_situation: SituationState,
              n_sensors: int = 8) -> List[SensorReading]:
        """Generate synthetic sensor readings based on true situation."""
        readings = []
        sensor_types = ["ir", "smoke", "gas", "video", "gps",
                        "gps", "weather", "smoke"]
        for i in range(min(n_sensors, len(sensor_types))):
            stype = sensor_types[i]
            # True signal strength increases with fire phase
            true_signal = (true_situation.phase.value / 6.0
                           if true_situation.phase != FirePhase.NORMAL else 0.0)
            noisy = true_signal + self._rng.normal(0, self.noise_std)
            noisy = float(np.clip(noisy, 0.0, 1.0))
            r = SensorReading(
                sensor_id=f"S{i + 1:02d}",
                sensor_type=stype,
                x_m=self._rng.uniform(0, 600),
                y_m=self._rng.uniform(0, 600),
                value=noisy,
                confidence=self._weight(stype),
                timestamp=t,
            )
            readings.append(r)
        self._reading_history.extend(readings)
        return readings

    def fuse(self, readings: List[SensorReading], t: float) -> SituationState:
        """Multi-sensor fusion -> estimated situation state."""
        if not readings:
            return SituationState(timestamp=t)

        # Weighted average of sensor values
        total_w = sum(r.confidence for r in readings)
        weighted_signal = sum(r.value * r.confidence for r in readings) / max(total_w, 1e-9)

        # Map signal to phase estimate
        if weighted_signal < 0.05:
            est_phase = FirePhase.NORMAL
        elif weighted_signal < 0.20:
            est_phase = FirePhase.S1
        elif weighted_signal < 0.40:
            est_phase = FirePhase.S2
        elif weighted_signal < 0.60:
            est_phase = FirePhase.S3
        elif weighted_signal < 0.80:
            est_phase = FirePhase.S4
        else:
            est_phase = FirePhase.S5

        fire_area = max(0.0, (weighted_signal ** 2) * 5000.0)

        return SituationState(
            phase=est_phase,
            fire_area_m2=fire_area,
            fire_spread_rate=weighted_signal * 2.0,
            timestamp=t,
        )

    def detect(self, readings: List[SensorReading]) -> bool:
        """Anomaly detection: returns True if fire event detected."""
        for r in readings:
            threshold = self._THRESHOLDS.get(r.sensor_type, 0.6)
            if r.value >= threshold:
                return True
        return False

    def validate(self, readings: List[SensorReading]) -> float:
        """Estimate P_COP from reading consistency."""
        if not readings:
            return 0.0
        values = [r.value for r in readings]
        cv = np.std(values) / (np.mean(values) + 1e-9)
        # High consistency -> high confidence
        confidence = max(0.0, 1.0 - cv)
        return float(np.clip(confidence, 0.0, 1.0))

    def update(self, t: float, true_situation: SituationState) -> COP:
        """Full L1 cycle: sense -> fuse -> detect -> validate."""
        readings = self.sense(t, true_situation)
        estimated = self.fuse(readings, t)
        fire_detected = self.detect(readings)
        p_cop = self.validate(readings)

        alerts = []
        if fire_detected and estimated.phase != FirePhase.NORMAL:
            alerts.append(f"ALERT: Fire detected, estimated phase {estimated.phase.name}")

        self.cop = COP(
            timestamp=t,
            situation=estimated,
            confidence=p_cop,
            alerts=alerts,
            sensor_count=len(readings),
        )
        return self.cop

    def adapt_weights(self, feedback: dict) -> None:
        """Dynamically adjust sensor weights based on accuracy feedback."""
        for stype, accuracy in feedback.items():
            current = self._sensor_weights.get(stype, 1.0)
            self._sensor_weights[stype] = 0.9 * current + 0.1 * accuracy
