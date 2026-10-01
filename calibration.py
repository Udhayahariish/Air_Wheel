"""Automatic centre calibration."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from config import Settings
from geometry import circular_mean_deg, wrap_angle
from models import CalibrationData, HandState
from steering_engine import measure_hands

TARGET_FRAMES = 45          # 30-60 frames are collected
MAX_ANGLE_SPREAD_DEG = 6.0  # reject calibrations where the hands moved too much
GAP_RESET_SECONDS = 0.5     # hands missing longer than this restart calibration

_Sample = Tuple[float, float, float, float, float]   # angle, axis, distance, cx, cy


class Calibrator:
    """Collects frames while both hands are held at the centre."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.result: Optional[CalibrationData] = None
        self.progress = 0.0
        self.status = ""
        self._samples: List[_Sample] = []
        self._t0: Optional[float] = None
        self._last_seen: Optional[float] = None
        self.start()

    def start(self) -> None:
        """Begin (or restart) a calibration run."""
        self._samples = []
        self._t0 = None
        self._last_seen = None
        self.progress = 0.0
        self.result = None
        self.status = "HOLD HANDS AT CENTER..."

    def update(self, left: Optional[HandState], right: Optional[HandState],
               aspect: float, now: float) -> bool:
        """Feed one frame. Returns True once calibration has completed."""
        if left is None or right is None:
            if self._last_seen is not None and now - self._last_seen > GAP_RESET_SECONDS:
                self.start()
                self.status = "SHOW BOTH HANDS"
            return False
        self._last_seen = now
        if self._t0 is None:
            self._t0 = now
        m = measure_hands(left, right, aspect)
        self._samples.append((m.raw_angle, m.hand_axis, m.distance, m.center[0], m.center[1]))

        frame_prog = len(self._samples) / TARGET_FRAMES
        time_prog = (now - self._t0) / max(self.settings.calibration_seconds, 0.1)
        self.progress = min(1.0, frame_prog, time_prog)
        self.status = "HOLD HANDS AT CENTER..."
        if self.progress < 1.0:
            return False

        angles = [s[0] for s in self._samples]
        mean_angle = circular_mean_deg(angles)
        spread = math.sqrt(sum(wrap_angle(a - mean_angle) ** 2 for a in angles) / len(angles))
        if spread > MAX_ANGLE_SPREAD_DEG:
            self.start()
            self.status = "HOLD STEADY..."
            return False

        n = len(self._samples)
        self.result = CalibrationData(
            center_angle=mean_angle,
            center_distance=sum(s[2] for s in self._samples) / n,
            center_position=(sum(s[3] for s in self._samples) / n,
                             sum(s[4] for s in self._samples) / n),
            center_rotation=circular_mean_deg(s[1] for s in self._samples),
            valid=True,
        )
        self.status = "CALIBRATION COMPLETE"
        return True
