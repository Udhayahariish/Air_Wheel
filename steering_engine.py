"""Continuous steering model: hands -> steering_value in [-1, +1]."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, Optional, Tuple

import numpy as np

from config import Settings
from geometry import circular_mean_deg, clamp, lerp, sign, wrap_angle
from models import CalibrationData, HandState, SteeringState
from smoothing import Hysteresis, SmoothFilter

WRIST_WEIGHT = 0.4          # blend of wrist-vector and palm-vector axis
PALM_WEIGHT = 0.6
VELOCITY_REF_DEG_S = 200.0  # speed at which adaptive gain saturates
STABILITY_WINDOW = 20
JITTER_FULL_SCALE = 0.04    # frame-to-frame std that maps to 0 % stability

# Hand-position limits (distance in fractions of frame width, size in frame heights)
MIN_HAND_DISTANCE = 0.15
MAX_HAND_DISTANCE = 0.80
MAX_HAND_SIZE = 0.30
REGION = (0.04, 0.08, 0.96, 0.92)   # x0, y0, x1, y1 of the steering region


@dataclass
class AxisMeasure:
    """Geometry of the two-hand steering axis for one frame."""
    raw_angle: float      # degrees, direction of the wheel axis (left hand -> right hand)
    hand_axis: float      # degrees, mean orientation of the two hands
    distance: float       # wrist-to-wrist, fraction of frame width
    center: Tuple[float, float]


def measure_hands(left: HandState, right: HandState, aspect: float) -> AxisMeasure:
    """Measure the steering axis from the image-left and image-right hand.

    Combines the wrist-to-wrist vector (A) with the palm-to-palm vector (B).
    ``aspect`` (w/h) makes angles correct on non-square frames.
    """
    ax = (right.wrist[0] - left.wrist[0]) * aspect
    ay = right.wrist[1] - left.wrist[1]
    bx = (right.palm_center[0] - left.palm_center[0]) * aspect
    by = right.palm_center[1] - left.palm_center[1]
    vx = WRIST_WEIGHT * ax + PALM_WEIGHT * bx
    vy = WRIST_WEIGHT * ay + PALM_WEIGHT * by
    return AxisMeasure(
        raw_angle=math.degrees(math.atan2(vy, vx)),
        hand_axis=circular_mean_deg([left.axis_angle, right.axis_angle]),
        distance=math.hypot(ax, ay) / max(aspect, 1e-6),
        center=((left.palm_center[0] + right.palm_center[0]) / 2.0,
                (left.palm_center[1] + right.palm_center[1]) / 2.0),
    )


@dataclass
class PositionReport:
    """Result of the hand-position quality check."""
    good: bool
    message: str


def assess_position(left: Optional[HandState], right: Optional[HandState],
                    aspect: float, min_confidence: float) -> PositionReport:
    """Check whether both hands are in a usable steering posture."""
    if left is None or right is None:
        return PositionReport(False, "SHOW BOTH HANDS")
    if min(left.confidence, right.confidence) < min_confidence:
        return PositionReport(False, "LOW CONFIDENCE")
    dist = measure_hands(left, right, aspect).distance
    if dist < MIN_HAND_DISTANCE:
        return PositionReport(False, "SPREAD HANDS APART")
    if dist > MAX_HAND_DISTANCE:
        return PositionReport(False, "MOVE HANDS CLOSER")
    if max(left.size, right.size) > MAX_HAND_SIZE:
        return PositionReport(False, "MOVE HANDS BACK")
    x0, y0, x1, y1 = REGION
    for hand in (left, right):
        px, py = hand.palm_center
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            return PositionReport(False, "KEEP HANDS IN THE AREA")
    return PositionReport(True, "GOOD")


class PositionMonitor:
    """Adds a grace period so brief posture glitches do not flash warnings."""

    def __init__(self, grace: float = 0.5) -> None:
        self.grace = grace
        self._bad_since: Optional[float] = None
        self.report = PositionReport(False, "SHOW BOTH HANDS")

    def update(self, report: PositionReport, now: float) -> PositionReport:
        if report.good:
            self._bad_since = None
            self.report = report
        else:
            if self._bad_since is None:
                self._bad_since = now
            if now - self._bad_since >= self.grace:
                self.report = report
        return self.report


class SteeringEngine:
    """Turns hand poses into a smooth, continuous steering value."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calibration = CalibrationData()
        self.aspect = 4.0 / 3.0
        self.mirrored = True
        self._filter = SmoothFilter(settings.smoothing, settings.filter_mode)
        self._dir = Hysteresis(settings.steer_start_threshold, settings.steer_release_threshold)
        self._history: Deque[float] = deque(maxlen=STABILITY_WINDOW)
        self._prev_angle: Optional[float] = None
        self._velocity = 0.0
        self._speed_factor = 0.0
        self._in_zone = True
        self._last: Optional[SteeringState] = None
        self._lost_time = 0.0
        self._lost_frames = 0

    # ------------------------------------------------------------ configuration
    def set_aspect(self, aspect: float) -> None:
        """Frame aspect ratio (width / height)."""
        self.aspect = max(aspect, 0.1)

    def calibrate(self, data: CalibrationData) -> None:
        """Store a new calibration and reset all dynamic state."""
        self.calibration = data
        self.reset()

    def reset(self, prime_zero: bool = False) -> None:
        """Clear filters/histories (optionally priming the filter at 0)."""
        self._filter.base_alpha = self.settings.smoothing
        self._filter.mode = self.settings.filter_mode
        self._filter.reset(0.0 if prime_zero else None)
        self._dir.reset()
        self._history.clear()
        self._prev_angle = None
        self._velocity = 0.0
        self._speed_factor = 0.0
        self._in_zone = True
        self._last = None
        self._lost_time = 0.0
        self._lost_frames = 0

    @property
    def in_grace(self) -> bool:
        """True while hands are briefly missing but control is still held."""
        return self._lost_frames > 0 and self._last is not None

    # ------------------------------------------------------------ pipeline steps
    def calculate_raw_angle(self, left: HandState, right: HandState) -> AxisMeasure:
        """Step 1: measure the wheel axis."""
        return measure_hands(left, right, self.aspect)

    def apply_center_offset(self, m: AxisMeasure) -> float:
        """Step 2: angle relative to calibrated centre, wrapped to -180..+180.

        A small share (``rotation_weight``) comes from the hands' own rotation.
        A non-mirrored image reverses the visual sense of rotation, so the sign
        is flipped in that case.
        """
        cal = self.calibration
        rel_axis = wrap_angle(m.raw_angle - cal.center_angle)
        rel_rot = clamp(wrap_angle(m.hand_axis - cal.center_rotation), -90.0, 90.0)
        w = self.settings.rotation_weight
        angle = (1.0 - w) * rel_axis + w * rel_rot
        return angle * (1 if self.mirrored else -1)

    def normalize_angle(self, angle: float) -> float:
        """Step 3: angle / MAX_STEERING_DEG clamped to -1..+1."""
        return clamp(angle / self.settings.max_steering_degrees, -1.0, 1.0)

    def apply_curve(self, x: float) -> float:
        """Step 4: sign(x) * |x|^curve (curve < 1 = stronger response)."""
        return sign(x) * math.pow(abs(x), self.settings.steering_curve)

    def apply_sensitivity(self, x: float, sensitivity: float) -> float:
        """Step 5: scale and clamp."""
        return clamp(x * sensitivity, -1.0, 1.0)

    def effective_sensitivity(self, velocity: float) -> float:
        """Base sensitivity plus a bounded speed-dependent boost."""
        s = self.settings
        if not s.adaptive_sensitivity:
            return s.sensitivity
        target = s.adaptive_gain * clamp(abs(velocity) / VELOCITY_REF_DEG_S, 0.0, 1.0)
        self._speed_factor = lerp(self._speed_factor, target, 0.2)
        lo = min(s.min_sensitivity, s.sensitivity)
        hi = max(s.max_sensitivity, s.sensitivity)
        return clamp(s.sensitivity + self._speed_factor, lo, hi)

    def apply_smoothing(self, x: float, velocity: float, dt: float, stability: float) -> float:
        """Step 6: speed-adaptive smoothing (stronger when the signal is unstable)."""
        self._filter.base_alpha = self.settings.smoothing
        self._filter.mode = self.settings.filter_mode
        scale = 0.85 if stability < 65.0 else 1.0
        return self._filter.update(x, velocity, dt, scale)

    def calculate_velocity(self, angle: float, dt: float) -> float:
        """Steering velocity in deg/s (lightly filtered)."""
        if self._prev_angle is None:
            v = 0.0
        else:
            v = wrap_angle(angle - self._prev_angle) / dt
        self._prev_angle = angle
        self._velocity = lerp(self._velocity, v, 0.35)
        return self._velocity

    def calculate_stability(self, value: float) -> Tuple[float, str]:
        """Stability 0..100 from the frame-to-frame jitter of recent values."""
        self._history.append(value)
        if len(self._history) < 6:
            return 100.0, "STABLE"
        jitter = float(np.std(np.diff(np.fromiter(self._history, dtype=np.float32))))
        stability = 100.0 * (1.0 - clamp(jitter / JITTER_FULL_SCALE, 0.0, 1.0))
        label = "STABLE" if stability >= 85 else "GOOD" if stability >= 65 else "UNSTABLE"
        return stability, label

    def _apply_center_zone(self, v: float) -> float:
        """Small centre dead zone with hysteresis (leave at dz, re-enter at 0.6 dz)."""
        dz = self.settings.center_deadzone
        if self._in_zone:
            if abs(v) > dz:
                self._in_zone = False
        elif abs(v) < dz * 0.6:
            self._in_zone = True
        return 0.0 if self._in_zone else v

    # ------------------------------------------------------------ main entry
    def update(self, left: Optional[HandState], right: Optional[HandState],
               dt: float) -> SteeringState:
        """Process one frame. ``left``/``right`` are the image-left/right hands."""
        dt = clamp(dt, 1e-3, 0.25)
        if left is None or right is None:
            return self._handle_lost(dt)
        self._lost_time = 0.0
        self._lost_frames = 0

        m = self.calculate_raw_angle(left, right)
        angle = self.apply_center_offset(m)
        velocity = self.calculate_velocity(angle, dt)
        normalized = self.normalize_angle(angle)
        curved = self.apply_curve(normalized)
        sens = self.effective_sensitivity(velocity)
        scaled = self.apply_sensitivity(curved, sens)
        stability, label = self.calculate_stability(scaled)
        smoothed = self.apply_smoothing(scaled, velocity, dt, stability)
        out = self._apply_center_zone(smoothed)
        if self.settings.auto_center and abs(out) < 0.18 and abs(velocity) < 25.0:
            out *= 1.0 - 0.25 * (1.0 - abs(out) / 0.18)     # gentle pull toward 0
        out = clamp(out, -1.0, 1.0)

        self._dir.on = self.settings.steer_start_threshold
        self._dir.off = self.settings.steer_release_threshold
        d = self._dir.update(out)
        state = SteeringState(
            raw_angle=m.raw_angle, calibrated_angle=angle, normalized=normalized,
            smoothed=smoothed, output=out, velocity=velocity, stability=stability,
            stability_label=label, direction="LEFT" if d < 0 else "RIGHT" if d > 0 else "CENTER",
            active=True, sensitivity=sens, hand_distance=m.distance, lost_frames=0,
        )
        self._last = state
        return state

    def _handle_lost(self, dt: float) -> SteeringState:
        """Hold the last steering briefly, then fall back to neutral."""
        self._lost_time += dt
        self._lost_frames += 1
        if self._last is not None and self._lost_time <= self.settings.lost_hand_grace:
            return replace(self._last, lost_frames=self._lost_frames)
        frames = self._lost_frames
        self.reset(prime_zero=True)          # on return, steering ramps up from 0
        self._lost_time = self.settings.lost_hand_grace + 1.0
        self._lost_frames = frames
        return SteeringState(active=False, lost_frames=frames)
