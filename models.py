"""Plain data models (dataclasses / enums) shared between modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import numpy as np

Point = Tuple[float, float]


class AppState(Enum):
    """Application state machine."""
    STARTING = "STARTING"
    WAITING_FOR_HANDS = "WAITING_FOR_HANDS"
    CALIBRATING = "CALIBRATING"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    CAMERA_ERROR = "CAMERA_ERROR"
    EXITING = "EXITING"


class Gesture(Enum):
    """Recognised hand gestures."""
    NONE = "NONE"
    OPEN_PALM = "OPEN PALM"
    FIST = "FIST"
    THUMBS_UP = "THUMBS UP"
    POINT = "POINT"
    PINCH = "PINCH"


def _empty_landmarks() -> np.ndarray:
    return np.zeros((21, 2), dtype=np.float32)


@dataclass
class HandState:
    """One tracked hand. Coordinates are normalised (0..1) image coordinates."""
    handedness: str = "Unknown"          # "Left" / "Right" from the USER's point of view
    wrist: Point = (0.0, 0.0)
    thumb: Point = (0.0, 0.0)            # fingertip positions
    index: Point = (0.0, 0.0)
    middle: Point = (0.0, 0.0)
    ring: Point = (0.0, 0.0)
    pinky: Point = (0.0, 0.0)
    palm_center: Point = (0.0, 0.0)
    openness: float = 0.0                # 0 = closed fist .. 1 = fully open
    confidence: float = 0.0
    visible: bool = False
    size: float = 0.0                    # wrist->middle-MCP length (height units)
    axis_angle: float = 0.0              # orientation of wrist->middle-MCP (deg)
    fingers: Tuple[bool, bool, bool, bool, bool] = (False,) * 5   # thumb..pinky extended
    gesture: Gesture = Gesture.NONE      # temporally filtered gesture
    raw_gesture: Gesture = Gesture.NONE  # single-frame gesture
    landmarks: np.ndarray = field(default_factory=_empty_landmarks)  # (21,2) normalised
    pts: np.ndarray = field(default_factory=_empty_landmarks)        # (21,2) aspect-corrected


@dataclass
class SteeringState:
    """Output of the steering engine for one frame."""
    raw_angle: float = 0.0          # degrees, straight from the hand axis
    calibrated_angle: float = 0.0   # degrees relative to the calibrated centre
    normalized: float = 0.0         # -1..+1 linear
    smoothed: float = 0.0           # -1..+1 after curve/sensitivity/smoothing
    output: float = 0.0             # final steering value after centre zone
    velocity: float = 0.0           # deg / s
    stability: float = 100.0        # 0..100
    stability_label: str = "STABLE"
    direction: str = "CENTER"       # LEFT / CENTER / RIGHT
    active: bool = False
    sensitivity: float = 1.0        # effective sensitivity this frame
    hand_distance: float = 0.0      # wrist-to-wrist, fraction of frame width
    lost_frames: int = 0


@dataclass
class ControlState:
    """What is sent to the output backend."""
    steering: float = 0.0   # -1..+1
    throttle: float = 0.0   # 0..1
    brake: float = 0.0      # 0..1
    paused: bool = False
    emergency_stop: bool = False


@dataclass
class CalibrationData:
    """Neutral ("hands at centre") reference measured during calibration."""
    center_angle: float = 0.0
    center_distance: float = 0.4
    center_position: Point = (0.5, 0.5)
    center_rotation: float = 0.0
    valid: bool = False
