"""Settings: dataclass, defaults, validation and JSON persistence."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Tuple

SETTINGS_PATH = Path(__file__).with_name("settings.json")
WINDOW_NAME = "Advanced Virtual Steering Wheel"

# name -> (sensitivity, curve)
PRESETS: Dict[str, Tuple[float, float]] = {
    "LOW": (0.85, 1.15),
    "NORMAL": (1.25, 0.85),
    "HIGH": (1.60, 0.70),
}

DEFAULT_KEY_MAP: Dict[str, str] = {
    "left": "left", "right": "right", "accelerate": "up", "brake": "down",
}


@dataclass
class Settings:
    """All user-configurable options (see README for descriptions)."""
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    fps: int = 60
    display_scale: float = 1.5

    max_steering_degrees: float = 65.0
    sensitivity: float = 1.25
    steering_curve: float = 0.85
    preset: str = "NORMAL"
    center_deadzone: float = 0.025
    smoothing: float = 0.18
    filter_mode: str = "adaptive"            # "adaptive" | "one_euro"
    adaptive_sensitivity: bool = True
    min_sensitivity: float = 0.5
    max_sensitivity: float = 2.2
    adaptive_gain: float = 0.35
    rotation_weight: float = 0.15            # contribution of hand rotation
    auto_center: bool = False
    steer_start_threshold: float = 0.035
    steer_release_threshold: float = 0.020
    lost_hand_grace: float = 0.20
    calibration_seconds: float = 2.0

    mirror_camera: bool = True
    show_hud: bool = True
    show_landmarks: bool = True
    show_debug: bool = False
    fullscreen: bool = False

    model_complexity: int = 1
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.65

    gesture_control: bool = True
    gesture_hold_time: float = 0.35
    throttle_mode: int = 1                   # 1 fist/palm, 2 right hand only, 3 gesture+keyboard (W/S)

    output_mode: str = "keyboard"
    key_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_KEY_MAP))


# name -> (min, max) for numeric settings
_RANGES: Dict[str, Tuple[float, float]] = {
    "camera_index": (0, 16), "camera_width": (160, 3840), "camera_height": (120, 2160),
    "fps": (5, 240), "display_scale": (0.5, 3.0),
    "max_steering_degrees": (20, 120), "sensitivity": (0.1, 3.0), "steering_curve": (0.3, 2.0),
    "center_deadzone": (0.0, 0.2), "smoothing": (0.01, 0.9),
    "min_sensitivity": (0.1, 3.0), "max_sensitivity": (0.1, 4.0), "adaptive_gain": (0.0, 2.0),
    "rotation_weight": (0.0, 0.6), "steer_start_threshold": (0.005, 0.5),
    "steer_release_threshold": (0.001, 0.5), "lost_hand_grace": (0.0, 2.0),
    "calibration_seconds": (0.5, 10.0), "model_complexity": (0, 1),
    "min_detection_confidence": (0.1, 1.0), "min_tracking_confidence": (0.1, 1.0),
    "gesture_hold_time": (0.05, 3.0), "throttle_mode": (1, 3),
}
_CHOICES: Dict[str, Tuple[str, ...]] = {
    "output_mode": ("keyboard", "virtual_joystick", "vjoy", "gamepad"),
    "filter_mode": ("adaptive", "one_euro"),
    "preset": ("LOW", "NORMAL", "HIGH", "CUSTOM"),
}


def _coerce(name: str, value: Any, default: Any) -> Any:
    """Return ``value`` if valid for setting ``name`` else ``default``."""
    try:
        if isinstance(default, bool):
            return value if isinstance(value, bool) else default
        if isinstance(default, dict):
            if not isinstance(value, dict):
                return default
            merged = dict(default)
            for k, v in value.items():
                if k in merged and isinstance(v, str) and v.strip():
                    merged[k] = v.strip().lower()
            return merged
        if isinstance(default, str):
            if isinstance(value, str) and value in _CHOICES.get(name, (value,)):
                return value
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        if isinstance(value, float) and not math.isfinite(value):
            return default
        lo, hi = _RANGES.get(name, (-math.inf, math.inf))
        if not (lo <= value <= hi):
            return default
        if isinstance(default, int):
            return int(round(value))
        return float(value)
    except Exception:
        return default


def validate_settings(raw: Any) -> Settings:
    """Build a valid Settings object from arbitrary JSON data (never raises)."""
    defaults = Settings()
    if not isinstance(raw, dict):
        return defaults
    values = {f.name: _coerce(f.name, raw.get(f.name, getattr(defaults, f.name)),
                              getattr(defaults, f.name)) for f in fields(Settings)}
    settings = Settings(**values)
    if settings.steer_release_threshold >= settings.steer_start_threshold:
        settings.steer_start_threshold = defaults.steer_start_threshold
        settings.steer_release_threshold = defaults.steer_release_threshold
    if settings.min_sensitivity > settings.max_sensitivity:
        settings.min_sensitivity = defaults.min_sensitivity
        settings.max_sensitivity = defaults.max_sensitivity
    return settings


def save_settings(settings: Settings, path: Path = SETTINGS_PATH) -> None:
    """Write settings to JSON (errors are reported, never raised)."""
    try:
        path.write_text(json.dumps(asdict(settings), indent=4), encoding="utf-8")
    except OSError as exc:
        print(f"[config] Could not save settings: {exc}")


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    """Load settings; create the file with defaults if missing."""
    if not path.exists():
        settings = Settings()
        save_settings(settings, path)
        return settings
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[config] Invalid settings file ({exc}); using defaults.")
        return Settings()
    return validate_settings(raw)


def apply_preset(settings: Settings, name: str) -> None:
    """Apply LOW / NORMAL / HIGH sensitivity preset."""
    if name in PRESETS:
        settings.sensitivity, settings.steering_curve = PRESETS[name]
        settings.preset = name
