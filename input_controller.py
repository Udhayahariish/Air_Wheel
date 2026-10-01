"""Output backends.  Internal steering is analog; keyboard output is digital."""
from __future__ import annotations

import atexit
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from config import DEFAULT_KEY_MAP, Settings
from smoothing import Hysteresis, Latch

LOGICAL_KEYS = ("left", "right", "accelerate", "brake")
PEDAL_ON = 0.30      # pedal value that presses the key
PEDAL_OFF = 0.15     # pedal value that releases it (hysteresis)


class InputOutput(ABC):
    """Interface every output backend (keyboard / vJoy / gamepad) implements."""

    name = "base"

    @abstractmethod
    def set_steering(self, value: float) -> None:
        """Steering, -1.0 (left) .. +1.0 (right)."""

    @abstractmethod
    def set_throttle(self, value: float) -> None:
        """Throttle, 0.0 .. 1.0."""

    @abstractmethod
    def set_brake(self, value: float) -> None:
        """Brake, 0.0 .. 1.0."""

    @abstractmethod
    def release_all(self, force: bool = False) -> None:
        """Release everything and centre all axes."""

    @property
    def held_keys(self) -> List[str]:
        """Names of currently pressed inputs (for diagnostics)."""
        return []


class NullOutput(InputOutput):
    """Does nothing (dry-run / fallback when pynput is unavailable)."""

    name = "none"

    def __init__(self) -> None:
        self.steering = self.throttle = self.brake = 0.0

    def set_steering(self, value: float) -> None:
        self.steering = value

    def set_throttle(self, value: float) -> None:
        self.throttle = value

    def set_brake(self, value: float) -> None:
        self.brake = value

    def release_all(self, force: bool = False) -> None:
        self.steering = self.throttle = self.brake = 0.0


def _resolve_key(name: str):
    """Convert 'left', 'up', 'w' ... to a pynput key object."""
    from pynput.keyboard import Key, KeyCode
    n = name.strip().lower()
    if len(n) == 1:
        return KeyCode.from_char(n)
    key = getattr(Key, n, None)
    if key is None:
        raise ValueError(f"unknown key '{name}'")
    return key


class KeyboardOutput(InputOutput):
    """Digital keyboard output: key_down / key_up are sent only on state changes."""

    name = "keyboard"

    def __init__(self, key_map: Dict[str, str], start: float, release: float) -> None:
        from pynput.keyboard import Controller
        self._kb = Controller()
        self._keys = {}
        for logical in LOGICAL_KEYS:
            try:
                self._keys[logical] = _resolve_key(key_map.get(logical, DEFAULT_KEY_MAP[logical]))
            except Exception as exc:
                print(f"[input] {exc}; using default key for '{logical}'.")
                self._keys[logical] = _resolve_key(DEFAULT_KEY_MAP[logical])
        self._down: Dict[str, bool] = {k: False for k in LOGICAL_KEYS}
        self._steer = Hysteresis(start, release)
        self._accel = Latch(PEDAL_ON, PEDAL_OFF)
        self._brake = Latch(PEDAL_ON, PEDAL_OFF)
        atexit.register(self.release_all, True)       # last line of defence

    def _set(self, logical: str, want: bool) -> None:
        if self._down[logical] == want:
            return
        try:
            if want:
                self._kb.press(self._keys[logical])
            else:
                self._kb.release(self._keys[logical])
            self._down[logical] = want
        except Exception as exc:
            print(f"[input] key '{logical}' failed: {exc}")

    def set_steering(self, value: float) -> None:
        d = self._steer.update(value)
        want_left, want_right = d < 0, d > 0
        if not want_left:
            self._set("left", False)
        if not want_right:
            self._set("right", False)
        if want_left:
            self._set("left", True)
        if want_right:
            self._set("right", True)

    def set_throttle(self, value: float) -> None:
        self._set("accelerate", self._accel.update(value))

    def set_brake(self, value: float) -> None:
        self._set("brake", self._brake.update(value))

    def release_all(self, force: bool = False) -> None:
        """Release every key. ``force`` sends key-up even if we think it is up."""
        for logical in LOGICAL_KEYS:
            if force or self._down[logical]:
                try:
                    self._kb.release(self._keys[logical])
                except Exception:
                    pass
                self._down[logical] = False
        self._steer.reset()
        self._accel.reset()
        self._brake.reset()

    @property
    def held_keys(self) -> List[str]:
        return [k for k, v in self._down.items() if v]


def create_output(settings: Settings, dry_run: bool = False) -> Tuple[InputOutput, str]:
    """Create the backend for ``settings.output_mode``; returns (backend, notice)."""
    if dry_run:
        return NullOutput(), "DRY-RUN: no keys will be sent"
    notice = ""
    if settings.output_mode != "keyboard":
        notice = (f"output_mode '{settings.output_mode}' is not implemented yet; "
                  "using keyboard output (see README: future vJoy support).")
    try:
        return KeyboardOutput(settings.key_map, settings.steer_start_threshold,
                              settings.steer_release_threshold), notice
    except Exception as exc:
        return NullOutput(), f"pynput unavailable ({exc}); running without key output"
