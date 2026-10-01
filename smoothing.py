"""Filters and small stateful helpers (smoothing, hysteresis, rate limiting)."""
from __future__ import annotations

import math
from typing import Optional

from geometry import clamp


class OneEuroFilter:
    """One Euro Filter: low jitter when slow, low lag when fast."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 6.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: Optional[float] = None
        self._dx = 0.0

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def reset(self, initial: Optional[float] = None) -> None:
        self._x = initial
        self._dx = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if self._x is None:
            self._x = x
            return x
        dt = max(dt, 1e-3)
        dx = (x - self._x) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(dt, cutoff)
        self._x = a * x + (1.0 - a) * self._x
        return self._x


class SmoothFilter:
    """Speed-adaptive exponential smoothing.

    ``smoothed = alpha * current + (1 - alpha) * previous`` where alpha grows
    with movement speed: slow movement -> ~0.12 (jitter removal), fast
    movement -> ~0.35 (responsive).  Alpha is frame-rate compensated (tuned
    for 60 FPS).  ``mode="one_euro"`` switches to a One Euro Filter.
    """

    def __init__(self, alpha: float = 0.18, mode: str = "adaptive",
                 speed_low: float = 20.0, speed_high: float = 220.0) -> None:
        self.base_alpha = alpha
        self.mode = mode
        self.speed_low = speed_low
        self.speed_high = speed_high
        self._value: Optional[float] = None
        self._euro = OneEuroFilter()

    @property
    def alpha_slow(self) -> float:
        return clamp(self.base_alpha * 0.67, 0.01, 0.9)

    @property
    def alpha_fast(self) -> float:
        return clamp(self.base_alpha * 1.95, 0.01, 0.95)

    def reset(self, initial: Optional[float] = None) -> None:
        self._value = initial
        self._euro.reset(initial)

    def update(self, value: float, speed: float, dt: float = 1.0 / 60.0,
               alpha_scale: float = 1.0) -> float:
        """Filter ``value``; ``speed`` is the movement speed in deg/s."""
        if self._value is None:
            self._value = value
            self._euro.reset(value)
            return value
        if self.mode == "one_euro":
            self._value = self._euro(value, dt)
            return self._value
        t = clamp((abs(speed) - self.speed_low) / (self.speed_high - self.speed_low), 0.0, 1.0)
        alpha = (self.alpha_slow + (self.alpha_fast - self.alpha_slow) * t) * alpha_scale
        alpha = clamp(alpha, 0.01, 0.95)
        a = 1.0 - (1.0 - alpha) ** (max(dt, 1e-3) * 60.0)   # frame-rate independent
        self._value += (value - self._value) * a
        return self._value


class Hysteresis:
    """Three-state (-1/0/+1) Schmitt trigger: engage at ``on``, release at ``off``."""

    def __init__(self, on: float, off: float) -> None:
        self.on = on
        self.off = off
        self.state = 0

    def reset(self) -> None:
        self.state = 0

    def update(self, value: float) -> int:
        if self.state == 0:
            if value >= self.on:
                self.state = 1
            elif value <= -self.on:
                self.state = -1
        elif self.state == 1:
            if value < self.off:
                self.state = -1 if value <= -self.on else 0
        else:
            if value > -self.off:
                self.state = 1 if value >= self.on else 0
        return self.state


class Latch:
    """Boolean Schmitt trigger for 0..1 pedal values."""

    def __init__(self, on: float, off: float) -> None:
        self.on = on
        self.off = off
        self.state = False

    def reset(self) -> None:
        self.state = False

    def update(self, value: float) -> bool:
        if self.state:
            if value < self.off:
                self.state = False
        elif value >= self.on:
            self.state = True
        return self.state


class RateLimiter:
    """Moves a value toward a target at limited rise/fall rates (units/s)."""

    def __init__(self, rise: float, fall: float) -> None:
        self.rise = rise
        self.fall = fall
        self.value = 0.0

    def reset(self, value: float = 0.0) -> None:
        self.value = value

    def update(self, target: float, dt: float) -> float:
        if target > self.value:
            self.value = min(target, self.value + self.rise * dt)
        else:
            self.value = max(target, self.value - self.fall * dt)
        return self.value


class ExpAverage:
    """Exponential moving average (used for FPS and timings)."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.value = 0.0
        self._init = False

    def update(self, sample: float) -> float:
        if not self._init:
            self.value = sample
            self._init = True
        else:
            self.value += (sample - self.value) * self.alpha
        return self.value
