"""Small, dependency-free math helpers shared by all modules."""
from __future__ import annotations

import math
from typing import Iterable, Tuple

Point = Tuple[float, float]


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the closed interval [lo, hi]."""
    return lo if value < lo else hi if value > hi else value


def sign(value: float) -> int:
    """Return -1, 0 or +1."""
    return (value > 0) - (value < 0)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def wrap_angle(deg: float) -> float:
    """Wrap an angle in degrees into the range [-180, +180)."""
    return (deg + 180.0) % 360.0 - 180.0


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def circular_mean_deg(values: Iterable[float]) -> float:
    """Mean of angles (degrees) that is safe across the +/-180 seam."""
    vals = list(values)
    if not vals:
        return 0.0
    s = sum(math.sin(math.radians(v)) for v in vals)
    c = sum(math.cos(math.radians(v)) for v in vals)
    return math.degrees(math.atan2(s, c))


def rotate_point(point: Point, center: Point, angle_deg: float) -> Point:
    """Rotate ``point`` around ``center`` by ``angle_deg``.

    Uses the standard rotation matrix::

        x' = cx + dx*cos(a) - dy*sin(a)
        y' = cy + dx*sin(a) + dy*cos(a)

    Image coordinates have y pointing DOWN, so a positive angle rotates
    clockwise on screen (i.e. towards "steer right").
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = point[0] - center[0], point[1] - center[1]
    return (center[0] + dx * ca - dy * sa, center[1] + dx * sa + dy * ca)
