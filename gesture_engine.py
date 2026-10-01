"""Hand gesture recognition, temporal filtering, commands and pedal values."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Settings
from geometry import clamp
from models import Gesture, HandState
from smoothing import RateLimiter

TIPS = (4, 8, 12, 16, 20)
PIPS = (3, 6, 10, 14, 18)
MCPS = (2, 5, 9, 13, 17)

DRIVE_DEBOUNCE = 0.10        # s a gesture must persist to count (throttle/brake)
PAUSE_HOLD_TIME = 1.5        # s single open palm must be held to pause
RECALIBRATE_COOLDOWN = 3.0   # s between pinch recalibrations
EXTENDED_RATIO = 1.45        # tip-to-wrist / mcp-to-wrist for an extended finger
THUMB_EXTENDED_RATIO = 0.70  # thumb-tip to index-MCP, in hand-size units
PINCH_RATIO = 0.28           # thumb-tip to index-tip, in hand-size units


class GestureCommand(Enum):
    """High-level commands triggered by held gestures."""
    RESUME = "RESUME"
    RECALIBRATE = "RECALIBRATE"
    PAUSE = "PAUSE"


@dataclass
class _Track:
    candidate: Gesture = Gesture.NONE
    candidate_since: float = 0.0
    stable: Gesture = Gesture.NONE
    stable_since: float = 0.0


class GestureEngine:
    """Classifies hand gestures and maps them to controls."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tracks: Dict[str, _Track] = {"Left": _Track(), "Right": _Track()}
        self._fired: Dict[str, bool] = {}
        self._last_recalibrate = -1e9
        self._throttle = RateLimiter(rise=8.0, fall=12.0)
        self._brake = RateLimiter(rise=8.0, fall=12.0)

    # ------------------------------------------------------------ static analysis
    def detect_hand_openness(self, hand: HandState) -> float:
        """Compute finger-extension flags and an openness score (0..1).

        Rotation invariant: compares fingertip and MCP distances from the wrist.
        """
        p = hand.pts
        wrist = p[0]
        ratios = []
        for tip, mcp in zip(TIPS[1:], MCPS[1:]):
            d_mcp = float(np.linalg.norm(p[mcp] - wrist)) + 1e-6
            ratios.append(float(np.linalg.norm(p[tip] - wrist)) / d_mcp)
        four = [r > EXTENDED_RATIO for r in ratios]
        size = hand.size + 1e-6
        thumb = float(np.linalg.norm(p[4] - p[5])) / size > THUMB_EXTENDED_RATIO
        hand.fingers = (thumb, four[0], four[1], four[2], four[3])
        hand.openness = float(np.mean([clamp((r - 1.1) / 0.7, 0.0, 1.0) for r in ratios]))
        return hand.openness

    def detect_open_palm(self, hand: HandState) -> bool:
        n = sum(hand.fingers[1:])
        return n == 4 or (n == 3 and hand.openness > 0.7)

    def detect_fist(self, hand: HandState) -> bool:
        return sum(hand.fingers[1:]) == 0 and hand.openness < 0.35 and not self.detect_thumbs_up(hand)

    def detect_thumbs_up(self, hand: HandState) -> bool:
        if not hand.fingers[0] or any(hand.fingers[1:]):
            return False
        # thumb tip must point clearly upward (smaller y) relative to the index MCP
        return float(hand.pts[4][1]) < float(hand.pts[5][1]) - 0.45 * hand.size

    def detect_point(self, hand: HandState) -> bool:
        f = hand.fingers
        return f[1] and not (f[2] or f[3] or f[4])

    def detect_pinch(self, hand: HandState) -> bool:
        gap = float(np.linalg.norm(hand.pts[4] - hand.pts[8])) / (hand.size + 1e-6)
        return gap < PINCH_RATIO and sum(hand.fingers[2:]) >= 2   # "OK" sign

    def classify(self, hand: HandState) -> Gesture:
        """Single-frame gesture classification."""
        self.detect_hand_openness(hand)
        if self.detect_pinch(hand):
            return Gesture.PINCH
        if self.detect_thumbs_up(hand):
            return Gesture.THUMBS_UP
        if self.detect_point(hand):
            return Gesture.POINT
        if self.detect_fist(hand):
            return Gesture.FIST
        if self.detect_open_palm(hand):
            return Gesture.OPEN_PALM
        return Gesture.NONE

    # ------------------------------------------------------------ temporal filtering
    def update_gesture(self, hand: HandState, now: float) -> Gesture:
        """Classify and debounce; a gesture only changes after DRIVE_DEBOUNCE s."""
        raw = self.classify(hand)
        hand.raw_gesture = raw
        tr = self._tracks.setdefault(hand.handedness, _Track())
        if raw != tr.candidate:
            tr.candidate = raw
            tr.candidate_since = now
        if raw != tr.stable and now - tr.candidate_since >= DRIVE_DEBOUNCE:
            tr.stable = raw
            tr.stable_since = now
        hand.gesture = tr.stable
        return tr.stable

    def process_hands(self, hands: List[HandState], now: float) -> None:
        """Update every visible hand and forget the ones that disappeared."""
        present = set()
        for hand in hands:
            self.update_gesture(hand, now)
            present.add(hand.handedness)
        for side, tr in self._tracks.items():
            if side not in present:
                tr.candidate = tr.stable = Gesture.NONE
                tr.candidate_since = tr.stable_since = now

    # ------------------------------------------------------------ commands
    def poll_commands(self, hands: List[HandState], now: float) -> List[GestureCommand]:
        """Edge-triggered commands from held gestures."""
        if not self.settings.gesture_control:
            return []
        commands: List[GestureCommand] = []
        hold = self.settings.gesture_hold_time
        for hand in hands:
            tr = self._tracks.get(hand.handedness)
            if tr is None:
                continue
            for g in (Gesture.THUMBS_UP, Gesture.PINCH, Gesture.OPEN_PALM):
                key = f"{hand.handedness}:{g.value}"
                if tr.stable != g:
                    self._fired[key] = False
                    continue
                if self._fired.get(key):
                    continue
                held = now - tr.stable_since
                if g == Gesture.THUMBS_UP and held >= hold:
                    commands.append(GestureCommand.RESUME)
                    self._fired[key] = True
                elif g == Gesture.PINCH and held >= hold and now - self._last_recalibrate > RECALIBRATE_COOLDOWN:
                    commands.append(GestureCommand.RECALIBRATE)
                    self._last_recalibrate = now
                    self._fired[key] = True
                elif g == Gesture.OPEN_PALM and len(hands) == 1 and held >= PAUSE_HOLD_TIME:
                    commands.append(GestureCommand.PAUSE)
                    self._fired[key] = True
        return commands

    # ------------------------------------------------------------ pedals
    def compute_pedals(self, hands: List[HandState], dt: float,
                       latch_throttle: bool = False, latch_brake: bool = False) -> Tuple[float, float]:
        """Return (throttle, brake) in 0..1.

        Mode 1: both fists = throttle, both open palms = brake.
        Mode 2: only the RIGHT hand's gesture is used.
        Mode 3: mode 1 plus keyboard latches (W = throttle, S = brake).
        Strength is analog: a tighter fist / flatter palm gives a larger value.
        """
        mode = self.settings.throttle_mode
        t_target = b_target = 0.0
        if self.settings.gesture_control:
            by_side = {h.handedness: h for h in hands}
            if mode == 2:
                pool = [by_side["Right"]] if "Right" in by_side else []
            else:
                pool = [by_side["Left"], by_side["Right"]] if len(by_side) == 2 and len(hands) == 2 else []
            if pool:
                if all(h.gesture == Gesture.FIST for h in pool):
                    closure = float(np.mean([1.0 - h.openness for h in pool]))
                    t_target = clamp((closure - 0.55) / 0.35, 0.0, 1.0)
                elif all(h.gesture == Gesture.OPEN_PALM for h in pool):
                    openness = float(np.mean([h.openness for h in pool]))
                    b_target = clamp((openness - 0.65) / 0.30, 0.0, 1.0)
        if mode == 3:
            t_target = max(t_target, 1.0 if latch_throttle else 0.0)
            b_target = max(b_target, 1.0 if latch_brake else 0.0)
        if b_target > 0.05:
            t_target = 0.0                       # brake has priority
        return self._throttle.update(t_target, dt), self._brake.update(b_target, dt)

    def reset_pedals(self) -> None:
        """Set both pedals to zero immediately."""
        self._throttle.reset()
        self._brake.reset()
