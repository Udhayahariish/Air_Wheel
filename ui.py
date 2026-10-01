"""Rendering: modern dark HUD, rotating virtual wheel, hand overlays.

Only ASCII text is drawn (OpenCV's Hershey fonts have no unicode); the degree
sign is drawn as a small ring by :func:`put_text`.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Settings
from geometry import clamp, rotate_point
from models import AppState, ControlState, Gesture, HandState, SteeringState
from steering_engine import REGION

Color = Tuple[int, int, int]
FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

PANEL: Color = (24, 20, 18)
BORDER: Color = (84, 74, 66)
TEXT: Color = (240, 240, 240)
DIM: Color = (160, 154, 148)
ACCENT: Color = (255, 190, 40)
GOOD: Color = (120, 230, 120)
WARN: Color = (60, 190, 255)
BAD: Color = (80, 80, 255)
LEFT_COLOR: Color = (255, 200, 60)
RIGHT_COLOR: Color = (90, 150, 255)

HAND_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
                    (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))
TIP_IDS = (4, 8, 12, 16, 20)

STATE_STYLE: Dict[AppState, Tuple[str, Color]] = {
    AppState.STARTING: ("STARTING", DIM),
    AppState.WAITING_FOR_HANDS: ("WAITING FOR HANDS", WARN),
    AppState.CALIBRATING: ("CALIBRATING", ACCENT),
    AppState.ACTIVE: ("ACTIVE", GOOD),
    AppState.PAUSED: ("PAUSED", WARN),
    AppState.EMERGENCY_STOP: ("EMERGENCY STOP", BAD),
    AppState.CAMERA_ERROR: ("CAMERA ERROR", BAD),
    AppState.EXITING: ("EXITING", DIM),
}


# ----------------------------------------------------------------------------- helpers
def mix(a: Color, b: Color, t: float) -> Color:
    """Blend two BGR colours."""
    t = clamp(t, 0.0, 1.0)
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


def put_text(img: np.ndarray, s: str, x: int, y: int, scale: float = 0.5, color: Color = TEXT,
             thick: int = 1, anchor: str = "l") -> int:
    """Draw text (anchor l/c/r). A '°' is rendered as a small ring. Returns width."""
    parts = s.split("\u00b0")
    sizes = [cv2.getTextSize(p, FONT, scale, thick)[0][0] if p else 0 for p in parts]
    th = cv2.getTextSize("A", FONT, scale, thick)[0][1]
    ring_r = max(2, int(th * 0.16))
    ring_w = ring_r * 2 + 3
    total = sum(sizes) + ring_w * (len(parts) - 1)
    x = x - total // 2 if anchor == "c" else x - total if anchor == "r" else x
    cx = x
    for i, part in enumerate(parts):
        if part:
            cv2.putText(img, part, (cx, y), FONT, scale, color, thick, AA)
        cx += sizes[i]
        if i < len(parts) - 1:
            cv2.circle(img, (cx + ring_r + 1, y - th + ring_r), ring_r, color, max(1, thick), AA)
            cx += ring_w
    return total


_mask_cache: Dict[Tuple[int, int, int], np.ndarray] = {}


def _rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    key = (w, h, r)
    m = _mask_cache.get(key)
    if m is None:
        r = max(1, min(r, w // 2, h // 2))
        m = np.zeros((h, w), np.uint8)
        cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 255, -1)
        cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 255, -1)
        for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
            cv2.circle(m, (cx, cy), r, 255, -1)
        if len(_mask_cache) > 64:
            _mask_cache.clear()
        _mask_cache[key] = m
    return m


def draw_panel(img: np.ndarray, x: int, y: int, w: int, h: int, radius: int = 12,
               color: Color = PANEL, alpha: float = 0.58, border: Optional[Color] = BORDER) -> None:
    """Semi-transparent rounded panel with a thin border."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    roi = img[y0:y1, x0:x1]
    mask = _rounded_mask(x1 - x0, y1 - y0, radius) > 0
    blended = cv2.addWeighted(roi, 1.0 - alpha, np.full_like(roi, color), alpha, 0)
    roi[mask] = blended[mask]
    if border is not None:
        r = max(1, min(radius, w // 2, h // 2))
        cv2.line(img, (x + r, y), (x + w - r, y), border, 1, AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), border, 1, AA)
        cv2.line(img, (x, y + r), (x, y + h - r), border, 1, AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), border, 1, AA)
        cv2.ellipse(img, (x + r, y + r), (r, r), 0, 180, 270, border, 1, AA)
        cv2.ellipse(img, (x + w - r, y + r), (r, r), 0, 270, 360, border, 1, AA)
        cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, border, 1, AA)
        cv2.ellipse(img, (x + r, y + h - r), (r, r), 0, 90, 180, border, 1, AA)


def draw_dot(img: np.ndarray, x: int, y: int, r: int, color: Color) -> None:
    """Glowing status dot."""
    cv2.circle(img, (x, y), r + 3, mix(color, (0, 0, 0), 0.65), -1, AA)
    cv2.circle(img, (x, y), r, color, -1, AA)


def draw_hbar(img: np.ndarray, x: int, y: int, w: int, h: int, value: float, color: Color) -> None:
    """Horizontal 0..1 progress bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (46, 42, 40), -1)
    fill = int(w * clamp(value, 0.0, 1.0))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDER, 1)


class Animator:
    """Exponentially smoothed UI values so numbers/bars glide instead of jumping."""

    def __init__(self) -> None:
        self._v: Dict[str, float] = {}

    def get(self, key: str, target: float, dt: float, rate: float = 12.0) -> float:
        cur = self._v.get(key, target)
        cur += (target - cur) * (1.0 - math.exp(-rate * dt))
        self._v[key] = cur
        return cur


@dataclass
class UIContext:
    """Everything the UI needs to draw one frame."""
    app_state: AppState
    steering: SteeringState
    control: ControlState
    hands: List[HandState]
    fps: float = 0.0
    process_ms: float = 0.0
    camera_size: Tuple[int, int] = (640, 480)
    position_text: str = ""
    position_good: bool = False
    calibration_progress: float = 0.0
    calibration_message: str = ""
    toast: str = ""
    toast_alpha: float = 0.0
    camera_message: str = ""
    center_marker: Optional[Tuple[float, float]] = None
    throttle_mode: int = 1


class UIManager:
    """Draws the whole interface onto the (already scaled) camera frame."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._anim = Animator()
        self._visual_angle = 0.0
        self._last_t = time.perf_counter()
        self._t = self._last_t
        self._dt = 1.0 / 60.0
        self.s = 1.0
        self.w = 640
        self.h = 480

    # ------------------------------------------------------------ orchestration
    def render(self, frame: np.ndarray, ctx: UIContext) -> np.ndarray:
        """Draw everything for one frame and return it."""
        now = time.perf_counter()
        self._dt = clamp(now - self._last_t, 1e-3, 0.1)
        self._last_t = self._t = now
        self.h, self.w = frame.shape[:2]
        self.s = self.h / 720.0

        calibrating = ctx.app_state in (AppState.STARTING, AppState.WAITING_FOR_HANDS,
                                        AppState.CALIBRATING)
        self.draw_background(frame, dim=calibrating)
        if self.settings.show_hud:
            self.draw_region(frame, ctx)
            if not calibrating:
                self.draw_wheel(frame, ctx)
            if self.settings.show_landmarks:
                self.draw_hand_tracking(frame, ctx)
            self.draw_status(frame, ctx, with_steering=not calibrating)
            self.draw_performance(frame, ctx)
            if calibrating:
                self.draw_calibration(frame, ctx)
            else:
                self.draw_hud(frame, ctx)
                self.draw_gesture_panel(frame, ctx)
                if self.settings.show_debug:
                    self.draw_debug(frame, ctx)
            self.draw_footer(frame)
        else:
            if self.settings.show_landmarks:
                self.draw_hand_tracking(frame, ctx)
            label, color = STATE_STYLE[ctx.app_state]
            draw_dot(frame, int(20 * self.s), int(20 * self.s), int(5 * self.s), color)
            put_text(frame, label, int(34 * self.s), int(26 * self.s), 0.5 * self.s, color, 1)

        self._draw_banners(frame, ctx)
        if ctx.toast and ctx.toast_alpha > 0.02:
            self.draw_toast(frame, ctx)
        return frame

    def _th(self, base: int = 1) -> int:
        return max(1, int(round(base * self.s)))

    # ------------------------------------------------------------ background / region
    def draw_background(self, frame: np.ndarray, dim: bool = False) -> None:
        """Darken the top/bottom bands (and the whole frame while calibrating)."""
        if dim:
            cv2.convertScaleAbs(frame, alpha=0.6, dst=frame)
            return
        for y0, y1 in ((0, int(130 * self.s)), (self.h - int(150 * self.s), self.h)):
            roi = frame[y0:y1]
            cv2.convertScaleAbs(roi, alpha=0.72, dst=roi)

    def draw_region(self, frame: np.ndarray, ctx: UIContext) -> None:
        """Corner brackets marking the virtual steering area."""
        x0, y0 = int(REGION[0] * self.w), int(REGION[1] * self.h)
        x1, y1 = int(REGION[2] * self.w), int(REGION[3] * self.h)
        color = mix(GOOD if ctx.position_good else WARN, (0, 0, 0), 0.45)
        n, t = int(30 * self.s), self._th(2)
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * n, cy), color, t, AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * n), color, t, AA)

    # ------------------------------------------------------------ hands
    def draw_hand_tracking(self, frame: np.ndarray, ctx: UIContext) -> None:
        """Glowing skeletons, markers, labels, steering axis and midpoint marker."""
        h, w = frame.shape[:2]
        s, t = self.s, self._t
        scale = np.array([w, h], np.float32)
        centers: Dict[str, Tuple[int, int]] = {}
        for hand in ctx.hands:
            if not hand.visible:
                continue
            color = LEFT_COLOR if hand.handedness == "Left" else RIGHT_COLOR
            pts = (hand.landmarks * scale).astype(np.int32)
            P = [(int(p[0]), int(p[1])) for p in pts]
            x0 = max(0, int(pts[:, 0].min()) - 24)
            y0 = max(0, int(pts[:, 1].min()) - 24)
            x1 = min(w, int(pts[:, 0].max()) + 24)
            y1 = min(h, int(pts[:, 1].max()) + 24)
            if x1 - x0 > 4 and y1 - y0 > 4:                       # glow
                roi = frame[y0:y1, x0:x1]
                glow = roi.copy()
                for a, b in HAND_CONNECTIONS:
                    cv2.line(glow, (P[a][0] - x0, P[a][1] - y0), (P[b][0] - x0, P[b][1] - y0),
                             color, self._th(9), AA)
                for i in TIP_IDS:
                    cv2.circle(glow, (P[i][0] - x0, P[i][1] - y0), int(12 * s), color, -1, AA)
                cv2.addWeighted(glow, 0.30, roi, 0.70, 0, dst=roi)
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, P[a], P[b], mix(color, (255, 255, 255), 0.25), self._th(2), AA)
            for i, p in enumerate(P):
                cv2.circle(frame, p, max(2, int(3 * s)), (245, 245, 245), -1, AA)
                if i in TIP_IDS:
                    cv2.circle(frame, p, max(3, int(6 * s)), color, -1, AA)
                    cv2.circle(frame, p, max(3, int(6 * s)), (255, 255, 255), 1, AA)
            cv2.circle(frame, P[0], int(9 * s), color, self._th(2), AA)    # wrist marker
            pc = (int(hand.palm_center[0] * w), int(hand.palm_center[1] * h))
            n = int(7 * s)
            cv2.line(frame, (pc[0] - n, pc[1]), (pc[0] + n, pc[1]), (255, 255, 255), 1, AA)
            cv2.line(frame, (pc[0], pc[1] - n), (pc[0], pc[1] + n), (255, 255, 255), 1, AA)
            centers[hand.handedness] = pc
            side = "LEFT HAND" if hand.handedness == "Left" else "RIGHT HAND"
            ty = max(int(18 * s), int(pts[:, 1].min()) - int(12 * s))
            tx = int(pts[:, 0].min())
            put_text(frame, f"{side} {hand.confidence * 100:.0f}%", tx, ty, 0.45 * s, color, self._th())
            if hand.gesture != Gesture.NONE:
                put_text(frame, hand.gesture.value, tx, ty + int(16 * s), 0.4 * s, TEXT, 1)

        if len(ctx.hands) >= 2:                                        # steering axis
            a = ctx.hands[0].palm_center
            b = ctx.hands[1].palm_center
            pa, pb = (int(a[0] * w), int(a[1] * h)), (int(b[0] * w), int(b[1] * h))
            col = GOOD if ctx.position_good else WARN
            cv2.line(frame, pa, pb, col, self._th(2), AA)
            mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
            pulse = 0.5 + 0.5 * math.sin(t * 5.0)
            cv2.circle(frame, mid, int((7 + 5 * pulse) * s), col, 1, AA)
            cv2.circle(frame, mid, max(2, int(4 * s)), col, -1, AA)
        if ctx.center_marker is not None:
            cm = (int(ctx.center_marker[0] * w), int(ctx.center_marker[1] * h))
            cv2.circle(frame, cm, int(14 * s), DIM, 1, AA)

    # ------------------------------------------------------------ wheel
    def draw_wheel(self, frame: np.ndarray, ctx: UIContext) -> None:
        """Rotating dashboard wheel with degree scale, markers and indicators."""
        s = self.s
        value = ctx.control.steering
        max_deg = self.settings.max_steering_degrees
        target = value * max_deg
        k = 1.0 - (1.0 - 0.22) ** (self._dt * 60.0)          # visual smoothing 0.22
        self._visual_angle += (target - self._visual_angle) * k
        theta = self._visual_angle

        R = max(36, int(0.15 * self.h))
        S = int(R * 2.9)
        c = S // 2
        C = (float(c), float(c))
        cv = np.zeros((S, S, 3), np.uint8)
        ip = lambda p: (int(round(p[0])), int(round(p[1])))
        rim_t = max(4, int(R * 0.13))

        cv2.circle(cv, (c, c), int(R * 1.08), (30, 26, 24), -1, AA)                  # backing
        cv2.circle(cv, (c, c), R, (96, 90, 86), rim_t, AA)                           # outer rim
        cv2.circle(cv, (c, c), R + rim_t // 2, mix(ACCENT, (0, 0, 0), 0.5), 1, AA)
        cv2.circle(cv, (c, c), int(R * 0.90), (150, 140, 130), 1, AA)                # inner rim
        for centre_deg in (180.0, 0.0):                                              # grips
            cv2.ellipse(cv, (c, c), (R, R), 0, centre_deg - 28 + theta, centre_deg + 28 + theta,
                        mix(ACCENT, (0, 0, 0), 0.35), rim_t, AA)

        hub_r = int(R * 0.30)
        for base in (180.0, 0.0, 90.0):                                              # 3 spokes
            ang = math.radians(base)
            d = (math.cos(ang), math.sin(ang))
            n = (-d[1], d[0])
            r_in, r_out = hub_r * 0.8, R * 0.92
            hw_in, hw_out = R * 0.12, R * 0.07
            quad = [(c + d[0] * r_in + n[0] * hw_in, c + d[1] * r_in + n[1] * hw_in),
                    (c + d[0] * r_out + n[0] * hw_out, c + d[1] * r_out + n[1] * hw_out),
                    (c + d[0] * r_out - n[0] * hw_out, c + d[1] * r_out - n[1] * hw_out),
                    (c + d[0] * r_in - n[0] * hw_in, c + d[1] * r_in - n[1] * hw_in)]
            rq = np.array([ip(rotate_point(q, C, theta)) for q in quad], np.int32)
            cv2.fillPoly(cv, [rq], (112, 106, 100), AA)
            cv2.polylines(cv, [rq], True, (170, 160, 150), 1, AA)

        cv2.circle(cv, (c, c), hub_r, (46, 42, 40), -1, AA)                          # hub
        cv2.circle(cv, (c, c), hub_r, ACCENT, max(1, int(2 * s)), AA)
        cv2.circle(cv, (c, c), int(hub_r * 0.75), (90, 84, 78), 1, AA)
        fs = max(0.3, R / 330.0)
        put_text(cv, "STEER", c, c + int(R * 0.04), fs, TEXT, 1, "c")

        p0 = rotate_point((c, c - R * 0.80), C, theta)                              # centre marker
        p1 = rotate_point((c, c - R * 1.00), C, theta)
        cv2.line(cv, ip(p0), ip(p1), (0, 200, 255), max(3, int(5 * s)), AA)

        Rs = int(R * 1.17)                                                           # scale ring
        mag = abs(theta) / max(max_deg, 1.0)
        arc_col = mix(ACCENT, BAD, clamp((mag - 0.6) / 0.4, 0.0, 1.0))
        if abs(theta) > 0.5:
            a0, a1 = sorted((-90.0, -90.0 + clamp(theta, -90, 90)))
            cv2.ellipse(cv, (c, c), (Rs, Rs), 0, a0, a1, arc_col, max(2, int(4 * s)), AA)
        for deg in range(-90, 91, 15):
            major = deg % 45 == 0 or abs(deg) in (30, 60)
            ln = R * (0.11 if major else 0.05)
            cv2.line(cv, ip(rotate_point((c, c - Rs), C, deg)), ip(rotate_point((c, c - Rs - ln), C, deg)),
                     (200, 200, 200) if major else (120, 118, 114), 1, AA)
        for deg in (-60, -45, -30, 0, 30, 45, 60):
            lp = rotate_point((c, c - Rs - R * 0.24), C, deg)
            put_text(cv, f"{deg}" if deg else "0", int(lp[0]), int(lp[1]) + 4, max(0.28, R / 360.0),
                     DIM if deg else TEXT, 1, "c")
        pt = clamp(theta, -90.0, 90.0)
        tri = np.array([ip(rotate_point(q, C, pt)) for q in
                        ((c, c - Rs + 2), (c - R * 0.06, c - Rs - R * 0.12), (c + R * 0.06, c - Rs - R * 0.12))],
                       np.int32)
        cv2.fillPoly(cv, [tri], (0, 200, 255), AA)

        lit = ctx.steering.direction
        for side in (-1, 1):                                                         # L / R arrows
            xo = c + side * R * 1.43
            xi = c + side * R * 1.31
            tri = np.array([(xo, c), (xi, c - R * 0.08), (xi, c + R * 0.08)], np.int32)
            on = (side < 0 and lit == "LEFT") or (side > 0 and lit == "RIGHT")
            cv2.fillPoly(cv, [tri], ACCENT if on else (74, 70, 66), AA)
        put_text(cv, f"{value * 100:+.0f}%", c, c + int(R * 1.40), max(0.4, R / 230.0), TEXT, 1, "c")

        x0, y0 = int(self.w // 2 - c), int(self.h * 0.58 - c)
        self._blend_canvas(frame, cv, x0, y0, 0.62)

    @staticmethod
    def _blend_canvas(frame: np.ndarray, canvas: np.ndarray, x0: int, y0: int, alpha: float) -> None:
        """Alpha-blend the non-black pixels of ``canvas`` into ``frame`` at (x0, y0)."""
        H, W = frame.shape[:2]
        h, w = canvas.shape[:2]
        fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        sub = canvas[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        roi = frame[fy0:fy1, fx0:fx1]
        mask = sub.any(axis=2)
        blended = cv2.addWeighted(roi, 1.0 - alpha, sub, alpha, 0)
        roi[mask] = blended[mask]

    # ------------------------------------------------------------ panels
    def draw_status(self, frame: np.ndarray, ctx: UIContext, with_steering: bool = True) -> None:
        """Top-left title/state and top-centre steering readout."""
        s = self.s
        x, y = int(12 * s), int(12 * s)
        draw_panel(frame, x, y, int(262 * s), int(62 * s))
        put_text(frame, "VIRTUAL STEERING", x + int(14 * s), y + int(26 * s), 0.58 * s, TEXT, self._th())
        label, color = STATE_STYLE[ctx.app_state]
        pulse = 0.7 + 0.3 * math.sin(self._t * 4.0) if ctx.app_state == AppState.ACTIVE else 1.0
        draw_dot(frame, x + int(20 * s), y + int(46 * s), int(5 * s), mix((0, 0, 0), color, pulse))
        put_text(frame, label, x + int(34 * s), y + int(51 * s), 0.5 * s, color, self._th())
        if not with_steering:
            return
        cw, ch = int(230 * s), int(104 * s)
        cx = (self.w - cw) // 2
        draw_panel(frame, cx, y, cw, ch)
        deg = self._anim.get("steer_deg", ctx.control.steering * self.settings.max_steering_degrees,
                             self._dt, 14.0)
        put_text(frame, "STEERING", cx + cw // 2, y + int(24 * s), 0.46 * s, DIM, 1, "c")
        put_text(frame, f"{deg:+.1f}\u00b0", cx + cw // 2, y + int(64 * s), 1.05 * s, TEXT, self._th(2), "c")
        d = ctx.steering.direction if ctx.steering.active else "---"
        dcol = ACCENT if d in ("LEFT", "RIGHT") else DIM
        put_text(frame, d, cx + cw // 2, y + int(90 * s), 0.55 * s, dcol, self._th(), "c")
        # hand-position pill
        pill_w, pill_h = int(300 * s), int(28 * s)
        px, py = (self.w - pill_w) // 2, y + ch + int(8 * s)
        draw_panel(frame, px, py, pill_w, pill_h, radius=int(14 * s), alpha=0.5)
        col = GOOD if ctx.position_good else WARN
        draw_dot(frame, px + int(16 * s), py + pill_h // 2, int(4 * s), col)
        txt = "HAND POSITION: GOOD" if ctx.position_good else (ctx.position_text or "SHOW BOTH HANDS")
        put_text(frame, txt, px + int(30 * s), py + int(19 * s), 0.46 * s, col, self._th())

    def draw_performance(self, frame: np.ndarray, ctx: UIContext) -> None:
        """Top-right FPS / processing / hands / camera panel."""
        s = self.s
        w, h = int(230 * s), int(98 * s)
        x, y = self.w - w - int(12 * s), int(12 * s)
        draw_panel(frame, x, y, w, h)
        fps = self._anim.get("fps", ctx.fps, self._dt, 4.0)
        ms = self._anim.get("ms", ctx.process_ms, self._dt, 4.0)
        n_hands = sum(1 for hnd in ctx.hands if hnd.visible)
        rows = [("FPS", f"{fps:.1f}", GOOD if fps >= 45 else WARN if fps >= 25 else BAD),
                ("PROCESS", f"{ms:.1f} ms", TEXT),
                ("HANDS", f"{n_hands}/2", GOOD if n_hands == 2 else WARN),
                ("CAMERA", f"{ctx.camera_size[0]}x{ctx.camera_size[1]}", TEXT)]
        for i, (k, v, col) in enumerate(rows):
            yy = y + int((24 + i * 21) * s)
            put_text(frame, k, x + int(14 * s), yy, 0.45 * s, DIM, 1)
            put_text(frame, v, x + w - int(14 * s), yy, 0.5 * s, col, self._th(), "r")
        # side panel: sensitivity / stability / velocity
        sw, sh = int(190 * s), int(150 * s)
        sx, sy = self.w - sw - int(12 * s), y + h + int(10 * s)
        draw_panel(frame, sx, sy, sw, sh)
        st = ctx.steering
        stab = self._anim.get("stab", st.stability, self._dt, 6.0)
        vel = self._anim.get("vel", st.velocity, self._dt, 8.0)
        stab_col = GOOD if st.stability_label == "STABLE" else WARN if st.stability_label == "GOOD" else BAD
        put_text(frame, "SENSITIVITY", sx + int(14 * s), sy + int(22 * s), 0.42 * s, DIM, 1)
        put_text(frame, f"{self.settings.sensitivity:.2f}", sx + sw - int(14 * s), sy + int(22 * s),
                 0.55 * s, TEXT, self._th(), "r")
        put_text(frame, f"{self.settings.preset}  eff {st.sensitivity:.2f}", sx + int(14 * s),
                 sy + int(40 * s), 0.38 * s, DIM, 1)
        put_text(frame, "STABILITY", sx + int(14 * s), sy + int(68 * s), 0.42 * s, DIM, 1)
        put_text(frame, f"{stab:.0f}%", sx + sw - int(14 * s), sy + int(68 * s), 0.55 * s, stab_col,
                 self._th(), "r")
        draw_hbar(frame, sx + int(14 * s), sy + int(76 * s), sw - int(28 * s), int(6 * s), stab / 100.0, stab_col)
        put_text(frame, st.stability_label, sx + int(14 * s), sy + int(98 * s), 0.38 * s, stab_col, 1)
        put_text(frame, "SPEED", sx + int(14 * s), sy + int(124 * s), 0.42 * s, DIM, 1)
        put_text(frame, f"{vel:.1f}\u00b0/s", sx + sw - int(14 * s), sy + int(124 * s), 0.5 * s, TEXT,
                 self._th(), "r")

    def draw_gesture_panel(self, frame: np.ndarray, ctx: UIContext) -> None:
        """Left panel: per-hand gesture, throttle mode."""
        s = self.s
        x, y, w, h = int(12 * s), int(84 * s), int(262 * s), int(86 * s)
        draw_panel(frame, x, y, w, h)
        by = {hnd.handedness: hnd for hnd in ctx.hands}
        modes = {1: "FIST=ACCEL / PALM=BRAKE", 2: "RIGHT HAND ONLY", 3: "GESTURE + KEYS W/S"}
        for i, side in enumerate(("Left", "Right")):
            hnd = by.get(side)
            g = hnd.gesture.value if hnd and hnd.visible else "-"
            put_text(frame, f"{side.upper()}", x + int(14 * s), y + int((24 + i * 20) * s), 0.44 * s, DIM, 1)
            put_text(frame, g, x + w - int(14 * s), y + int((24 + i * 20) * s), 0.48 * s, TEXT, 1, "r")
        put_text(frame, f"MODE {ctx.throttle_mode}: {modes.get(ctx.throttle_mode, '')}", x + int(14 * s),
                 y + int(72 * s), 0.38 * s, ACCENT, 1)

    def draw_hud(self, frame: np.ndarray, ctx: UIContext) -> None:
        """Bottom panels: hand status, steering bar, throttle/brake."""
        s = self.s
        ph = int(96 * s)
        py = self.h - int(34 * s) - ph
        # hands
        x, w = int(12 * s), int(220 * s)
        draw_panel(frame, x, py, w, ph)
        by = {hnd.handedness: hnd for hnd in ctx.hands}
        for i, side in enumerate(("Left", "Right")):
            hnd = by.get(side)
            ok = hnd is not None and hnd.visible
            yy = py + int((30 + i * 40) * s)
            put_text(frame, f"{side.upper()} HAND", x + int(14 * s), yy, 0.5 * s,
                     LEFT_COLOR if side == "Left" else RIGHT_COLOR, self._th())
            draw_dot(frame, x + int(20 * s), yy + int(16 * s), int(4 * s), GOOD if ok else BAD)
            put_text(frame, "DETECTED" if ok else "LOST", x + int(32 * s), yy + int(21 * s), 0.42 * s,
                     GOOD if ok else BAD, 1)
            if ok:
                put_text(frame, f"{hnd.confidence * 100:.0f}%", x + w - int(14 * s), yy + int(21 * s),
                         0.42 * s, DIM, 1, "r")
        # steering bar
        bw = int(self.w * 0.46)
        bx = (self.w - bw) // 2
        draw_panel(frame, bx, py, bw, ph)
        v = self._anim.get("bar", ctx.control.steering, self._dt, 14.0)
        d = ctx.steering.direction if ctx.steering.active else "CENTER"
        put_text(frame, "STEERING", bx + int(16 * s), py + int(24 * s), 0.44 * s, DIM, 1)
        put_text(frame, f"{d} {v * 100:+.0f}%", bx + bw - int(16 * s), py + int(24 * s), 0.55 * s,
                 ACCENT if d != "CENTER" else TEXT, self._th(), "r")
        self.draw_steering_bar(frame, bx + int(24 * s), py + int(52 * s), bw - int(48 * s), v)
        # throttle / brake
        tw = int(220 * s)
        tx = self.w - tw - int(12 * s)
        draw_panel(frame, tx, py, tw, ph)
        thr = self._anim.get("thr", ctx.control.throttle, self._dt, 12.0)
        brk = self._anim.get("brk", ctx.control.brake, self._dt, 12.0)
        for i, (label, val, col) in enumerate((("THROTTLE", thr, GOOD), ("BRAKE", brk, BAD))):
            yy = py + int((28 + i * 38) * s)
            put_text(frame, label, tx + int(14 * s), yy, 0.44 * s, DIM, 1)
            put_text(frame, f"{val * 100:.0f}%", tx + tw - int(14 * s), yy, 0.5 * s, TEXT, self._th(), "r")
            draw_hbar(frame, tx + int(14 * s), yy + int(8 * s), tw - int(28 * s), int(9 * s), val, col)

    def draw_steering_bar(self, frame: np.ndarray, x: int, y: int, w: int, value: float) -> None:
        """LEFT --- CENTER --- RIGHT bar with -100..+100 % ticks and a moving marker."""
        s = self.s
        bh = max(6, int(10 * s))
        cx = x + w // 2
        cv2.rectangle(frame, (x, y), (x + w, y + bh), (46, 42, 40), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + bh), BORDER, 1)
        mx = int(cx + clamp(value, -1.0, 1.0) * w / 2)
        if mx != cx:
            cv2.rectangle(frame, (min(cx, mx), y), (max(cx, mx), y + bh), mix(ACCENT, BAD, abs(value) ** 3), -1)
        step = 1 if (w / 8) >= 34 else 2
        for i in range(-4, 5, step):
            tx = int(cx + i * w / 8)
            cv2.line(frame, (tx, y + bh + 2), (tx, y + bh + 6), DIM, 1, AA)
            put_text(frame, f"{i * 25:+d}" if i else "0", tx, y + bh + int(20 * s), 0.3 * s, DIM, 1, "c")
        cv2.line(frame, (cx, y - 4), (cx, y + bh + 4), TEXT, 1, AA)
        cv2.circle(frame, (mx, y + bh // 2), int(8 * s), (0, 200, 255), -1, AA)
        cv2.circle(frame, (mx, y + bh // 2), int(8 * s), (255, 255, 255), 1, AA)
        put_text(frame, "LEFT", x, y - int(8 * s), 0.38 * s, DIM, 1)
        put_text(frame, "RIGHT", x + w, y - int(8 * s), 0.38 * s, DIM, 1, "r")

    def draw_debug(self, frame: np.ndarray, ctx: UIContext) -> None:
        """Debug values (toggle with D)."""
        s = self.s
        st = ctx.steering
        by = {hnd.handedness: hnd for hnd in ctx.hands}
        lc = by["Left"].confidence if "Left" in by else 0.0
        rc = by["Right"].confidence if "Right" in by else 0.0
        lg = by["Left"].gesture.value if "Left" in by else "-"
        rg = by["Right"].gesture.value if "Right" in by else "-"
        lines = [("raw angle", f"{st.raw_angle:+.2f}"), ("calibrated", f"{st.calibrated_angle:+.2f}"),
                 ("normalized", f"{st.normalized:+.3f}"), ("smoothed", f"{st.smoothed:+.3f}"),
                 ("output", f"{st.output:+.3f}"), ("sensitivity", f"{st.sensitivity:.3f}"),
                 ("velocity", f"{st.velocity:+.1f}"), ("hand dist", f"{st.hand_distance:.3f}"),
                 ("L conf", f"{lc:.2f}"), ("R conf", f"{rc:.2f}"),
                 ("gestures", f"{lg} / {rg}"), ("lost frames", f"{st.lost_frames}")]
        lh = int(18 * s)
        x, y, w = int(12 * s), int(180 * s), int(262 * s)
        draw_panel(frame, x, y, w, lh * len(lines) + int(20 * s), alpha=0.5)
        for i, (k, v) in enumerate(lines):
            yy = y + int(22 * s) + i * lh
            put_text(frame, k.upper(), x + int(12 * s), yy, 0.38 * s, DIM, 1)
            put_text(frame, v, x + w - int(12 * s), yy, 0.4 * s, TEXT, 1, "r")

    def draw_footer(self, frame: np.ndarray) -> None:
        """Keyboard hint line."""
        txt = "C Calibrate   SPACE Pause   X Stop   R Resume   1/2/3 Preset   +/- Sens   D Debug   H HUD   F Full   Q Quit"
        put_text(frame, txt, self.w // 2, self.h - int(12 * self.s), 0.4 * self.s, DIM, 1, "c")

    # ------------------------------------------------------------ overlays
    def draw_calibration(self, frame: np.ndarray, ctx: UIContext) -> None:
        """Calibration panel with progress bar."""
        s = self.s
        pw, ph = int(480 * s), int(200 * s)
        x, y = (self.w - pw) // 2, int(96 * s)
        draw_panel(frame, x, y, pw, ph, alpha=0.78)
        put_text(frame, "STEERING CALIBRATION", x + pw // 2, y + int(36 * s), 0.8 * s, TEXT, self._th(2), "c")
        lines = (ctx.calibration_message or "HOLD HANDS AT CENTER...").split("\n")
        for i, line in enumerate(lines):
            put_text(frame, line, x + pw // 2, y + int((72 + i * 24) * s), 0.58 * s, ACCENT, self._th(), "c")
        prog = self._anim.get("calib", ctx.calibration_progress, self._dt, 10.0)
        bx, by_, bw = x + int(40 * s), y + int(128 * s), pw - int(80 * s)
        draw_hbar(frame, bx, by_, bw, int(16 * s), prog, ACCENT)
        put_text(frame, f"{prog * 100:.0f}%", x + pw // 2, y + int(176 * s), 0.6 * s, TEXT, self._th(), "c")

    def draw_toast(self, frame: np.ndarray, ctx: UIContext) -> None:
        """Short confirmation message with a drawn check mark."""
        s = self.s
        pw, ph = int(380 * s), int(54 * s)
        x, y = (self.w - pw) // 2, int(self.h * 0.20)
        draw_panel(frame, x, y, pw, ph, radius=int(14 * s), alpha=0.8 * ctx.toast_alpha + 0.1,
                   border=GOOD)
        cx, cy = x + int(32 * s), y + ph // 2
        cv2.line(frame, (cx - int(9 * s), cy), (cx - int(2 * s), cy + int(8 * s)), GOOD, self._th(3), AA)
        cv2.line(frame, (cx - int(2 * s), cy + int(8 * s)), (cx + int(11 * s), cy - int(8 * s)), GOOD,
                 self._th(3), AA)
        put_text(frame, ctx.toast, x + int(56 * s), y + int(34 * s), 0.7 * s, GOOD, self._th(2))

    def draw_warning(self, frame: np.ndarray, title: str, subtitle: str, color: Color) -> None:
        """Large banner for pauses, stops and errors."""
        s = self.s
        pw, ph = int(440 * s), int(70 * s)
        x, y = (self.w - pw) // 2, int(self.h * 0.30)
        draw_panel(frame, x, y, pw, ph, radius=int(14 * s), alpha=0.8, border=color)
        put_text(frame, title, x + pw // 2, y + int(36 * s), 0.9 * s, color, self._th(2), "c")
        if subtitle:
            put_text(frame, subtitle, x + pw // 2, y + int(58 * s), 0.42 * s, TEXT, 1, "c")

    def _draw_banners(self, frame: np.ndarray, ctx: UIContext) -> None:
        st = ctx.app_state
        if st == AppState.EMERGENCY_STOP:
            self.draw_warning(frame, "EMERGENCY STOP", "ALL KEYS RELEASED - PRESS R TO RESUME", BAD)
        elif st == AppState.PAUSED:
            self.draw_warning(frame, "PAUSED", "PRESS R / SPACE (OR THUMBS UP) TO RESUME", WARN)
        elif st == AppState.CAMERA_ERROR:
            self.draw_warning(frame, "CAMERA ERROR", ctx.camera_message or "RECONNECTING...", BAD)
        elif st == AppState.ACTIVE and not ctx.steering.active:
            self.draw_warning(frame, "CONTROL PAUSED", "SHOW BOTH HANDS TO RESUME", WARN)
