"""Advanced Virtual Steering Wheel - application entry point and state machine."""
from __future__ import annotations

import argparse
import dataclasses
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from calibration import Calibrator
from camera import Camera
from config import (PRESETS, SETTINGS_PATH, WINDOW_NAME, Settings, apply_preset, load_settings,
                    save_settings)
from geometry import clamp
from gesture_engine import GestureCommand, GestureEngine
from hand_tracker import HandTracker, split_hands
from input_controller import InputOutput, create_output
from models import AppState, ControlState, SteeringState
from smoothing import ExpAverage
from steering_engine import PositionMonitor, SteeringEngine, assess_position
from ui import UIContext, UIManager

MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_INTERVAL = 1.0
WAIT_SETTLE_SECONDS = 0.3
TOAST_SECONDS = 1.8
SENSITIVITY_STEP = 0.05
BANNER = """========================================
   ADVANCED VIRTUAL STEERING WHEEL
========================================
"""


class VirtualSteeringApp:
    """Wires camera -> tracker -> gestures -> steering -> output -> UI."""

    def __init__(self, settings: Settings, settings_path: Path, dry_run: bool = False) -> None:
        self.settings = settings
        self.settings_path = settings_path
        self.dry_run = dry_run
        self._initial_settings = dataclasses.asdict(settings)
        self.state = AppState.STARTING
        self.control = ControlState()
        self.steering = SteeringState()
        self.camera: Optional[Camera] = None
        self.tracker: Optional[HandTracker] = None
        self.output: Optional[InputOutput] = None
        self.gestures = GestureEngine(settings)
        self.engine = SteeringEngine(settings)
        self.calibrator = Calibrator(settings)
        self.position = PositionMonitor()
        self.ui = UIManager(settings)
        self._fps = ExpAverage(0.1)
        self._proc_ms = ExpAverage(0.1)
        self._calibrated = False
        self._both_since: Optional[float] = None
        self._toast = ""
        self._toast_until = 0.0
        self._latch_throttle = False
        self._latch_brake = False
        self._reconnect_attempts = 0
        self._next_retry = 0.0
        self._camera_message = ""
        self._printed_ready = False
        self._frames = 0
        self._position_text = ""
        self._center_marker = None

    # ------------------------------------------------------------ lifecycle
    def run(self) -> int:
        """Run the application; always releases every key on exit."""
        code = 0
        try:
            self._startup()
            self._loop()
        except KeyboardInterrupt:
            pass
        except Exception:
            code = 1
            traceback.print_exc()
        finally:
            self._shutdown()
        return code

    def _startup(self) -> None:
        print(BANNER)
        signal.signal(signal.SIGINT, self._on_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._on_signal)
        self.output, notice = create_output(self.settings, self.dry_run)
        if notice:
            print(f"Input: {notice}")
        self.camera = Camera(self.settings)
        if not self.camera.open():
            print("Camera: ERROR - could not open camera index", self.settings.camera_index)
            raise RuntimeError("camera unavailable")
        print(f"Camera: READY ({self.camera.resolution[0]}x{self.camera.resolution[1]})")
        self.tracker = HandTracker(self.settings)
        print("MediaPipe: READY\n\nPlace both hands in front of camera.\n")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if self.settings.fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.state = AppState.WAITING_FOR_HANDS

    def _shutdown(self) -> None:
        """Release keys first, then everything else (each step is guarded)."""
        self.state = AppState.EXITING
        if self.output is not None:
            try:
                self.output.release_all(force=True)
            except Exception:
                pass
        for closer in (lambda: self.tracker and self.tracker.close(),
                       lambda: self.camera and self.camera.release(),
                       cv2.destroyAllWindows):
            try:
                closer()
            except Exception:
                pass
        if dataclasses.asdict(self.settings) != self._initial_settings:
            save_settings(self.settings, self.settings_path)
        print("Shutdown complete - all keys released.")

    def _on_signal(self, signum, frame) -> None:  # noqa: ARG002
        self.state = AppState.EXITING

    # ------------------------------------------------------------ main loop
    def _loop(self) -> None:
        prev = time.perf_counter()
        while self.state != AppState.EXITING:
            start = time.perf_counter()
            dt = clamp(start - prev, 1e-3, 0.1)
            prev = start
            self._fps.update(1.0 / dt)
            ok, frame = self.camera.read()
            if ok and frame is not None:
                if self.state == AppState.CAMERA_ERROR:
                    self._on_camera_recovered()
                display = self._process_frame(frame, start, dt)
                self._proc_ms.update((time.perf_counter() - start) * 1000.0)
            else:
                display = self._handle_camera_failure(start)
            if self.state == AppState.EXITING:
                break
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)
            self._frames += 1
            if self._frames % 30 == 0 and self._window_closed():
                self.state = AppState.EXITING

    @staticmethod
    def _window_closed() -> bool:
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    # ------------------------------------------------------------ camera failure
    def _handle_camera_failure(self, now: float) -> np.ndarray:
        if self.state != AppState.CAMERA_ERROR:
            self.state = AppState.CAMERA_ERROR
            self._neutral()
            self._reconnect_attempts = 0
            self._next_retry = now
            print("Camera: ERROR - attempting to reconnect...")
        if now >= self._next_retry:
            self._reconnect_attempts += 1
            self._camera_message = f"RECONNECTING ({self._reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})"
            if self.camera.reconnect():
                return self._black_canvas()
            self._next_retry = now + RECONNECT_INTERVAL
            if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                print("Camera: could not reconnect - exiting safely.")
                self.state = AppState.EXITING
        canvas = self._black_canvas()
        ctx = self._context([], now)
        return self.ui.render(canvas, ctx)

    def _on_camera_recovered(self) -> None:
        print("Camera: READY (reconnected)")
        self.state = AppState.WAITING_FOR_HANDS
        self._both_since = None
        self.calibrator.start()

    def _black_canvas(self) -> np.ndarray:
        w, h = self.camera.resolution
        sc = self.settings.display_scale
        return np.zeros((int(h * sc), int(w * sc), 3), np.uint8)

    # ------------------------------------------------------------ per-frame processing
    def _process_frame(self, frame: np.ndarray, now: float, dt: float) -> np.ndarray:
        s = self.settings
        h, w = frame.shape[:2]
        if s.mirror_camera:
            frame = cv2.flip(frame, 1)
        hands = self.tracker.process(frame, s.mirror_camera)
        self.gestures.process_hands(hands, now)
        img_left, img_right = split_hands(hands)
        self.engine.set_aspect(w / float(h))
        self.engine.mirrored = s.mirror_camera
        report = self.position.update(
            assess_position(img_left, img_right, w / float(h), s.min_tracking_confidence), now)
        self._position_text = report.message

        self._run_state_machine(hands, img_left, img_right, now, dt)

        if s.display_scale != 1.0:
            frame = cv2.resize(frame, None, fx=s.display_scale, fy=s.display_scale,
                               interpolation=cv2.INTER_LINEAR)
        return self.ui.render(frame, self._context(hands, now, report.good))

    def _run_state_machine(self, hands, img_left, img_right, now: float, dt: float) -> None:
        both = img_left is not None and img_right is not None
        aspect = self.engine.aspect
        commands = self.gestures.poll_commands(hands, now)

        if self.state == AppState.STARTING:
            self.state = AppState.WAITING_FOR_HANDS

        elif self.state == AppState.WAITING_FOR_HANDS:
            self._neutral()
            if both:
                self._both_since = self._both_since or now
                if now - self._both_since >= WAIT_SETTLE_SECONDS:
                    self.calibrator.start()
                    self.state = AppState.CALIBRATING
            else:
                self._both_since = None

        elif self.state == AppState.CALIBRATING:
            self._neutral()
            done = self.calibrator.update(img_left, img_right, aspect, now)
            if done and self.calibrator.result is not None:
                self.engine.calibrate(self.calibrator.result)
                self._center_marker = self.calibrator.result.center_position
                self._calibrated = True
                self._toast_text("CALIBRATION COMPLETE", now)
                self.state = AppState.ACTIVE
                if not self._printed_ready:
                    print("SYSTEM READY\n\nSteering: ACTIVE\nThrottle: ACTIVE\n")
                    self._printed_ready = True
            elif not both and self.calibrator.status == "SHOW BOTH HANDS":
                self.state = AppState.WAITING_FOR_HANDS
                self._both_since = None

        elif self.state == AppState.ACTIVE:
            self._run_active(hands, img_left, img_right, dt)
            if GestureCommand.PAUSE in commands:
                self._pause()
            elif GestureCommand.RECALIBRATE in commands:
                self.begin_calibration()

        elif self.state == AppState.PAUSED:
            self._neutral()
            if GestureCommand.RESUME in commands:
                self.resume()

        elif self.state == AppState.EMERGENCY_STOP:
            self._neutral()

    def _run_active(self, hands, img_left, img_right, dt: float) -> None:
        self.steering = self.engine.update(img_left, img_right, dt)
        both = img_left is not None and img_right is not None
        throttle, brake = self.control.throttle, self.control.brake
        if both:
            throttle, brake = self.gestures.compute_pedals(hands, dt, self._latch_throttle,
                                                           self._latch_brake)
        elif not self.engine.in_grace:               # hand-loss grace expired -> neutral
            self.gestures.reset_pedals()
            throttle = brake = 0.0
        if not self.steering.active:
            throttle = brake = 0.0
        self.control = ControlState(
            steering=self.steering.output if self.steering.active else 0.0,
            throttle=throttle, brake=brake)
        self.output.set_steering(self.control.steering)
        self.output.set_throttle(self.control.throttle)
        self.output.set_brake(self.control.brake)

    def _neutral(self) -> None:
        """Zero all controls and release every key."""
        paused = self.state == AppState.PAUSED
        stopped = self.state == AppState.EMERGENCY_STOP
        self.control = ControlState(paused=paused, emergency_stop=stopped)
        self.steering = SteeringState()
        self.gestures.reset_pedals()
        if self.output is not None:
            self.output.release_all()

    # ------------------------------------------------------------ commands
    def begin_calibration(self) -> None:
        """Start a fresh calibration."""
        self._neutral()
        self.engine.reset()
        self.calibrator.start()
        self._both_since = None
        self.state = AppState.WAITING_FOR_HANDS

    def _pause(self) -> None:
        self.state = AppState.PAUSED
        self._neutral()

    def resume(self) -> None:
        """Resume from PAUSED / EMERGENCY_STOP."""
        if self.state in (AppState.PAUSED, AppState.EMERGENCY_STOP):
            self.engine.reset()
            self._latch_throttle = self._latch_brake = False
            if self._calibrated:
                self.state = AppState.ACTIVE
            else:
                self.begin_calibration()

    def emergency_stop(self) -> None:
        """Release all keys and zero all controls immediately."""
        self.state = AppState.EMERGENCY_STOP
        self._neutral()
        if self.output is not None:
            self.output.release_all(force=True)

    def _toast_text(self, text: str, now: float) -> None:
        self._toast = text
        self._toast_until = now + TOAST_SECONDS

    def _handle_key(self, key: int) -> None:
        if key == 255:
            return
        s = self.settings
        now = time.perf_counter()
        ch = chr(key).lower() if 32 <= key < 127 else ""
        if key == 27:                                   # ESC: stop, second press quits
            if self.state == AppState.EMERGENCY_STOP:
                self.state = AppState.EXITING
            else:
                self.emergency_stop()
        elif ch == "q":
            self.state = AppState.EXITING
        elif ch == "x":
            self.emergency_stop()
        elif ch == "c":
            self.begin_calibration()
        elif ch == "r":
            self.resume()
        elif ch == " ":
            if self.state == AppState.ACTIVE:
                self._pause()
            elif self.state == AppState.PAUSED:
                self.resume()
        elif ch in ("+", "="):
            s.sensitivity = round(clamp(s.sensitivity + SENSITIVITY_STEP, 0.1, 3.0), 2)
            s.preset = "CUSTOM"
        elif ch == "-":
            s.sensitivity = round(clamp(s.sensitivity - SENSITIVITY_STEP, 0.1, 3.0), 2)
            s.preset = "CUSTOM"
        elif ch in ("1", "2", "3"):
            apply_preset(s, ("LOW", "NORMAL", "HIGH")[int(ch) - 1])
            self._toast_text(f"SENSITIVITY: {s.preset}", now)
        elif ch == "h":
            s.show_hud = not s.show_hud
        elif ch == "d":
            s.show_debug = not s.show_debug
        elif ch == "f":
            s.fullscreen = not s.fullscreen
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if s.fullscreen else cv2.WINDOW_NORMAL)
        elif ch == "m":
            s.mirror_camera = not s.mirror_camera
            self.begin_calibration()
        elif ch == "w" and s.throttle_mode == 3:
            self._latch_throttle = not self._latch_throttle
            self._latch_brake = False
        elif ch == "s" and s.throttle_mode == 3:
            self._latch_brake = not self._latch_brake
            self._latch_throttle = False

    # ------------------------------------------------------------ UI context
    def _context(self, hands, now: float, position_good: bool = False) -> UIContext:
        if self.state == AppState.WAITING_FOR_HANDS:
            message = f"Place both hands in the center\nHold for {self.settings.calibration_seconds:g} seconds"
        else:
            message = self.calibrator.status
        remaining = self._toast_until - now
        return UIContext(
            app_state=self.state, steering=self.steering, control=self.control, hands=hands,
            fps=self._fps.value, process_ms=self._proc_ms.value,
            camera_size=self.camera.resolution if self.camera else (0, 0),
            position_text=self._position_text, position_good=position_good,
            calibration_progress=self.calibrator.progress if self.state == AppState.CALIBRATING else 0.0,
            calibration_message=message, toast=self._toast if remaining > 0 else "",
            toast_alpha=clamp(remaining / 0.4, 0.0, 1.0), camera_message=self._camera_message,
            center_marker=self._center_marker if self._calibrated else None,
            throttle_mode=self.settings.throttle_mode,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Advanced AI hand-tracking virtual steering wheel")
    parser.add_argument("--settings", default=str(SETTINGS_PATH), help="path to settings.json")
    parser.add_argument("--camera", type=int, default=None, help="camera index override")
    parser.add_argument("--dry-run", action="store_true", help="do not send any keyboard input")
    args = parser.parse_args(argv)
    path = Path(args.settings)
    settings = load_settings(path)
    if args.camera is not None:
        settings.camera_index = args.camera
    return VirtualSteeringApp(settings, path, dry_run=args.dry_run).run()


if __name__ == "__main__":
    sys.exit(main())
