"""Threaded webcam capture with failure detection and reconnection."""
from __future__ import annotations

import sys
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from config import Settings

MAX_CONSECUTIVE_FAILURES = 30


class Camera:
    """Reads frames on a background thread so processing never waits on I/O."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._new_frame = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._failed = False
        self.resolution: Tuple[int, int] = (settings.camera_width, settings.camera_height)

    def open(self) -> bool:
        """Open the camera and start the capture thread."""
        self.release()
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._s.camera_index, backend)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._s.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._s.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self._s.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, first = cap.read()
        if not ok or first is None:
            cap.release()
            return False
        self.resolution = (first.shape[1], first.shape[0])
        self._cap = cap
        self._failed = False
        self._stop.clear()
        self._new_frame.clear()
        with self._lock:
            self._frame = first
        self._new_frame.set()
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()
        return True

    def _reader(self) -> None:
        failures = 0
        cap = self._cap
        while not self._stop.is_set() and cap is not None:
            ok, frame = cap.read()
            if ok and frame is not None:
                failures = 0
                with self._lock:
                    self._frame = frame
                self._new_frame.set()
            else:
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    self._failed = True
                    self._new_frame.set()
                    return
                self._stop.wait(0.01)

    def read(self, timeout: float = 0.5) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the newest frame; (False, None) if the camera stalled/failed."""
        if self._failed or self._cap is None:
            return False, None
        if not self._new_frame.wait(timeout):
            return False, None
        self._new_frame.clear()
        if self._failed:
            return False, None
        with self._lock:
            frame = self._frame
        return (frame is not None), frame

    def reconnect(self) -> bool:
        """Try to re-open the camera."""
        return self.open()

    def release(self) -> None:
        """Stop the thread and release the device."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
