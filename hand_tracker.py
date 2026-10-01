"""MediaPipe Hands wrapper that produces HandState objects."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import Settings
from models import HandState

PALM_IDS = (0, 5, 9, 13, 17)
TIP_IDS = (4, 8, 12, 16, 20)


class HandTracker:
    """Detects up to two hands and converts them to :class:`HandState`."""

    def __init__(self, settings: Settings) -> None:
        import mediapipe as mp  # imported lazily so other modules can be tested without it
        if not hasattr(mp, "solutions"):
            raise RuntimeError("This MediaPipe build has no legacy 'solutions' API. "
                               "Install: pip install mediapipe==0.10.14")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=settings.model_complexity,
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray, mirrored: bool) -> List[HandState]:
        """Detect hands in the (already mirrored, if enabled) frame.

        Returns hands sorted by wrist x (image-left first).  ``handedness`` is
        expressed from the user's point of view.
        """
        h, w = frame_bgr.shape[:2]
        aspect = w / float(h)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []
        hands: List[HandState] = []
        for i, lms in enumerate(result.multi_hand_landmarks):
            cls = result.multi_handedness[i].classification[0]
            label = cls.label
            if not mirrored:                      # MediaPipe assumes a selfie-mirrored image
                label = "Left" if label == "Right" else "Right"
            hands.append(self._build_state(lms, float(cls.score), label, aspect))
        hands.sort(key=lambda hs: hs.wrist[0])
        if len(hands) >= 2:                       # position is more reliable than the label
            hands = hands[:2]
            hands[0].handedness = "Left" if mirrored else "Right"
            hands[1].handedness = "Right" if mirrored else "Left"
        return hands

    @staticmethod
    def _build_state(landmarks, confidence: float, label: str, aspect: float) -> HandState:
        pts = np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=np.float32)
        ar = pts.copy()
        ar[:, 0] *= aspect
        palm = pts[list(PALM_IDS)].mean(axis=0)
        vec = ar[9] - ar[0]
        tip = lambda idx: (float(pts[idx][0]), float(pts[idx][1]))
        return HandState(
            handedness=label,
            wrist=tip(0), thumb=tip(4), index=tip(8), middle=tip(12), ring=tip(16), pinky=tip(20),
            palm_center=(float(palm[0]), float(palm[1])),
            confidence=confidence, visible=True,
            size=float(np.linalg.norm(vec)),
            axis_angle=math.degrees(math.atan2(float(vec[1]), float(vec[0]))),
            landmarks=pts, pts=ar,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._hands.close()
        except Exception:
            pass


def split_hands(hands: List[HandState]) -> Tuple[Optional[HandState], Optional[HandState]]:
    """Return (image_left_hand, image_right_hand) or (None, None) unless both are visible."""
    if len(hands) >= 2:
        return hands[0], hands[1]
    return None, None
