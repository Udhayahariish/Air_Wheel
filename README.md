<div align="center">

# 🎮 Advanced Virtual Steering Wheel

### Steer any PC driving game with your bare hands, using just a webcam.

![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-00A67E)
![Platform](https://img.shields.io/badge/Platform-Windows%20|%20Linux%20|%20macOS-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

**Euro Truck Simulator 2 • American Truck Simulator • Racing games • Driving simulators**

</div>

---

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| 🎯 | **Continuous steering** | Internal analog value from `-1.0` (left) to `+1.0` (right), not just LEFT/RIGHT/STRAIGHT |
| 🧠 | **Multi-measurement tracking** | Combines the wrist vector, palm-center vector and hand rotation, not just the wrist angle |
| 🎚️ | **Response curve + adaptive sensitivity** | Precise near center, faster for large or fast movements |
| 🧈 | **Smart smoothing** | Speed-adaptive filter (or One Euro Filter) removes jitter without adding lag |
| 🎛️ | **Auto calibration** | Measures your neutral hand position at startup, recalibrate any time |
| 🖐️ | **Gesture control** | Fist, open palm, thumbs up, point and pinch with debounce protection |
| ⚡ | **Analog throttle / brake** | Tighter fist = more throttle (percentage shown on the HUD) |
| 🛡️ | **Safety first** | Hand-loss grace period, emergency stop, keys always released on exit or crash |
| 📷 | **Camera recovery** | Threaded capture with automatic reconnect |
| 🖥️ | **Modern HUD** | Rotating wheel, steering bar, FPS, stability, velocity, debug panel |
| ⚙️ | **Fully configurable** | `settings.json` with validation, so a broken file never crashes the app |
| 🔌 | **Future-proof** | Output backend interface ready for vJoy / virtual gamepad |

---

## 🏗️ How It Works

```
 Webcam
   │
   ▼
 HandTracker (MediaPipe)
   │
   ▼
 GestureEngine ──────────────┐
   │                         │
   ▼                         ▼
 SteeringEngine         Throttle / Brake
   │                         │
   └──────────┬──────────────┘
              ▼
        ControlState
              │
              ▼
        OutputBackend
        ├── ⌨️  Keyboard (pynput)   ← available now
        └── 🕹️  vJoy / Gamepad      ← future
```

**Steering pipeline:**
`hand axis angle → subtract calibrated center → ÷ max degrees → response curve → sensitivity → smoothing → center dead zone → steering (-1…+1)`

> ⚠️ **Note:** Keyboard output is *digital* (key held / released). The steering value is analog internally, but the game receives LEFT/RIGHT key presses. A virtual joystick backend would give true analog steering.

---

## 📁 Project Structure

```
virtual_steering/
├── 📄 main.py               # Entry point, state machine, key handling
├── ⚙️ config.py              # Settings, validation, presets
├── 📷 camera.py              # Threaded capture + reconnect
├── ✋ hand_tracker.py        # MediaPipe → HandState
├── 🖐️ gesture_engine.py      # Gestures, commands, throttle/brake
├── 🎯 steering_engine.py     # Continuous steering model
├── 📐 calibration.py         # Center calibration
├── 🧈 smoothing.py           # Filters, hysteresis, rate limiter
├── ⌨️ input_controller.py    # Output interface + keyboard backend
├── 🎨 ui.py                  # HUD and rotating wheel
├── 🧩 models.py              # Dataclasses and enums
├── 📏 geometry.py            # Math helpers
├── 🛠️ settings.json          # User configuration
├── 📦 requirements.txt
└── 📘 README.md
```

| Module | Responsibility |
|---|---|
| `main.py` | App states (`STARTING → WAITING_FOR_HANDS → CALIBRATING → ACTIVE / PAUSED / EMERGENCY_STOP / CAMERA_ERROR`) |
| `hand_tracker.py` | Detects up to 2 hands, builds `HandState` |
| `gesture_engine.py` | Classifies and debounces gestures, computes pedals |
| `steering_engine.py` | Angle → steering, velocity, stability, hand-position check |
| `input_controller.py` | `InputOutput` interface, `KeyboardOutput`, `NullOutput` |
| `ui.py` | `UIManager`: wheel, HUD panels, hand overlays |

---

## 🚀 Installation

**Requirements:** Python **3.10 – 3.12** and a webcam.

```bash
# 1. Clone
git clone https://github.com/<your-username>/virtual-steering.git
cd virtual-steering

# 2. Virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> 💡 If you see `module 'mediapipe' has no attribute 'solutions'`, run:
> `pip install mediapipe==0.10.14`

---

## ▶️ Running

```bash
python main.py               # normal mode
python main.py --dry-run     # test mode: no keys are sent
python main.py --camera 1    # use another webcam
```

| Option | Description |
|---|---|
| `--dry-run` | Run everything but send no keyboard input |
| `--camera N` | Camera index override |
| `--settings PATH` | Use a custom settings file |

---

## 🎛️ Calibration

1. Start the app and show **both hands** to the camera.
2. Hold them at the **center**, as if on a wheel at 9 and 3 o'clock.
3. Wait about 2 seconds until the progress bar fills: **✅ CALIBRATION COMPLETE**.
4. Press **`C`** at any time to recalibrate.

> If your hands move too much, calibration restarts with `HOLD STEADY...`

---

## ✋ Gesture Controls

| Gesture | Action | Notes |
|---|---|---|
| ✊ + ✊ Both fists | 🟢 **Accelerate** | Tighter fist = more throttle |
| 🖐️ + 🖐️ Both open palms | 🔴 **Brake** | Brake has priority over throttle |
| 👍 Thumbs up (hold 0.35 s) | ▶️ Resume | From pause |
| 👌 Pinch / OK sign (hold 0.35 s) | 🎯 Recalibrate | 3 s cooldown |
| 🖐️ One open palm (hold 1.5 s, other hand hidden) | ⏸️ Pause | |

**Throttle modes** (`throttle_mode` in settings):

| Mode | Behaviour |
|---|---|
| `1` | Both hands: fists = accelerate, open palms = brake |
| `2` | Only the **right hand** gesture controls throttle/brake |
| `3` | Gestures **plus** keyboard latches: `W` toggles throttle, `S` toggles brake |

---

## ⌨️ Keyboard Shortcuts (in the video window)

| Key | Action |
|---|---|
| `C` | 🎛️ Recalibrate |
| `SPACE` | ⏸️ Pause / resume |
| `X` | 🛑 Emergency stop |
| `ESC` | 🛑 Emergency stop (press again to quit) |
| `R` | ▶️ Resume |
| `Q` | ❌ Quit |
| `1` `2` `3` | 🎚️ Sensitivity preset LOW / NORMAL / HIGH |
| `+` `-` | 🔧 Fine-tune sensitivity |
| `H` | 👁️ Toggle HUD |
| `D` | 🐞 Toggle debug panel |
| `F` | 🖥️ Toggle fullscreen |
| `M` | 🪞 Toggle camera mirror (recalibrates) |
| `W` / `S` | 🅿️ Throttle / brake latch (mode 3 only) |

---

## 🎚️ Sensitivity Presets

| Preset | Key | Sensitivity | Curve | Feel |
|---|---|---|---|---|
| 🐢 LOW | `1` | 0.85 | 1.15 | Gentle, very precise |
| ⚖️ NORMAL | `2` | 1.25 | 0.85 | Balanced (default) |
| 🚀 HIGH | `3` | 1.60 | 0.70 | Aggressive, fast response |

---

## ⚙️ Configuration (`settings.json`)

The file is created automatically. Invalid values silently fall back to defaults.

| Setting | Default | Range / Options | Description |
|---|---|---|---|
| `camera_index` | `0` | 0–16 | Webcam index |
| `camera_width` / `camera_height` | `640` / `480` | | Capture resolution |
| `fps` | `60` | 5–240 | Requested camera FPS |
| `display_scale` | `1.5` | 0.5–3.0 | Window size multiplier |
| `max_steering_degrees` | `65` | 20–120 | Hand angle that equals full lock |
| `sensitivity` | `1.25` | 0.1–3.0 | Base sensitivity |
| `steering_curve` | `0.85` | 0.3–2.0 | Below 1 = stronger response |
| `center_deadzone` | `0.025` | 0–0.2 | Small center zone (with hysteresis) |
| `smoothing` | `0.18` | 0.01–0.9 | Smoothing strength |
| `filter_mode` | `adaptive` | `adaptive`, `one_euro` | Smoothing algorithm |
| `adaptive_sensitivity` | `true` | bool | Boost sensitivity on fast movement |
| `auto_center` | `false` | bool | Gently pull toward center |
| `steer_start_threshold` | `0.035` | | Key press threshold |
| `steer_release_threshold` | `0.020` | | Key release threshold (hysteresis) |
| `lost_hand_grace` | `0.20` | 0–2 s | Hold steering briefly when hands vanish |
| `calibration_seconds` | `2.0` | 0.5–10 | Calibration duration |
| `mirror_camera` | `true` | bool | Mirror the image |
| `show_hud` / `show_landmarks` / `show_debug` | `true` / `true` / `false` | bool | Display options |
| `model_complexity` | `1` | 0, 1 | `0` is faster, `1` is more accurate |
| `min_detection_confidence` / `min_tracking_confidence` | `0.65` | 0.1–1.0 | MediaPipe thresholds |
| `gesture_control` | `true` | bool | Enable gestures |
| `gesture_hold_time` | `0.35` | 0.05–3 s | Hold time for commands |
| `throttle_mode` | `1` | 1–3 | See "Throttle modes" |
| `output_mode` | `keyboard` | `keyboard` (others fall back) | Output backend |
| `key_map` | arrows | key names | Keys for left, right, accelerate, brake |

**Example `key_map` for WASD:**
```json
"key_map": { "left": "a", "right": "d", "accelerate": "w", "brake": "s" }
```

---

## 🚛 Using with ETS2 / ATS

1. In the game, go to **Options → Controls** and choose **Keyboard + Mouse**.
2. Start `python main.py` and finish calibration.
3. **Click the game window** so it has focus (keys go to the focused window).
4. Rotate your hands to steer, make two fists to accelerate and open both palms to brake.

> 💡 Tip: raise the in-game steering sensitivity and enable steering assist, because keyboard steering is digital.

---

## 🛡️ Safety Features

| Situation | Behaviour |
|---|---|
| Hands leave the camera | Steering held for 0.2 s, then everything resets to neutral and all keys are released |
| Emergency stop (`X` / `ESC`) | Releases all keys instantly, zeroes all controls |
| Camera failure | Releases keys, shows `CAMERA ERROR`, retries 5 times, then exits safely |
| Crash / exception / Ctrl+C | `finally` block and `atexit` hook release every key |
| Hands return after loss | Steering ramps up from 0, with no sudden full lock |

---

## 🩺 Troubleshooting

| Problem | Solution |
|---|---|
| 📷 Camera won't open | Close other apps using the webcam, try `--camera 1` |
| ⌨️ Keys don't reach the game | Run terminal and game as administrator (Windows) |
| 🍎 macOS: no key output | System Settings → Privacy & Security → Accessibility → allow your terminal |
| 🐧 Linux: no key output | Use an X11 session (pynput doesn't work on pure Wayland) |
| ↔️ Steering is inverted | Press `M`, then `C` to recalibrate |
| 📉 Jittery steering | Improve lighting, raise `smoothing`, or set `"filter_mode": "one_euro"` |
| 🐌 Low FPS | See "Performance Tips" below |
| 🖐️ Gestures misfire | Show a flat, clear hand; tune thresholds at the top of `gesture_engine.py` |

---

## ⚡ Performance Tips

- Keep the default **640×480** resolution (1280×720 only on strong hardware).
- Set `"model_complexity": 0` for faster tracking.
- Lower `display_scale` to `1.0`.
- Use **good front lighting** because webcams drop FPS in dim rooms.
- Close heavy background apps.

---

## 🗺️ Roadmap

- [x] Continuous steering model
- [x] Calibration, gestures, safety systems, HUD
- [ ] 🕹️ vJoy / `vgamepad` analog output backend
- [ ] 🧩 Tkinter settings window
- [ ] 🌐 Per-game profiles
- [ ] 🔊 Audio feedback

### 🔌 Adding a new output backend

Implement `InputOutput` in `input_controller.py`:

```python
class VJoyOutput(InputOutput):
    def set_steering(self, value): ...   # -1.0 .. +1.0  -> X axis
    def set_throttle(self, value): ...   #  0.0 .. 1.0   -> axis
    def set_brake(self, value): ...      #  0.0 .. 1.0   -> axis
    def release_all(self, force=False): ...
```

Then return it from `create_output()`. Tracking, gestures and steering need no changes.

---

## 🤝 Contributing

Contributions are welcome!
1. 🍴 Fork the repo
2. 🌿 Create a branch: `git checkout -b feature/my-feature`
3. 💾 Commit: `git commit -m "Add my feature"`
4. 📤 Push and open a **Pull Request**

---

## 📜 License

Released under the **MIT License**. See [`LICENSE`](LICENSE).

---

## 🙏 Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) for hand tracking
- [OpenCV](https://opencv.org/) for vision and rendering
- [pynput](https://github.com/moses-palmer/pynput) for keyboard control

<div align="center">


</div>
