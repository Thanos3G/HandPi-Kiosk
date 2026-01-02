# main_pi.py (Raspberry Pi headless entrypoint)
#
# CAMERA DESIGN NOTES (Raspberry Pi):
#
# This file is designed to work on Raspberry Pi OS (Bookworm) with BOTH:
#   1) USB webcams (V4L2, OpenCV VideoCapture)
#   2) Official Raspberry Pi CSI ribbon cameras (libcamera / Picamera2)
#
# How camera selection works:
# - The script FIRST tries USB cameras using OpenCV (V4L2).
# - If no USB camera works, it FALLS BACK to Picamera2 for CSI cameras.
# - This avoids hardcoding camera indices and prevents boot crash loops.
#
# Environment constraints:
# - MediaPipe runs inside the Python virtual environment.
# - Picamera2 is provided by the system (apt-installed).
# - If Picamera2 is unavailable inside the venv, hands are disabled.
#
# Uses: MediaPipe hand tracking + evdev keyboard, menu is rendered via mpv OSD

import argparse
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from video_core import (
    ActionQueue,
    MPVController,
    VideoApp,
    discover_videos,
    is_windows,
    now,
    to_forward_slashes,
)

# Menu behavior (how long it stays open + how often it refreshes)
MENU_IDLE_CLOSE_S = 22.0
MENU_REFRESH_S = 1.0

# Menu selection stability
SELECTION_DEBOUNCE_S = 0.20
SELECTION_STABLE_SAMPLES = 3
MENU_TOUCH_FROM_POINTING = True

# Volume step
VOLUME_STEP = 5

OSD = {
    "MARGIN_X": 28,
    "MARGIN_Y": 22,
    "FONT_SIZE": 26,
    "BORDER_SIZE": 2,
    "CHAR_W": 0.58,
    "MAX_CHARS": 44,
    "LINE_H": 1.28,
    "EXTRA_PAD_X": 18,
    "EXTRA_PAD_Y": 12,
    "LIST_TOP_LINES": 6,
    "LIST_BOTTOM_PADDING_LINES": 9,
}


def render_menu_main(main_label: str, items: List[str], selected: int) -> str:
    lines: List[str] = []
    lines.append("◆ PLAYLIST ◆")
    lines.append("")
    lines.append(f"MAIN: {main_label}")
    lines.append("")
    if not items:
        lines.append("No secondary videos found.")
        lines.append("Upload into: videos/secondary/")
        lines.append("")
    else:
        lines.append("Secondary:")
        for i, name in enumerate(items):
            prefix = "➤ " if i == selected else "  "
            lines.append(f"{prefix}{i+1:02d}. {name}")
        lines.append("")

    lines.append("◆ HOT KEYS ◆")
    lines.append("M: Menu")
    lines.append("Enter: Play selected")
    lines.append("Esc: Back to MAIN")
    lines.append("N: Next Video")
    lines.append("Space: Pause/Resume")
    lines.append("Left/Right: Seek")
    lines.append("Q: Quit")
    lines.append("Up/Down: Move selection")
    lines.append("+/-: Volume (SECONDARY)")
    lines.append("")
    lines.append("◆ HAND ◆")
    lines.append("Thumb+Index: Toggle menu (MAIN) / Return MAIN")
    lines.append("Thumb+Middle: Scroll list (MAIN) / Volume Up")
    lines.append("Thumb+Ring: Play selected (MAIN) / Volume Down")
    lines.append("Thumb+Pinky: Next video")
    lines.append("Palm: Pause/Resume")
    return "\n".join(lines)


# Gesture ratios are relative to "hand scale" so they work at different distances from camera
GESTURE = {
    "PINCH_RATIO": 0.24,
    "SCROLL_RATIO": 0.30,
    "PLAY_RATIO": 0.26,
    "NEXT_RATIO": 0.34,
    "PALM_RATIO": 1.65,
    # Frames required before a gesture is considered "stable" (reduces flicker)
    "PINCH_FRAMES": 6,
    "SCROLL_FRAMES": 4,
    "PLAY_FRAMES": 5,
    "NEXT_FRAMES": 5,
    "PALM_FRAMES": 9,
}

# Cooldowns prevent repeated triggers from one sustained pose
COOLDOWN_S = {
    "palm": 1.0,
    "pinch": 0.9,
    "next": 0.7,
    "play": 0.6,
    "back": 0.8,
    "scroll": 0.55,
    "vol_up": 0.18,
    "vol_down": 0.18,
}

# Hold times: gesture must be held this long before firing (tunes responsiveness vs false positives)
HOLD_S = {
    "pinch": 0.22,
    "scroll": 0.16,
    "play": 0.16,
    "next": 0.18,
    "palm": 0.30,
}

# Scroll direction from pointing: above/below MID_Y with hysteresis to avoid rapid flipping
SCROLL = {"MID_Y": 0.50, "HYST": 0.06}

# Pointer smoothing for index fingertip position (lower ALPHA = smoother, higher = more responsive)
POINTER = {"ALPHA": 0.22, "DEADZONE": 0.004, "MAX_STEP": 0.06}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

CAM = {"WIDTH": 640, "HEIGHT": 480, "FPS": 30, "SCAN_MAX_INDEX": 12}


def mono_ms() -> int:
    return int(time.monotonic() * 1000)


def ensure_hand_model(model_path: Path, url: str = MODEL_URL) -> Path:
    # Downloads the MediaPipe model once; on subsequent boots it's reused if size looks valid
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        return model_path

    tmp = model_path.with_suffix(model_path.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    print(f"Downloading hand model to: {model_path}", flush=True)
    last_err: Optional[Exception] = None
    for attempt in range(1, 6):
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.exists() and tmp.stat().st_size > 1_000_000:
                tmp.replace(model_path)
                print("Model download complete.", flush=True)
                return model_path
            raise RuntimeError("Downloaded file too small/corrupted.")
        except Exception as e:
            last_err = e
            print(f"Attempt {attempt}/5 failed: {repr(e)}", flush=True)
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to download model after retries: {last_err}")


def _try_open_camera_v4l2(index: int) -> bool:
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM["WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM["HEIGHT"])
    cap.set(cv2.CAP_PROP_FPS, CAM["FPS"])
    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None)


def _find_working_v4l2_index(preferred: Optional[int]) -> Optional[int]:
    if preferred is not None and preferred >= 0:
        return preferred if _try_open_camera_v4l2(preferred) else None

    for i in range(int(CAM["SCAN_MAX_INDEX"]) + 1):
        if _try_open_camera_v4l2(i):
            return i
    return None


def _picamera2_available() -> bool:
    try:
        import picamera2  
        return True
    except Exception:
        return False


def choose_camera_backend(preferred: Optional[int]) -> Tuple[str, Optional[int]]:
    v4l2_idx = _find_working_v4l2_index(preferred)
    if v4l2_idx is not None:
        return ("v4l2", v4l2_idx)

    if _picamera2_available():
        return ("picamera2", None)

    raise RuntimeError(
        f"No working V4L2 camera found (tried indices 0..{CAM['SCAN_MAX_INDEX']}) and Picamera2 not available. "
        "For CSI ribbon camera install: sudo apt install -y python3-picamera2"
    )


# ---------------------------
# AUDIO 
# ---------------------------
#   - If a USB playback card is detected, we force mpv to use ALSA on that card
#   - Otherwise we at least force --ao=alsa (so HDMI/analog can still work)
#
def detect_usb_playback_card() -> Optional[int]:
    try:
        p = Path("/proc/asound/cards")
        if not p.exists():
            return None
        txt = p.read_text(errors="ignore")
        for line in txt.splitlines():
            s = line.strip()
            if not s or not s[0].isdigit():
                continue
            idx_str = s.split()[0]
            if ("USB" in s) or ("USB-Audio" in s) or ("USB Audio" in s):
                try:
                    return int(idx_str)
                except Exception:
                    pass
        return None
    except Exception:
        return None


class HoldGate:
    def __init__(self, hold_s: float, repeat: bool = False, repeat_every_s: float = 0.18, release_frames: int = 4):
        self.hold_s = float(hold_s)
        self.repeat = bool(repeat)
        self.repeat_every_s = float(repeat_every_s)
        self.release_frames = int(release_frames)
        self.t0: Optional[float] = None
        self.fired = False
        self.next_repeat_t = 0.0
        self._release_count = self.release_frames

    def reset(self) -> None:
        self.t0 = None
        self.fired = False
        self.next_repeat_t = 0.0
        self._release_count = self.release_frames

    def update(self, is_true: bool) -> bool:
        t = now()
        if not is_true:
            self._release_count = min(self.release_frames, self._release_count + 1)
            if self._release_count >= self.release_frames:
                self.t0 = None
                self.fired = False
                self.next_repeat_t = 0.0
            return False

        self._release_count = 0

        if self.t0 is None:
            self.t0 = t
            return False

        if (t - self.t0) < self.hold_s:
            return False

        if not self.fired:
            self.fired = True
            self.next_repeat_t = t + self.repeat_every_s
            return True

        if self.repeat and t >= self.next_repeat_t:
            self.next_repeat_t = t + self.repeat_every_s
            return True

        return False


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])


class HandControl(threading.Thread):
    def __init__(
        self,
        actions: ActionQueue,
        model_path: Path,
        *,
        backend: str = "v4l2",
        camera_index: int = 0,
        flip: bool = True,
    ):
        super().__init__(daemon=True)
        self.actions = actions
        self.running = True
        self.flip = bool(flip)
        self.backend = str(backend)

        self.cooldown_until: Dict[str, float] = {
            "palm": 0.0,
            "pinch": 0.0,
            "next": 0.0,
            "play": 0.0,
            "back": 0.0,
            "scroll": 0.0,
            "vol_up": 0.0,
            "vol_down": 0.0,
        }

        self.alpha = float(POINTER["ALPHA"])
        self.deadzone = float(POINTER["DEADZONE"])
        self.max_step = float(POINTER["MAX_STEP"])
        self.cx, self.cy = 0.5, 0.5

        self._scroll_dir = +1
        self._ignore_volume_until = 0.0
        self._prev_playing_secondary = False

        import cv2

        self.cv2 = cv2
        self.cap = None
        self.picam2 = None

        if self.backend == "v4l2":
            self.cap = (
                cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
                if hasattr(cv2, "CAP_V4L2")
                else cv2.VideoCapture(camera_index)
            )
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM["WIDTH"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM["HEIGHT"])
            self.cap.set(cv2.CAP_PROP_FPS, CAM["FPS"])
        elif self.backend == "picamera2":
            from picamera2 import Picamera2

            self.picam2 = Picamera2()
            cfg = self.picam2.create_video_configuration(
                main={"size": (CAM["WIDTH"], CAM["HEIGHT"]), "format": "RGB888"}
            )
            self.picam2.configure(cfg)
            self.picam2.start()
        else:
            raise ValueError(f"Unknown camera backend: {self.backend}")

        import mediapipe as mp

        self.mp = mp

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self._stable_counts: Dict[str, int] = {}

        self.gate_pinch = HoldGate(HOLD_S["pinch"], repeat=True, repeat_every_s=0.22, release_frames=5)
        self.gate_scroll = HoldGate(HOLD_S["scroll"], repeat=True, repeat_every_s=0.16, release_frames=3)
        self.gate_play = HoldGate(HOLD_S["play"], repeat=True, repeat_every_s=0.24, release_frames=5)
        self.gate_next = HoldGate(HOLD_S["next"], repeat=True, repeat_every_s=0.35, release_frames=5)
        self.gate_palm = HoldGate(HOLD_S["palm"], repeat=True, repeat_every_s=0.60, release_frames=6)
        self.gate_vol_up = HoldGate(HOLD_S["scroll"], repeat=True, repeat_every_s=0.14, release_frames=3)
        self.gate_vol_down = HoldGate(HOLD_S["play"], repeat=True, repeat_every_s=0.14, release_frames=3)

        self.menu_open_getter = None
        self.playing_secondary_getter = None

    def stop(self):
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.picam2 is not None:
                self.picam2.stop()
        except Exception:
            pass
        try:
            self.landmarker.close()
        except Exception:
            pass

    def _can_fire(self, name: str) -> bool:
        return now() >= self.cooldown_until[name]

    def _fire(self, name: str) -> None:
        self.cooldown_until[name] = now() + COOLDOWN_S[name]

    def _stable(self, key: str, condition: bool, frames_required: int) -> bool:
        c = self._stable_counts.get(key, 0)
        c = c + 1 if condition else 0
        self._stable_counts[key] = c
        return c >= frames_required

    def _update_scroll_dir(self, idx_y_norm: float) -> int:
        mid = float(SCROLL["MID_Y"])
        hyst = float(SCROLL["HYST"])
        if idx_y_norm < (mid - hyst):
            self._scroll_dir = +1
        elif idx_y_norm > (mid + hyst):
            self._scroll_dir = -1
        return self._scroll_dir

    def run(self):
        cv2 = self.cv2

        while self.running:
            if self.backend == "v4l2":
                ok, frame = self.cap.read() if self.cap is not None else (False, None)
                if not ok or frame is None:
                    time.sleep(0.03)
                    continue
                if self.flip:
                    frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = self.picam2.capture_array() if self.picam2 is not None else None
                if frame is None:
                    time.sleep(0.03)
                    continue
                if self.flip:
                    frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = frame

            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect_for_video(mp_image, mono_ms())

            pinch_ok = scroll_ok = play_ok = next_ok = palm_ok = False
            vol_up_ok = vol_down_ok = False
            idx_y_norm = 0.5

            if res and res.hand_landmarks and len(res.hand_landmarks) > 0:
                lm = res.hand_landmarks[0]

                def pt(i: int) -> Tuple[float, float]:
                    return (lm[i].x * w, lm[i].y * h)

                wrist = pt(0)
                middle_mcp = pt(9)

                thumb_tip = pt(4)
                index_tip = pt(8)
                middle_tip = pt(12)
                ring_tip = pt(16)
                pinky_tip = pt(20)

                sx = float(index_tip[0] / max(1, w))
                sy = float(index_tip[1] / max(1, h))
                idx_y_norm = sy

                dx = sx - self.cx
                dy = sy - self.cy
                if dx > self.max_step:
                    dx = self.max_step
                if dx < -self.max_step:
                    dx = -self.max_step
                if dy > self.max_step:
                    dy = self.max_step
                if dy < -self.max_step:
                    dy = -self.max_step
                sx = self.cx + dx
                sy = self.cy + dy

                nx = self.cx * (1 - self.alpha) + sx * self.alpha
                ny = self.cy * (1 - self.alpha) + sy * self.alpha

                if abs(nx - self.cx) < self.deadzone:
                    nx = self.cx
                if abs(ny - self.cy) < self.deadzone:
                    ny = self.cy

                self.cx, self.cy = nx, ny

                scale = max(40.0, _dist(wrist, middle_mcp))

                d_ti = _dist(thumb_tip, index_tip)
                d_tm = _dist(thumb_tip, middle_tip)
                d_tr = _dist(thumb_tip, ring_tip)
                d_tp = _dist(thumb_tip, pinky_tip)

                pinch = d_ti < (GESTURE["PINCH_RATIO"] * scale)
                scroll_touch = d_tm < (GESTURE["SCROLL_RATIO"] * scale)
                play_touch = d_tr < (GESTURE["PLAY_RATIO"] * scale)
                next_touch = d_tp < (GESTURE["NEXT_RATIO"] * scale)

                pinch_ok = self._stable("pinch", pinch, GESTURE["PINCH_FRAMES"])
                scroll_ok = self._stable("scroll", scroll_touch, GESTURE["SCROLL_FRAMES"])
                play_ok = self._stable("play", play_touch, GESTURE["PLAY_FRAMES"])
                next_ok = self._stable("next", next_touch, GESTURE["NEXT_FRAMES"])

                tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
                avg_tip_to_wrist = sum(_dist(t, wrist) for t in tips) / len(tips)

                finger_ext = 1.35 * scale
                palm = (
                    avg_tip_to_wrist > (GESTURE["PALM_RATIO"] * scale)
                    and _dist(index_tip, wrist) > finger_ext
                    and _dist(middle_tip, wrist) > finger_ext
                    and _dist(ring_tip, wrist) > finger_ext
                    and _dist(pinky_tip, wrist) > finger_ext
                )

                palm_ok = self._stable(
                    "palm",
                    palm and (not pinch) and (not scroll_touch) and (not play_touch) and (not next_touch),
                    GESTURE["PALM_FRAMES"],
                )

                vol_up_ok = scroll_ok
                vol_down_ok = play_ok

            menu_open = bool(self.menu_open_getter()) if callable(self.menu_open_getter) else False
            playing_secondary = bool(self.playing_secondary_getter()) if callable(self.playing_secondary_getter) else False

            if playing_secondary and (not self._prev_playing_secondary):
                self._ignore_volume_until = now() + 0.75
                self._stable_counts.clear()
                self.gate_vol_up.reset()
                self.gate_vol_down.reset()
                self.gate_pinch.reset()
                self.gate_play.reset()
                self.gate_next.reset()
                self.gate_palm.reset()
                self.gate_scroll.reset()

            self._prev_playing_secondary = playing_secondary

            if self.gate_pinch.update(pinch_ok) and self._can_fire("pinch"):
                self._fire("pinch")
                if playing_secondary and self._can_fire("back"):
                    self._fire("back")
                    self.actions.push("RETURN_MAIN")
                else:
                    self.actions.push("TOGGLE_MENU")

            if (not playing_secondary) and menu_open:
                if self.gate_scroll.update(scroll_ok) and self._can_fire("scroll"):
                    self._fire("scroll")
                    direction = self._update_scroll_dir(idx_y_norm)
                    if direction < 0:
                        self.actions.push("MENU_SCROLL_UP")
                    else:
                        self.actions.push("MENU_SCROLL_DOWN")
                    if MENU_TOUCH_FROM_POINTING:
                        self.actions.push("MENU_TOUCH")

            if (not playing_secondary) and menu_open:
                if self.gate_play.update(play_ok) and self._can_fire("play"):
                    self._fire("play")
                    self._ignore_volume_until = now() + 0.75
                    self.actions.push("MENU_PLAY")

            if self.gate_next.update(next_ok) and self._can_fire("next"):
                self._fire("next")
                self.actions.push("NEXT")

            if self.gate_palm.update(palm_ok) and self._can_fire("palm"):
                self._fire("palm")
                self.actions.push("TOGGLE_PAUSE")

            if playing_secondary and now() >= self._ignore_volume_until:
                if self.gate_vol_up.update(vol_up_ok) and self._can_fire("vol_up"):
                    self._fire("vol_up")
                    self.actions.push("VOL_UP")
                if self.gate_vol_down.update(vol_down_ok) and self._can_fire("vol_down"):
                    self._fire("vol_down")
                    self.actions.push("VOL_DOWN")

            time.sleep(0.01)


def start_evdev_keyboard(app: VideoApp, stop_flag: threading.Event) -> None:
    if is_windows():
        return

    def worker():
        try:
            from evdev import InputDevice, ecodes, list_devices
        except Exception as e:
            print(f"evdev not available: {e}", flush=True)
            return

        devs = [InputDevice(p) for p in list_devices()]
        kbs: List[str] = []
        for d in devs:
            name = (d.name or "").lower()
            if "keyboard" in name or "kbd" in name:
                kbs.append(d.path)


        if not kbs:
            kbs = [d.path for d in devs]

        if not kbs:
            # stdin fallback only if we actually have a TTY (systemd usually doesn't)
            try:
                import sys
                if not sys.stdin.isatty():
                    print("No evdev devices and no TTY stdin. Keyboard disabled.", flush=True)
                    return

                import termios, tty
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                tty.setraw(fd)
                print("No evdev keyboard found. Using stdin (SSH) fallback.", flush=True)

                while not stop_flag.is_set():
                    ch = sys.stdin.read(1)
                    if ch.lower() == "q":
                        actions = getattr(app, "actions", None)
                        if actions is not None:
                            actions.push("QUIT")
                        else:
                            stop_flag.set()
                        break
                    elif ch.lower() == "m":
                        app.toggle_menu()
                    elif ch.lower() == "n":
                        app.next_secondary()
                    elif ch == " ":
                        app.toggle_pause_secondary()
                    elif ch == "\x1b":
                        app.play_main()
                    elif ch in ("\r", "\n"):
                        if app.get_menu_open() and (not app.get_playing_secondary()):
                            app.menu_select()
                        else:
                            app.show_menu()

                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception as e:
                print(f"stdin fallback failed: {e}", flush=True)
            return

        def listen_one(path: str):
            from evdev import InputDevice, ecodes

            try:
                kb = InputDevice(path)
                print(f"Listening: {kb.name} ({path})", flush=True)

                for ev in kb.read_loop():
                    if stop_flag.is_set():
                        break
                    if ev.type != ecodes.EV_KEY:
                        continue
                    if ev.value != 1:  
                        continue

                    code = ev.code

                    if code == ecodes.KEY_Q:
                        actions = getattr(app, "actions", None)
                        if actions is not None:
                            actions.push("QUIT")
                        else:
                            stop_flag.set()
                        break
                    elif code == ecodes.KEY_M:
                        app.toggle_menu()
                    elif code == ecodes.KEY_N:
                        app.next_secondary()
                    elif code == ecodes.KEY_SPACE:
                        app.toggle_pause_secondary()
                    elif code == ecodes.KEY_ESC:
                        app.play_main()
                    elif code == ecodes.KEY_UP:
                        app.menu_move_selected(-1)
                    elif code == ecodes.KEY_DOWN:
                        app.menu_move_selected(+1)
                    elif code == ecodes.KEY_ENTER or code == ecodes.KEY_KPENTER:
                        if app.get_menu_open() and (not app.get_playing_secondary()):
                            app.menu_select()
                        else:
                            app.show_menu()
                    elif code == ecodes.KEY_LEFT:
                        app.seek_backward(5)
                    elif code == ecodes.KEY_RIGHT:
                        app.seek_forward(5)
                    elif code in (ecodes.KEY_EQUAL, ecodes.KEY_KPPLUS):
                        if app.get_playing_secondary():
                            try:
                                app.mpv.command({"command": ["add", "volume", int(VOLUME_STEP)]})
                                app.mpv.show_osd(f"🔊 +{int(VOLUME_STEP)}", 500)
                            except Exception:
                                pass
                    elif code in (ecodes.KEY_MINUS, ecodes.KEY_KPMINUS):
                        if app.get_playing_secondary():
                            try:
                                app.mpv.command({"command": ["add", "volume", -int(VOLUME_STEP)]})
                                app.mpv.show_osd(f"🔉 -{int(VOLUME_STEP)}", 500)
                            except Exception:
                                pass

            except Exception as e:
                print(f"Keyboard error {path}: {e}", flush=True)

        for p in kbs:
            threading.Thread(target=listen_one, args=(p,), daemon=True).start()

    threading.Thread(target=worker, daemon=True).start()


def run_headless_loop(app: VideoApp, actions: ActionQueue, stop_flag: threading.Event) -> None:
    def menu_scroll_wrap(delta: int) -> None:
        with app._state_lock:
            if app.state.playing_secondary or (not app.menu.open):
                return
            n = len(app.secondary_titles)
            if n <= 0:
                return
            app.menu.selected = (app.menu.selected + delta) % n
            t = now()
            app.menu.last_select_change = t
            app.menu.last_interaction = t
        app.show_menu()

    while not stop_flag.is_set():
        if app.mpv.proc is not None and app.mpv.proc.poll() is not None:
            stop_flag.set()
            break

        playing_secondary = app.get_playing_secondary()
        menu_open = app.get_menu_open()
        acts = actions.pop_all()

        if "QUIT" in acts:
            stop_flag.set()
            try:
                app.stop()
            except Exception:
                pass
            break

        acts = [a for a in acts if a != "QUIT"]

        # volume only in SECONDARY
        if playing_secondary:
            for act in acts:
                if act == "VOL_UP":
                    try:
                        app.mpv.command({"command": ["add", "volume", int(VOLUME_STEP)]})
                        app.mpv.show_osd(f"🔊 +{int(VOLUME_STEP)}", 500)
                    except Exception:
                        pass
                elif act == "VOL_DOWN":
                    try:
                        app.mpv.command({"command": ["add", "volume", -int(VOLUME_STEP)]})
                        app.mpv.show_osd(f"🔉 -{int(VOLUME_STEP)}", 500)
                    except Exception:
                        pass

        # remove volume actions
        acts = [a for a in acts if a not in ("VOL_UP", "VOL_DOWN")]

        if playing_secondary:
            for act in acts:
                if act == "RETURN_MAIN":
                    app.play_main()
                elif act == "NEXT":
                    app.next_secondary()
                elif act == "TOGGLE_PAUSE":
                    app.toggle_pause_secondary()
                elif act == "SYNC_MAIN_HINT":
                    app.hide_menu(show_hint=True)
            time.sleep(0.016)
            continue

        if not menu_open:
            for act in acts:
                if act == "TOGGLE_MENU":
                    app.toggle_menu()
                elif act == "RETURN_MAIN":
                    app.play_main()
                elif act == "TOGGLE_PAUSE":
                    app.toggle_pause_secondary()
                elif act == "NEXT":
                    app.next_secondary()
                elif act == "SYNC_MAIN_HINT":
                    app.hide_menu(show_hint=True)
            time.sleep(0.016)
            continue

        for act in acts:
            if act == "TOGGLE_MENU":
                app.toggle_menu()
            elif act == "RETURN_MAIN":
                app.play_main()
            elif act == "NEXT":
                app.next_secondary()
            elif act == "TOGGLE_PAUSE":
                app.toggle_pause_secondary()
            elif act == "MENU_PLAY":
                if app.get_menu_open():
                    app.menu_select()
            elif act == "MENU_SCROLL_UP":
                menu_scroll_wrap(-1)
            elif act == "MENU_SCROLL_DOWN":
                menu_scroll_wrap(+1)
            elif act == "MENU_TOUCH":
                app.touch_menu()
            elif act == "SYNC_MAIN_HINT":
                app.hide_menu(show_hint=True)

        time.sleep(0.016)


def main():
    default_root = "/home/pi/handpi_kiosk/pi5"

    parser = argparse.ArgumentParser(description="Headless Video Kiosk + MPV + Lua + MediaPipe + evdev")
    parser.add_argument("--root", default=default_root)
    parser.add_argument("--mpv-path", default="mpv")
    parser.add_argument("--ipc", default="/tmp/mpvsocket")
    parser.add_argument("--extra-mpv-args", default="")
    parser.add_argument("--camera-index", type=int, default=-1, help="camera index; -1 = auto-detect")
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--no-flip", action="store_true", help="disable horizontal flip")

    # AUDIO (USER-SELECTABLE):
    #   * default: --audio auto  -> choose an ALSA default 
    #   * --audio none          -> do not add any audio defaults
    #   * --ao / --audio-device -> explicit override 
    parser.add_argument("--audio", default="auto", choices=["auto", "none"])
    parser.add_argument("--ao", default="", help="mpv audio output, e.g. alsa")
    parser.add_argument("--audio-device", default="", help="mpv audio device, e.g. alsa/plughw:0,0")

    args = parser.parse_args()

    root = Path(args.root).resolve()
    lib = discover_videos(root / "videos")

    lua_script = root / "mpv" / "scripts" / "automain.lua"
    if not lua_script.exists():
        raise FileNotFoundError(f"Missing Lua script: {lua_script}")

    main_for_lua = to_forward_slashes(lib.main_video)
    lua_script_arg = str(lua_script.resolve()).replace("\\", "/")

    headless_vo_args = [
        "--vo=gpu",
        "--gpu-context=drm",
        "--no-osc",
        "--no-input-default-bindings",
        "--really-quiet",
        "--idle=yes",
        "--keep-open=yes",
        "--drm-draw-plane=overlay",
    ]

    osd_left_args: List[str] = [
        "--osd-align-x=left",
        "--osd-align-y=top",
        f"--osd-margin-x={int(OSD['MARGIN_X'])}",
        f"--osd-margin-y={int(OSD['MARGIN_Y'])}",
        "--osd-level=1",
        f"--osd-font-size={int(OSD['FONT_SIZE'])}",
        f"--osd-border-size={int(OSD['BORDER_SIZE'])}",
        "--osd-font=/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
    ]

    audio_debug_args = [
    "--log-file=/tmp/mpv.log",
    "--msg-level=ao=v,alsa=v,audio=v",
    ]

    extra_args: List[str] = [
        *audio_debug_args,
        f"--script={lua_script_arg}",
        f"--script-opts=automain-main={main_for_lua}",
        *osd_left_args,
        *headless_vo_args,
    ]

    # AUDIO DEFAULTS (AUTO):
    user_extra = args.extra_mpv_args.strip()
    user_overrode_audio = ("--ao=" in user_extra) or ("--audio-device=" in user_extra)
    user_overrode_audio = user_overrode_audio or bool(args.ao) or bool(args.audio_device)

    if args.audio == "auto" and (not user_overrode_audio):
        usb_card = detect_usb_playback_card()
        if usb_card is not None:
            # Prefer USB playback 
            extra_args += ["--ao=alsa", f"--audio-device=alsa/plughw:{usb_card},0"]
        else:
            # No USB detected -> force ALSA so HDMI
            extra_args += ["--ao=alsa"]
            
    if args.ao:
        extra_args.append(f"--ao={args.ao}")
    if args.audio_device:
        extra_args.append(f"--audio-device={args.audio_device}")

    if args.extra_mpv_args.strip():
        extra_args += args.extra_mpv_args.split()

    mpv = MPVController(args.mpv_path, args.ipc, fullscreen=True, extra_args=extra_args)

    app = VideoApp(
        mpv,
        lib,
        menu_idle_close_s=MENU_IDLE_CLOSE_S,
        menu_refresh_s=MENU_REFRESH_S,
        selection_debounce_s=SELECTION_DEBOUNCE_S,
        selection_stable_samples=SELECTION_STABLE_SAMPLES,
        render_menu_fn=render_menu_main,
    )

    stop_flag = threading.Event()
    actions = ActionQueue()
    app.actions = actions

    hand: Optional[HandControl] = None
    model_path = root / "models" / "hand_landmarker.task"

    try:
        app.start()
        start_evdev_keyboard(app, stop_flag)

        if not args.no_hands:
            try:
                ensure_hand_model(model_path)
                preferred = None if args.camera_index is None or args.camera_index < 0 else int(args.camera_index)
                backend, cam_idx = choose_camera_backend(preferred)

                if backend == "v4l2":
                    print(f"Using V4L2 camera index: {cam_idx}", flush=True)
                    hand = HandControl(
                        actions=actions,
                        model_path=model_path,
                        backend="v4l2",
                        camera_index=int(cam_idx),
                        flip=(not args.no_flip),
                    )
                else:
                    print("Using CSI camera via Picamera2.", flush=True)
                    hand = HandControl(
                        actions=actions,
                        model_path=model_path,
                        backend="picamera2",
                        flip=(not args.no_flip),
                    )

                hand.menu_open_getter = app.get_menu_open
                hand.playing_secondary_getter = app.get_playing_secondary
                hand.start()
            except Exception as e:
                print(f"⚠️ Hands disabled (camera init failed): {e}", flush=True)
                hand = None

        run_headless_loop(app, actions, stop_flag)

    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        try:
            if hand is not None:
                hand.stop()
        except Exception:
            pass
        try:
            app.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
