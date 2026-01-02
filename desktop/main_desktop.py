# main.py (Desktop / Windows)
# UI + inputs only:
# - Menu policy + menu renderer
# - MediaPipe hand tracking (gestures -> actions)
# - Keyboard hotkeys (pynput)
# - main() wiring + main event loop (no Qt / no overlay / no dot)
#

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

# Menu behavior 
MENU_IDLE_CLOSE_S = 22.0
MENU_REFRESH_S = 1.0

# Desktop tends to run higher FPS, so require more stable samples for selection to avoid jitter
SELECTION_DEBOUNCE_S = 0.5
SELECTION_STABLE_SAMPLES = 10

# Volume step )
VOLUME_STEP = 5

OSD = {
    "MARGIN_X": 28,
    "MARGIN_Y": 22,
    "FONT_SIZE": 22,
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
    lines.append("📼 Playlist")
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
    lines.append("←/→: Fast Forward")
    lines.append("Q: Quit")
    lines.append("↑/↓: Move selection")
    lines.append("")
    lines.append("◆ HAND ◆")
    lines.append("Thumb+Index: Toggle menu (MAIN) / Return MAIN")
    lines.append("Thumb+Middle: Scroll list (MAIN) / Volume Up")
    lines.append("Thumb+Ring: Play selected (MAIN) / Volume Down")
    lines.append("Thumb+Pinky: Next Video")
    lines.append("Palm: Pause/Resume")
    return "\n".join(lines)

# Gesture ratios are relative to a per-frame hand scale
GESTURE = {
    "PINCH_RATIO": 0.20,
    "SCROLL_RATIO": 0.24,
    "PLAY_RATIO": 0.20,
    "NEXT_RATIO": 0.30,
    "PALM_RATIO": 1.45,
    
# Higher FPS -> more frames required before a gesture is treated as stable
    "PALM_FRAMES": 22,
    "PINCH_FRAMES": 22,
    "SCROLL_FRAMES": 18,
    "PLAY_FRAMES": 15,
    "NEXT_FRAMES": 18,
}

# Cooldowns prevent repeating triggers from one sustained pose 
COOLDOWN_S = {
    "palm": 1.3,
    "pinch": 0.8,
    "next": 0.8,
    "play": 0.7,
    "back": 0.9,
    "scroll": 0.35,
    "vol_up": 0.18,
    "vol_down": 0.18,
}

# Hold times: gesture must be held this long before firing
HOLD_S = {
    "pinch": 0.30,
    "scroll": 0.14,
    "play": 0.25,
    "next": 0.30,
    "palm": 0.35,
    "vol": 0.14,
}

# Scroll direction from pointing Y: above/below MID_Y with hysteresis to avoid rapid flipping
SCROLL = {"MID_Y": 0.50, "HYST": 0.07}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

def mono_ms() -> int:
    return int(time.monotonic() * 1000)

def ensure_hand_model(model_path: Path, url: str = MODEL_URL) -> Path:
    # Download-once: keep the model file and reuse on future 
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        return model_path

    tmp = model_path.with_suffix(model_path.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    print(f"⬇️  Downloading hand model to: {model_path}")
    last_err: Optional[Exception] = None
    for attempt in range(1, 6):
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.exists() and tmp.stat().st_size > 1_000_000:
                tmp.replace(model_path)
                print("✅ Model download complete.")
                return model_path
            raise RuntimeError("Downloaded file too small/corrupted.")
        except Exception as e:
            last_err = e
            print(f"⚠️  Attempt {attempt}/5 failed: {repr(e)}")
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"Failed to download model after retries: {last_err}")

class HoldGate:
    def __init__(self, hold_s: float, repeat: bool = False, repeat_every_s: float = 0.18, release_frames: int = 6):
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
    # MediaPipe runs in its own thread and pushes actions into the shared Queue
    def __init__(self, actions: ActionQueue, model_path: Path, camera_index: int = 0, flip: bool = True):
        super().__init__(daemon=True)
        self.actions = actions
        self.running = True
        self.flip = bool(flip)

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
        self._stable_counts: Dict[str, int] = {}
        self._scroll_dir = +1

        self._ignore_volume_until = 0.0
        self._prev_playing_secondary = False

        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if is_windows() else 0)

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
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.gate_pinch = HoldGate(HOLD_S["pinch"], repeat=True, repeat_every_s=0.18, release_frames=8)
        self.gate_scroll = HoldGate(HOLD_S["scroll"], repeat=True, repeat_every_s=0.16, release_frames=4)
        self.gate_play = HoldGate(HOLD_S["play"], repeat=True, repeat_every_s=0.18, release_frames=8)
        self.gate_next = HoldGate(HOLD_S["next"], repeat=True, repeat_every_s=0.35, release_frames=6)
        self.gate_palm = HoldGate(HOLD_S["palm"], repeat=True, repeat_every_s=0.18, release_frames=10)

        # Volume in SECONDARY
        self.gate_vol_up = HoldGate(HOLD_S["vol"], repeat=True, repeat_every_s=0.14, release_frames=4)
        self.gate_vol_down = HoldGate(HOLD_S["vol"], repeat=True, repeat_every_s=0.14, release_frames=4)

        self.menu_open_getter = None
        self.playing_secondary_getter = None

    def stop(self):
        self.running = False
        try:
            self.cap.release()
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
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            if self.flip:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                idx_y_norm = float(index_tip[1] / max(1, h))
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

                # Open palm detection (pause)
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
                palm_ok = self._stable("palm", palm and (not pinch) and (not scroll_touch) and (not play_touch) and (not next_touch), GESTURE["PALM_FRAMES"])
                
                vol_up_ok = self._stable("vol_up", d_tm < (GESTURE["SCROLL_RATIO"] * scale), GESTURE["SCROLL_FRAMES"])
                vol_down_ok = self._stable("vol_down", d_tr < (GESTURE["PLAY_RATIO"] * scale), GESTURE["PLAY_FRAMES"])

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

            self._prev_playing_secondary = playing_secondary

            if self.gate_pinch.update(pinch_ok) and self._can_fire("pinch"):
                self._fire("pinch")
                if playing_secondary and self._can_fire("back"):
                    self._fire("back")
                    self.actions.push("RETURN_MAIN")
                else:
                    self.actions.push("TOGGLE_MENU")

            # MAIN + menu open: Thumb+Middle is scroll 
            if (not playing_secondary) and menu_open:
                if self.gate_scroll.update(scroll_ok) and self._can_fire("scroll"):
                    self._fire("scroll")
                    direction = self._update_scroll_dir(idx_y_norm)
                    if direction < 0:
                        self.actions.push("MENU_SCROLL_UP")
                    else:
                        self.actions.push("MENU_SCROLL_DOWN")
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

            # SECONDARY only: volume  
            if playing_secondary and now() >= self._ignore_volume_until:
                if self.gate_vol_up.update(vol_up_ok) and self._can_fire("vol_up"):
                    self._fire("vol_up")
                    self.actions.push("VOL_UP")
                if self.gate_vol_down.update(vol_down_ok) and self._can_fire("vol_down"):
                    self._fire("vol_down")
                    self.actions.push("VOL_DOWN")

            time.sleep(0.01)

def start_global_hotkeys_windows(app: VideoApp, stop_flag: threading.Event) -> None:
    from pynput import keyboard

    def on_press(key):
        if hasattr(key, "char") and key.char:
            ch = key.char.lower()
            if ch == "m":
                app.toggle_menu()
            elif ch == "n":
                app.next_secondary()
            elif ch == "q":
                stop_flag.set()
                return False
            elif ch == "=":
                if app.get_playing_secondary():
                    try:
                        app.mpv.command({"command": ["add", "volume", int(VOLUME_STEP)]})
                        app.mpv.show_osd(f"🔊 +{int(VOLUME_STEP)}", 500)
                    except Exception:
                        pass
            elif ch == "-":
                if app.get_playing_secondary():
                    try:
                        app.mpv.command({"command": ["add", "volume", -int(VOLUME_STEP)]})
                        app.mpv.show_osd(f"🔉 -{int(VOLUME_STEP)}", 500)
                    except Exception:
                        pass

        if key == keyboard.Key.space:
            app.toggle_pause_secondary()
        elif key == keyboard.Key.left:
            app.seek_backward(5)
        elif key == keyboard.Key.right:
            app.seek_forward(5)
        elif key == keyboard.Key.esc:
            app.play_main()
        elif key == keyboard.Key.up:
            app.menu_move_selected(-1)
        elif key == keyboard.Key.down:
            app.menu_move_selected(+1)
        elif key == keyboard.Key.enter:
            if app.get_menu_open() and (not app.get_playing_secondary()):
                app.menu_select()
            else:
                app.show_menu()

    keyboard.Listener(on_press=on_press).start()

def run_desktop_loop(app: VideoApp, actions: ActionQueue, stop_flag: threading.Event) -> None:
    #consume actions from hand/keyboard and call VideoApp methods
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

        # Apply volume only in SECONDARY 
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

        # Remove VOL actions 
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
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Video Kiosk + MPV + Lua + MediaPipe (no Qt overlay)")
    parser.add_argument("--root", default=str(script_dir), help="Project root (defaults to folder containing main.py)")
    parser.add_argument("--mpv-path", default="mpv.exe" if is_windows() else "mpv")
    parser.add_argument("--ipc", default=r"\\.\pipe\mpvpipe" if is_windows() else "/tmp/mpvsocket")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--extra-mpv-args", default="")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--no-flip", action="store_true", help="disable horizontal flip")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    lib = discover_videos(root / "videos")

    lua_script = root / "mpv" / "scripts" / "automain.lua"
    if not lua_script.exists():
        raise FileNotFoundError(f"Missing Lua script: {lua_script}")

    main_for_lua = to_forward_slashes(lib.main_video)
    lua_script_arg = str(lua_script.resolve()).replace("\\", "/")

    osd_left_args: List[str] = [
        "--osd-align-x=left",
        "--osd-align-y=top",
        f"--osd-margin-x={int(OSD['MARGIN_X'])}",
        f"--osd-margin-y={int(OSD['MARGIN_Y'])}",
        "--osd-level=1",
        f"--osd-font-size={int(OSD['FONT_SIZE'])}",
        f"--osd-border-size={int(OSD['BORDER_SIZE'])}",
    ]

    extra_args: List[str] = [
        f"--script={lua_script_arg}",
        f"--script-opts=automain-main={main_for_lua}",
        *osd_left_args,
    ]
    if args.extra_mpv_args.strip():
        extra_args += args.extra_mpv_args.split()

    mpv = MPVController(args.mpv_path, args.ipc, fullscreen=not args.windowed, extra_args=extra_args)

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

        if is_windows():
            start_global_hotkeys_windows(app, stop_flag)

        if not args.no_hands:
            ensure_hand_model(model_path)
            hand = HandControl(
                actions=actions,
                model_path=model_path,
                camera_index=args.camera_index,
                flip=(not args.no_flip),
            )
            hand.menu_open_getter = app.get_menu_open
            hand.playing_secondary_getter = app.get_playing_secondary
            hand.start()

        run_desktop_loop(app, actions, stop_flag)

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
