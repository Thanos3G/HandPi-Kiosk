# video_core.py (CORE ONLY)
#
# Contains:
# - video discovery
# - mpv IPC controller + event listener
# - VideoApp state machine (no UI policy)
#

import json
import platform
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, List, Optional

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}  # supported video extensions


def is_windows() -> bool:
    return platform.system().lower().startswith("win")


def now() -> float:
    return time.time()


def safe_title_from_filename(p: Path) -> str:
    name = p.stem
    s = name
    while s and s[0].isdigit():
        s = s[1:]
    s = s.lstrip("_- .")
    return s if s else name


def to_forward_slashes(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def norm_path(p: str) -> str:
    # normalize mpv-reported paths vs filesystem paths (Windows case-insensitive; slash + resolve)
    if not p:
        return ""
    try:
        s = str(Path(p).resolve())
    except Exception:
        s = str(p)
    s = s.replace("\\", "/")
    return s.casefold() if is_windows() else s


def _natural_key(name: str):
    # human sort: video2 < video10
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _is_video_file(p: Path) -> bool:
    if not (p.is_file() and p.suffix.lower() in VIDEO_EXTS):
        return False
    n = p.name
    if n.startswith(".") or n.startswith("._") or n.endswith("~"):
        return False
    return True


@dataclass
class VideoLibrary:
    main_video: Path
    secondary_videos: List[Path]


def discover_videos(videos_dir: Path) -> VideoLibrary:
    # expects videos_dir/main (1 file) and videos_dir/secondary (0..N files)
    main_dir = videos_dir / "main"
    secondary_dir = videos_dir / "secondary"

    if not main_dir.exists():
        raise FileNotFoundError(f"Missing main dir: {main_dir}")

    if not secondary_dir.exists():
        secondary_dir.mkdir(parents=True, exist_ok=True)

    main_candidates = sorted(
        [p for p in main_dir.iterdir() if _is_video_file(p)],
        key=lambda p: _natural_key(p.name),
    )
    if not main_candidates:
        raise FileNotFoundError(f"No main video found in {main_dir} (put 1 video file there)")
    if len(main_candidates) > 1:
        print(f"⚠️  Multiple main videos found. Using: {main_candidates[0].name}")

    main_video = main_candidates[0].resolve()

    secondary_all = sorted(
        [p for p in secondary_dir.iterdir() if _is_video_file(p)],
        key=lambda p: _natural_key(p.name),
    )

    secondary: List[Path] = []
    for p in secondary_all:
        try:
            if p.resolve() == main_video:
                print(f"⚠️  Ignoring secondary duplicate of MAIN: {p.name}")
                continue
        except Exception:
            pass
        secondary.append(p)

    return VideoLibrary(main_video=main_video, secondary_videos=secondary)


class ActionQueue:
    # action queue for decoupling inputs from state machine
    def __init__(self):
        self._q: Deque[str] = deque()
        self._lock = threading.Lock()

    def push(self, action: str) -> None:
        with self._lock:
            self._q.append(action)

    def pop_all(self) -> List[str]:
        out: List[str] = []
        with self._lock:
            while self._q:
                out.append(self._q.popleft())
        return out


class MPVEventListener(threading.Thread):
    # persistent IPC listener; keep Python state aligned with mpv/Lua
    def __init__(self, ipc_name: str, on_path: Callable[[str], None]):
        super().__init__(daemon=True)
        self.ipc = ipc_name
        self.on_path = on_path
        self._running = True

    def stop(self):
        self._running = False

    def _connect_windows(self):
        last_err = None
        for _ in range(80):
            try:
                f = open(self.ipc, "r+", encoding="utf-8", newline="\n")
                f.write(json.dumps({"command": ["observe_property", 1, "path"]}) + "\n")
                f.flush()
                return f
            except Exception as e:
                last_err = e
                time.sleep(0.05)
        raise RuntimeError(f"Failed to connect to MPV IPC pipe: {last_err}") from last_err

    def _connect_unix(self):
        import socket

        last_err = None
        for _ in range(80):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self.ipc)
                cmd = {"command": ["observe_property", 1, "path"]}
                s.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
                s_file = s.makefile("rwb", buffering=0)
                return s, s_file
            except Exception as e:
                last_err = e
                time.sleep(0.05)
        raise RuntimeError(f"Failed to connect to MPV IPC socket: {last_err}") from last_err

    def run(self):
        if is_windows():
            f = self._connect_windows()
            try:
                while self._running:
                    line = f.readline()
                    if not line:
                        break
                    try:
                        msg = json.loads(line)
                    except Exception:
                        continue
                    if msg.get("event") == "property-change" and msg.get("name") == "path":
                        data = msg.get("data")
                        if isinstance(data, str) and data:
                            self.on_path(data)
            finally:
                try:
                    f.close()
                except Exception:
                    pass
        else:
            s = None
            sf = None
            try:
                s, sf = self._connect_unix()
                while self._running:
                    line = sf.readline()
                    if not line:
                        break
                    try:
                        msg = json.loads(line.decode("utf-8", errors="ignore"))
                    except Exception:
                        continue
                    if msg.get("event") == "property-change" and msg.get("name") == "path":
                        data = msg.get("data")
                        if isinstance(data, str) and data:
                            self.on_path(data)
            finally:
                try:
                    if sf:
                        sf.close()
                except Exception:
                    pass
                try:
                    if s:
                        s.close()
                except Exception:
                    pass


class MPVController:
    # starts mpv and sends JSON IPC commands (loadfile, pause, seek, OSD)
    def __init__(self, mpv_path: str, ipc_name: str, fullscreen: bool = True, extra_args: Optional[List[str]] = None):
        self.mpv_path = mpv_path
        self.fullscreen = fullscreen
        self.proc: Optional[subprocess.Popen] = None
        self.ipc = ipc_name
        self.extra_args = extra_args or []
        self._lock = threading.Lock()

        self._listener: Optional[MPVEventListener] = None
        self._on_path_cb: Optional[Callable[[str], None]] = None

    def set_on_path_changed(self, cb: Callable[[str], None]) -> None:
        self._on_path_cb = cb

    def _kill_stale_mpv_and_ipc_unix(self) -> None:
        # only remove stale IPC socket
        if is_windows():
            return
        try:
            p = Path(self.ipc)
            if p.exists():
                p.unlink()
        except Exception:
            pass

    def _wait_for_ipc_ready(self, timeout_s: float = 6.0) -> None:
        t0 = time.time()
        ipc_path = Path(self.ipc)
        while time.time() - t0 < timeout_s:
            if ipc_path.exists():
                return
            time.sleep(0.05)
        raise RuntimeError(f"MPV IPC not ready (did not appear): {self.ipc}")

    def start(self) -> None:
        self._kill_stale_mpv_and_ipc_unix()

        args = [self.mpv_path]
        if self.fullscreen:
            args.append("--fs")

        args += [
            "--idle=yes",
            "--keep-open=yes",
            "--really-quiet",
            "--no-osc",
            "--no-input-default-bindings",
            f"--input-ipc-server={self.ipc}",
        ]
        args += self.extra_args

        creationflags = 0
        if is_windows():
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

        if is_windows():
            time.sleep(0.4)
        else:
            self._wait_for_ipc_ready(timeout_s=6.0)

        if callable(self._on_path_cb):
            self._listener = MPVEventListener(self.ipc, self._on_path_cb)
            self._listener.start()

    def stop(self) -> None:
        try:
            if self._listener is not None:
                self._listener.stop()
        except Exception:
            pass

        if self.proc and self.proc.poll() is None:
            try:
                self.command({"command": ["quit"]})
            except Exception:
                pass
            time.sleep(0.2)
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _request_unix(self, obj: dict, timeout_s: float = 0.6) -> dict:
        import socket

        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        last_err: Optional[Exception] = None

        for _ in range(80):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(timeout_s)
                    s.connect(self.ipc)
                    s.sendall(data)

                    buf = b""
                    while b"\n" not in buf:
                        chunk = s.recv(65536)
                        if not chunk:
                            break
                        buf += chunk

                if not buf:
                    return {"error": "no-reply"}

                first_line = buf.split(b"\n", 1)[0].strip()
                if not first_line:
                    return {"error": "empty-reply"}

                try:
                    return json.loads(first_line.decode("utf-8", errors="ignore"))
                except Exception:
                    return {"error": "bad-reply", "raw": first_line.decode("utf-8", errors="ignore")}

            except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError, OSError) as e:
                last_err = e
                time.sleep(0.05)

        raise RuntimeError(f"MPV IPC request failed after retries: {last_err}")

    def _write_ipc_windows(self, payload: str) -> None:
        last_err: Optional[Exception] = None
        for _ in range(60):
            try:
                with open(self.ipc, "w", encoding="utf-8", newline="") as f:
                    f.write(payload + "\n")
                return
            except FileNotFoundError as e:
                last_err = e
                time.sleep(0.1)
            except OSError as e:
                last_err = e
                time.sleep(0.1)
        raise RuntimeError(f"Failed writing to MPV named pipe {self.ipc}: {last_err}") from last_err

    def command(self, obj: dict) -> None:
        if is_windows():
            payload = json.dumps(obj, ensure_ascii=False)
            with self._lock:
                self._write_ipc_windows(payload)
            return

        with self._lock:
            reply = self._request_unix(obj)

        if isinstance(reply, dict) and reply.get("error") not in (None, "success"):
            raise RuntimeError(f"MPV command failed: {reply}")

    def loadfile(self, path: Path, replace: bool = True) -> None:
        mode = "replace" if replace else "append"
        self.command({"command": ["loadfile", str(path.resolve()), mode]})

    def set_loop_inf(self) -> None:
        self.command({"command": ["set_property", "loop", "inf"]})

    def set_loop_off(self) -> None:
        self.command({"command": ["set_property", "loop", "no"]})

    def set_pause(self, pause: bool) -> None:
        self.command({"command": ["set_property", "pause", bool(pause)]})

    def show_osd(self, text: str, duration_ms: int = 2000) -> None:
        self.command({"command": ["show-text", text, duration_ms]})

    def set_osd_msg1(self, text: str) -> None:
        self.command({"command": ["set_property", "options/osd-msg1", text]})

    def clear_osd_msg1(self) -> None:
        self.set_osd_msg1("")

    def seek(self, seconds: float) -> None:
        self.command({"command": ["seek", float(seconds), "relative"]})

    def seek_forward(self, seconds: float = 5.0) -> None:
        self.seek(+seconds)

    def seek_backward(self, seconds: float = 5.0) -> None:
        self.seek(-seconds)


@dataclass
class MenuState:
    open: bool = False
    selected: int = 0
    last_interaction: float = 0.0
    last_select_change: float = 0.0
    candidate_idx: int = -1
    candidate_count: int = 0


@dataclass
class AppState:
    playing_secondary: bool = False
    secondary_index: int = 0
    paused: bool = False


class VideoApp:
    # core state machine; UI policy is injected from main 
    def __init__(
        self,
        mpv: MPVController,
        library: VideoLibrary,
        *,
        menu_idle_close_s: float,
        menu_refresh_s: float,
        selection_debounce_s: float,
        selection_stable_samples: int,
        render_menu_fn: Callable[[str, List[str], int], str],
    ):
        self.mpv = mpv
        self.lib = library

        self.menu = MenuState(open=False, selected=0, last_interaction=0.0, last_select_change=0.0)
        self.state = AppState(playing_secondary=False, secondary_index=0, paused=False)

        self.secondary_titles = [safe_title_from_filename(p) for p in self.lib.secondary_videos]
        self.main_title = safe_title_from_filename(self.lib.main_video)

        self.main_norm = norm_path(str(self.lib.main_video))
        self.secondary_norms = [norm_path(str(p)) for p in self.lib.secondary_videos]

        self.menu_idle_close_s = float(menu_idle_close_s)
        self.menu_refresh_s = float(menu_refresh_s)
        self.selection_debounce_s = float(selection_debounce_s)
        self.selection_stable_samples = int(selection_stable_samples)

        self.render_menu_fn = render_menu_fn

        self._running = True
        self._menu_thread = threading.Thread(target=self._menu_watchdog, daemon=True)

        self._state_lock = threading.RLock()
        self.actions: Optional[ActionQueue] = None

        self._suppress_next_main_hint = False

        self.mpv.set_on_path_changed(self._on_mpv_path_changed)

    def get_menu_open(self) -> bool:
        with self._state_lock:
            return self.menu.open

    def get_playing_secondary(self) -> bool:
        with self._state_lock:
            return self.state.playing_secondary

    def start(self) -> None:
        self.mpv.start()
        self.play_main()
        self._menu_thread.start()

    def stop(self) -> None:
        self._running = False
        self.mpv.stop()

    def _sync_pause_banner_locked(self) -> None:
        self.mpv.clear_osd_msg1()


    def _on_mpv_path_changed(self, path: str) -> None:
        # single source of truth for MAIN/SECONDARY transitions 
        p = norm_path(path)
        if not p:
            return

        with self._state_lock:
            if p == self.main_norm:
                self.state.playing_secondary = False
                self.state.paused = False
                self._sync_pause_banner_locked()
                if (not self.menu.open) and (self.actions is not None):
                    if self._suppress_next_main_hint:
                        self._suppress_next_main_hint = False
                    else:
                        self.actions.push("SYNC_MAIN_HINT")
                return

            try:
                idx = self.secondary_norms.index(p)
                self.state.playing_secondary = True
                self.state.secondary_index = idx
                self._sync_pause_banner_locked()
                return
            except ValueError:
                return

    def touch_menu(self) -> None:
        with self._state_lock:
            self.menu.last_interaction = now()

    def play_main(self) -> None:
        with self._state_lock:
            self.state.playing_secondary = False
            self.state.secondary_index = 0
            self.state.paused = False
            self._sync_pause_banner_locked()
            self._suppress_next_main_hint = True

        self.mpv.loadfile(self.lib.main_video, replace=True)
        self.mpv.set_pause(False)
        self.mpv.set_loop_inf()
        self.hide_menu(show_hint=True)

    def play_secondary(self, idx: int) -> None:
        if not self.lib.secondary_videos:
            self.mpv.show_osd("No secondary videos available.", 1500)
            return

        idx = max(0, min(idx, len(self.lib.secondary_videos) - 1))
        with self._state_lock:
            self.state.playing_secondary = True
            self.state.secondary_index = idx
            self.state.paused = False
            self._sync_pause_banner_locked()

        self.mpv.loadfile(self.lib.secondary_videos[idx], replace=True)
        self.mpv.set_loop_off()
        self.mpv.set_pause(False)

        self.hide_menu(show_hint=False)
        self.mpv.show_osd("", 10)

    def next_secondary(self) -> None:
        if not self.lib.secondary_videos:
            self.mpv.show_osd("No secondary videos.", 1200)
            return

        with self._state_lock:
            playing_secondary = self.state.playing_secondary
            secondary_index = self.state.secondary_index

        if not playing_secondary:
            self.play_secondary(0)
            return

        nxt = (secondary_index + 1) % len(self.lib.secondary_videos)
        self.play_secondary(nxt)

    def toggle_pause_secondary(self) -> None:
        with self._state_lock:
            playing_secondary = self.state.playing_secondary

        if not playing_secondary:
            self.mpv.show_osd("Pause disabled on MAIN.", 1000)
            return

        with self._state_lock:
            self.state.paused = not self.state.paused
            paused = self.state.paused
            self._sync_pause_banner_locked()

        self.mpv.set_pause(paused)

        if not paused:
            self.mpv.show_osd("", 10)
            self.mpv.show_osd("▶ RESUMED", 900)

    def seek_forward(self, seconds: float = 5.0) -> None:
        with self._state_lock:
            playing_secondary = self.state.playing_secondary
        if not playing_secondary:
            self.mpv.show_osd("Fast-forward disabled on MAIN.", 1000)
            return
        self.mpv.seek_forward(seconds)
        self.mpv.show_osd(f"⏩ +{int(seconds)}s", 600)

    def seek_backward(self, seconds: float = 5.0) -> None:
        with self._state_lock:
            playing_secondary = self.state.playing_secondary
        if not playing_secondary:
            self.mpv.show_osd("Seek disabled on MAIN.", 1000)
            return
        self.mpv.seek_backward(seconds)
        self.mpv.show_osd(f"⏪ -{int(seconds)}s", 600)

    def show_menu(self) -> None:
        with self._state_lock:
            if self.state.playing_secondary:
                return
            self.menu.open = True
            self.menu.last_interaction = now()
            if self.secondary_titles:
                self.menu.selected = max(0, min(self.menu.selected, len(self.secondary_titles) - 1))
            else:
                self.menu.selected = 0

            selected = self.menu.selected
            main_title = self.main_title
            items = list(self.secondary_titles)

        text = self.render_menu_fn(main_title, items, selected)
        self.mpv.show_osd(text, duration_ms=3000)

    def hide_menu(self, show_hint: bool = False) -> None:
        with self._state_lock:
            self.menu.open = False
            playing_secondary = self.state.playing_secondary

    def toggle_menu(self) -> None:
        with self._state_lock:
            if self.state.playing_secondary:
                return
            is_open = self.menu.open

        if is_open:
            self.hide_menu(show_hint=False)
        else:
            self.show_menu()

    def menu_select(self) -> None:
        with self._state_lock:
            if not self.menu.open or self.state.playing_secondary:
                return
            sel = self.menu.selected
            self.menu.last_interaction = now()
        self.play_secondary(sel)

    def menu_move_selected(self, delta: int) -> None:
        with self._state_lock:
            if not self.menu.open or self.state.playing_secondary:
                return
            if not self.secondary_titles:
                return
            n = len(self.secondary_titles)
            self.menu.selected = max(0, min(self.menu.selected + delta, n - 1))
            self.menu.last_select_change = now()
            self.menu.last_interaction = now()
        self.show_menu()

    def menu_set_selected(self, idx: int) -> None:
        with self._state_lock:
            if not self.menu.open or self.state.playing_secondary:
                return
            if not self.secondary_titles:
                return

            idx = max(0, min(idx, len(self.secondary_titles) - 1))

            if idx != self.menu.candidate_idx:
                self.menu.candidate_idx = idx
                self.menu.candidate_count = 1
                return
            else:
                self.menu.candidate_count += 1

            if self.menu.candidate_count < self.selection_stable_samples:
                return

            t = now()
            if (t - self.menu.last_select_change) < self.selection_debounce_s:
                return

            if idx != self.menu.selected:
                self.menu.selected = idx
                self.menu.last_select_change = t
                self.menu.last_interaction = t

        self.show_menu()

    def _menu_watchdog(self) -> None:
        last_refresh = 0.0
        while self._running:
            with self._state_lock:
                playing_secondary = self.state.playing_secondary
                menu_open = self.menu.open
                last_interaction = self.menu.last_interaction
                paused = self.state.paused

            if playing_secondary:
                if menu_open:
                    self.hide_menu(show_hint=False)
                time.sleep(0.2)
                continue

            if menu_open:
                if now() - last_interaction > self.menu_idle_close_s:
                    self.hide_menu(show_hint=False)
                else:
                    if (not paused) and (now() - last_refresh > self.menu_refresh_s):
                        last_refresh = now()
                        self.show_menu()

            time.sleep(0.2)
