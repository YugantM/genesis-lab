"""Fast local Genesis dashboard server with MLX-rendered habitat frames."""

from __future__ import annotations

import argparse
import json
import queue
import struct
import threading
import time
import zlib
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import mlx.core as mx
import numpy as np

from .core import GenesisConfig, GenesisWorld
from .genome import Specimen, load_specimen
from .metrics import center_of_mass
from .multispecies_benchmark import centered_state
from .robust_search import calibrated_damage


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
SIZE = 128
GRID = 3


def _chunk(name: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + name
        + payload
        + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
    )


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an uint8 RGB image without adding a server-side image dependency."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an uint8 image with three channels")
    height, width = rgb.shape[:2]
    scanlines = b"".join(b"\0" + row.tobytes() for row in rgb)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        # Level 1 is deliberate: localhost bandwidth is cheap, frame latency is not.
        + _chunk(b"IDAT", zlib.compress(scanlines, 1))
        + _chunk(b"IEND", b"")
    )


def _smoothstep(low: float, high: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def render_atlas(states: np.ndarray) -> np.ndarray:
    """Render nine scalar Lenia states into the observatory's 3×3 colour atlas."""
    atlas = np.zeros((SIZE * GRID, SIZE * GRID, 3), np.float32)
    deep = np.asarray([0.008, 0.025, 0.020], np.float32)
    for index, state in enumerate(states):
        halo = _smoothstep(0.008, 0.25, state)[..., None]
        core = _smoothstep(0.3, 0.88, state)[..., None]
        phase_a = (index * 0.381) % 1
        phase_b = (index * 0.217) % 1
        moss = (1 - phase_a) * np.asarray([0.055, 0.25, 0.15]) + phase_a * np.asarray([0.08, 0.19, 0.25])
        acid = (1 - phase_b) * np.asarray([0.62, 1.0, 0.36]) + phase_b * np.asarray([0.35, 1.0, 0.72])
        colour = deep * (1 - halo) + moss * halo
        body = _smoothstep(0.1, 0.68, state)[..., None]
        colour = colour * (1 - body) + acid * body
        colour = colour * (1 - core * 0.75) + np.asarray([0.9, 1.0, 0.84]) * core * 0.75
        row, column = divmod(index, GRID)
        atlas[row * SIZE : (row + 1) * SIZE, column * SIZE : (column + 1) * SIZE] = colour
    return np.asarray(np.clip(atlas * 255, 0, 255), np.uint8)


def load_specimens() -> list[Specimen]:
    manifest = json.loads((WEB_ROOT / "specimens/species-benchmark.json").read_text())
    paths = [WEB_ROOT / "specimens/genesis-001.json"]
    paths.extend(WEB_ROOT / "specimens" / entry["path"] for entry in manifest["entries"])
    return [load_specimen(path) for path in paths]


class DashboardEngine:
    def __init__(self, *, frames_per_second: float = 30, steps_per_frame: int = 3):
        self.specimens = load_specimens()
        self.frames_per_second = frames_per_second
        self.steps_per_frame = steps_per_frame
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.generation = 0
        self.frame_number = 0
        self.paused = False
        self.running = True
        self.measured_fps = 0.0
        self.last_frame_ms = 0.0
        self.world = self._new_world()
        self.commands: queue.Queue[dict] = queue.Queue()
        state = self.world.numpy().copy()
        self.masses = state.sum(axis=(1, 2)).tolist()
        self.png = encode_png(render_atlas(state))
    def _new_world(self) -> GenesisWorld:
        states = np.stack([centered_state(specimen, SIZE) for specimen in self.specimens])
        world = GenesisWorld(
            GenesisConfig(size=SIZE, batch=len(self.specimens), radius=13, dt=0.1)
        )
        world.state = mx.array(states)
        world.set_growth_parameters(
            [specimen.parameters["growth_center"] for specimen in self.specimens],
            [specimen.parameters["growth_width"] for specimen in self.specimens],
        )
        mx.eval(world.state)
        return world

    def run(self) -> None:
        target = 1 / self.frames_per_second
        previous = time.perf_counter()
        while self.running:
            started = time.perf_counter()
            with self.lock:
                while True:
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        self._apply_control(command["action"], command["index"])
                        command["result"] = self.status()
                    except ValueError as error:
                        command["error"] = error
                    finally:
                        command["event"].set()
                if not self.paused:
                    self.world.step(self.steps_per_frame)
                    self.generation += self.steps_per_frame
                snapshot = self.world.numpy().copy()
            png = encode_png(render_atlas(snapshot))
            finished = time.perf_counter()
            interval = finished - previous
            previous = finished
            with self.condition:
                self.png = png
                self.masses = snapshot.sum(axis=(1, 2)).tolist()
                self.frame_number += 1
                self.last_frame_ms = (finished - started) * 1000
                instant_fps = 1 / interval if interval > 0 else 0
                self.measured_fps = (
                    instant_fps
                    if self.measured_fps == 0
                    else self.measured_fps * 0.9 + instant_fps * 0.1
                )
                self.condition.notify_all()
            time.sleep(max(0, target - (time.perf_counter() - started)))

    def status(self) -> dict:
        with self.lock:
            return {
                "mode": "mlx-server-render",
                "device": mx.device_info().get("device_name", "Apple GPU"),
                "generation": self.generation,
                "frame": self.frame_number,
                "paused": self.paused,
                "target_fps": self.frames_per_second,
                "render_fps": round(self.measured_fps, 1),
                "last_frame_ms": round(self.last_frame_ms, 2),
                "simulation_steps_per_second": round(
                    self.measured_fps * self.steps_per_frame, 1
                ),
                "masses": [round(value, 1) for value in self.masses],
                "names": [specimen.name for specimen in self.specimens],
            }

    def _apply_control(self, action: str, index: int | None = None) -> None:
        if action == "toggle_pause":
            self.paused = not self.paused
        elif action == "reset":
            self.world = self._new_world()
            self.generation = 0
        elif action in {"stress_all", "stress_one"}:
            state = self.world.numpy().copy()
            indexes = range(len(self.specimens)) if action == "stress_all" else [index]
            for habitat in indexes:
                if habitat is None or not 0 <= habitat < len(self.specimens):
                    raise ValueError("habitat index is out of range")
                centre = center_of_mass(state[habitat][None])[0]
                state[habitat], _, _ = calibrated_damage(
                    state[habitat], centre, 0.05
                )
            self.world.state = mx.array(state)
            mx.eval(self.world.state)
        else:
            raise ValueError(f"unknown action: {action}")

    def control(self, action: str, index: int | None = None) -> dict:
        command = {
            "action": action,
            "index": index,
            "event": threading.Event(),
            "result": None,
            "error": None,
        }
        self.commands.put(command)
        if not command["event"].wait(timeout=3):
            raise ValueError("render engine did not acknowledge the command")
        if command["error"]:
            raise command["error"]
        return command["result"]


class DashboardHandler(SimpleHTTPRequestHandler):
    engine: DashboardEngine

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:
        if not self.path.startswith("/api/status"):
            super().log_message(format, *args)

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._json(self.engine.status())
            return
        if path == "/api/frame.png":
            with self.engine.lock:
                body = self.engine.png
                generation = self.engine.generation
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Genesis-Generation", str(generation))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/stream/zoo.mjpeg":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=genesis")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            previous = -1
            try:
                while self.engine.running:
                    with self.engine.condition:
                        self.engine.condition.wait_for(
                            lambda: self.engine.frame_number != previous, timeout=2
                        )
                        previous = self.engine.frame_number
                        body = self.engine.png
                    self.wfile.write(b"--genesis\r\nContent-Type: image/png\r\n")
                    self.wfile.write(f"Content-Length: {len(body)}\r\n\r\n".encode())
                    self.wfile.write(body + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/control":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            request = json.loads(self.rfile.read(length) or b"{}")
            self._json(self.engine.control(request.get("action", ""), request.get("index")))
        except (ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=3)
    args = parser.parse_args()
    engine = DashboardEngine(
        frames_per_second=args.fps, steps_per_frame=args.steps_per_frame
    )
    DashboardHandler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server_thread = threading.Thread(
        target=server.serve_forever, name="genesis-http", daemon=True
    )
    server_thread.start()
    print(
        f"Genesis dashboard: http://{args.host}:{args.port} "
        f"({engine.status()['device']}, server-rendered at {args.fps:g} fps)",
        flush=True,
    )
    try:
        engine.run()
    except KeyboardInterrupt:
        pass
    finally:
        engine.running = False
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
