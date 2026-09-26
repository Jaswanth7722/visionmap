"""
PS26053 Dashboard Launcher (one command, zero manual steps).
Builds if needed, starts the native C++ server, waits until it answers,
then opens the dashboard in your browser.

Usage:
    python scripts/run_dashboard.py [port] [scan|sequence_dir] [--watch DIR]
        [--watch-interval SEC] [--loop] [--no-browser] [--no-build]

Examples:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py 8080 data/raw --watch data/raw --watch-interval 3
    python scripts/run_dashboard.py 8080 C:/lidar_logs --watch C:/lidar_logs
"""

import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXE_NAMES = (
    "live_dashboard_server.exe" if os.name == "nt" else "live_dashboard_server"
)


def server_binary() -> Path:
    return REPO_ROOT / "build" / "bin" / EXE_NAMES


def ensure_built() -> None:
    if server_binary().is_file():
        return
    print("[Launcher] Server binary missing; building (one-time cost)...")
    subprocess.run(
        ["cmake", "--build", "build", "--parallel", "4"],
        cwd=str(REPO_ROOT),
        check=True,
    )


def wait_for_health(port: str, timeout_s: int = 120) -> None:
    url = f"http://localhost:{port}/api/status"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[Launcher] Server answering at {url}")
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server did not answer {url} within {timeout_s}s.")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    port = "8080"
    passthrough = []
    open_browser = True
    do_build = True
    positional = 0
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--no-browser":
            open_browser = False
        elif a == "--no-build":
            do_build = False
        elif a in ("--watch", "--watch-interval"):
            passthrough.extend([a, args[i + 1]])
            i += 1
        elif a == "--loop":
            passthrough.append(a)
        elif positional == 0:
            port, positional = a, 1
        elif positional == 1:
            passthrough.append(a)
            positional = 2
        else:
            passthrough.append(a)
        i += 1

    if do_build:
        ensure_built()
    if not server_binary().is_file():
        raise FileNotFoundError(
            f"C++ dashboard server not found at {server_binary()}. "
            "Build it: cmake --build build --parallel 4"
        )

    print("==============================================================")
    print("  PS 26053: Live 2.5D Adaptive LiDAR Perception Dashboard      ")
    print("  DRDO Smart Vehicles | Native C++ Runtime                     ")
    print("==============================================================")
    proc = subprocess.Popen(
        [str(server_binary()), port] + passthrough,
        cwd=str(REPO_ROOT),
    )
    try:
        wait_for_health(port)
    except RuntimeError as exc:
        proc.terminate()
        raise SystemExit(f"[Launcher] {exc}")
    url = f"http://localhost:{port}/"
    print(f"[Launcher] Dashboard live at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[Launcher] Stopping server...")
        proc.terminate()


if __name__ == "__main__":
    main()
