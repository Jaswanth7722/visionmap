"""
PS26053 Dashboard Launcher
Starts the native C++ live dashboard server (the Streamlit dashboard was
removed: the runtime is C++-only for low RAM and higher FPS).
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXE_NAMES = (
    "live_dashboard_server.exe" if os.name == "nt" else "live_dashboard_server"
)


def find_server_binary() -> Path:
    candidate = REPO_ROOT / "build" / "bin" / EXE_NAMES
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"C++ dashboard server not found at {candidate}. "
        "Build it first: cmake --build build --parallel 4"
    )


def main(argv=None):
    port = "8080"
    extra = []
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        port = args[0]
    if len(args) > 1:
        # Optional scan file or sequence directory, forwarded to the server.
        extra = [args[1]]
    server = find_server_binary()
    print("==============================================================")
    print("  PS 26053: Live 2.5D Adaptive LiDAR Perception Dashboard      ")
    print("  DRDO Smart Vehicles | Native C++ Runtime                     ")
    print("==============================================================")
    print(f"Serving dashboard: {server} on http://localhost:{port} ...")
    os.execv(str(server), [str(server), port] + extra)


if __name__ == "__main__":
    main()
