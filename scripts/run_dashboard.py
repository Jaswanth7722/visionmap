"""
PS26053 Dashboard Launcher
Starts the Streamlit Live Dynamic Perception Dashboard.
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    dashboard_script = REPO_ROOT / "python" / "visualization" / "dashboard.py"
    print("==============================================================")
    print("  PS 26053: Live 2.5D Adaptive LiDAR Perception Dashboard      ")
    print("  DRDO Smart Vehicles | Software Autonomous Navigation        ")
    print("==============================================================")
    print(f"Launching dashboard: {dashboard_script}")
    print("Starting Streamlit server on http://localhost:8501 ...")

    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        str(dashboard_script),
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
