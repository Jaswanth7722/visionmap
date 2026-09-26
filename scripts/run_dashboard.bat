@echo off
REM ==============================================================
REM  PS 26053: Live 2.5D Adaptive LiDAR Perception Dashboard
REM  DRDO Smart Vehicles | One-Click Real-Time Dashboard Launcher
REM ==============================================================

cd /d "%~dp0\.."
echo Starting PS26053 Live Perception Dashboard...
REM H12: resolve the interpreter from PATH (with an opt-out override) instead
REM of a hardcoded machine-specific absolute path.
if defined PS26053_PYTHON (
    "%PS26053_PYTHON%" scripts\run_dashboard.py
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [Error] No 'python' on PATH. Install Python 3.10+ with the packages
        echo in requirements.txt, or set PS26053_PYTHON to your interpreter.
        pause
        exit /b 1
    )
    python scripts\run_dashboard.py
)
pause
