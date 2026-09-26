@echo off
REM ==============================================================
REM  PS 26053: Live 2.5D Adaptive LiDAR Perception Dashboard
REM  DRDO Smart Vehicles | One-Click Real-Time Dashboard Launcher
REM ==============================================================

cd /d "%~dp0\.."
echo Starting PS26053 Live Perception Dashboard...
D:\python\python.exe scripts\run_dashboard.py
pause
