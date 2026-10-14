@echo off
REM Jigo dashboard launcher. Prefers the real Python install (the Windows Store "python" stub fails silently).
cd /d "%~dp0"
set "PY=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" web_ui.py
pause
