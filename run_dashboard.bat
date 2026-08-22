@echo off
REM Jigo dashboard launcher - uses the real Python install, not the Store stub
cd /d "%~dp0"
"C:\Users\kadam\AppData\Local\Python\pythoncore-3.14-64\python.exe" web_ui.py
pause
