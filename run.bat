@echo off
title Sealing Assembly Monitor
echo Starting Sealing Assembly Monitor...

:: Open the default browser after a 2-second delay to let the server bind
start "" cmd /c "timeout /t 2 >nul && start http://127.0.0.1:5000"

call .\venv\Scripts\activate.bat
python app.py
pause
