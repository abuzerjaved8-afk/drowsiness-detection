@echo off
cd /d "%~dp0"
if exist "%~dp0.facedetection.pid" (
    set /p PID=<"%~dp0.facedetection.pid"
    taskkill /PID %PID% /T /F >nul 2>&1
    del "%~dp0.facedetection.pid"
    echo Face detection stopped.
) else (
    echo No PID file found. Searching for a running instance...
    powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*face_detection.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    echo Done.
)
timeout /t 2 >nul
