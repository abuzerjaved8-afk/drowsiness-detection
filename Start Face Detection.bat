@echo off
cd /d "%~dp0"
echo Starting face detection...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -FilePath 'python' -ArgumentList 'face_detection.py' -WorkingDirectory '%~dp0' -PassThru; $p.Id | Out-File -Encoding ascii '%~dp0.facedetection.pid'"
echo Camera window should open in a few seconds. You can close this window.
timeout /t 2 >nul
