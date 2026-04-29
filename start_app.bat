@echo off
title SignTranslate Server Manager

echo [1/3] Starting SignTranslate Backend API...
start "SignTranslate Backend" cmd /k "cd /d "%~dp0" && uvicorn backend.app:app --reload"

echo [2/3] Starting SignTranslate React Frontend...
start "SignTranslate Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo [3/3] Waiting for servers to initialize...
timeout /t 6 /nobreak >nul

echo Opening your default browser...
start http://localhost:5173

echo Both servers are running in separate windows. 
echo Close those windows to stop the servers.
pause
