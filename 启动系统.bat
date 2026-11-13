@echo off
title Bike Demand Prediction System
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo  ============================================
echo    Bike Sharing Demand Prediction System
echo  ============================================
echo.

REM ---------- 1. Free port 8501 (kill old instance) ----------
echo  [1/3] Checking port 8501 ...
set PORT_BUSY=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
    set PORT_BUSY=1
    echo        Port busy PID %%a, killing old instance...
    taskkill /F /PID %%a >nul 2>&1
)
if "!PORT_BUSY!"=="1" (
    timeout /t 2 /nobreak >nul
    echo        [OK] Old instance stopped
) else (
    echo        [OK] Port free
)

REM  Also kill leftover streamlit processes of this project
REM  NOTE(2026-10-17): wmic.exe was removed in Windows 11 26H2.
REM  Replaced by a PowerShell CIM query that works on all supported builds.
powershell -NoProfile -Command "foreach ($x in (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)) { if ($x.Name -like 'python*.exe' -and $x.CommandLine -like '*streamlit*') { Write-Host ('       Killing leftover streamlit PID ' + $x.ProcessId); Stop-Process -Id $x.ProcessId -Force -ErrorAction SilentlyContinue } }"


REM ---------- 2. Check Python ----------
echo  [2/3] Checking Python ...
set PYTHON_CMD=python
if not exist "%PYTHON_CMD%" (
    echo        [ERROR] Python not found: %PYTHON_CMD%
    echo        Please edit PYTHON_CMD at the top of this script.
    pause
    exit /b 1
)
echo        [OK] Python: %PYTHON_CMD%

REM ---------- 3. Start Streamlit ----------
echo  [3/3] Starting system, browser will open automatically ...
echo.
echo        URL: http://localhost:8501
echo        Close this window to stop the server.
echo.
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
start "" cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8501"
"%PYTHON_CMD%" -m streamlit run app/app.py --server.port 8501 --server.headless true

echo.
echo  Server stopped.
pause
