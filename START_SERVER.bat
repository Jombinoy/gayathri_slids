@echo off
echo ========================================
echo   RL Course Presentation Server
echo ========================================
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r server\requirements.txt

echo.
echo Starting server...
echo.
echo ========================================
echo   Server will start on:
echo   http://localhost:5000
echo.
echo   Press CTRL+C to stop
echo ========================================
echo.

cd server
python app.py

pause
