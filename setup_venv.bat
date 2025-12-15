@echo off
echo ========================================
echo   RL Presentation - Virtual Environment Setup
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.8+ is installed
    pause
    exit /b 1
)
echo ✓ Virtual environment created
echo.

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

echo [3/4] Installing requirements...
pip install -r server\requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)
echo ✓ Requirements installed
echo.

echo [4/4] Setup complete!
echo.
echo ========================================
echo   Setup Successful!
echo ========================================
echo.
echo To start the server in the future:
echo   1. Run: run_server.bat
echo   2. Or manually: venv\Scripts\activate ^&^& cd server ^&^& python app.py
echo.
pause
