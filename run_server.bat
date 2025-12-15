@echo off
echo ========================================
echo   RL Presentation Server
echo ========================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run setup_venv.bat first
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting server...
echo.
echo ========================================
echo   Server starting on:
echo   http://localhost:5000
echo.
echo   Press CTRL+C to stop
echo ========================================
echo.

cd server
python app.py

pause
