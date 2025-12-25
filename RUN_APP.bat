@echo off
echo ========================================
echo AI Stock Trading Assistant - Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if packages are installed
echo Checking if required packages are installed...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages (this may take a few minutes)...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install packages
        pause
        exit /b 1
    )
    echo Packages installed successfully!
    echo.
) else (
    echo All packages are already installed!
    echo.
)

REM Run the Streamlit app
echo Starting Streamlit application...
echo.
echo The app will open in your browser automatically.
echo To stop the app, press Ctrl+C in this window.
echo.
python -m streamlit run app.py

pause

