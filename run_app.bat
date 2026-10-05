@echo off
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed or not added to PATH.
    echo Install Python from https://www.python.org/downloads/
    echo During installation, tick "Add python.exe to PATH".
    pause
    exit /b 1
)

python -m streamlit --version >nul 2>nul
if errorlevel 1 (
    echo Streamlit is not installed yet.
    echo Installing project packages now...
    python -m pip install -r requirements.txt
)

python -m streamlit run app.py
pause
