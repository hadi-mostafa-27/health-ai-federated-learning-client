@echo off
setlocal

cd /d "%~dp0"

echo Building HospitalFLSystem Windows executable...
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe not found.
    echo Create/activate your virtual environment and install requirements first.
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean HospitalFLSystem.spec
if errorlevel 1 exit /b 1

echo.
echo Build complete:
echo dist\HospitalFLSystem\HospitalFLSystem.exe
echo.
echo Test the executable before building the installer.

endlocal
