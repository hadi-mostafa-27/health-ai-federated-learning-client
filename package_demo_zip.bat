@echo off
setlocal

set "INPUT=installer\Output\HospitalFLSystem_Setup.exe"
set "OUTPUT_DIR=release"
set "OUTPUT=%OUTPUT_DIR%\HospitalFLSystem_Demo_Setup.zip"

if not exist "%INPUT%" (
    echo Missing installer: %INPUT%
    echo Build the installer first using Inno Setup, then run this script again.
    exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Creating demo ZIP...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -LiteralPath '%INPUT%' -DestinationPath '%OUTPUT%' -Force"

if errorlevel 1 (
    echo Failed to create demo ZIP.
    exit /b 1
)

echo Created: %OUTPUT%
echo Upload this ZIP to GitHub Releases. Do not commit it to the source repository.
endlocal
