@echo off
title EduBot - Dependency Installer
color 0a
echo ===========================================================
echo       EduBot Dependency Installer (Offline Ready)
echo ===========================================================
echo.

:: --- Check Python Installation ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado o no esta en el PATH.
    echo Por favor instala Python 3.8 o superior y vuelve a intentar.
    pause
    exit /b
)

:: --- Detect Python Version ---
for /f "tokens=2 delims= " %%a in ('python --version 2^>^&1') do set PYVER=%%a
for /f "tokens=1 delims=." %%a in ("%PYVER%") do set PYMAJOR=%%a
for /f "tokens=2 delims=." %%b in ("%PYVER%") do set PYMINOR=%%b
echo Detectando version de Python: %PYVER%

:: --- Create virtual environment ---
if not exist venv (
    echo Creando entorno virtual...
    python -m venv venv
) else (
    echo Entorno virtual ya existe, saltando creacion.
)

:: --- Activate virtual environment ---
set VENV_PATH=%~dp0venv
if exist "%VENV_PATH%\Scripts\activate" (
    call "%VENV_PATH%\Scripts\activate"
) else (
    echo [ERROR] No se encontro el entorno virtual en "%VENV_PATH%"
    pause
    exit /b
)

:: --- Upgrade pip and tools ---
echo.
echo Actualizando pip, setuptools y wheel...
python -m pip install --upgrade pip setuptools wheel

:: --- Clean pip cache ---
echo.
echo Limpiando cache de pip...
pip cache purge >nul 2>&1

:: --- Create dynamic requirements file ---
echo.
echo Creando archivo temporal de dependencias...
(
echo fastapi==0.115.2
echo uvicorn==0.32.0
echo RestrictedPython==8.1
echo PyYAML==6.0.2
echo websockets==13.1
echo flask==3.0.0
echo requests==2.31.0
) > temp_requirements.txt

:: --- Install dependencies ---
echo.
echo Instalando dependencias, por favor espere...
pip install -r temp_requirements.txt --upgrade

:: --- Cleanup ---
del temp_requirements.txt
echo.
echo Instalacion completa.

:: --- Final message ---
echo ===========================================================
echo EduBot listo para ejecutarse offline
echo Activa el entorno con:
echo    venv\Scripts\activate
echo Y ejecuta el servidor con:
echo    python server.py
echo ===========================================================
pause
