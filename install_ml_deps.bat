@echo off
:: =======================================================
:: EduBot - Instalador de dependencias para módulos de ML
:: =======================================================
:: Activa el entorno virtual y verifica las librerías requeridas
:: Autor: Michego Takoro
:: =======================================================

setlocal enabledelayedexpansion

echo.
echo ==============================================
echo     Instalador de dependencias ML - EduBot
echo ==============================================
echo.

:: Ruta del entorno virtual
set "VENV_PATH=%~dp0venv"

:: Verificar que el entorno exista
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo No se encontro el entorno virtual en "%VENV_PATH%"
    echo Por favor crea el entorno antes de ejecutar este instalador.
    pause
    exit /b 1
)

:: Activar entorno virtual
echo Activando entorno virtual...
call "%VENV_PATH%\Scripts\activate.bat"

echo.
echo Comprobando e instalando dependencias necesarias...
echo --------------------------------------------------------------

:: Lista de dependencias requeridas
set DEPS=scikit-learn, joblib

:: Crear carpeta logs si no existe
if not exist "%~dp0logs" mkdir "%~dp0logs"
set LOG_FILE=%~dp0logs\install_ml_deps.log

:: Verificar e instalar dependencias
for %%D in (%DEPS%) do (
    echo Verificando %%D...
    python -c "import %%D" 2>nul
    if errorlevel 1 (
        echo Instalando %%D...
        pip install %%D --quiet >> "%LOG_FILE%" 2>&1
        if errorlevel 1 (
            echo Error instalando %%D. Revisa el log: %LOG_FILE%
        ) else (
            echo %%D instalado correctamente.
        )
    ) else (
        echo %%D ya esta instalado.
    )
    echo --------------------------------------------------------------
)

echo.
echo ==============================================
echo   Instalacion de dependencias completada
echo   Log guardado en: %LOG_FILE%
echo ==============================================
echo.

pause
endlocal