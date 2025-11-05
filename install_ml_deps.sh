#!/bin/bash
# =======================================================
# EduBot - Instalador de dependencias para módulos de ML
# =======================================================
# Activa el entorno virtual, verifica e instala dependencias
# Autor: Michego Takoro
# =======================================================

echo ""
echo "=============================================="
echo "     Instalador de dependencias ML - EduBot"
echo "=============================================="
echo ""

# Ruta del entorno virtual
VENV_PATH="$(dirname "$0")/venv"

# Verificar existencia del entorno virtual
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "No se encontro el entorno virtual en: $VENV_PATH"
    echo "Por favor crea el entorno ejecutando:"
    echo "python3 -m venv venv"
    exit 1
fi

# Activar entorno virtual
echo "Activando entorno virtual..."
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

echo ""
echo "Comprobando e instalando dependencias necesarias..."
echo "--------------------------------------------------------------"

# Lista de dependencias requeridas
DEPS=("pandas" "numpy" "scikit-learn" "joblib" "matplotlib")

# Carpeta para logs
LOG_DIR="$(dirname "$0")/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/install_ml_deps.log"

# Verificar e instalar dependencias
for dep in "${DEPS[@]}"; do
    echo "Verificando $dep..."
    python -c "import $dep" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo " Instalando $dep..."
        pip install "$dep" --quiet >>"$LOG_FILE" 2>&1
        if [ $? -ne 0 ]; then
            echo "Error instalando $dep. Revisa el log: $LOG_FILE"
        else
            echo "$dep instalado correctamente."
        fi
    else
        echo "$dep ya esta instalado."
    fi
    echo "--------------------------------------------------------------"
done

echo ""
echo "=============================================="
echo "  Instalacion de dependencias completada"
echo "  Log guardado en: $LOG_FILE"
echo "=============================================="
echo ""

deactivate
exit 0