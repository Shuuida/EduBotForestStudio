#!/bin/bash
echo "==========================================================="
echo "      EduBot Dependency Installer"
echo "==========================================================="

# --- Step 1: Check Python installation ---
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 no esta instalado. Por favor, instalalo primero."
    echo "Ejemplo en Ubuntu: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

# --- Step 2: Detect Python version ---
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detectando version de Python: $PYVER"

# --- Step 3: Create virtual environment ---
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
else
    echo "Entorno virtual ya existe, saltando creacion."
fi

source venv/bin/activate

# --- Step 4: Upgrade pip and tools ---
echo
echo "Actualizando pip, setuptools y wheel..."
pip install --upgrade pip setuptools wheel

# --- Step 5: Create temporary requirements file ---
echo
echo "Creando archivo temporal de dependencias..."
cat > temp_requirements.txt <<EOF
eel==0.16.0
bottle==0.12.25
bottle-websocket==0.2.9
cryptography==41.0.3
requests==2.31.0
PyYAML==6.0.2
pyinstaller==6.3.0
EOF

# --- Step 6: Install dependencies ---
echo
echo "Instalando dependencias..."
pip install -r temp_requirements.txt --upgrade

# --- Step 7: Cleanup ---
rm temp_requirements.txt

echo
echo "==========================================================="
echo "EduBot listo para ejecutarse"
echo "Activa el entorno con:"
echo "   source venv/bin/activate"
echo "Y ejecuta la aplicacion con:"
echo "   python main.py"
echo "==========================================================="