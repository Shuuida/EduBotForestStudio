"""
EduBot Unified File Handler
=========================================
Sistema integral para gestión de archivos y persistencia.

Correcciones:
- save_model: Ajustado para enviar solo model_obj a serialize_model.
- Directorios: Autocreación robusta.
"""

import os
import json
import sys
#import shutil
from datetime import datetime
from typing import Any, Dict, List

#-----------------------------------------------
# Función para obtener la ruta real donde está el .exe
def get_base_path():
    if getattr(sys, 'frozen', False):
        # Si estamos ejecutando desde el .exe (PyInstaller)
        return os.path.dirname(sys.executable)
    else:
        # Si estamos ejecutando desde código fuente
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(current_dir)

BASE_PATH = get_base_path()

# Directorios base
PROJECTS_DIR = os.path.join(BASE_PATH, "projects")
TRASH_DIR = os.path.join(BASE_PATH, "trash")

# ---------------------------------------------
# UTILIDADES BÁSICAS

def ensure_dir_exist():
    """Garantiza que todas las carpetas esenciales existan."""
    for path in [PROJECTS_DIR, TRASH_DIR]:
        os.makedirs(path, exist_ok=True)

def _get_path(directory: str, name: str, extension: str) -> str:
    """Genera una ruta segura asegurando la extensión."""
    if not name.endswith(extension):
        name += extension
    return os.path.join(directory, name)

# -----------------------------------------------------
# GESTIÓN DE PROYECTOS (.edubotproj)

def save_proj(project_data: Dict, filename: str) -> bool:
    try:
        ensure_dir_exist()
        path = _get_path(PROJECTS_DIR, filename, ".edubotproj")
        
        # Validar estructura mínima
        if "blocks" not in project_data and "code" not in project_data:
            print("[WARN] Guardando proyecto vacío.")

        project_data["_meta"] = {"saved_at": datetime.now().isoformat()}
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=4)
        return True
    except Exception as e:
        print(f"[ERROR] Save Project: {e}")
        return False

def load_proj(filename: str) -> Dict:
    try:
        path = _get_path(PROJECTS_DIR, filename, ".edubotproj")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Load Project: {e}")
        return None

def list_projs() -> List[str]:
    ensure_dir_exist()
    return [f for f in os.listdir(PROJECTS_DIR) if f.endswith(".edubotproj")]