"""
EduBot Unified File Handler
=========================================
Sistema integral para gestión de archivos y persistencia.
Sincronizado con ml_runtime v2.3 y ml_exporter v2.5.

Correcciones:
- save_model: Ajustado para enviar solo model_obj a serialize_model.
- export_to_blocks: Conectado a la nueva API de visualización.
- Directorios: Autocreación robusta.
"""

import os
import json
import sys
#import shutil
from datetime import datetime
from typing import Any, Dict, List

# Dependencias internas
from storage import ml_exporter
from core import ml_manager
#from core import ml_struct_rules
#from core import ml_adapter

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
#MODELS_DIR = os.path.join(BASE_PATH, "models")
#DATASETS_DIR = os.path.join(BASE_PATH, "datasets")
#EXPORTS_DIR = os.path.join(BASE_PATH, "exports")
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

# -------------------------------------------------
# GESTIÓN DE MODELOS (.edubotml)

#*def save_model(model_name: str, filename: str) -> bool:
    """
    Guarda un modelo entrenado (desde la memoria) al disco.
    """
    try:
        ensure_dir_exist()
        
        # Obtener modelo de memoria
        model = ml_manager.get_model(model_name)
        if not model:
            print(f"[ERROR] Modelo '{model_name}' no encontrado en registro.")
            return False

        # Serializar
        model_data = ml_exporter.serialize_model(model)
        
        # Agregar metadatos de archivo
        model_data["_meta"] = {
            "filename": filename,
            "saved_at": datetime.now().isoformat(),
            "original_name": model_name
        }

        # Escribir a disco
        path = _get_path(MODELS_DIR, filename, ".edubotml")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=4)
            
        print(f"[INFO] Modelo guardado: {path}")
        return True

    except Exception as e:
        print(f"[ERROR] No se pudo guardar el modelo: {e}")
        return False

#*def load_model(filename: str) -> Any:
    """
    Carga un modelo desde disco y lo registra en ml_manager.
    Retorna el objeto modelo instanciado.
    """
    try:
        path = _get_path(MODELS_DIR, filename, ".edubotml")
        if not os.path.exists(path):
            print(f"[ERROR] Archivo no encontrado: {path}")
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Deserializar (Reconstruir objeto Python)
        model = ml_exporter.deserialize_model(data)
        
        if model:
            # Registrar en Manager (usando el nombre original o el filename)
            name = data.get("_meta", {}).get("original_name", filename)
            # Registrar explícitamente en el manager
            ml_manager.register_model(name, model)
            print(f"[INFO] Modelo cargado y registrado como '{name}'")
            return model
        
        return None

    except Exception as e:
        print(f"[ERROR] Falló carga de modelo: {e}")
        return None

# ------------------------------------------------
# GESTIÓN DE DATASETS (.json)

#*def save_dataset(data: List[List[float]], name: str) -> bool:
    try:
        ensure_dir_exist()
        path = _get_path(DATASETS_DIR, name, ".json")
        wrapper = {
            "type": "dataset",
            "name": name,
            "data": data,
            "created_at": datetime.now().isoformat()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, indent=None) # Minificado para velocidad
        return True
    except Exception as e:
        print(f"[ERROR] Guardando dataset: {e}")
        return False

#*def _parse_csv_line(line: str) -> List[float]:
    """Helper seguro para convertir línea CSV a lista de floats."""
    try:
        # Divide por comas y limpia espacios
        parts = [p.strip() for p in line.split(',')]
        # Intenta convertir todo a float
        return [float(p) for p in parts if p]
    except ValueError:
        # Si falla (ej: es un header 'x1,x2,y'), retorna None
        return None

#*def load_dataset(name: str) -> List[List[float]]:
    """
    Carga datasets soportando tanto JSON nativo como CSV importado.
    Prioriza CSV si se indica explícitamente o si existe el archivo.
    """
    try:
        # Limpieza del nombre (quitar extensiones si el usuario las puso)
        clean_name = name.replace(".json", "").replace(".csv", "")
        
        # INTENTO DE CARGA CSV (Prioridad para importaciones externas)
        csv_path = os.path.join(DATASETS_DIR, clean_name + ".csv")
        
        if os.path.exists(csv_path):
            print(f"[INFO] Detectado dataset CSV: {csv_path}")
            data = []
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                if not line.strip(): continue # Saltar líneas vacías
                
                # Parsear línea
                row = _parse_csv_line(line)
                
                if row is not None:
                    data.append(row)
                elif i == 0:
                    # Si la primera línea falla, asumimos que es Header y no hacemos nada
                    print(f"[INFO] Header CSV detectado e ignorado: {line.strip()}")
                else:
                    print(f"[WARN] Fila {i+1} inválida en CSV (ignorada): {line.strip()}")
            
            if not data:
                print(f"[WARN] El archivo CSV '{csv_path}' existe pero no contiene datos numéricos válidos.")
                return None
                
            return data

        # INTENTO DE CARGA JSON (Formato nativo EduBot)
        json_path = os.path.join(DATASETS_DIR, clean_name + ".json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                wrapper = json.load(f)
            return wrapper.get("data", [])

        # No encontrado
        print(f"[ERROR] Dataset no encontrado (ni .csv ni .json): {clean_name}")
        return None

    except Exception as e:
        print(f"[ERROR] Falló carga de dataset '{name}': {e}")
        return None

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

# -------------------------------------------------------
# EXPORTACIÓN INVERSA (MODELO -> BLOQUES)

#*def export_to_blocks(model_obj: Any, model_name: str) -> Dict[str, Any]:
    """
    Convierte un objeto modelo en memoria a su representación visual (Bloque).
    Usa la nueva API de ml_exporter.
    """
    try:
        return ml_exporter.export_model_to_block_structure(model_obj, model_name)
    except Exception as e:
        print(f"[ERROR] Falló exportación a bloques: {e}")
        return None

# --------------------------------------------------
# AUTO-EXPORT (Wrapper Genérico)

#*def auto_export(content: Any, name: str) -> bool:
    """Detecta el tipo de contenido y lo guarda apropiadamente."""
    if isinstance(content, dict) and ("blocks" in content or "code" in content):
        return save_proj(content, name)
    elif isinstance(content, list):
        return save_dataset(content, name)
    # Si es un objeto de modelo desconocido, habría que serializarlo primero
    # Por seguridad, no guardamos objetos arbitrarios aquí.
    return False

#*def auto_import(path: str) -> Any:
    """Carga inteligente basada en extensión."""
    if path.endswith(".edubotproj"):
        return load_proj(os.path.basename(path).replace(".edubotproj", ""))
    elif path.endswith(".json"):
        # Asumir dataset
        return load_dataset(os.path.basename(path).replace(".json", ""))
    elif path.endswith(".edubotml"):
        return load_model(os.path.basename(path).replace(".edubotml", ""))
    return None