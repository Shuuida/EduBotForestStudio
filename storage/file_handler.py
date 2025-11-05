"""
EduBot Unified File Handler
===========================

Sistema integral para gestión de archivos, modelos ML y traducción bidireccional Python ⇄ Bloques visuales.

Incluye:
 - Guardado y carga de proyectos, datasets y modelos (.edubotproj, .edubotml, .json)
 - Exportación e importación segura sin exec()
 - Conversión entre código Python y estructuras visuales (bloques)
 - Exportación inversa: cualquier modelo/dataset se puede representar como bloque visual

Depende de:
  - core.ml_exporter
  - core.ml_struct_rules
  - core.ml_adapter (para traducción Python <-> Bloques)
"""

import os
import json
#import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Dependencias internas del ecosistema EduBot
from storage import ml_exporter
from core import ml_struct_rules
from core import ml_adapter

# Directorios base
PROJECTS_DIR = "./projects"
BACKUP_DIR = "./backups"
TRASH_DIR = "./trash"
MODELS_DIR = "./models"
DATASETS_DIR = "./datasets"


# ============================================================
# UTILIDADES BÁSICAS

def ensure_dir_exist():
    """Garantiza que todas las carpetas esenciales existan."""
    for path in [PROJECTS_DIR, BACKUP_DIR, TRASH_DIR, MODELS_DIR, DATASETS_DIR]:
        os.makedirs(path, exist_ok=True)


def get_path(base: str, filename: str, ext: str) -> str:
    """Normaliza rutas de archivos por tipo."""
    if not filename.endswith(ext):
        filename += ext
    return os.path.join(base, filename)


# ============================================================
# PROYECTOS EDUCATIVOS

def save_proj(project_data: Dict[str, Any], filename: str) -> bool:
    """Guarda un proyecto EduBot (.edubotproj)"""
    ensure_dir_exist()
    path = get_path(PROJECTS_DIR, filename, ".edubotproj")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Proyecto guardado: {path}")
        return True
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el proyecto: {e}")
        return False

def backup_proj(project_data: Dict[str, Any], filename: str) -> bool:
    """Guarda una copia de seguridad de un proyecto EduBot (.edubotproj)"""
    ensure_dir_exist()
    backup_path = get_path(BACKUP_DIR, filename + "_" + datetime.now().isoformat(), ".edubotproj")
    try:
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Copia de seguridad del proyecto guardada: {backup_path}")
        return True
    except Exception as e:
        print(f"[ERROR] No se pudo guardar la copia de seguridad del proyecto: {e}")
        return False

def load_proj(filename: str) -> Optional[Dict[str, Any]]:
    """Carga un proyecto .edubotproj"""
    ensure_dir_exist()
    path = get_path(PROJECTS_DIR, filename, ".edubotproj")
    if not os.path.exists(path):
        print(f"[WARN] Proyecto no encontrado: {filename}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] No se pudo leer el proyecto: {e}")
        return None


def list_projs() -> List[str]:
    """Lista todos los proyectos existentes"""
    ensure_dir_exist()
    return [f[:-11] for f in os.listdir(PROJECTS_DIR) if f.endswith(".edubotproj")]


# ============================================================
# MODELOS ML (MiniML o scikit-learn)

def save_model(model_obj: Any, filename: str, metadata: Optional[dict] = None) -> bool:
    """
    Guarda un modelo ML como .edubotml
    Incluye metadatos para identificar el framework, modo y fecha.
    """
    ensure_dir_exist()
    path = get_path(MODELS_DIR, filename, ".edubotml")
    try:
        metadata = metadata or {}
        metadata.update({
            "saved_at": datetime.now().isoformat(),
            "framework": "MiniML/sklearn hybrid",
        })
        data = ml_exporter.serialize_model(model_obj, metadata)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Modelo ML exportado: {path}")
        return True
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el modelo: {e}")
        return False
    
def backup_model(model_obj: Any, filename: str, metadata: Optional[dict] = None) -> bool:
    """
    Guarda una copia de seguridad de un modelo ML como .edubotml
    Incluye metadatos para identificar el framework, modo y fecha.
    """
    ensure_dir_exist()
    backup_path = get_path(BACKUP_DIR, filename + "_" + datetime.now().isoformat(), ".edubotml")
    try:
        metadata = metadata or {}
        metadata.update({
            "saved_at": datetime.now().isoformat(),
            "framework": "MiniML/sklearn hybrid",
        })
        data = ml_exporter.serialize_model(model_obj, metadata)
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Copia de seguridad del modelo ML exportada: {backup_path}")
        return True
    except Exception as e:
        print(f"[ERROR] No se pudo guardar la copia de seguridad del modelo: {e}")
        return False

def load_model(filename: str) -> Optional[Any]:
    """Carga un modelo .edubotml"""
    ensure_dir_exist()
    path = get_path(MODELS_DIR, filename, ".edubotml")
    if not os.path.exists(path):
        print(f"[WARN] Modelo no encontrado: {filename}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = ml_exporter.deserialize_model(data)
        print(f"[INFO] Modelo ML cargado: {filename}")
        return model
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo ML: {e}")
        return None


# ============================================================
# DATASETS

def save_dataset(data: List[List[Any]], filename: str) -> bool:
    """Guarda un dataset JSON estructurado."""
    ensure_dir_exist()
    path = get_path(DATASETS_DIR, filename, ".json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Dataset guardado: {path}")
        return True
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el dataset: {e}")
        return False
    
def backup_dataset(data: List[List[Any]], filename: str) -> bool:
    """Guarda una copia de seguridad de un dataset JSON estructurado."""
    ensure_dir_exist()
    backup_path = get_path(BACKUP_DIR, filename + "_" + datetime.now().isoformat(), ".json")
    try:
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Copia de seguridad del dataset guardada: {backup_path}")
        return True
    except Exception as e:
        print(f"[ERROR] No se pudo guardar la copia de seguridad del dataset: {e}")
        return False

def load_dataset(filename: str) -> Optional[List[List[Any]]]:
    """Carga un dataset JSON estructurado."""
    ensure_dir_exist()
    path = get_path(DATASETS_DIR, filename, ".json")
    if not os.path.exists(path):
        print(f"[WARN] Dataset no encontrado: {filename}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] No se pudo leer el dataset: {e}")
        return None


# ============================================================
# TRADUCCIÓN BIDIRECCIONAL PYTHON ⇄ BLOQUES

def python_to_blocks(code: str) -> Optional[Dict[str, Any]]:
    """
    Convierte código Python a estructura de bloques visuales usando ml_adapter.Translator.
    """
    try:
        translator = ml_adapter.Translator()
        blocks = translator.translate_to_blocks(code)
        print(f"[INFO] Código Python convertido a bloques visuales.")
        return {"result": blocks}
    except Exception as e:
        print(f"[ERROR] Fallo en traducción Python→Bloques: {e}")
        return None


def blocks_to_python(blocks: List[Dict[str, Any]]) -> Optional[str]:
    """
    Convierte bloques visuales a código Python usando ml_adapter.Translator.
    """
    try:
        translator = ml_adapter.Translator()
        code = translator.translate_to_python(blocks)
        print(f"[INFO] Bloques visuales convertidos a código Python.")
        return code
    except Exception as e:
        print(f"[ERROR] Fallo en traducción Bloques→Python: {e}")
        return None


# ============================================================
# EXPORTACIÓN / IMPORTACIÓN INTELIGENTE

def auto_import(file_path: str) -> Union[Dict[str, Any], List, Any]:
    """Detecta el tipo de archivo e invoca el método de carga adecuado."""
    if file_path.endswith(".edubotproj"):
        return load_proj(os.path.basename(file_path))
    elif file_path.endswith(".edubotml"):
        return load_model(os.path.basename(file_path))
    elif file_path.endswith(".json"):
        return load_dataset(os.path.basename(file_path))
    elif file_path.endswith(".py"):
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        return python_to_blocks(code)
    else:
        print(f"[WARN] Tipo de archivo no reconocido: {file_path}")
        return None


def auto_export(obj: Any, filename: str) -> bool:
    """
    Guarda automáticamente un objeto (modelo, bloques, código, dataset o proyecto)
    en el formato adecuado según su tipo.
    """
    ensure_dir_exist()
    try:
        # Modelos ML (MiniML / sklearn)
        if hasattr(obj, "fit") or hasattr(obj, "predict"):
            return save_model(obj, filename)

        # Bloques visuales
        elif isinstance(obj, dict) and "type" in obj:
            path = get_path(PROJECTS_DIR, filename, ".edublock.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=4, ensure_ascii=False)
            print(f"[INFO] Bloques exportados a {path}")
            return True

        # Código Python
        elif isinstance(obj, str):
            path = get_path(PROJECTS_DIR, filename, ".py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(obj)
            print(f"[INFO] Código Python guardado en {path}")
            return True

        # Dataset
        elif isinstance(obj, list) and all(isinstance(row, list) for row in obj):
            return save_dataset(obj, filename)

        # Proyecto EduBot
        elif isinstance(obj, dict):
            return save_proj(obj, filename)

        else:
            print(f"[WARN] Tipo de objeto desconocido: {type(obj).__name__}")
            return False

    except Exception as e:
        print(f"[ERROR] Error durante auto_export: {e}")
        return False


# ============================================================
# EXPORTACIÓN INVERSA A BLOQUES

def export_to_blocks(obj: Any, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Convierte cualquier modelo, dataset o estructura ML a su representación en bloques visuales EduBot.
    También soporta conversión inversa desde código Python.
    """
    ensure_dir_exist()
    block_struct = None

    if isinstance(obj, dict):
        if 'model' in obj:
            obj = obj['model']  # Extrae la instancia real
        elif all(k in obj for k in ['mode', 'type', 'framework']):
            # Es meta puro, intenta recuperar model si hay 'model_name'
            if 'model_name' in obj:
                from core.ml_manager import get_model
                obj = get_model(obj['model_name'])
            else:
                print("[WARN] Meta dict sin 'model' o 'model_name'. Usando fallback.")
                struct = obj
                return {"type": "unsupported", "raw": struct}

    # Caso 1: modelo ML (ampliado para MiniML nativo de EduBot)
    if any(hasattr(obj, m) for m in ["predict", "fit", "train", "classify", "forward", "run"]):
        try:
            struct = ml_exporter.extract_model_structure(obj)
            if struct:
                block_struct = ml_struct_rules.struct_to_visual_block(struct)  # Corregido: era struct_to_block
                print(f"[INFO] Modelo convertido a bloque visual ML.")
        except Exception as e:
            print(f"[WARN] No se pudo convertir modelo a estructura visual: {e}")

    # Caso 2: dataset
    elif isinstance(obj, list) and all(isinstance(row, list) for row in obj):
        block_struct = {
            "type": "ml_dataset",
            "source": "inline",
            "data": obj,
            "name": "dataset_auto"
        }
        print(f"[INFO] Dataset convertido a bloque visual ML.")

    # Caso 3: código Python -> Bloques
    elif isinstance(obj, str):
        block_struct = python_to_blocks(obj)

    elif block_struct is None:
        print(f"[WARN] Estructura del modelo no reconocida por ml_exporter: {type(obj).__name__}")
        struct = {"framework": "MiniML", "type": "UnknownModel", "repr": repr(obj)}
        block_struct = {"type": "unsupported", "raw": struct}

    else:
        raise TypeError("Objeto no reconocido o no soportado para exportación a bloques.")

    if output_file:
        path = get_path(PROJECTS_DIR, output_file, ".edublock.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(block_struct, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Bloque exportado: {path}")

    return block_struct


# ============================================================
# INICIALIZACIÓN

if __name__ == "__main__":
    ensure_dir_exist()
    print("[INFO] Sistema unificado EduBot (ML + Traducción + Archivos) inicializado correctamente.")