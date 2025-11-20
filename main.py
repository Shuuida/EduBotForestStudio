"""
EduBot Unified Desktop Entry Point
==================================

Este módulo fusiona la capacidad de interfaz de Eel con la lógica
de backend de EduBot en server.py.

"""

import eel
import os
import json
import yaml

#IMPORTACIONES DEL NÚCLEO DE EDUBOT
from core.ml_adapter import Translator, execute_structs
from core.ml_struct_rules import block_to_struct as ml_block_to_struct
from storage import ml_exporter
from storage import file_handler
from core.executor import execute_user_code

# 1. Inicialización de Eel
# Aseguramos que la carpeta web exista
if not os.path.exists('web'):
    print("[ERROR CRÍTICO] La carpeta 'web' no existe. Asegúrate de haber creado el index.html allí.")
    exit()

# Inicializamos apuntando a la carpeta donde está index.html
eel.init('web')

# ---------------------------------------------------------
# FUNCIONES EXPUESTAS A JAVASCRIPT (API DE EEL)

@eel.expose
def get_app_status():
    """Equivalente a @app.route('/status')"""
    return {
        "status": "online",
        "modules": {
            "translator": True,
            "ml_adapter": True,
            "ml_exporter": True,
            "file_handler": True
        }
    }

# TRADUCCIÓN
@eel.expose
def api_translate(direction, blocks=None, code=None, mode="code"):
    """Equivalente a @app.route('/translate')"""
    try:
        translator = Translator()
        
        if mode == "struct" and direction == "to_struct":
            result = ml_exporter.export_blocks_to_struct(blocks)
            return result

        elif direction == "to_python":
            python_code = translator.translate_to_python(blocks)
            return {"result": python_code}

        elif direction == "to_blocks":
            blocks_result = translator.translate_to_blocks(code)
            return {"result": blocks_result}
        
        else:
            return {"error": "Dirección o modo inválido"}

    except Exception as e:
        return {"error": f"Error de traducción: {str(e)}"}

#EJECUCIÓN DE CÓDIGO
@eel.expose
def api_run_code(code):
    """Equivalente a @app.route('/run')"""
    try:
        if not code or not code.strip():
            return {"error": "Código vacío"}
        
        result = execute_user_code(code)
        # Retornamos el dict directamente
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

# EJECUCIÓN ML
@eel.expose
def api_ml_execute(blocks):
    """Equivalente a @app.route('/ml_execute')"""
    try:
        if not blocks:
            return {"error": "No se recibieron bloques"}

        structs = []
        for b in blocks:
            if isinstance(b, dict) and "action" in b:
                structs.append(b)
            else:
                structs.append(ml_block_to_struct(b))

        result = execute_structs(structs)
        return result
    except Exception as e:
        return {"error": str(e)}

# GESTIÓN DE ARCHIVOS
@eel.expose
def api_file_save(name, content):
    """Equivalente a @app.route('/file/save')"""
    try:
        success = file_handler.auto_export(content, name)
        return {"status": "saved" if success else "failed"}
    except Exception as e:
        return {"error": str(e)}

@eel.expose
def api_file_load(path):
    """Equivalente a @app.route('/file/load')"""
    try:
        content = file_handler.auto_import(path)
        return {"content": content}
    except Exception as e:
        return {"error": str(e)}

@eel.expose
def api_file_list():
    """Equivalente a @app.route('/file/list')"""
    try:
        # Aseguramos que existan los directorios antes de listar
        file_handler.ensure_dir_exist()
        
        files = {
            "projects": file_handler.list_projs(),
            "models": os.listdir("./models") if os.path.exists("./models") else [],
            "datasets": os.listdir("./datasets") if os.path.exists("./datasets") else []
        }
        return files
    except Exception as e:
        return {"error": str(e)}

# --------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL

if __name__ == "__main__":
    print("Iniciando EduBot Forest Studio...")
    print("Backend integrado listo. Abriendo interfaz...")

    # Aseguramos directorios críticos
    os.makedirs("exports", exist_ok=True)
    file_handler.ensure_dir_exist()

    # Iniciamos la App
    # port=0 busca un puerto libre automáticamente (evita conflictos con 5000 u 8000)
    try:
        eel.start('index.html', mode='default', size=(1280, 800), port=0)
    except EnvironmentError:
        # Si no encuentra Chrome/Edge, intenta abrir en el navegador del sistema con fallback
        print("Navegador no detectado automáticamente. Intentando modo fallback...")
        eel.start('index.html', mode='user_default', size=(1280, 800), port=0)