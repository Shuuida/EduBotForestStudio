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

# Inicialización de Eel
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
    return {
        "status": "online",
        "modules": {
            "translator": True,
            "ml_adapter": True,
            "ml_exporter": True,
            "file_handler": True
        }
    }

# GESTIÓN DE PROYECTOS (.edubotproj)
@eel.expose
def api_save_project(data, filename):
    """Guarda el estado completo del editor (nodos y conexiones)"""
    try:
        # file_handler.save_proj maneja la extensión y directorios
        success = file_handler.save_proj(data, filename)
        return {'success': success}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_load_project(filename):
    """Carga un proyecto guardado"""
    try:
        data = file_handler.load_proj(filename)
        if data:
            return data # Devuelve el JSON directo al frontend
        else:
            return {'error': 'Proyecto no encontrado o ilegible'}
    except Exception as e:
        return {'error': str(e)}

# TRADUCCIÓN Y ESTRUCTURAS
@eel.expose
def api_translate(direction, blocks=None, code=None, mode="code"):
    """
    Maneja traducciones:
    - Bloques -> Código Python
    - Código -> Bloques
    - Bloques -> Estructura JSON (para exportar a C)
    """
    try:
        translator = Translator()
        
        if mode == "struct" and direction == "to_struct":
            # Usado para exportar a C / Firmware
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

# EJECUCIÓN (ML y Código Usuario)
@eel.expose
def api_execute(blocks, mode='auto'):
    """
    Ejecuta el pipeline de bloques ML directamente.
    Esta es la función que llama el botón 'Ejecutar Flujo'.
    """
    try:
        if not blocks:
            return {"error": "No se recibieron bloques para ejecutar"}

        # Convertir bloques visuales a estructuras lógicas
        structs = []
        for b in blocks:
            # Si ya tiene formato de acción, lo usamos, si no, lo convertimos
            if isinstance(b, dict) and "action" in b:
                structs.append(b)
            else:
                # Usa las reglas de EduBot para interpretar el bloque visual
                structs.append(ml_block_to_struct(b))

        # Ejecutar la lógica (entrenamiento, predicción, etc.)
        # execute_structs devuelve un dict con {success, message, model_name, etc.}
        result = execute_structs(structs)
        return result

    except Exception as e:
        return {"success": False, "error": f"Error interno de ejecución: {str(e)}"}

@eel.expose
def api_run_code(code):
    """Ejecuta código Python arbitrario (sandbox)"""
    try:
        if not code or not code.strip():
            return {"error": "Código vacío"}
        return execute_user_code(code)
    except Exception as e:
        return {"success": False, "error": str(e)}

# GESTIÓN DE ARCHIVOS GENÉRICOS
@eel.expose
def api_file_save(content, name, file_type="auto"):
    """
    Guarda archivos generados (ej. exportaciones a C, datasets).
    NOTA: El orden de argumentos se ajustó para coincidir con JS (content, name).
    """
    try:
        # file_handler.auto_export deduce la extensión o usa el contenido
        success = file_handler.auto_export(content, name)
        return {"success": success}
    except Exception as e:
        return {"success": False, "error": str(e)}

@eel.expose
def api_file_list():
    try:
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
    print("---------------------------------------------------")
    print("Iniciando EduBot Forest Studio...")
    print("Backend integrado listo. Abriendo interfaz...")
    print("👉 Abre esta dirección en tu navegador: http://localhost:8080")
    print("---------------------------------------------------")

    # Aseguramos directorios críticos
    os.makedirs("exports", exist_ok=True)
    file_handler.ensure_dir_exist()

    # Iniciamos la App
    try:
        eel.start(
                    'index.html',
                    mode=None,           # IMPORTANTE: No intentar abrir navegador automático (falla en nube)
                    host='localhost',    # IMPORTANTE: Escuchar en todas las interfaces para que el Proxy de Google entre
                    port=8080,           # IMPORTANTE: Puerto fijo para configurar el Port Forwarding
                    block=True           # Mantiene el script corriendo
                )
    except EnvironmentError:
        # Si no encuentra Chrome/Edge, intenta abrir en el navegador del sistema con fallback
        print("Navegador no detectado automáticamente. Intentando modo fallback...")
        eel.start('index.html', mode='user_default', size=(1280, 800), host= 'localhost', port=8080, block=True)