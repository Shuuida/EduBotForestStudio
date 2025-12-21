"""
EduBot Unified Desktop Entry Point
==================================

Este módulo fusiona la capacidad de interfaz de Eel con la lógica
de backend de EduBot en server.py.

"""

import eel
import os
#import json
#import yaml

#IMPORTACIONES DEL NÚCLEO DE EDUBOT
from core.ml_adapter import Translator, execute_structs
from core.ml_struct_rules import block_to_struct as ml_block_to_struct
from storage import ml_exporter
from storage import file_handler
from core.executor import execute_user_code
from core import ml_manager
from estimators import memory_estimator

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

# EJECUCIÓN INTELIGENTE(ML y Código Python)
@eel.expose
def api_execute(blocks, mode='auto'):
    """
    Ejecuta el pipeline de bloques.
    Detecta automáticamente si es una simulación ML o un script de Python.
    """
    try:
        if not blocks:
            return {"error": "No se recibieron bloques para ejecutar"}

        # Detección de Modo
        # Si encontramos bloques que empiezan con 'py_', asumimos modo Python Fundamentos
        has_python_logic = any(b.get('type', '').startswith('py_') for b in blocks)

        if has_python_logic:
            # RUTA PYTHON (Ejecución de Script)
            translator = Translator()
            
            # Traducir bloques a código real
            code = translator.translate_to_python(blocks)
            
            # Ejecutar en Sandbox
            # Retorna {'success': bool, 'output': str, 'error': str}
            return execute_user_code(code)

        else:
            # RUTA ML (Pipeline de Datos)
            structs = []
            for b in blocks:
                if isinstance(b, dict) and "action" in b:
                    structs.append(b)
                else:
                    structs.append(ml_block_to_struct(b))

            # Retorna {'success': bool, 'results': [...]}
            return execute_structs(structs)

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

# MANEJO DE NODOS DESDE EL EDITOR VISUAL
@eel.expose
def api_delete_node(node_id):
    """Notifica al backend que un nodo ha sido eliminado del editor visual."""
    # Aunque React Flow maneja la eliminación visualmente, esta función 
    # es el punto de entrada si Python necesita actualizar sus estructuras internas.
    print(f"[Backend/Delete] Nodo con ID '{node_id}' eliminado del editor visual.")
    return True

@eel.expose
def api_add_node_manually(node_data):
    """Notifica al backend que un nuevo nodo ha sido agregado al editor visual."""
    # De manera similar, esta función notifica a Python sobre un nuevo nodo.
    node_type = node_data.get('type', 'N/A')
    node_id = node_data.get('id', 'N/A')
    print(f"[Backend/Add] Nuevo nodo '{node_type}' con ID '{node_id}' agregado al editor visual.")
    return True

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
        
        # Filtro inteligente para datasets
        datasets_raw = os.listdir("./datasets") if os.path.exists("./datasets") else []
        valid_datasets = [f for f in datasets_raw if f.endswith('.json') or f.endswith('.csv')]

        files = {
            "projects": file_handler.list_projs(),
            "models": os.listdir("./models") if os.path.exists("./models") else [],
            "datasets": valid_datasets
        }
        return files
    except Exception as e:
        return {"error": str(e)}

@eel.expose
def api_load_dataset_data(name):
    """
    Conecta la UI con la capacidad de lectura híbrida (CSV/JSON).
    Permite a la interfaz previsualizar datos sin importar el formato origen.
    """
    try:
        data = file_handler.load_dataset(name)
        if data:
            return {"success": True, "data": data}
        else:
            return {"success": False, "error": "Dataset vacío o no encontrado"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ESTIMACIÓN DE MEMORIA
@eel.expose
def api_estimate_memory_desktop(model_name):
    try:
        entry = ml_manager._MODEL_REGISTRY.get(model_name)
        if not entry:
            return {"error": "Modelo no encontrado"}
        
        model = entry['model']
        stats = memory_estimator.estimate_memory(
            model, 
            quantized=True,
            target_flash=32256,
            target_sram=2048
        )
        return stats
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
        eel.start('index.html', mode='user_default', size=(1280, 800), host='localhost', port=8080, block=True)