"""
EduBot Unified Desktop Entry Point
==================================

Este módulo fusiona la capacidad de interfaz de Eel con la lógica
de backend de EduBot en server.py.

"""

import eel
import os
import sys
import platform
import shutil
import json
#import yaml
import hashlib
from datetime import datetime

#IMPORTACIONES DEL NÚCLEO DE EDUBOT
from core.ml_adapter import Translator, execute_structs
from core.ml_struct_rules import block_to_struct as ml_block_to_struct
from storage import ml_exporter
from storage import file_handler
from core.executor import execute_user_code
from core import ml_manager
from estimators import memory_estimator
# import maker_edu.auth
# import maker_edu.autograder
# import maker_edu.dashboard



BASE_PATH = file_handler.BASE_PATH
# Definir ruta al navegador portable (dentro del proyecto)
if platform.system() == "Windows":
    BROWSER_PATH = os.path.join(BASE_PATH, "chromium", "chrome.exe")
else:
    # Asumimos Linux/Mac
    BROWSER_PATH = os.path.join(BASE_PATH, "chromium", "chrome")

# Configurar Eel para usar este navegador Específico
if os.path.exists(BROWSER_PATH):
    print(f"[INFO] Usando Chromium Portable: {BROWSER_PATH}")
    eel.browsers.set_path('chrome', BROWSER_PATH)
else:
    print("[WARN] Chromium portable no encontrado. Intentando navegador del sistema...")

# Inicialización de Eel
# Aseguramos que la carpeta web exista
if not os.path.exists('web'):
    print("[ERROR CRÍTICO] La carpeta 'web' no existe. Asegúrate de haber creado el index.html allí.")
    sys.exit()

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
    """
    Carga inteligente de Proyectos Python.
    Delega la lectura a file_handler para usar las rutas absolutas correctas.
    """
    try:
        # Limpieza del nombre por si el usuario escribe la extensión
        clean_name = filename.replace('.edubotproj', '')
        
        # Le pasamos el trabajo al gestor de archivos unificado
        data = file_handler.load_proj(clean_name)
        
        if data:
            # Aseguramos consistencia de la metadata
            if 'mode' not in data: 
                data['mode'] = 'python'
            data['_found_name'] = clean_name
            return data
        else:
            return {'error': f'El proyecto "{clean_name}" no existe o la ruta es incorrecta.'}

    except Exception as e:
        return {'error': f"Error de lectura: {str(e)}"}

@eel.expose
def api_save_model_file(data, filename):
    """
    Guarda la arquitectura del modelo ML (nodos/conexiones) 
    como archivo .edubotml en la carpeta 'models'.
    """
    try:
        # Asegurar extensión
        if not filename.endswith('.edubotml'):
            filename += '.edubotml'
        
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        full_path = os.path.join(models_dir, filename)
        
        # Guardamos usando el file_handler genérico o escritura directa
        # Aquí usamos escritura directa para asegurar el path
        import json
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        return {'success': True, 'path': full_path}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_move_to_trash(filename, file_type):
    """
    Mueve un proyecto o modelo a la carpeta 'trash'.
    file_type: 'project' (.edubotproj) o 'model' (.edubotml)
    """
    try:
        trash_dir = "trash"
        if not os.path.exists(trash_dir):
            os.makedirs(trash_dir)
            
        # Determinar directorio origen y extensión
        if file_type == 'model':
            src_dir = "models"
            ext = ".edubotml"
        else:
            src_dir = "projects" # Asumiendo que file_handler usa esta carpeta base
            ext = ".edubotproj"
            
        if not filename.endswith(ext):
            filename += ext
            
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(trash_dir, filename)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            return {'success': True}
        else:
            return {'success': False, 'error': 'Archivo no encontrado'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

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

# EXPORTACIÓN A C
@eel.expose
def api_export_c_model(model_name):
    """
    Genera el código C/C++ para Arduino del modelo entrenado y lo guarda en disco.
    Incluye automáticamente cabeceras y guardas para Arduino Uno.
    """
    try:
        # Recuperar el modelo de la memoria RAM
        model = ml_manager.get_model(model_name)
        if not model:
            return {"success": False, "error": f"El modelo '{model_name}' no está entrenado. Ejecuta el entrenamiento primero."}

        # Verificar si soporta exportación
        if not hasattr(model, 'to_arduino_code'):
             return {"success": False, "error": f"El modelo '{type(model).__name__}' no soporta exportación a C."}

        # Generar el código fuente crudo (Lógica matemática)
        raw_c_code = model.to_arduino_code(fn_name=model_name)
        
        # Preparar el código final (Inyección de Cabeceras)
        # Generamos un nombre seguro para el Header Guard (ej. UNSABI_MODEL_H)
        header_guard = f"{model_name.upper()}_MODEL_H"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Plantilla profesional para Arduino Uno
        final_c_code = f"""
/**
* EduBot ML Export: {model_name}
* Target: Arduino Uno (AVR)
* Generated at: {timestamp}
**/

#ifndef {header_guard}
#define {header_guard}

#include <Arduino.h>       // Tipos básicos (float, int, etc.)
#include <avr/pgmspace.h>  // Gestión de memoria Flash (PROGMEM)
#include <math.h>          // Funciones matemáticas (exp, sqrt)

// --- INICIO DEL MODELO GENERADO ---
{raw_c_code}
// --- FIN DEL MODELO GENERADO ---

#endif // {header_guard}
"""

        # Guardar archivo .h en carpeta 'exports'
        filename = f"{model_name}.h"
            
        # Asegurar directorio
        export_dir = "exports"
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
                
        full_path = os.path.join(export_dir, filename)
            
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(final_c_code)
                
            # Retornamos la ruta absoluta
        abs_path = os.path.abspath(full_path)
        return {"success": True, "path": abs_path}

    except Exception as e:
        return {"success": False, "error": f"Error interno exportando: {str(e)}"}

# ------------------------------------
# MÓDULO DE SEGURIDAD Y AUTENTICACIÓN
# ------------------------------------

DB_FOLDER = os.path.join(BASE_PATH, "db")
USERS_FILE = os.path.join(DB_FOLDER, "users.json")
STUDENTS_FILE = os.path.join(DB_FOLDER, "students.json")

def _ensure_users_db_exists():
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    if not os.path.exists(USERS_FILE):
        # Usuario por defecto: admin / admin
        default_users = [{
            "username": "admin",
            "password_hash": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918",
            "name": "Profesor Admin"
        }]
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_users, f, indent=4)

@eel.expose
def api_login_teacher(username, password):
    """
    Valida el acceso del profesor contra users.json
    """
    try:
        _ensure_users_db_exists()
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
            
        input_hash = hashlib.sha256(password.encode()).hexdigest()
        
        for user in users:
            if user.get('username') == username and user.get('password_hash') == input_hash:
                return {'success': True, 'role': 'teacher', 'name': user.get('name', username)}
                
        return {'success': False, 'error': 'Usuario o contraseña incorrectos'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_create_teacher(username, password, name):
    """
    Crea un nuevo perfil de profesor en users.json
    """
    try:
        _ensure_users_db_exists()
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
            
        # Validar duplicados
        if any(u.get('username') == username for u in users):
            return {'success': False, 'error': 'El nombre de usuario ya existe.'}
            
        new_hash = hashlib.sha256(password.encode()).hexdigest()
        
        users.append({
            "username": username,
            "password_hash": new_hash,
            "name": name
        })
        
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=4)
            
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_check_admin_access(password):
    """Valida la contraseña maestra de administrador (Default: 'root')"""
    try:
        # Hash SHA256 de "root"
        master_hash = "4813494d137e1631bba301d5acab6e7bb7aa74ce1185d456565ef51d737677b2"
        input_hash = hashlib.sha256(password.encode()).hexdigest()
        return {'success': input_hash == master_hash}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_get_users():
    """Obtiene lista de profesores para el panel de admin"""
    try:
        _ensure_users_db_exists()
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        # Retornamos solo datos seguros
        return {'success': True, 'users': [{'username': u['username'], 'name': u.get('name', 'Sin Nombre')} for u in users]}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_delete_user(username):
    """Elimina un usuario profesor"""
    try:
        _ensure_users_db_exists()
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        new_users = [u for u in users if u.get('username') != username]
        
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_users, f, indent=4)
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_login_student(name):
    """
    Registra el inicio de sesión de un estudiante.
    """
    try:
        if not name or len(name.strip()) < 1:
            return {'success': False, 'error': 'El nombre no puede estar vacío'}
        
        # Asegurar existencia de DB y archivo de estudiantes
        if not os.path.exists(DB_FOLDER):
            os.makedirs(DB_FOLDER)
        if not os.path.exists(STUDENTS_FILE):
            with open(STUDENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)

        with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
            students = json.load(f)

        # Registrar o actualizar estudiante
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        found = False
        for s in students:
            if s.get('name') == name:
                s['last_login'] = now
                found = True
                break
        
        if not found:
            students.append({'name': name, 'first_login': now, 'last_login': now})
            
        with open(STUDENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(students, f, indent=4)

        print(f"[AUTH] Estudiante conectado: {name}")
        return {'success': True, 'role': 'student', 'name': name}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# -------------------------------------
# SISTEMA DE EVALUACIÓN Y PERSISTENCIA
# -------------------------------------

GRADES_FILE = os.path.join(DB_FOLDER, "grades.json")

def ensure_db_exists():
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    if not os.path.exists(GRADES_FILE):
        with open(GRADES_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)

@eel.expose
def api_save_grade(student_name, challenge_id, score):
    """
    Guarda o actualiza la nota de un estudiante en un reto específico.
    """
    try:
        ensure_db_exists()
        
        with open(GRADES_FILE, 'r', encoding='utf-8') as f:
            grades = json.load(f)
            
        # Buscar si ya existe una entrada para este estudiante y reto
        found = False
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Asumimos que score >= 100 es aprobado (lógica simple)
        status = "Aprobado" if float(score) >= 100 else "Reprobado" 
        
        for record in grades:
            if record.get('student') == student_name and record.get('challenge') == challenge_id:
                record['score'] = score
                record['status'] = status
                record['timestamp'] = timestamp
                found = True
                break
        
        if not found:
            grades.append({
                'student': student_name,
                'challenge': challenge_id,
                'score': score,
                'status': status,
                'timestamp': timestamp
            })
            
        with open(GRADES_FILE, 'w', encoding='utf-8') as f:
            json.dump(grades, f, indent=4)
            
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_get_grades():
    """
    Recupera todas las notas para el Dashboard del profesor.
    """
    try:
        ensure_db_exists()
        with open(GRADES_FILE, 'r', encoding='utf-8') as f:
            grades = json.load(f)
        return {'success': True, 'grades': grades}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# --------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL

if __name__ == "__main__":

    try:
        os.chdir(file_handler.BASE_PATH)
        PROFILE_PATH = os.path.join(file_handler.BASE_PATH, 'browser_profile')

        print("---------------------------------------------------")
        print("Iniciando EduBot Forest Studio...")
        print("Backend integrado listo. Abriendo interfaz...")
        print("👉 Abre esta dirección en tu navegador: http://localhost:8080")
        print("---------------------------------------------------")

        # Aseguramos directorios críticos
        file_handler.ensure_dir_exist()

        # Iniciamos la App
        try:
            eel.start(
                'index.html',
                mode='chrome',       # Al haber hecho set_path, 'chrome' ahora apunta al portable
                host='localhost',
                port=0,              # Puerto 0 elige un puerto libre aleatorio (vital para evitar conflictos)
                size=(1280, 800),    # Tamaño inicial de la ventana
                # Flags para evitar errores en entornos restringidos (Liceos)
                cmdline_args=[
                    '--no-sandbox', 
                    '--disable-http-cache',
                    f'--user-data-dir={PROFILE_PATH}',
                    '--disable-gpu' if False else '', # A veces necesario en PCs viejas
                    '--kiosk' if False else ''
                ]
            )
        except EnvironmentError:
            # Fallback si no hay Chrome/Edge instalado: abrir en navegador por defecto
            eel.start('index.html', mode='user_default', host='localhost', port=8080, size=(1280, 800))

    except Exception as e:
        import traceback
        error_msg = f"ERROR FATAL DE EDUBOT:\n{str(e)}\n\n{traceback.format_exc()}"
        with open("crash_log.txt", "w", encoding="utf-8") as f:
            f.write(error_msg)
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, f"EduBot falló al iniciar. Revisa crash_log.txt\n\nError: {str(e)}", "Error Fatal", 0x10)
        except:
            pass
        sys.exit(1)