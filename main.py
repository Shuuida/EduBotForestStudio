"""
EduBot Unified Desktop Entry Point
==================================

Este módulo fusiona la capacidad de interfaz de Eel con la lógica
de backend de EduBot en server.py.

"""

import eel
import eel.browsers
import os
import sys
import platform
import shutil
import json
#import yaml
import webbrowser
import hashlib
from datetime import datetime
from cryptography.fernet import Fernet

#IMPORTACIONES DEL NÚCLEO DE EDUBOT
from core.ml_adapter import Translator, execute_structs
from core.ml_struct_rules import block_to_struct as ml_block_to_struct
from storage import ml_exporter
from storage import file_handler
from core.executor import execute_user_code, input_queue
from core import ml_manager
from estimators import memory_estimator
# import maker_edu.auth
# import maker_edu.autograder
# import maker_edu.dashboard

# Esta llave es estática y vivirá compilada dentro del .exe.
# Si un estudiante abre el grades.json, solo verá texto ilegible.
SECRET_KEY = b'Z1h2U3E0dDdyOHU5eXpBMkMzRDRFNUY2RzhIOUkxSjI='
cipher = Fernet(SECRET_KEY)

BASE_PATH = file_handler.BASE_PATH
def get_chromium_path():
    """Detecta la ruta de Chromium considerando PyInstaller (_internal) y el Sistema Operativo"""
    # Detectar si estamos en el .exe o en VS Code
    if hasattr(sys, '_MEIPASS'):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        
    # Detectar el Sistema Operativo para el nombre del ejecutable
    if platform.system() == "Windows":
        exe_name = "chrome.exe"
    else:
        # En Linux/Mac los binarios no llevan .exe
        exe_name = "chrome"     # O "chromium"
        
    return os.path.join(base_dir, 'chromium', exe_name)

CHROMIUM_PATH = get_chromium_path()

# Configurar Eel para usar este navegador específico si existe
if os.path.exists(CHROMIUM_PATH):
    print(f"[INFO] Usando Chromium Portable en: {CHROMIUM_PATH}")
    eel.browsers.set_path('chrome', CHROMIUM_PATH)
else:
    print(f"[WARN] Chromium portable no encontrado en: {CHROMIUM_PATH}")
    print("El sistema usará el navegador por defecto (Edge/Chrome/Safari) como respaldo.")

def get_web_path():
    """Detecta inteligentemente dónde está la carpeta web (VS Code vs .exe)"""
    if hasattr(sys, '_MEIPASS'):
        # Si estamos dentro del .exe, PyInstaller guarda los datos en _MEIPASS (carpeta _internal)
        return os.path.join(sys._MEIPASS, 'web')
    else:
        # Si estamos ejecutando el script normal en VS Code
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'web')

WEB_PATH = get_web_path()

if not os.path.exists(WEB_PATH):
    print(f"[ERROR CRÍTICO] La carpeta 'web' no existe en la ruta esperada: {WEB_PATH}")
    input("Presiona Enter para salir...")
    sys.exit()

# Inicializamos Eel apuntando a la ruta absoluta blindada
eel.init(WEB_PATH)

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

#@eel.expose
#*def api_save_model_file(data, filename):
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
#@eel.expose
#*def api_file_save(content, name, file_type="auto"):
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

#@eel.expose
#*def api_file_list():
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

#@eel.expose
#*def api_load_dataset_data(name):
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
#@eel.expose
#*def api_estimate_memory_desktop(model_name):
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
#@eel.expose
#*def api_export_c_model(model_name):
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

@eel.expose
def api_open_manual():
    """Busca y abre el manual de usuario en PDF con el visor predeterminado del sistema."""
    try:
        # Busca el archivo en la raíz del proyecto
        manual_path = os.path.abspath(os.path.join(file_handler.BASE_PATH, "Manual de Usuario EduBot.pdf"))
        
        # Buscar dentro de la carpeta 'docs'
        if not os.path.exists(manual_path):
            manual_path = os.path.abspath(os.path.join(file_handler.BASE_PATH, "docs", "user", "Manual de Usuario EduBot.pdf"))
            
        if not os.path.exists(manual_path):
            return {'success': False, 'error': 'No se encontró el archivo manual_usuario.pdf en el directorio de instalación.'}
        
        # Convertir la ruta a formato URI seguro para todos los sistemas operativos
        file_url = f"file:///{manual_path.replace(chr(92), '/')}"
        webbrowser.open(file_url)
        
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

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
def api_create_student(username, password, name):
    """Crea un nuevo perfil de estudiante con contraseña en students.json"""
    try:
        if not os.path.exists(DB_FOLDER):
            os.makedirs(DB_FOLDER)
        if not os.path.exists(STUDENTS_FILE):
            with open(STUDENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)

        with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
            students = json.load(f)

        # Validar duplicados
        if any(s.get('username') == username for s in students):
            return {'success': False, 'error': 'El nombre de usuario ya existe.'}

        new_hash = hashlib.sha256(password.encode()).hexdigest()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        students.append({
            "username": username,
            "password_hash": new_hash,
            "name": name,
            "first_login": now,
            "last_login": now
        })

        with open(STUDENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(students, f, indent=4)

        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_login_student(username, password):
    """Valida el acceso del estudiante usando usuario y contraseña cifrada"""
    try:
        if not os.path.exists(DB_FOLDER) or not os.path.exists(STUDENTS_FILE):
            return {'success': False, 'error': 'No hay estudiantes registrados en el sistema.'}

        with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
            students = json.load(f)

        input_hash = hashlib.sha256(password.encode()).hexdigest()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for s in students:
            if s.get('username') == username and s.get('password_hash') == input_hash:
                s['last_login'] = now # Actualizamos su última conexión
                with open(STUDENTS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(students, f, indent=4)
                
                print(f"[AUTH] Estudiante conectado: {s.get('name')}")
                # Devolvemos 'name' para no romper el sistema de calificaciones actual
                return {'success': True, 'role': 'student', 'name': s.get('name')}

        return {'success': False, 'error': 'Usuario o contraseña incorrectos'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_change_password(username, old_password, new_password, role):
    """Permite cambiar la contraseña desde adentro de la app si conoces la actual."""
    try:
        file_path = USERS_FILE if role == 'teacher' else STUDENTS_FILE
        if not os.path.exists(file_path): 
            return {'success': False, 'error': 'Base de datos no encontrada.'}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            users = json.load(f)
            
        old_hash = hashlib.sha256(old_password.encode()).hexdigest()
        new_hash = hashlib.sha256(new_password.encode()).hexdigest()
        
        for u in users:
            if u.get('username') == username and u.get('password_hash') == old_hash:
                u['password_hash'] = new_hash
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(users, f, indent=4)
                return {'success': True}
                
        return {'success': False, 'error': 'La contraseña actual es incorrecta.'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_reset_password(username, full_name, new_password, role):
    """Permite recuperar la contraseña si conoces tu Usuario y tu Nombre Completo exacto."""
    try:
        file_path = USERS_FILE if role == 'teacher' else STUDENTS_FILE
        if not os.path.exists(file_path): 
            return {'success': False, 'error': 'Base de datos no encontrada.'}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            users = json.load(f)
            
        new_hash = hashlib.sha256(new_password.encode()).hexdigest()
        
        for u in users:
            # Verificación de seguridad básica para entornos offline: Coincidencia de Usuario + Nombre
            if u.get('username') == username and u.get('name').lower() == full_name.lower():
                u['password_hash'] = new_hash
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(users, f, indent=4)
                return {'success': True}
                
        return {'success': False, 'error': 'Los datos no coinciden con ninguna cuenta registrada.'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_get_all_accounts():
    """Obtiene todos los profesores y estudiantes para el panel de administración interna."""
    try:
        teachers = []
        students = []
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                t_data = json.load(f)
                teachers = [{'username': u['username'], 'name': u.get('name', 'Sin Nombre')} for u in t_data]
        if os.path.exists(STUDENTS_FILE):
            with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
                s_data = json.load(f)
                students = [{'username': u['username'], 'name': u.get('name', 'Sin Nombre')} for u in s_data]
        return {'success': True, 'teachers': teachers, 'students': students}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_delete_account(username, role):
    """Elimina una cuenta de profesor o estudiante."""
    try:
        file_path = USERS_FILE if role == 'teacher' else STUDENTS_FILE
        if not os.path.exists(file_path): 
            return {'success': False, 'error': 'Base de datos no encontrada.'}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        new_users = [u for u in users if u.get('username') != username]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(new_users, f, indent=4)
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_edit_account(old_username, new_username, new_name, new_password, role):
    """Edita los datos de una cuenta existente."""
    try:
        file_path = USERS_FILE if role == 'teacher' else STUDENTS_FILE
        if not os.path.exists(file_path): 
            return {'success': False, 'error': 'Base de datos no encontrada.'}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        # Verificar si el nuevo usuario ya existe (si el administrador lo cambió)
        if old_username != new_username:
            if any(u.get('username') == new_username for u in users):
                return {'success': False, 'error': 'El nuevo nombre de usuario ya está en uso.'}
        
        for u in users:
            if u.get('username') == old_username:
                u['username'] = new_username
                u['name'] = new_name
                # Solo cambia la contraseña si el administrador escribió una nueva
                if new_password and len(new_password) > 0: 
                    u['password_hash'] = hashlib.sha256(new_password.encode()).hexdigest()
                break
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=4)
        return {'success': True}
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
        # Creamos un archivo vacío cifrado
        empty_data = cipher.encrypt(b"[]")
        with open(GRADES_FILE, 'wb') as f:
            f.write(empty_data)

@eel.expose
def api_save_grade(student_name, challenge_id, score):
    """Guarda o actualiza la nota cifrando el archivo completo en AES."""
    try:
        ensure_db_exists()
        grades = []
        
        # LEER Y DESCIFRAR
        if os.path.getsize(GRADES_FILE) > 0:
            with open(GRADES_FILE, 'rb') as f:
                file_content = f.read()
                try:
                    decrypted_data = cipher.decrypt(file_content)
                    grades = json.loads(decrypted_data.decode('utf-8'))
                except Exception:
                    # Fallback: Si falla el descifrado, asume que es un JSON viejo en texto plano
                    grades = json.loads(file_content.decode('utf-8'))
            
        # ACTUALIZAR DATOS
        found = False
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            
        # CIFRAR Y GUARDAR
        json_str = json.dumps(grades, indent=4)
        encrypted_output = cipher.encrypt(json_str.encode('utf-8'))
        
        with open(GRADES_FILE, 'wb') as f:
            f.write(encrypted_output)
            
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_get_grades():
    """Descifra las notas en memoria para enviarlas al Dashboard del profesor."""
    try:
        ensure_db_exists()
        grades = []
        
        if os.path.getsize(GRADES_FILE) > 0:
            with open(GRADES_FILE, 'rb') as f:
                file_content = f.read()
                try:
                    decrypted_data = cipher.decrypt(file_content)
                    grades = json.loads(decrypted_data.decode('utf-8'))
                except Exception:
                    # Fallback JSON viejo
                    grades = json.loads(file_content.decode('utf-8'))
                    
        return {'success': True, 'grades': grades}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@eel.expose
def api_provide_input(user_text):
    """Recibe la respuesta de la terminal visual y desbloquea el hilo de Python"""
    input_queue.put(user_text)

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