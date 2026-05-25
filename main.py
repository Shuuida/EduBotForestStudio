"""
EduBot Unified Desktop Entry Point
==================================

Este módulo fusiona la capacidad de interfaz de Eel con la lógica
de backend de EduBot en server.py.

"""

import ast
import difflib
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
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from cryptography.fernet import Fernet
import multiprocessing

#IMPORTACIONES DEL NÚCLEO DE EDUBOT
from storage import file_handler
from core.executor import execute_user_code, submit_input, kill_execution
from core.translator import Translator
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
            "executor": True,
            "file_handler": True
        }
    }

# GESTIÓN DE PROYECTOS (.edubotproj)
@eel.expose
def api_save_project(data, filename):
    """Guarda el estado completo del editor (nodos y conexiones)"""
    try:
        data['mode'] = 'python' # Aseguramos que el modo siempre se guarde como 'python' para consistencia
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
def api_move_to_trash(filename, file_type=None):
    """
    Mueve un proyecto a la carpeta 'trash'.
    (El parámetro file_type se recibe para no romper el frontend, pero se ignora)
    """
    try:
        # Usamos rutas absolutas seguras para entornos empaquetados en .exe
        trash_dir = os.path.join(file_handler.BASE_PATH, "trash")
        projects_dir = os.path.join(file_handler.BASE_PATH, "projects")
        
        if not os.path.exists(trash_dir):
            os.makedirs(trash_dir)
            
        ext = ".edubotproj"
            
        if not filename.endswith(ext):
            filename += ext
            
        src_path = os.path.join(projects_dir, filename)
        dst_path = os.path.join(trash_dir, filename)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            return {'success': True}
        else:
            return {'success': False, 'error': 'Archivo de proyecto no encontrado.'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

# TRADUCCIÓN Y ESTRUCTURAS
PYTHON_BUILTINS = {
    'range', 'len', 'int', 'float', 'str', 'list', 'dict', 'set', 'tuple', 'bool',
    'abs', 'min', 'max', 'sum', 'round', 'zip', 'map', 'filter', 'sorted', 'enumerate',
    'print', 'input', 'True', 'False', 'None'
}

def _extract_names(expr: str):
    if not expr or not isinstance(expr, str) or expr.strip() == '':
        return set(), None
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        return set(), str(e)

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names, None


def detect_typo_errors(blocks):
    defined_names = {}
    errors = []

    def add_definition(name, node_id):
        if name and isinstance(name, str) and name.strip():
            defined_names[name.strip()] = node_id

    # PRIMERA PASADA (RECURSIVA): Registrar todas las variables en todos los niveles
    def gather_definitions(block_list):
        for block in block_list:
            if not isinstance(block, dict):
                continue
            b_type = block.get('type', '')
            if b_type == 'py_var':
                add_definition(block.get('var_name'), block.get('id'))
            elif b_type == 'py_math':
                add_definition(block.get('target'), block.get('id'))
            elif b_type in ['py_int', 'py_float', 'py_input']:
                add_definition(block.get('target'), block.get('id'))
            elif b_type == 'py_loop':
                add_definition(block.get('iterator'), block.get('id'))
            elif b_type == 'py_func':
                add_definition(block.get('func_name'), block.get('id'))
            elif b_type == 'py_class':
                add_definition(block.get('name'), block.get('id'))

            # Si el nodo tiene hijos (If, For, etc), entramos a revisarlos
            body = block.get('body', [])
            if isinstance(body, list):
                gather_definitions(body)

    gather_definitions(blocks)

    def suggest(name):
        candidates = difflib.get_close_matches(name, defined_names.keys(), n=1, cutoff=0.6)
        return candidates[0] if candidates else None

    def check_expression(expr, block_id, description):
        if expr is None:
            return
            
        # str() protege contra números puros que puedan crashear el AST
        names, parse_error = _extract_names(str(expr))
        
        if parse_error:
            errors.append({
                'node_id': block_id,
                'message': f"Sintaxis inválida en {description}",
                'suggestion': None
            })
            return

        for name in names:
            if name in PYTHON_BUILTINS or name in defined_names:
                continue
            suggestion = suggest(name)
            if suggestion:
                errors.append({
                    'node_id': block_id,
                    'message': f"'{name}' no existe. ¿Quisiste decir '{suggestion}'?",
                    'suggestion': suggestion
                })
            else:
                errors.append({
                    'node_id': block_id,
                    'message': f"La variable '{name}' no ha sido creada.",
                    'suggestion': None
                })

    # SEGUNDA PASADA (RECURSIVA): Buscar errores en cualquier profundidad
    def check_blocks(block_list):
        for block in block_list:
            if not isinstance(block, dict):
                continue
            b_type = block.get('type', '')
            block_id = block.get('id')
            
            if b_type == 'py_var':
                check_expression(block.get('value', ''), block_id, 'el valor')
            elif b_type == 'py_math':
                check_expression(block.get('left', ''), block_id, 'el lado izquierdo')
                check_expression(block.get('right', ''), block_id, 'el lado derecho')
            elif b_type == 'py_print':
                check_expression(block.get('content', ''), block_id, 'lo que se va a imprimir')
            elif b_type in ['py_int', 'py_float']:
                check_expression(block.get('value', ''), block_id, 'la conversión')
            elif b_type == 'py_compare':
                check_expression(block.get('left', ''), block_id, 'la comparación (izq)')
                check_expression(block.get('right', ''), block_id, 'la comparación (der)')
            elif b_type == 'py_if':
                check_expression(block.get('condition', ''), block_id, 'la condición del If')
            elif b_type == 'py_elif':
                check_expression(block.get('condition', ''), block_id, 'la condición del Elif')
            elif b_type == 'py_loop':
                check_expression(block.get('iterable', ''), block_id, 'el rango del For')
            elif b_type == 'py_while':
                check_expression(block.get('condition', ''), block_id, 'la condición del While')
            elif b_type == 'py_call':
                func_expr = block.get('func', '')
                args_expr = block.get('args', '')
                check_expression(func_expr, block_id, 'la llamada a función')
                if args_expr:
                    check_expression(f"({args_expr})", block_id, 'los parámetros')
            elif b_type == 'py_return':
                check_expression(block.get('value', ''), block_id, 'el valor a retornar')

            # Volvemos a bajar en cascada para revisar el interior de los contenedores
            body = block.get('body', [])
            if isinstance(body, list):
                check_blocks(body)

    check_blocks(blocks)

    # Filtrar duplicados
    unique = {}
    for err in errors:
        key = (err['node_id'], err['message'])
        if key not in unique:
            unique[key] = err
    return list(unique.values())

@eel.expose
def api_translate(direction, blocks=None, code=None):
    """
    Maneja traducciones exclusivas del motor Python:
    - Bloques -> Código Python (to_python)
    - Código Python -> Bloques (to_blocks)
    """
    try:
        translator = Translator()
        
        if direction == "to_python":
            python_code = translator.translate_to_python(blocks)
            return {"result": python_code}

        elif direction == "to_blocks":
            blocks_result = translator.translate_to_blocks(code)
            return {"result": blocks_result}
        
        else:
            return {"error": "Dirección inválida. Use 'to_python' o 'to_blocks'."}

    except Exception as e:
        return {"error": f"Error de traducción: {str(e)}"}

# EJECUCIÓN INTELIGENTE(ML y Código Python)
@eel.expose
def api_execute(blocks):
    """
    Ejecuta el pipeline de bloques puramente en el entorno seguro de Python.
    """
    try:
        if not blocks:
            return {"error": "No se recibieron bloques para ejecutar"}

        # 1. RASTREO DE ERRORES TIPOGRÁFICOS
        typo_errors = detect_typo_errors(blocks)
        if typo_errors:
            return {
                'success': False,
                'error': 'Se detectaron errores de tipografía/sintaxis en los bloques.',
                'error_nodes': typo_errors
            }

        # 2. TRADUCCIÓN A CÓDIGO PYTHON
        translator = Translator()
        code = translator.translate_to_python(blocks)
        
        # 3. EJECUCIÓN EN SANDBOX
        return execute_user_code(code)

    except Exception as e:
        return {"success": False, "error": f"Error interno de ejecución: {str(e)}"}

@eel.expose
def api_kill_execution():
    """Detiene de forma inmediata la ejecución Python activa."""
    try:
        stopped = kill_execution()
        return {"success": stopped, "error": None if stopped else "No hay ejecución activa."}
    except Exception as e:
        return {"success": False, "error": str(e)}

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

@eel.expose
def api_export_python_file(code, default_name):
    """
    Abre una ventana nativa del SO para guardar el código Python.
    Bypassea las advertencias de seguridad de descargas de Chromium.
    """
    try:
        # Inicializa una ventana de Tkinter oculta
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True) # Fuerza la ventana al frente de EduBot

        # Abre el diálogo nativo de Guardar Como
        file_path = filedialog.asksaveasfilename(
            defaultextension=".py",
            initialfile=default_name,
            title="Exportar Código Python",
            filetypes=[("Archivos Python", "*.py"), ("Todos los archivos", "*.*")]
        )

        root.destroy() # Destruye la ventana oculta para liberar memoria

        # Si el usuario presiona "Cancelar" o cierra la ventana
        if not file_path:
            return {'success': False, 'error': 'cancelado'}

        # Si el usuario elige una ruta, guardamos el archivo físicamente
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        return {'success': True, 'path': file_path}
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
    """
    Lee las bases de datos de usuarios (profesores y estudiantes) por separado,
    las unifica y las devuelve en formato de lista para el Dashboard.
    """
    try:
        teachers = []
        students = []

        if os.path.exists(USERS_FILE) and os.path.getsize(USERS_FILE) > 0:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                try:
                    users_db = json.load(f)
                    # Forzamos que todos aquí tengan el rol 'teacher'
                    for user in users_db:
                        if isinstance(user, dict):
                            teachers.append({
                                'username': user.get('username', 'Desconocido'),
                                'name': user.get('name', 'Profesor Sin Nombre'),
                                'role': 'teacher'
                            })
                except Exception as e:
                    print(f"[WARN] Error leyendo users.json: {e}")

        if os.path.exists(STUDENTS_FILE) and os.path.getsize(STUDENTS_FILE) > 0:
            with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
                try:
                    students_db = json.load(f)
                    # Forzamos que todos aquí tengan el rol 'student'
                    for user in students_db:
                        if isinstance(user, dict):
                            students.append({
                                'username': user.get('username', 'Desconocido'),
                                'name': user.get('name', 'Estudiante Sin Nombre'),
                                'role': 'student'
                            })
                except Exception as e:
                    print(f"[WARN] Error leyendo students.json: {e}")

        # Retornamos la data consolidada al frontend
        return {
            'success': True, 
            'teachers': teachers, 
            'students': students
        }

    except Exception as e:
        print(f"❌ Error crítico en api_get_all_accounts: {e}")
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
def api_save_grade(student_name, challenge_id, score, year="General", section="Única", topic="Sin Asignar", old_timestamp=None):
    """Guarda o actualiza la nota, soportando metadatos de Año, Sección y Tema."""
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
                    grades = json.loads(file_content.decode('utf-8'))
            
        # ACTUALIZAR O INSERTAR
        found = False
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "Aprobado" if float(score) >= 10 else "Reprobado" 
        
        for record in grades:
            # Si estamos editando un registro existente (usamos el timestamp original como ID único)
            if old_timestamp and record.get('timestamp') == old_timestamp:
                record['student'] = student_name
                record['challenge'] = challenge_id
                record['score'] = score
                record['status'] = status
                record['year'] = year
                record['section'] = section
                record['topic'] = topic
                # Opcional: actualizar el timestamp o dejar el original de la evaluación
                record['timestamp'] = timestamp 
                found = True
                break
            # Fallback lógico si no hay old_timestamp pero coincide el reto y estudiante
            elif not old_timestamp and record.get('student') == student_name and record.get('challenge') == challenge_id:
                record['score'] = score
                record['status'] = status
                record['year'] = year
                record['section'] = section
                record['topic'] = topic
                record['timestamp'] = timestamp
                found = True
                break
        
        if not found:
            grades.append({
                'student': student_name,
                'challenge': challenge_id,
                'score': score,
                'status': status,
                'year': year,
                'section': section,
                'topic': topic,
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
def api_delete_grade(timestamp_id):
    """Elimina un registro específico usando su timestamp como identificador único."""
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
                    grades = json.loads(file_content.decode('utf-8'))
        
        # Filtrar eliminando el que coincida con el timestamp
        initial_length = len(grades)
        grades = [g for g in grades if g.get('timestamp') != timestamp_id]
        
        if len(grades) == initial_length:
            return {'success': False, 'error': 'Registro no encontrado.'}

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
    submit_input(user_text)

# --------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL

if __name__ == "__main__":
    multiprocessing.freeze_support()

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
                    '--disable-gpu',
                    '--disable-software-rasterizer'
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