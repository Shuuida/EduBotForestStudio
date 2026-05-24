"""
Módulo ejecutor para EduBot
----------------------
Este módulo se encarga de ejecutar el código Python generado a partir de los bloques.
"""

import eel
import traceback
import threading
import queue
import multiprocessing
import importlib
import time
from typing import Dict, Any

# Variables globales dinámicas (Se regenerarán en cada ejecución)
input_queue = None
gui_queue = None
wait_flag = None

current_process = None
current_process_lock = threading.Lock()

# Worker independiente: Nace y muere con cada ejecución
def _log_worker_routine(g_queue):
    import gevent
    while True:
        # Extraemos todo lo que el proceso haya escupido
        while not g_queue.empty():
            try:
                msg = g_queue.get_nowait()
                if msg == "__TRIGGER_INPUT__":
                    eel.trigger_frontend_input() 
                else:
                    eel.api_realtime_log(msg) 
            except Exception:
                pass
        
        with current_process_lock:
            if current_process is None or not current_process.is_alive():
                if g_queue.empty():
                    break
        gevent.sleep(0.05)

# -------------------------------------
# CONFIGURACIÓN DE SEGURIDAD
ALLOWED_MODULES = {
    'math', 'random', 'time', 'datetime',
    'core.ml_runtime', 'core.ml_manager',
}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ALLOWED_MODULES:
        return importlib.__import__(name, globals, locals, fromlist, level)
    base_name = name.split('.')[0]
    if base_name == 'core' and name in ALLOWED_MODULES:
         return importlib.__import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Importación bloqueada: '{name}' no está permitido.")

def sanitize_code(code: str) -> str:
    forbidden = [
        'import os', 'import sys', 'import subprocess', '__import__', 
        'eval(', 'exec(', 'open(', 'compile(', 
        'globals()', 'locals()', 'vars()'
    ]
    for kw in forbidden:
        if kw in code:
            code = code.replace(kw, f"# BLOCKED: {kw}")
    return code

def _execution_target(result_queue, input_q, gui_q, wait_flag_obj, code):
    try:
        safe_code = sanitize_code(code)

        def custom_print(*args, sep=' ', end='\n', file=None, flush=False):
            text = sep.join(map(str, args))
            gui_q.put(str(text))

        def interactive_input(prompt=""):
            if prompt:
                gui_q.put(str(prompt))
            gui_q.put("__TRIGGER_INPUT__")
            wait_flag_obj.value = True
            user_response = input_q.get()
            wait_flag_obj.value = False
            gui_q.put(f"❯ {user_response}")
            return str(user_response)

        safe_builtins_map = {
            'print': custom_print,
            'input': interactive_input,
            'range': range, 'len': len, 'int': int, 'float': float, 'str': str,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
            'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
            'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted, 'enumerate': enumerate,
            'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
            '__build_class__': __build_class__,
            'object': object, 'super': super, 'classmethod': classmethod,
            'staticmethod': staticmethod, 'property': property, 'type': type,
            'isinstance': isinstance, '__import__': safe_import
        }

        env = {
            '__builtins__': safe_builtins_map,
            '__name__': '__main__'
        }

        exec(safe_code, env)
        result_queue.put({'success': True, 'output': '', 'error': ''})

    except SyntaxError as se:
        result_queue.put({'success': False, 'output': '', 'error': f"Error de Sintaxis: {se}"})
    except ImportError as ie:
        result_queue.put({'success': False, 'output': '', 'error': f"Error de Importación: {ie}"})
    except Exception:
        result_queue.put({'success': False, 'output': '', 'error': traceback.format_exc()})


def submit_input(val: str):
    global input_queue, wait_flag
    if input_queue is not None:
        input_queue.put(val)
    if wait_flag is not None:
        wait_flag.value = False

def kill_execution():
    global current_process, gui_queue, wait_flag
    with current_process_lock:
        if current_process and current_process.is_alive():
            current_process.terminate()
            
            # Vaciamos la cola para destruir "inputs fantasmas" de último milisegundo
            if gui_queue is not None:
                while not gui_queue.empty():
                    try:
                        gui_queue.get_nowait()
                    except:
                        break
                        
            if wait_flag is not None:
                wait_flag.value = False
                
            current_process.join(timeout=1)
            current_process = None
            
            try:
                eel.api_realtime_log("\n[SISTEMA] Ejecución detenida forzosamente.")
            except:
                pass
                
            return True 
            
    return False

# ---------------------------------------------
# EJECUTOR PRINCIPAL
def execute_user_code(code: str, timeout: int = 5) -> Dict[str, Any]:
    global current_process, input_queue, gui_queue, wait_flag
    
    if current_process is not None and current_process.is_alive():
        return {'success': False, 'output': '', 'error': 'Ejecutor ocupado.'}

    # Creamos colas completamente nuevas y limpias para cada ejecución.
    # Esto ignora cualquier candado roto que haya dejado un "Kill" anterior.
    input_queue = multiprocessing.Queue()
    gui_queue = multiprocessing.Queue()
    wait_flag = multiprocessing.Value('b', False)
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=_execution_target,
        args=(result_queue, input_queue, gui_queue, wait_flag, code)
    )

    with current_process_lock:
        current_process = process
        
    process.start()
    
    # Iniciamos el obrero de la terminal específico para estas colas
    eel.spawn(_log_worker_routine, gui_queue)

    time_elapsed = 0.0
    while process.is_alive() and time_elapsed < timeout:
        eel.sleep(0.05)
        # Solo sumamos tiempo si no estamos esperando input del usuario
        if not wait_flag.value:
            time_elapsed += 0.05

    # Si se excedió el tiempo límite natural
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        with current_process_lock:
            current_process = None
        return {'success': False, 'output': '', 'error': 'Tiempo excedido (Timeout). ¿El proceso se colgó?'}

    with current_process_lock:
        current_process = None

    try:
        result = result_queue.get_nowait()
        return result
    except queue.Empty:
        return {'success': False, 'output': '', 'error': 'Proceso terminado inesperadamente (posible SIGKILL).'}