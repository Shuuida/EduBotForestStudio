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

# Use a Manager so queues and shared state can be passed across the worker process.
if multiprocessing.current_process().name == 'MainProcess':
    _multiprocessing_manager = multiprocessing.Manager()
    input_queue = _multiprocessing_manager.Queue()
    gui_queue = _multiprocessing_manager.Queue()
    wait_flag = _multiprocessing_manager.Value('b', False)
else:
    _multiprocessing_manager = None
    input_queue = None
    gui_queue = None
    wait_flag = None

current_process = None
current_process_lock = threading.Lock()

def _global_log_worker():
    import gevent
    while True:
        if gui_queue is not None:
            while not gui_queue.empty():
                try:
                    msg = gui_queue.get_nowait()
                    if msg == "__TRIGGER_INPUT__":
                        eel.trigger_frontend_input()()
                    else:
                        eel.api_realtime_log(msg)()
                except Exception:
                    pass
        gevent.sleep(0.05)

if multiprocessing.current_process().name == 'MainProcess':
    eel.spawn(_global_log_worker)

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


def kill_execution() -> bool:
    global current_process
    if current_process is None:
        return False

    with current_process_lock:
        if current_process is not None and current_process.is_alive():
            current_process.terminate()
            current_process.join(timeout=1)
            if wait_flag is not None:
                wait_flag.value = False
            current_process = None
            return True

    return False

# ---------------------------------------------
# EJECUTOR PRINCIPAL
def execute_user_code(code: str, timeout: int = 5) -> Dict[str, Any]:
    global current_process
    result = {'success': False, 'output': '', 'error': ''}

    if input_queue is None or gui_queue is None or wait_flag is None or _multiprocessing_manager is None:
        return {'success': False, 'output': '', 'error': 'El entorno de ejecución no está inicializado correctamente.'}

    if current_process is not None and current_process.is_alive():
        return {'success': False, 'output': '', 'error': 'Ejecutor ocupado. Espere a que termine la ejecución actual.'}

    while not input_queue.empty():
        try:
            input_queue.get_nowait()
        except queue.Empty:
            break

    result_queue = _multiprocessing_manager.Queue()
    process = multiprocessing.Process(
        target=_execution_target,
        args=(result_queue, input_queue, gui_queue, wait_flag, code)
    )

    with current_process_lock:
        current_process = process
    process.start()

    time_elapsed = 0.0
    while process.is_alive() and time_elapsed < timeout:
        eel.sleep(0.05)
        if not wait_flag.value:
            time_elapsed += 0.05

    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        with current_process_lock:
            current_process = None
        wait_flag.value = False
        result['error'] = 'Tiempo excedido (Timeout). ¿El proceso se colgó?'
        return result

    with current_process_lock:
        current_process = None

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = {'success': False, 'output': '', 'error': 'Error desconocido de procesamiento.'}

    return result