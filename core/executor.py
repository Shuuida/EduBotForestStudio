"""
Módulo ejecutor para EduBot
----------------------
Este módulo se encarga de ejecutar el código Python generado a partir de los bloques.
"""

import eel
import traceback
import threading
import queue
import importlib
import time
from typing import Dict, Any

input_queue = queue.Queue()
gui_queue = queue.Queue()
is_waiting_for_user = False

def _global_log_worker():
    import gevent
    while True:
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

# ---------------------------------------------
# EJECUTOR PRINCIPAL
def execute_user_code(code: str, timeout: int = 5) -> Dict[str, Any]:
    global is_waiting_for_user
    result = {'success': False, 'output': '', 'error': ''}
    
    # Limpiar entradas zombis
    while not input_queue.empty(): input_queue.get_nowait()
    
    def target(q):
        global is_waiting_for_user
        try:
            safe_code = sanitize_code(code)
            
            def custom_print(*args, sep=' ', end='\n', file=None, flush=False):
                text = sep.join(map(str, args))
                gui_queue.put(str(text))

            def interactive_input(prompt=""):
                global is_waiting_for_user
                if prompt: gui_queue.put(str(prompt))
                gui_queue.put("__TRIGGER_INPUT__")
                is_waiting_for_user = True
                user_response = input_queue.get() 
                is_waiting_for_user = False
                gui_queue.put(f"❯ {user_response}")
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

            # __name__ se pasa a nivel de globals para que no haya falsos saltos
            env = {
                '__builtins__': safe_builtins_map,
                '__name__': '__main__'
            }

            exec(safe_code, env)
            q.put({'success': True, 'output': '', 'error': ''})

        except SyntaxError as se:
            q.put({'success': False, 'output': '', 'error': f"Error de Sintaxis: {se}"})
        except ImportError as ie:
            q.put({'success': False, 'output': '', 'error': f"Error de Importación: {ie}"})
        except Exception:
            q.put({'success': False, 'output': '', 'error': traceback.format_exc()})

    q = queue.Queue()
    thread = threading.Thread(target=target, args=(q,))
    thread.daemon = True 
    thread.start()
    
    time_elapsed = 0.0
    while thread.is_alive() and time_elapsed < timeout:
        eel.sleep(0.05) 
        if not is_waiting_for_user:
            time_elapsed += 0.05
            
    if thread.is_alive():
        result['error'] = "Tiempo excedido (Timeout). ¿El hilo principal se colgó?"
    else:
        if not q.empty():
            result = q.get()
        else:
            result['error'] = "Error desconocido de procesamiento."
            
    return result