"""
Módulo ejecutor para EduBot
Ejecuta código Python de forma segura a partir de la traducción basada en nodos.
Permite la importación controlada de módulos del núcleo (MiniML).
"""

import io
import sys
import eel
import traceback
import threading
import queue
import importlib
from contextlib import redirect_stdout
from typing import Dict, Any


input_queue = queue.Queue()
is_waiting_for_user = False

# -------------------------------------
# CONFIGURACIÓN DE SEGURIDAD

# Lista blanca de módulos permitidos
ALLOWED_MODULES = {
    'math', 
    'random', 
    'time', 
    'datetime',
    'core.ml_runtime',  # VITAL: Permitir el runtime
    'core.ml_manager',  # Opcional: si se quiere permitir gestión avanzada
}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    Sustituto seguro de __import__. Solo permite módulos en ALLOWED_MODULES.
    """
    # Manejo de imports relativos o paquetes
    if name in ALLOWED_MODULES:
        return importlib.__import__(name, globals, locals, fromlist, level)
    
    # Permitir submodulos si el padre está permitido (ej: core.ml_runtime)
    base_name = name.split('.')[0]
    if base_name == 'core' and name in ALLOWED_MODULES:
         return importlib.__import__(name, globals, locals, fromlist, level)

    raise ImportError(f"Importación bloqueada por seguridad: '{name}' no está permitido.")

# ---------------------------------------------
# EJECUTOR PRINCIPAL

class RealTimeStdout:
    """Intercepta los prints y los envía a la interfaz de React al instante."""
    def write(self, text):
        # Filtramos los saltos de línea vacíos que hace print por defecto
        if text and text != '\n': 
            try:
                eel.api_realtime_log(str(text))()
            except Exception:
                pass
    def flush(self):
        pass

def sanitize_code(code: str) -> str:
    """Evita palabras clave peligrosas antes de la ejecución."""
    forbidden = [
        'import os', 'import sys', 'import subprocess', '__import__', 
        'eval(', 'exec(', 'open(', 'compile(', 
        'globals()', 'locals()', 'vars()'
    ]
    # Nota: No bloqueamos 'import core' porque lo manejamos en safe_import
    for kw in forbidden:
        if kw in code:
            # Comentar la línea peligrosa en lugar de romper todo
            code = code.replace(kw, f"# BLOCKED: {kw}")
    return code

def execute_user_code(code: str, timeout: int = 5) -> Dict[str, Any]:
    global is_waiting_for_user
    result = {'success': False, 'output': '', 'error': ''}

    print("\n[DEBUG] CÓDIGO ENVIADO AL SANDBOX:")
    print(code)
    print("----------------------------------\n")
    
    # Limpiar cualquier residuo de inputs de ejecuciones pasadas
    while not input_queue.empty():
        input_queue.get_nowait()
        
    def target(q):
        global is_waiting_for_user
        try:
            safe_code = sanitize_code(code)
            
            def interactive_input(prompt=""):
                global is_waiting_for_user
                if prompt:
                    try: eel.api_realtime_log(str(prompt))()
                    except: pass
                
                # Avisar a React que habilite la caja de texto
                try: eel.trigger_frontend_input()()
                except: pass
                
                # Bloquear el hilo de Python hasta que React responda
                is_waiting_for_user = True
                user_response = input_queue.get() 
                is_waiting_for_user = False
                
                # Hacer eco de la respuesta en la terminal visual
                try: eel.api_realtime_log(f"❯ {user_response}")()
                except: pass
                
                return str(user_response)

            safe_builtins_map = {
                'print': print,
                'input': interactive_input,          # <- Nueva función interactiva
                'range': range, 'len': len, 'int': int, 'float': float, 'str': str,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
                'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
                'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted, 'enumerate': enumerate,
                'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
                '__build_class__': __build_class__,  # Vital para los nodos Class
                'object': object,                    
                'super': super,                      
                'classmethod': classmethod,          
                'staticmethod': staticmethod,        
                'property': property,                
                'type': type,                        
                'isinstance': isinstance,            
                '__import__': safe_import,
                '__name__': '__main__'               
            }

            env = {'__builtins__': safe_builtins_map}

            # Aplicar la redirección de consola en tiempo real
            original_stdout = sys.stdout
            sys.stdout = RealTimeStdout()
            
            try:
                exec(safe_code, env)
                q.put({'success': True, 'output': '', 'error': ''})
            finally:
                sys.stdout = original_stdout # Restaurar stdout por seguridad

        #Captura específica de errores
        except SyntaxError as se:
            q.put({'success': False, 'output': '', 'error': f"Error de Sintaxis: {se}"})
        except ImportError as ie:
            q.put({'success': False, 'output': '', 'error': f"Error de Importación: {ie}"})
        except Exception:
            q.put({'success': False, 'output': '', 'error': traceback.format_exc()})

    # Ejecución en hilo secundario
    q = queue.Queue()
    thread = threading.Thread(target=target, args=(q,))
    thread.start()
    
    # TIMEOUT INTELIGENTE
    time_elapsed = 0.0
    while thread.is_alive() and time_elapsed < timeout:
        # eel.sleep cede el control temporalmente a los WebSockets
        # permitiendo que la interfaz reaccione en tiempo real sin bloquear el hilo principal.
        eel.sleep(0.1) 
        
        if not is_waiting_for_user:
            time_elapsed += 0.1
            
    if thread.is_alive():
        result['error'] = "Tiempo de ejecución excedido (Timeout). ¿Tienes un bucle infinito?"
    else:
        if not q.empty():
            result = q.get()
        else:
            result['error'] = "Error desconocido: No se recibió respuesta del hilo."
            
    return result