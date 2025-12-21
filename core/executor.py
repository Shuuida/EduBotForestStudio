"""
Módulo ejecutor para EduBot
Ejecuta código Python de forma segura a partir de la traducción basada en nodos.
Permite la importación controlada de módulos del núcleo (MiniML).
"""

import io
import traceback
import threading
import queue
import importlib
from contextlib import redirect_stdout
from typing import Dict, Any

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

def sanitize_code(code: str) -> str:
    """Evita palabras clave peligrosas antes de la ejecución."""
    forbidden = [
        'import os', 'import sys', 'import subprocess', '__import__', 
        'eval(', 'exec(', 'open(', 'compile(', 'input(', 
        'globals()', 'locals()', 'vars()'
    ]
    # Nota: No bloqueamos 'import core' porque lo manejamos en safe_import
    for kw in forbidden:
        if kw in code:
            # Comentar la línea peligrosa en lugar de romper todo
            code = code.replace(kw, f"# BLOCKED: {kw}")
    return code

def execute_user_code(code: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Ejecuta el código de usuario en un hilo separado con stdout capturado.
    
    Args:
        code (str): Código Python a ejecutar.
        timeout (int): Tiempo máximo de ejecución en segundos.
        
    Returns:
        dict: {'success': bool, 'output': str, 'error': str}
    """
    result = {'success': False, 'output': '', 'error': ''}
    
    def target(q):
        buffer = io.StringIO()
        try:
            # Saneamiento básico
            safe_code = sanitize_code(code)
            
            # Configuración del entorno restringido (Sandbox)
            # Definimos qué funciones 'built-in' puede ver el estudiante
            safe_builtins_map = {
                # Básicos
                'print': print,
                'range': range,
                'len': len,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'bool': bool,
                
                # Matemáticas y Utilidades
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'round': round,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'enumerate': enumerate,
                
                # Excepciones
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                
                # Soporte para Clases y POO
                '__build_class__': __build_class__,  
                'object': object,                    # Necesario para herencia base
                'super': super,                      # Necesario para llamar al padre
                'classmethod': classmethod,          # Decorador útil para enseñar POO
                'staticmethod': staticmethod,        # Decorador útil para enseñar POO
                'property': property,                # Getters/Setters pythonicos
                'type': type,                        # Para introspección de tipos
                'isinstance': isinstance,            # Validación de tipos
                
                # Sistema
                '__import__': safe_import,
                '__name__': '__main__'               # Evita errores en algunos contextos de ejecución
            }

            # Entorno global inicial
            env = {'__builtins__': safe_builtins_map}

            # Ejecución
            with redirect_stdout(buffer):
                exec(safe_code, env)

            q.put({'success': True, 'output': buffer.getvalue(), 'error': ''})

        except SyntaxError as se:
            q.put({'success': False, 'output': buffer.getvalue(), 'error': f"Error de Sintaxis: {se}"})
        except ImportError as ie:
            q.put({'success': False, 'output': buffer.getvalue(), 'error': f"Error de Importación: {ie}"})
        except Exception:
            q.put({'success': False, 'output': buffer.getvalue(), 'error': traceback.format_exc()})

    # Gestión de Hilos (Threading) para evitar bloqueos infinitos (while True)
    q = queue.Queue()
    thread = threading.Thread(target=target, args=(q,))
    thread.start()
    
    try:
        thread.join(timeout)
        if thread.is_alive():
            result['error'] = "Tiempo de ejecución excedido (Timeout). ¿Tienes un bucle infinito?"
            # Nota: Python threads no se pueden matar forzosamente de forma segura, 
            # pero el frontend dejará de esperar.
        else:
            if not q.empty():
                result = q.get()
            else:
                result['error'] = "Error desconocido: No se recibió respuesta del hilo."
    except Exception as e:
        result['error'] = str(e)

    return result