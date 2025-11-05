"""Módulo ejecutor para EduBot
Ejecuta código Python de forma segura a partir de la traducción basada en nodos.
Si RestrictedPython y PrintCollector están disponibles y compatibles -> usarlo (wrap para compatibilidad).
Si no -> usar fallback exec() con __builtins__ reducido (determinista).
Siempre devolver: {'success': bool, 'output': str, 'error': str}"""

import io
import traceback
import threading
import queue
from contextlib import redirect_stdout
from typing import Dict

import importlib  # Para imports seguros

import sys
import time

# Intento de importaciones opcionales de RestrictedPython
try:
    from RestrictedPython import compile_restricted  # type: ignore
    try:
        from RestrictedPython.PrintCollector import PrintCollector  # type: ignore
    except Exception:
        PrintCollector = None
    try:
        from RestrictedPython import safe_builtins  # type: ignore
    except Exception:
        safe_builtins = {}
except Exception:
    compile_restricted = None
    PrintCollector = None
    safe_builtins = {}

# -------------------------
# Utilidades

def sanitize_code(code: str) -> str:
    forbidden = [
        'import os', 'import sys', '__import__', 'eval', 'exec',
        'open', 'compile', 'input', 'globals', 'locals', 'vars'
    ]
    for kw in forbidden:
        code = code.replace(kw, f"# {kw} (blocked)")
    return code

# Wrapper útil para PrintCollector
def _make_print_wrapper(pc):
    def _wrapper(*args, sep=' ', end='\n'):
        try:
            text = sep.join(str(a) for a in args) + end
            try:
                return pc(text)
            except TypeError:
                if hasattr(pc, "prints") and isinstance(pc.prints, list):
                    pc.prints.append(text)
                elif hasattr(pc, "result"):
                    pc.result = (pc.result or "") + text
                return None
        except Exception:
            return None
    return _wrapper

# Nueva: Imports seguros (solo permitidos de core.ml_runtime, etc.)
ALLOWED_IMPORTS = {'core.ml_runtime': ['DecisionTreeClassifier', 'RandomForestClassifier', 'MiniLinearModel', 'MiniSVM', 'MiniNeuralNetwork', 'accuracy_score']}

def safe_import(module_name, names):
    if module_name not in ALLOWED_IMPORTS:
        raise ImportError(f"Import bloqueado: {module_name}")
    module = importlib.import_module(module_name)
    imported = {}
    for name in names:
        if name == '*':
            for n in ALLOWED_IMPORTS[module_name]:
                imported[n] = getattr(module, n)
        elif name in ALLOWED_IMPORTS[module_name]:
            imported[name] = getattr(module, name)
        else:
            raise ImportError(f"Import bloqueado: {name} de {module_name}")
    return imported

# -------------------------
# Función principal con timeout

def execute_user_code(code: str, timeout=5) -> Dict[str, object]:
    result = {'success': False, 'output': '', 'error': ''}

    def target(q):
        try:
            code = sanitize_code(code)
            buffer = io.StringIO()

            if compile_restricted and PrintCollector is not None:
                builtins = dict(safe_builtins) if isinstance(safe_builtins, dict) else {}
                builtins.setdefault('range', range)
                builtins.setdefault('len', len)
                builtins.setdefault('abs', abs)
                builtins.setdefault('min', min)
                builtins.setdefault('max', max)
                builtins['import'] = safe_import  # Seguro import

                pc = PrintCollector()
                print_wrapper = _make_print_wrapper(pc)

                globals_for_restricted = {
                    '__builtins__': builtins,
                    '_print_': print_wrapper,
                    '_getattr_': getattr,
                    '_getitem_': lambda obj, key: obj[key],
                    '_getiter_': iter,
                    '_iter_unpack_sequence_': tuple,
                    '_write_': lambda x: x,
                }

                byte_code = compile_restricted(code, '<user_code>', 'exec')
                exec(byte_code, globals_for_restricted)

                printed = ''
                if hasattr(pc, 'getvalue'):
                    printed = pc.getvalue()
                elif hasattr(pc, 'prints') and isinstance(pc.prints, list):
                    printed = '\n'.join(str(p) for p in pc.prints)
                elif hasattr(pc, 'result'):
                    printed = str(pc.result)

                q.put({'success': True, 'output': printed, 'error': ''})
                return

            # FALLBACK
            safe_builtins_map = {
                'print': print,
                'range': range,
                'len': len,
                'abs': abs,
                'min': min,
                'max': max,
                'Exception': Exception,
                'BaseException': BaseException,
                'import': safe_import,
            }

            env = {'__builtins__': safe_builtins_map}

            with redirect_stdout(buffer):
                exec(code, env)

            q.put({'success': True, 'output': buffer.getvalue(), 'error': ''})

        except SyntaxError as se:
            q.put({'success': False, 'output': '', 'error': f"SyntaxError: {se}"})
        except Exception as e:
            q.put({'success': False, 'output': '', 'error': traceback.format_exc()})

    q = queue.Queue()
    thread = threading.Thread(target=target, args=(q,))
    thread.start()
    try:
        thread.join(timeout)
        if thread.is_alive():
            # No se puede matar thread en Python; fallback a error
            result['error'] = "Execution timeout"
        else:
            result = q.get()
    except queue.Empty:
        result['error'] = "Execution timeout"
    return result

# Demo
if __name__ == "__main__":
    demo = "for i in range(3):\n    print('Demo', i)\n"
    print(execute_user_code(demo))