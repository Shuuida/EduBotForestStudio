"""
========================================================
EduBot — Python <-> Blocks translation rules module
========================================================

This module defines the bidirectional translation rules
between Python code structures and visual blocks.
All rules are based on the official Python documentation:
https://docs.python.org/3/tutorial/index.html

Módulo de reglas de traducción bidireccional entre
estructuras de código Python y bloques visuales.
Basado en la documentación oficial de Python.
========================================================
"""

import ast
from typing import Dict, Any, List, Union, Optional

# Intenta cargar los módulos ML de forma segura
try:
    from core import ml_rules
    from core import ml_struct_rules
except ImportError:
    ml_rules = None
    ml_struct_rules = None

# ----------------------------------------------------------
# BLOQUES -> CÓDIGO PYTHON (Generación Recursiva)

def _safe_str(val: Any) -> str:
    """Convierte valores a string. Si es None o vacío, devuelve cadena vacía."""
    if val is None: return ""
    s = str(val).strip()
    return s

def _generate_body(body: List[Dict], level: int) -> str:
    """Genera el código para un bloque interno (indentado)."""
    if not body:
        return "    " * level + "pass\n"
    
    code = ""
    for block in body:
        code += block_to_code(block, level)
    return code

def block_to_code(block: Dict[str, Any], level: int = 0) -> str:
    """Convierte un bloque JSON a código Python."""
    if not isinstance(block, dict): return f"# Error: Bloque inválido\n"

    b_type = block.get('type', '')
    indent = "    " * level
    
    # FUNDAMENTOS PYTHON (Mapeo Directo)
    # Variables
    if b_type == 'py_var':
        target = _safe_str(block.get('var_name', 'x'))
        value = _safe_str(block.get('value', '0'))
        return f"{indent}{target} = {value}\n"

    # Operaciones Matemáticas (Math)
    if b_type == 'py_math':
        target = _safe_str(block.get('target', 'res'))
        left = _safe_str(block.get('left', '0'))
        right = _safe_str(block.get('right', '0'))
        # Leemos el operador, pero si viene vacío ("") o es inválido, forzamos el '+'
        op = block.get('op', '+')
        if not op or op not in ['+', '-', '*', '/']:
            op = '+'
        # Si no hay target, es una expresión suelta
        if not target:
            return f"{indent}{left} {op} {right}\n"
        return f"{indent}{target} = {left} {op} {right}\n"

    # Print
    if b_type == 'py_print':
        content = _safe_str(block.get('content', ''))
        return f"{indent}print({content})\n"

    # Input
    if b_type == 'py_input':
        target = _safe_str(block.get('target', 'entrada'))
        prompt = _safe_str(block.get('prompt', ''))
        safe_prompt = f"'{prompt}'" if prompt else "''"
        return f"{indent}{target} = input({safe_prompt})\n"

    # Transformación de Tipos (Casting a Int / Float)
    if b_type in ['py_int', 'py_float']:
        target = _safe_str(block.get('target', 'res'))
        val = _safe_str(block.get('value', '0'))
        func = 'int' if b_type == 'py_int' else 'float'
        return f"{indent}{target} = {func}({val})\n"

    # Comparaciones (Sueltas o para IF)
    if b_type == 'py_compare':
        left = _safe_str(block.get('left', 'a'))
        # Leemos el operador, pero si viene vacío ("") o es inválido, forzamos el '=='
        op = block.get('op', '==')
        valid_ops = ['==', '!=', '>', '<', '>=', '<=']
        if not op or op not in valid_ops:
            op = '=='
            
        right = _safe_str(block.get('right', 'b'))
        return f"{indent}{left} {op} {right}\n"

    # ESTRUCTURAS DE CONTROL (Con Body)

    # If
    if b_type == 'py_if':
        cond = _safe_str(block.get('condition', 'True'))
        header = f"{indent}if {cond}:\n"
        return header + _generate_body(block.get('body', []), level + 1)

    # Elif
    if b_type == 'py_elif':
        cond = _safe_str(block.get('condition', 'True'))
        header = f"{indent}elif {cond}:\n"
        return header + _generate_body(block.get('body', []), level + 1)

    # Else
    if b_type == 'py_else':
        header = f"{indent}else:\n"
        return header + _generate_body(block.get('body', []), level + 1)

    # Loops
    if b_type == 'py_loop': # For
        iterator = _safe_str(block.get('iterator', 'i'))
        iterable = _safe_str(block.get('iterable', 'range(10)'))
        header = f"{indent}for {iterator} in {iterable}:\n"
        return header + _generate_body(block.get('body', []), level + 1)

    if b_type == 'py_while':
        cond = _safe_str(block.get('condition', 'True'))
        header = f"{indent}while {cond}:\n"
        return header + _generate_body(block.get('body', []), level + 1)

    # Funciones
    if b_type == 'py_func':
        name = _safe_str(block.get('func_name', 'mi_funcion'))
        args = _safe_str(block.get('args', ''))
        header = f"{indent}def {name}({args}):\n"
        return header + _generate_body(block.get('body', []), level + 1)

    # Llamar funciones
    if b_type == 'py_call':
        func = _safe_str(block.get('func', 'metodo'))
        args = _safe_str(block.get('args', ''))
        return f"{indent}{func}({args})\n"

    # Clases
    if b_type == 'py_class':
        name = _safe_str(block.get('name', 'MiClase'))
        bases = _safe_str(block.get('bases', ''))
        base_str = f"({bases})" if bases else ""
        header = f"{indent}class {name}{base_str}:\n"
        return header + _generate_body(block.get('body', []), level + 1)

    # Constructor de Clase (__init__)
    if b_type == 'py_init':
        args = _safe_str(block.get('args', 'self'))
        header = f"{indent}def __init__({args}):\n"
        return header + _generate_body(block.get('body', []), level + 1)

    # Atributo de Clase (self.algo = valor)
    if b_type == 'py_self':
        attr = _safe_str(block.get('attr', 'propiedad'))
        val = _safe_str(block.get('value', 'None'))
        return f"{indent}self.{attr} = {val}\n"

    # Try
    if b_type == 'py_try':
        header = f"{indent}try:\n"
        return header + _generate_body(block.get('body', []), level + 1)

    # Except
    if b_type == 'py_except':
        err_type = _safe_str(block.get('error_type', 'Exception as e'))
        header = f"{indent}except {err_type}:\n"
        return header + _generate_body(block.get('body', []), level + 1)

    # Otros
    if b_type == 'py_return':
        return f"{indent}return {_safe_str(block.get('value', ''))}\n"

    # Nodo Validador (Autograding Python)
    if b_type == 'py_validator':
        challenge_id = _safe_str(block.get('challenge_id', 'reto_1'))
        target = _safe_str(block.get('target', 'res'))
        expected = _safe_str(block.get('expected', '10')).replace("'", "\\'")
        
        # Genera un if/else que imprime los códigos de telemetría ocultos
        return (
            f"{indent}# --- VALIDACIÓN DEL RETO: {challenge_id} ---\n"
            f"{indent}if str({target}) == str('{expected}'):\n"
            f"{indent}    print('EDUBOT_VAL_PASS|{challenge_id}')\n"
            f"{indent}    print('✅ [Validador] ¡Reto superado correctamente!')\n"
            f"{indent}else:\n"
            f"{indent}    print('EDUBOT_VAL_FAIL|{challenge_id}')\n"
            f"{indent}    print('❌ [Validador] Resultado incorrecto.')\n"
        )
    
    if b_type == 'py_import':
        names = block.get('names', '')
        src = block.get('from', '')
        if src: return f"{indent}from {src} import {names}\n"
        return f"{indent}import {names}\n"

    if b_type == 'py_control':
        # Si el usuario no interactúa con el desplegable, la variable viene vacía.
        # Forzamos a que el valor por defecto sea 'break', igual que en la interfaz visual.
        c_type = block.get('control_type', 'break')
        
        # Validación extra de seguridad para evitar inyección de código
        if c_type not in ['break', 'continue', 'pass']:
            c_type = 'break'
            
        return f"{indent}{c_type}\n"

    # MACHINE LEARNING
    if ml_rules and b_type.startswith('ml_'):
        return ml_rules.get_ml_code(block)

    return f"{indent}# Bloque desconocido: {b_type}\n"

# --------------------------------------------------------------------
# CÓDIGO PYTHON -> BLOQUES (AST Parsing - Ingeniería Inversa)

def _ast_op_to_str(op):
    if isinstance(op, ast.Add): return '+'
    if isinstance(op, ast.Sub): return '-'
    if isinstance(op, ast.Mult): return '*'
    if isinstance(op, ast.Div): return '/'
    if isinstance(op, ast.Eq): return '=='
    if isinstance(op, ast.NotEq): return '!='
    if isinstance(op, ast.Lt): return '<'
    if isinstance(op, ast.LtE): return '<='
    if isinstance(op, ast.Gt): return '>'
    if isinstance(op, ast.GtE): return '>='
    return '?'

def _ast_expr_to_str(node):
    if node is None: return ""
    if isinstance(node, ast.Constant): return repr(node.value)
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Str): return repr(node.s)
    if isinstance(node, ast.Num): return str(node.n)
    if isinstance(node, ast.Call):
        func = _ast_expr_to_str(node.func)
        args = ", ".join([_ast_expr_to_str(a) for a in node.args])
        return f"{func}({args})"
    if isinstance(node, ast.BinOp):
        return f"{_ast_expr_to_str(node.left)} {_ast_op_to_str(node.op)} {_ast_expr_to_str(node.right)}"
    if isinstance(node, ast.Compare):
        # Simplificación: solo primer operador
        return f"{_ast_expr_to_str(node.left)} {_ast_op_to_str(node.ops[0])} {_ast_expr_to_str(node.comparators[0])}"
    return ""

def _parse_body(stmts: List[ast.stmt]) -> List[Dict[str, Any]]:
    """Parsea recursivamente una lista de sentencias AST."""
    blocks = []
    for n in stmts:
        res = ast_node_to_block(n)
        if isinstance(res, list):
            blocks.extend(res)
        else:
            blocks.append(res)
    return blocks

def _ast_args_to_str(args_node):
    """
    Extrae los nombres de los argumentos de un nodo ast.arguments 
    y los convierte en un string separado por comas (ej: 'self, x, y').
    """
    if not args_node:
        return ""
        
    # Extraemos el nombre (.arg) de cada argumento en la lista de argumentos del AST
    arg_names = [arg.arg for arg in args_node.args]
    
    # Los unimos con una coma y un espacio
    return ", ".join(arg_names)

def ast_node_to_block(node: ast.AST) -> Dict[str, Any]:
    """Convierte un nodo AST en un diccionario de bloque."""
    
    # Asignaciones (Variables, Atributos, Input, Conversiones, Math)
    if isinstance(node, ast.Assign):
        # Detectar si es un Atributo de Clase (ej. self.nombre = ...)
        if isinstance(node.targets[0], ast.Attribute):
            target_attr = node.targets[0]
            if isinstance(target_attr.value, ast.Name) and target_attr.value.id == 'self':
                return {
                    "type": "py_self",
                    "attr": target_attr.attr,
                    "value": _ast_expr_to_str(node.value)
                }

        # Extraemos el nombre de la variable receptora
        target = _ast_expr_to_str(node.targets[0])

        # Input
        if isinstance(node.value, ast.Call) and _ast_expr_to_str(node.value.func) == 'input':
            prompt = _ast_expr_to_str(node.value.args[0]) if node.value.args else '""'
            return {"type": "py_input", "target": target, "prompt": prompt}

        # Castings (int, float)
        if isinstance(node.value, ast.Call):
            func_name = _ast_expr_to_str(node.value.func)
            if func_name in ['int', 'float']:
                arg_val = _ast_expr_to_str(node.value.args[0]) if node.value.args else '0'
                return {
                    "type": f"py_{func_name}", 
                    "target": target, 
                    "value": arg_val
                }

        # Math
        if isinstance(node.value, ast.BinOp):
            return {
                "type": "py_math", 
                "target": target, 
                "left": _ast_expr_to_str(node.value.left), 
                "op": _ast_op_to_str(node.value.op), 
                "right": _ast_expr_to_str(node.value.right)
            }

        # Variable Normal (Fallback)
        value = _ast_expr_to_str(node.value)
        return {"type": "py_var", "var_name": target, "value": value}

    # Control de Flujo (Recursivo)
    if isinstance(node, ast.If):
        if_block = {
            "type": "py_if", 
            "condition": _ast_expr_to_str(node.test),
            "body": _parse_body(node.body)
        }
        if node.orelse:
            # Detección de Elif (En AST, un elif es un if dentro de un orelse)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                elif_res = ast_node_to_block(node.orelse[0])
                if isinstance(elif_res, dict):
                    elif_res['type'] = 'py_elif'
                    return [if_block, elif_res]
                elif isinstance(elif_res, list):
                    elif_res[0]['type'] = 'py_elif'
                    return [if_block] + elif_res
            else:
                # Detección de Else tradicional
                else_block = {
                    "type": "py_else",
                    "body": _parse_body(node.orelse)
                }
                return [if_block, else_block]
                
        return if_block

    if isinstance(node, ast.For):
        return {
            "type": "py_loop", 
            "iterator": _ast_expr_to_str(node.target), 
            "iterable": _ast_expr_to_str(node.iter),
            "body": _parse_body(node.body)
        }
    if isinstance(node, ast.While):
        return {
            "type": "py_while", 
            "condition": _ast_expr_to_str(node.test),
            "body": _parse_body(node.body)
        }

    # Funciones y Constructores
    if isinstance(node, ast.FunctionDef):
        args_str = _ast_args_to_str(node.args)
        
        # Detectar si es el constructor
        if node.name == '__init__':
            return {
                "type": "py_init",
                "args": args_str,
                "body": _parse_body(node.body)
            }
        else:
            return {
                "type": "py_func",
                "func_name": node.name,
                "args": args_str,
                "body": _parse_body(node.body)
            }

    # Clases
    if isinstance(node, ast.ClassDef):
        bases = ", ".join([_ast_expr_to_str(b) for b in node.bases])
        return {
            "type": "py_class", 
            "name": node.name, 
            "bases": bases,
            "body": _parse_body(node.body)
        }

    # Try/Except
    if isinstance(node, ast.Try):
        try_block = {
            "type": "py_try",
            "body": _parse_body(node.body)
        }
        
        blocks = [try_block]
        
        # Iteramos sobre todos los bloques 'except' que tenga el código
        for handler in node.handlers:
            err_type = ""
            if handler.type:
                err_type = _ast_expr_to_str(handler.type)
                # Si el error tiene un alias (ej: except ValueError as e)
                if handler.name:
                    err_type += f" as {handler.name}"
            else:
                err_type = "Exception as e"
                
            except_block = {
                "type": "py_except",
                "error_type": err_type,
                "body": _parse_body(handler.body)
            }
            blocks.append(except_block)
            
        return blocks

    # Expresiones (Print o Llamadas a Funciones sueltas)
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Call):
            func = _ast_expr_to_str(node.value.func)
            args = [_ast_expr_to_str(a) for a in node.value.args]
            
            if func == 'print': 
                # Unimos los argumentos con coma por si hacen print(a, b)
                return {"type": "py_print", "content": ", ".join(args) if args else ""}
                
            return {
                "type": "py_call", 
                "func": func, 
                "args": ", ".join(args)
            }

    if isinstance(node, ast.Return):
        return {"type": "py_return", "value": _ast_expr_to_str(node.value) if node.value else ""}

    # Imports
    if isinstance(node, ast.Import): 
        return {"type": "py_import", "names": ", ".join([a.name for a in node.names])}
    if isinstance(node, ast.ImportFrom): 
        return {"type": "py_import", "from": node.module, "names": ", ".join([a.name for a in node.names])}

    # Control Simple
    if isinstance(node, ast.Break): return {"type": "py_control", "control_type": "break"}
    if isinstance(node, ast.Continue): return {"type": "py_control", "control_type": "continue"}
    if isinstance(node, ast.Pass): return {"type": "py_control", "control_type": "pass"}

    return {"type": "unsupported", "repr": str(type(node))}

def parse_code_to_blocks(code: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(code)
        return [ast_node_to_block(n) for n in tree.body]
    except Exception as e:
        return [{"type": "error", "msg": f"Parse Error: {e}"}]

def node_to_block(line: str) -> Dict[str, Any]:
    return {"type": "code", "code": line}