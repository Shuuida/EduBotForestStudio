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
from typing import Dict, Any, List, Union

# Intenta cargar los módulos ML sin romper el módulo base en caso de no existir
try:
    from core import ml_rules
    from core import ml_struct_rules
except ImportError:
    ml_rules = None
    ml_struct_rules = None

# ---------------------------
# Utilitidades internas

def _indent(text: str, level: int = 1) -> str:
    """Aplica indentación consistente según el nivel."""
    return "\n".join(("    " * level) + line if line.strip() else line for line in text.splitlines())

def _safe_repr_value(value: str) -> str:
    s = str(value)
    if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
        return s
    try:
        float(s)
        return s
    except Exception:
        pass
    return repr(s)

def _ast_expr_to_str(node: ast.AST) -> str:
    """Devuelve una representación compacta similar a una fuente para expresiones AST básicas."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_ast_expr_to_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        func = _ast_expr_to_str(node.func)
        args = ", ".join(_ast_expr_to_str(a) for a in node.args)
        return f"{func}({args})"
    if isinstance(node, ast.BinOp):
        left = _ast_expr_to_str(node.left)
        right = _ast_expr_to_str(node.right)
        op_map = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%",
            ast.Pow: "**", ast.FloorDiv: "//"
        }
        op_symbol = op_map.get(type(node.op), "?")
        return f"({left} {op_symbol} {right})"
    if isinstance(node, ast.Subscript):
        return f"{_ast_expr_to_str(node.value)}[{_ast_expr_to_str(node.slice)}]"
    return ast.dump(node)

# ---------------------------
# BLOCK -> PYTHON
# (Expansión de block_to_code para tipos de bloques avanzados)

def block_to_code(block: Dict[str, Any], level: int = 0) -> str:
    """
    Convierte un diccionario de bloques en una cadena de código Python. 
    Admite tipos de bloques básicos y avanzados con identación.
    """
    t = block.get("type")

    # bloques ML
    if t.startswith("ml_") and ml_rules:
        try:
            func_name = f"{t}_block_to_code"
            if hasattr(ml_rules, func_name):
                func = getattr(ml_rules, func_name)
                return func(block)
        except Exception as e:
            # No interrumpe flujo, solo anota el error
            print(f"[WARN] ML block_to_code error: {e}")
            pass

    # print
    if t == "print":
        # El 'value' ya es la expresión correcta (ej: "result" o "'hello'")
        return _indent(f"print({block.get('value', '')})", level)

    # asignación
    if t == "assign":
        return _indent(f"{block.get('name', 'var')} = {block.get('value', 'None')}", level)

    # if
    if t == "if":
        cond = block.get("condition", "True")
        body = block.get("body", [])
        else_block = block.get("else", [])
        body_code = "\n".join(block_to_code(b, level + 1) for b in body) if body else _indent("pass", level + 1)
        code = f"{_indent(f'if {cond}:', level)}\n{body_code}"
        if else_block:
            else_code = "\n".join(block_to_code(b, level + 1) for b in else_block)
            code += f"\n{_indent('else:', level)}\n{else_code}"
        return code

    # while
    if t == "while":
        cond = block.get("condition", "True")
        body = block.get("body", [])
        body_code = "\n".join(block_to_code(b, level + 1) for b in body) if body else _indent("pass", level + 1)
        return f"{_indent(f'while {cond}:', level)}\n{body_code}"

    # for
    if t == "for_range":
        var = block.get("var", "i")
        start = block.get("start")
        end = block.get("end")
        step = block.get("step")
        range_expr = "range(0)" if not end else f"range({start}, {end}, {step})" if step else f"range({start}, {end})"
        body = block.get("body", [])
        body_code = "\n".join(block_to_code(b, level + 1) for b in body) if body else _indent("pass", level + 1)
        return f"{_indent(f'for {var} in {range_expr}:', level)}\n{body_code}"

    # function
    if t == "function":
        name = block.get("name", "func")
        args = ", ".join(block.get("args", []))
        body = block.get("body", [])
        body_code = "\n".join(block_to_code(b, level + 1) for b in body) if body else _indent("pass", level + 1)
        return f"{_indent(f'def {name}({args}):', level)}\n{body_code}"

    # class
    if t == "class":
        name = block.get("name", "MyClass")
        body = block.get("body", [])
        body_code = "\n".join(block_to_code(b, level + 1) for b in body) if body else _indent("pass", level + 1)
        return f"{_indent(f'class {name}:', level)}\n{body_code}"

    # return
    if t == "return":
        return _indent(f"return {block.get('value', '')}", level)

    # try/except
    if t == "try":
        body = block.get("body", [])
        handlers = block.get("handlers", [])
        code = f"{_indent('try:', level)}\n" + "\n".join(block_to_code(b, level + 1) for b in body)
        for h in handlers:
            h_type = h.get("type", "Exception")
            name = h.get("name")
            header = f"except {h_type}" + (f" as {name}" if name else "") + ":"
            code += f"\n{_indent(header, level)}\n" + "\n".join(block_to_code(b, level + 1) for b in h.get("body", []))
        return code

    # import
    if t == "import":
        mods = ", ".join(block.get("modules", []))
        return _indent(f"import {mods}", level)

    # expr_call
    if t == "expr_call":
        func = block.get("func", "")
        args = ", ".join(block.get("args", []))
        return _indent(f"{func}({args})", level)

    return _indent(f"# unsupported block: {t}", level)

# ---------------------------
# PYTHON -> BLOCK (line-based fallback)
# (_keep node_to_block para compatibilidad con versiones anteriores)

def _strip_quotes_if_literal(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        return s[1:-1]
    return s

def node_to_block(code_line: str) -> Dict[str, Any]:
    """
    Conversión basada en líneas (compatible con versiones anteriores). 
    Ideal para conversiones simples de una línea y como alternativa.
    """
    if not isinstance(code_line, str):
        raise TypeError("code_line must be a string")

    line = code_line.strip()
    if line == "":
        return {"type": "noop"}

    # Soporte para llamadas de Machine Learning (ml_rules)
    # Detecta si la línea es una llamada tipo ml_* (por ejemplo: ml_train(...))
    line_str = str(code_line).strip()
    if isinstance(code_line, str) and line_str.startswith("ml_"):
        if ml_struct_rules:
            try:
                func_name = f"{line_str.split('(')[0]}_to_block"
                if hasattr(ml_struct_rules, func_name):
                    func = getattr(ml_struct_rules, func_name)
                    return func(code_line)
            except Exception as e:
                print(f"[WARN] ML node_to_block error: {e}")
                pass
    # print
    if line.startswith("print(") and line.endswith(")"):
        inner = line[len("print("):-1].strip()
        if len(inner) >= 2 and inner[0] in ('"', "'") and inner[-1] == inner[0]:
            inner_val = _strip_quotes_if_literal(inner)
        else:
            inner_val = inner
        return {"type": "print", "value": inner_val}

    # return
    if line.startswith("return "):
        return {"type": "return", "value": line[len("return "):].strip()}

    # break / continue
    if line == "break":
        return {"type": "break"}
    if line == "continue":
        return {"type": "continue"}

    # assignment
    if "=" in line and "==" not in line and not line.startswith("if ") and not line.startswith("for ") and not line.startswith("while ") and not line.startswith("def ") and not line.startswith("class "):
        parts = line.split("=", 1)
        name = parts[0].strip()
        value = parts[1].strip()
        return {"type": "assign", "name": name, "value": value}

    # basic def
    if line.startswith("def ") and line.endswith(":"):
        try:
            header = line[4:-1].strip()
            name, rest = header.split("(", 1)
            args = rest.split(")")[0]
            args_list = [a.strip() for a in args.split(",") if a.strip()]
            return {"type": "function", "name": name.strip(), "args": args_list, "body": []}
        except Exception:
            pass

    if line.startswith("class ") and line.endswith(":"):
        name_part = line[6:-1].strip()
        return {"type": "class", "name": name_part, "body": []}

    # import
    if line.startswith("import "):
        modules = [m.strip() for m in line[len("import "):].split(",")]
        return {"type": "import", "modules": modules}

    if line.startswith("from ") and " import " in line:
        try:
            left, right = line.split(" import ", 1)
            module = left[len("from "):].strip()
            names = [n.strip() for n in right.split(",")]
            return {"type": "from_import", "module": module, "names": names}
        except Exception:
            pass

    # with
    if line.startswith("with ") and line.endswith(":"):
        ctx = line[len("with "):-1].strip()
        return {"type": "with", "context": ctx, "body": []}

    # try / except headers
    if line.startswith("try:"):
        return {"type": "try", "body": [], "handlers": [], "finalbody": []}
    if line.startswith("except "):
        return {"type": "unsupported", "code": line}  # need context to be useful

    # comprehension naive detect
    if line.startswith("[") and " for " in line and "]" in line:
        # naive parse
        inner = line[1:line.rfind("]")]
        # Esto es aproximado
        return {"type": "listcomp", "code": line}

    return {"type": "unsupported", "code": line}

# ---------------------------
# PYTHON -> BLOCK (AST-based, richer)
# ---------------------------

def ast_node_to_block(node: ast.AST) -> Dict[str, Any]:
    """
    Convierte un nodo AST (declaración de nivel superior) en un diccionario de bloque.
    Esto es más completo que el análisis basado en líneas y mantiene la anidación además
    del identado.
    """
    # Asignación
    if isinstance(node, ast.Assign):
        target = node.targets[0]
        if isinstance(target, ast.Attribute):
            name = _ast_expr_to_str(target)  # Ej: self.x
        elif isinstance(target, ast.Name):
            name = target.id
        else:
            name = ast.dump(target)
        value = _ast_expr_to_str(node.value)
        return {"type": "assign", "name": name, "value": value}

    # Expresiones
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Call):
            func = _ast_expr_to_str(node.value.func)
            args = [_ast_expr_to_str(a) for a in node.value.args]
            if func == "print":
                return {"type": "print", "value": args[0] if args else ""}
            return {"type": "expr_call", "func": func, "args": args}
        return {"type": "unsupported", "repr": ast.dump(node.value)}

    # If
    if isinstance(node, ast.If):
        test = _ast_expr_to_str(node.test)
        body = [ast_node_to_block(n) for n in node.body]
        else_body = [ast_node_to_block(n) for n in node.orelse] if node.orelse else []
        return {"type": "if", "condition": test, "body": body, "else": else_body}

    # For
    if isinstance(node, ast.For):
        var = _ast_expr_to_str(node.target)
        it = _ast_expr_to_str(node.iter)
        body = [ast_node_to_block(n) for n in node.body]
        return {"type": "for_range", "var": var, "iter": it, "body": body}

    # While
    if isinstance(node, ast.While):
        test = _ast_expr_to_str(node.test)
        body = [ast_node_to_block(n) for n in node.body]
        return {"type": "while", "condition": test, "body": body}

    # Funciones
    if isinstance(node, ast.FunctionDef):
        name = node.name
        args = [a.arg for a in node.args.args]
        body = [ast_node_to_block(n) for n in node.body]
        return {"type": "function", "name": name, "args": args, "body": body}

    # Clases
    if isinstance(node, ast.ClassDef):
        name = node.name
        bases = [_ast_expr_to_str(b) for b in node.bases]
        body = [ast_node_to_block(n) for n in node.body]
        return {"type": "class", "name": name, "bases": bases, "body": body}

    # Nueva: ML-specific (e.g., si es Call a train)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name in ['DecisionTreeClassifier', 'RandomForestClassifier', 'MiniLinearModel', 'MiniSVM', 'MiniNeuralNetwork']:
            args = {kw.arg: _ast_expr_to_str(kw.value) for kw in node.keywords}
            return {"type": f"ml_train_{func_name.lower()}", "args": args}

    # Import / From Import
    if isinstance(node, ast.Import):
        modules = [alias.name for alias in node.names]
        return {"type": "import", "modules": modules}

    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        names = [alias.name for alias in node.names]
        return {"type": "from_import", "module": module, "names": names}

    # Try / Except
    if isinstance(node, ast.Try):
        body = [ast_node_to_block(n) for n in node.body]
        handlers = []
        for h in node.handlers:
            typ = _ast_expr_to_str(h.type) if h.type else "Exception"
            handler_body = [ast_node_to_block(n) for n in h.body]
            handlers.append({"type": typ, "name": h.name, "body": handler_body})
        return {"type": "try", "body": body, "handlers": handlers, "finalbody": []}

    # Return
    if isinstance(node, ast.Return):
        val = _ast_expr_to_str(node.value) if node.value else ""
        return {"type": "return", "value": val}

    # Break / Continue
    if isinstance(node, ast.Break):
        return {"type": "break"}
    if isinstance(node, ast.Continue):
        return {"type": "continue"}

    return {"type": "unsupported", "repr": ast.dump(node)}

def parse_code_to_blocks(code: str) -> List[Dict[str, Any]]:
    """
    Analiza el código fuente completo de Python (posiblemente de varias líneas) 
    con ast y devuelve una lista de bloques que representan sentencias de nivel superior.
    Recurre a una lista vacía en caso de SyntaxError (el que llama debe recurrir al analizador de líneas).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    blocks: List[Dict[str, Any]] = []
    for node in tree.body:
        try:
            block = ast_node_to_block(node)
            blocks.append(block)
        except Exception as e:
            blocks.append({"type": "error", "message": str(e)})
    return blocks