"""
========================================================
EduBot — Translation Core Module
========================================================

Main controller for the translation system between Python
source code and visual block structures (JSON-based).

Controlador principal del sistema de traducción entre el
código fuente Python y los bloques visuales en formato JSON.
Basado en el módulo translation_rules.py
========================================================
"""

import ast
from typing import Any, Dict, List
import traceback

# Importamos todas las reglas definidas
from core.translation_rules import node_to_block, block_to_code, parse_code_to_blocks

# ========================================================

class Translator:
    """
    Main Translator class
    ----------------------
    Handles bidirectional translation:
      • Blocks -> Python
      • Python -> Blocks

    Clase principal del traductor
    ------------------------------
    Gestiona la traducción bidireccional:
      • Bloques -> Python
      • Python -> Bloques
    """

    def __init__(self):
        """
        Inicializa el traductor.
        Los diccionarios de reglas ya no son necesarios,
        ya que el módulo translation_rules.py se encarga
        de las conversiones internas.
        """
        self.version = "0.1"
        self.author = "EduBot Development Team"
        self.description = "Módulo principal de traducción Python <-> Bloques"

    def _safe_block_to_code(self, block: Any) -> str:
        """
        Traduce *un* bloque a código Python de forma segura.
        Devuelve una línea o bloque de código (string). Nunca lanza excepción fuera.
        """
        try:
            # Validación básica (robustecida para None y tipos inválidos)
            if block is None:
                return "# Error translating block: block is None"
            if not isinstance(block, dict):
                return f"# Error translating block: expected dict, got {type(block).__name__}"
            block_type = block.get("type")
            if block_type is None or not isinstance(block_type, str) or block_type.strip() == "":
                return f"# Error translating block: missing or invalid 'type' field in block {block!r}"

            from core.ml_rules import get_ml_code
            if block_type.startswith('ml_'):
                return get_ml_code(block)
            
            # Delega directamente a block_to_code de translation_rules (maneja ML y todo)
            return block_to_code(block)

        except Exception as outer_e:
            tb = traceback.format_exc()
            return f"# Unexpected error translating block: {outer_e} \n# {tb.splitlines()[-1]}"

    def _inject_auto_runner(self, blocks: List[Dict]) -> str:
        """
        Genera el auto-runner. 
        Auto-rellena argumentos con valores "dummy" (1)
        para forzar la ejecución y garantizar que la terminal siempre dé una respuesta.
        """
        runner_code = "\n# --- Auto-Runner ---\n"
        has_executable = False

        # Recorremos los bloques en el orden estricto que dictó el Frontend
        for block in blocks:
            b_type = block.get('type')
            
            if b_type == 'py_class':
                cls_name = block.get('name', 'MiClase')
                body = block.get('body', [])
                methods = [b for b in body if b.get('type') == 'py_func']
                
                # Análisis de argumentos del constructor
                init_block = next((b for b in body if b.get('type') == 'py_init'), None)
                init_args = init_block.get('args', 'self').replace('self', '').strip(' ,') if init_block else ""
                
                # Contamos comas y generamos un "1" por cada parámetro exigido
                dummy_init = ", ".join(["1"] * len([a for a in init_args.split(',') if a.strip()]))
                
                runner_code += f"try:\n"
                runner_code += f"    _{cls_name}_inst = {cls_name}({dummy_init})\n"
                
                if methods:
                    for m in methods:
                        m_name = m.get('func_name', 'main')
                        if m_name != '__init__':
                            m_args = m.get('args', '').replace('self', '').strip(' ,')
                            dummy_m_args = ", ".join(["1"] * len([a for a in m_args.split(',') if a.strip()]))
                            
                            runner_code += f"    try:\n"
                            runner_code += f"        _{cls_name}_inst.{m_name}({dummy_m_args})\n"
                            runner_code += f"    except Exception as e:\n"
                            runner_code += f"        print(f'❌ Error en método {m_name}: {{e}}')\n"
                
                runner_code += f"except Exception as e:\n"
                runner_code += f"    print(f'❌ Error al instanciar clase {cls_name}: {{e}}')\n"
                has_executable = True

            elif b_type == 'py_func':
                func_name = block.get('func_name', 'main')
                args = block.get('args', '').strip()
                
                # Analizador léxico rápido para auto-rellenar
                # Si pide (a, b), inyectará (1, 1). Si no pide nada, inyectará ().
                dummy_args = ", ".join(["1"] * len([a for a in args.split(',') if a.strip()]))
                
                runner_code += f"try:\n"
                runner_code += f"    {func_name}({dummy_args})\n"
                runner_code += f"except Exception as e:\n"
                runner_code += f"    print(f'❌ Error ejecutando función {func_name}: {{e}}')\n"
                has_executable = True

        if not has_executable:
            return ""
            
        return runner_code

    # ----------------------------------------------------
    # Bloques -> Python
    def translate_to_python(self, blocks: List[Dict[str, Any]]) -> str:
        """
        Traduce una lista de bloques (lista de dicts) a código Python.
        - Valida entrada
        - Traduce bloque a bloque con aislamiento de errores
        - Devuelve string con el código completo
        """
        if blocks is None:
            raise ValueError("blocks is None (expected list of dicts)")

        if not isinstance(blocks, list):
            raise TypeError("Expected a list of blocks")

        # Pre-procesado: Si es una lista con un wrapper {'result': list}, desenvuélvelo
        # Esto resuelve el error con {'result': [...]} sin romper inputs válidos
        if len(blocks) == 1 and isinstance(blocks[0], dict) and 'result' in blocks[0] and isinstance(blocks[0]['result'], list):
            blocks = blocks[0]['result']  # Ahora blocks es la lista real de bloques

        lines: List[str] = []
        for idx, block in enumerate(blocks):
            code_line = self._safe_block_to_code(block)
            # Normaliza: si la regla retorna varias líneas, las añadimos tal cual
            if isinstance(code_line, str):
                lines.append(code_line)
            else:
                # protección extra
                lines.append(f"# Error: translator returned non-str for block index {idx}")

        base_code = "\n".join(lines)

        auto_runner = self._inject_auto_runner(blocks)

        return base_code + auto_runner

    # ----------------------------------------------------
    # Python -> Bloques
    def translate_to_blocks(self, code: str) -> List[Dict[str, Any]]:
        """
        Convierte código Python en una lista de bloques JSON equivalentes.

        :param code: código fuente en Python.
        :return: lista de bloques en formato JSON.

        Prueba el análisis basado en AST (parse_code_to_blocks) 
        para obtener un resultado completo y estructurado.
        Si el análisis de AST devuelve un resultado vacío (sintaxis o no compatible), 
        recurre al node_to_block basado en líneas.
        """
        if code is None:
            raise TypeError("code is None")
        if not isinstance(code, str):
            raise TypeError("Expected code as string")

        # Prueba primero el análisis basado en AST
        blocks = []
        try:
            ast_blocks = parse_code_to_blocks(code)
            if ast_blocks:
                return ast_blocks
        except Exception:
            # Si el análisis de AST falla inesperadamente, lo ignora y vuelva a la opción anterior.
            pass

        # Alternativa: análisis línea por línea (compatibilidad con versiones anteriores)
        for line in code.splitlines():
            if line.strip() == "":
                continue
            try:
                blk = node_to_block(line)
                blocks.append(blk)
            except Exception as e:
                blocks.append({"type": "error", "message": str(e)})
        return blocks

    # ----------------------------------------------------
    # Utilidad de Depuración
    def preview_translation(self, code: str):
        """
        Vista previa: traduce Python -> Bloques y Bloques -> Python.
        Muestra cómo se comporta el sistema completo.
        """
        blocks = self.translate_to_blocks(code)
        reconstructed = self.translate_to_python(blocks)

        print("========= PYTHON ORIGINAL =========")
        print(code)
        print("========= BLOQUES GENERADOS =========")
        print(blocks)
        print("========= RECONSTRUCCIÓN PYTHON =========")
        print(reconstructed)