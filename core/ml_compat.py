"""
ml_compat.py
------------
Módulo de compatibilidad y utilidades globales para MiniML Framework.

Proporciona funciones de comparación segura, extracción numérica y
normalización de estructuras de árbol, eliminando errores del tipo:
'<=' not supported between instances of 'float' and 'dict'.

No depende de librerías externas (sin NumPy, sin pandas).
Puede ser replicado en firmware C o hardware embebido.
"""

# =============================
#  Funciones básicas de tipo
# =============================

def _is_number(x):
    """Verifica si x es un número (int o float)."""
    return isinstance(x, (int, float))

def to_float_if_possible(x):
    """Convierte a float si es posible. Si no puede, devuelve None."""
    if _is_number(x):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except Exception:
            return None
    return None


# =============================
#  Resolución numérica
# =============================

def resolve_numeric(value):
    """
    Extrae un número flotante desde cualquier tipo simple o dict.
    - Si es int/float -> devuelve float(value)
    - Si es str numérico -> float(value)
    - Si es dict:
        * Busca clave 'value', 'score', 'threshold', 'val', 'acc', 'accuracy'
        * Si la clave contiene número, devuelve ese número
        * Si el dict tiene un solo valor numérico, devuelve ese número
    Devuelve None si no encuentra número válido.
    """
    f = to_float_if_possible(value)
    if f is not None:
        return f

    if isinstance(value, dict):
        for k in ("value", "score", "threshold", "val", "acc", "accuracy"):
            if k in value:
                f2 = to_float_if_possible(value[k])
                if f2 is not None:
                    return f2

        if len(value) == 1:
            sole = next(iter(value.values()))
            f2 = to_float_if_possible(sole)
            if f2 is not None:
                return f2
    return None


# =============================
#  Comparación segura
# =============================

def safe_compare_le(row_val, node_val, subnode_handler=None):
    """
    Compara row_val <= node_val de forma segura.
    - Convierte ambos a float con resolve_numeric().
    - Si node_val es dict sin número, se asume subnodo y se delega al handler.
    - Si row_val no es convertible, lanza TypeError.
    - Retorna True/False o lanza TypeError.
    """
    rv = resolve_numeric(row_val)
    if rv is None and isinstance(row_val, dict):
        for v in row_val.values():
            rv = resolve_numeric(v)
            if rv is not None:
                break
    if rv is None:
        raise TypeError(f"Row value not numeric: {type(row_val)} -> {row_val}")

    nv = resolve_numeric(node_val)
    if nv is not None:
        return rv <= nv

    # node_val puede ser subnodo
    if isinstance(node_val, dict):
        if callable(subnode_handler):
            return subnode_handler(node_val, row_val)
        raise TypeError("Node value is a subnode but no handler provided.")

    raise TypeError(f"Node limit not numeric: {type(node_val)} -> {node_val}")


# =============================
#  Normalización de árbol
# =============================

def normalize_tree_node(node):
    """
    Recorre recursivamente un nodo y normaliza los valores.
    Convierte los valores numéricos representados como string o dict simple.
    """
    if node is None:
        return None
    if not isinstance(node, dict):
        return node

    # Normalizar hijos
    if "left" in node:
        node["left"] = normalize_tree_node(node["left"])
    if "right" in node:
        node["right"] = normalize_tree_node(node["right"])

    # Normalizar valor del nodo
    if "value" in node:
        v = node["value"]
        f = resolve_numeric(v)
        if f is not None:
            node["value"] = f
        elif isinstance(v, dict):
            node["value"] = normalize_tree_node(v)
    return node


def normalize_tree(tree):
    """Normaliza un árbol completo."""
    if isinstance(tree, list):
        return [normalize_tree_node(n) for n in tree]
    if isinstance(tree, dict):
        return normalize_tree_node(tree)
    return tree


# =============================
#  Comparación de métricas
# =============================

def safe_metric_compare(metric, best_score):
    """
    Compara métricas de evaluación evitando errores de tipo.
    Extrae valores numéricos de dicts o strings.
    """
    m_val = resolve_numeric(metric)
    if m_val is None and isinstance(metric, dict):
        m_val = resolve_numeric(metric)
    if m_val is None:
        raise TypeError(f"Metric not numeric: {metric}")

    b_val = resolve_numeric(best_score)
    if b_val is None:
        raise TypeError(f"Best score not numeric: {best_score}")

    return m_val <= b_val