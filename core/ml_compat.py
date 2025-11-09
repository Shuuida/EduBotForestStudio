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
    - Si alguno no es convertible, devuelve False en lugar de error.
    """
    rv = resolve_numeric(row_val)
    nv = resolve_numeric(node_val)

    # Si ambos son números, comparar directamente
    if rv is not None and nv is not None:
        return rv <= nv

    # Si el valor del nodo no es numérico pero es un dict, puede ser un subnodo
    if nv is None and isinstance(node_val, dict):
        if callable(subnode_handler):
            # Devolver el resultado del sub-manejador
            return subnode_handler(node_val, row_val)
        else:
            # No hay manejador, no se puede comparar
            return False

    # Si el valor de la fila no es numérico, no se puede tomar una decisión
    if rv is None:
        return False

    # Fallback seguro: si no se puede comparar, no cumple la condición
    return False


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
    Devuelve False si la comparación no es posible.
    """
    m_val = resolve_numeric(metric)
    b_val = resolve_numeric(best_score)

    # Solo comparar si ambos son números válidos
    if m_val is not None and b_val is not None:
        return m_val <= b_val

    # Si no se pueden convertir, no se puede mejorar el score
    return False
