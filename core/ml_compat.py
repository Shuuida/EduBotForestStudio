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

from typing import List, Any, Dict

# -----------------------------------
#  Funciones básicas de tipo

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


# ------------------------------
#  Resolución numérica

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


# -------------------------------
#  Comparación segura

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


# ---------------------------------
#  Normalización de árbol

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

# ---------------------------------
#  Comparación de métricas

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

# ---------------------------------------------------
# FUNCIONES DE APLANADO DE ÁRBOLES DE DECISIÓN

def _flatten_tree_to_arrays(root_node: Any) -> Dict[str, List]:
    """
    Convierte un árbol de decisión (dict anidado) en arrays paralelos
    compatibles con firmware C.
    
    Usa un recorrido DFS (Pre-order) para construir los arrays.
    
    CORREGIDO: Esta función ahora maneja correctamente la diferencia entre
    nodos hoja (que son valores, ej: 0, 1, 3.14) y
    nodos de división (que son dicts con 'index', 'value', 'left', 'right').
    """
    # Arrays paralelos que se llenarán
    feature_index = []
    threshold = []
    left_child = []
    right_child = []
    value = [] # Almacenará el valor de la hoja (si es hoja) o 0.0 (si es división)

    def _traverse(node: Any) -> int:
        """
        Función interna recursiva.
        Añade el nodo actual a los arrays y devuelve su índice.
        """
        # Obtener el índice para este nodo
        current_index = len(feature_index)

        # CORRECCIÓN
        # Caso Base: Es un nodo Hoja (NO es un dict)
        if not isinstance(node, dict):
            feature_index.append(-1)          # Marcador C para hoja
            threshold.append(0.0)             # Valor dummy
            left_child.append(-1)             # Marcador C para hoja
            right_child.append(-1)            # Marcador C para hoja
            value.append(node)                # <-- Almacena el valor real de la hoja (ej: 0, 1, 3.14)
            return current_index

        # Caso Recursivo: Es un nodo de División (Split)
        # 1. Reservar espacio para el nodo actual (pre-order)
        feature_index.append(node['index'])
        threshold.append(node['value']) # <-- 'value' aquí es el umbral
        value.append(0.0)               # Valor dummy (solo las hojas tienen valor)
        
        # Índices de hijos (se llenarán después de la recursión)
        left_child.append(-1)
        right_child.append(-1)

        # 2. Recorrer hijos
        left_idx = _traverse(node['left'])
        right_idx = _traverse(node['right'])

        # 3. Actualizar los punteros de índice de este nodo
        left_child[current_index] = left_idx
        right_child[current_index] = right_idx

        return current_index

    # Iniciar el aplanado desde el nodo raíz
    _traverse(root_node)

    return {
        "feature_index": feature_index,
        "threshold": threshold,
        "left_child": left_child,
        "right_child": right_child,
        "value": value
    }

def _unflatten_arrays_to_tree(tree_struct: Dict[str, List]) -> Dict[str, Any]:
    """
    Reconstruye un árbol de decisión (dict anidado) a partir de los
    arrays paralelos.
    
    Esta es la función inversa de _flatten_tree_to_arrays, necesaria
    para la deserialización en Python.
    """
    # Extraer los arrays
    feature_index = tree_struct['feature_index']
    threshold = tree_struct['threshold']
    left_child = tree_struct['left_child']
    right_child = tree_struct['right_child']
    value = tree_struct['value']

    def _build_node(index: int) -> Dict[str, Any]:
        """
        Función interna recursiva.
        Construye el nodo (y sus sub-árboles) para el índice dado.
        """
        # Caso Base: Es un nodo Hoja
        if feature_index[index] == -1:
            # CORRECCIÓN
            # Devuelve el valor de la hoja directamente, no un dict
            return value[index]

        # Caso Recursivo: Es un nodo de División
        # Construir recursivamente los hijos
        left_node = _build_node(left_child[index])
        right_node = _build_node(right_child[index])

        # Construir el nodo actual
        return {
            'index': feature_index[index],
            'value': threshold[index], # <-- 'value' es el umbral
            'left': left_node,
            'right': right_node
        }

    # Iniciar la reconstrucción desde el nodo raíz (índice 0)
    return _build_node(0)

def _flatten_tree_internal(root: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Aplana un árbol de decisión recursivo a arrays lineales (Structure of Arrays).
    Usa BFS (Breadth-First Search) para generar índices secuenciales.
    Output compatible con iteración while() en C.
    """
    if not root:
        return {k: [] for k in ['feature_index', 'threshold', 'left_child', 'right_child', 'value']}

    # Colas para BFS: (nodo_dict, indice_asignado_en_array)
    queue = [(root, 0)]
    
    # Estructura temporal mapeada por índice para llenado desordenado si fuera necesario
    # (aunque BFS llena en orden, esto es seguro)
    flat_map = {}
    
    next_idx = 1 # Siguiente índice disponible para hijos
    
    while queue:
        node, current_idx = queue.pop(0)
        
        # Detectar si es nodo hoja
        is_leaf = False
        # Caso 1: El nodo es directamente un valor (terminal simplificado)
        if not isinstance(node, dict):
            val = node
            is_leaf = True
        # Caso 2: El nodo es dict pero no tiene índice de split o apunta a terminales directos mal formados
        elif node.get('index') is None:
            # Intentar sacar el valor de 'left' si existe, o del propio nodo
            val = node.get('left') if 'left' in node else node.get('value')
            is_leaf = True
        
        if is_leaf:
            # Nodo Hoja: Feature -1 indica fin del while en C
            flat_map[current_idx] = {
                'f': -1, 
                't': 0.0, 
                'l': -1, 
                'r': -1, 
                'v': val
            }
            continue

        # Nodo Interno
        left_idx = next_idx
        right_idx = next_idx + 1
        next_idx += 2
        
        flat_map[current_idx] = {
            'f': node['index'],
            't': float(node['value']), # Asegurar float para C
            'l': left_idx,
            'r': right_idx,
            'v': 0 # Valor dummy en nodos internos
        }
        
        # Encolar hijos
        # Nota: get('left') puede devolver un valor terminal o un dict
        queue.append((node.get('left'), left_idx))
        queue.append((node.get('right'), right_idx))

    # Convertir mapa a arrays ordenados
    max_idx = max(flat_map.keys())
    feature_index = []
    threshold = []
    left_child = []
    right_child = []
    value = []

    for i in range(max_idx + 1):
        if i in flat_map:
            d = flat_map[i]
            feature_index.append(d['f'])
            threshold.append(d['t'])
            left_child.append(d['l'])
            right_child.append(d['r'])
            value.append(d['v'])
        else:
            # Nodo inalcanzable (seguridad)
            feature_index.append(-2)
            threshold.append(0.0)
            left_child.append(-1)
            right_child.append(-1)
            value.append(0)

    return {
        'feature_index': feature_index,
        'threshold': threshold,
        'left_child': left_child,
        'right_child': right_child,
        'value': value
    }
