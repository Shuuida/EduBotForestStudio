"""
ml_compat.py
------------
Módulo de compatibilidad y utilidades globales para MiniML Framework.

ACTUALIZADO:
- Soporte para 'Scalar Leaves' (Hojas como valores crudos) en aplanado de árboles.
- Funciones de imputación y chequeo de dimensiones.
- Sin dependencias externas.
"""

from typing import List, Any, Dict, Union

# -----------------------------------
#  Funciones básicas de tipo

def _is_number(x):
    """Verifica si x es un número (int o float)."""
    return isinstance(x, (int, float))

def to_float_if_possible(x):
    """Convierte a float si es posible. Si no, devuelve None."""
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
    """
    f = to_float_if_possible(value)
    if f is not None:
        return f
    
    if isinstance(value, dict):
        # Intentar claves comunes
        for k in ['value', 'score', 'threshold', 'val', 'acc', 'accuracy']:
            if k in value:
                return to_float_if_possible(value[k])
        # Si tiene un solo valor, devolverlo
        if len(value) == 1:
            return to_float_if_possible(list(value.values())[0])
            
    return 0.0 # Fallback seguro

def safe_compare_le(val_a, val_b):
    """
    Comparación segura menor o igual (<=).
    Maneja tipos mixtos resolviendo a float.
    """
    a = resolve_numeric(val_a)
    b = resolve_numeric(val_b)
    return a <= b

# -------------------------------------------
#  Aplanado de Árboles (Exportación a C)

def _flatten_tree_to_arrays(node: Union[Dict, int, float]) -> Dict[str, List]:
    """
    Convierte un árbol recursivo (dict/escalar) en arrays paralelos para C (PROGMEM).
    Soporta:
     - Formato Legacy: Hojas son dicts {'index': -1, 'value': ...}
     - Formato v2.3: Hojas son escalares (int/float)
    """
    feature_index = []
    threshold = []
    left_child = []
    right_child = []
    value = []
    
    # Cola para BFS (Breadth-First Search) para índices deterministas
    # Guardamos (nodo_objeto, indice_padre, es_hijo_izq)
    # Pero para generar arrays planos indexados, es mejor un recorrido lineal
    # y asignar índices dinámicamente.
    
    # Estrategia: Recorrido recursivo primero para asignar IDs, luego llenar arrays.
    # O más simple: Usar una lista y un cursor.
    
    node_list = [node] # Lista de nodos pendientes de procesar
    processed_nodes = [] # Lista final en orden
    
    # Linearizar el árbol (BFS)
    cursor = 0
    while cursor < len(node_list):
        curr = node_list[cursor]
        cursor += 1
        
        # Detectar si es hoja
        is_leaf = False
        if not isinstance(curr, dict):
            is_leaf = True
        elif curr.get('index') == -1 or 'left' not in curr:
            is_leaf = True
            
        if not is_leaf:
            node_list.append(curr['left'])
            node_list.append(curr['right'])
            
    # Construir arrays
    # Ahora node_list tiene todos los nodos en orden BFS.
    # El nodo i no tiene hijos en 2*i+1, esto no es un heap completo.
    # Necesitamos recalcular los índices de los hijos basados en su posición en node_list.
    
    # Re-hacemos el paso 1 pero guardando referencias de índices
    # Estructura de la cola: (nodo, indice_en_arrays)
    queue = [node]
    
    # Mapeo de identidad no sirve porque los escalares no son únicos.
    # Necesitamos reconstruir recursivamente y luego aplanar.
    
    # Enfoque: Aplanado recursivo con punteros globales
    feature_index = []
    threshold = []
    left_child = []
    right_child = []
    value = []
    
    next_free_index = 0
    
    def register_node(n):
        nonlocal next_free_index
        idx = next_free_index
        next_free_index += 1
        return idx

    # Recorrido BFS con cola que guarda (nodo, index_asignado)
    q = [(node, register_node(node))]
    
    # Los arrays se llenarán en desorden, necesitamos pre-llenarlos o usar dicts
    temp_storage = {} # index -> data
    
    head = 0
    while head < len(q):
        curr_node, curr_idx = q[head]
        head += 1
        
        # Analizar nodo
        is_leaf = False
        val = 0.0
        
        if not isinstance(curr_node, dict):
            is_leaf = True
            val = float(curr_node)
        elif curr_node.get('index') == -1 or 'left' not in curr_node:
            is_leaf = True
            val = float(curr_node.get('value', 0.0))
            
        if is_leaf:
            temp_storage[curr_idx] = {
                'f': -1, 't': 0.0, 'l': -1, 'r': -1, 'v': val
            }
        else:
            # Es nodo de decisión
            # Registramos hijos
            l_idx = register_node(curr_node['left'])
            r_idx = register_node(curr_node['right'])
            
            # Encolamos hijos
            q.append((curr_node['left'], l_idx))
            q.append((curr_node['right'], r_idx))
            
            temp_storage[curr_idx] = {
                'f': int(curr_node['index']),
                't': float(curr_node['value']),
                'l': l_idx,
                'r': r_idx,
                'v': 0.0
            }
            
    # Convertir temp_storage a arrays ordenados
    # Como usamos register_node secuencialmente, los índices van de 0 a N-1
    count = len(temp_storage)
    for i in range(count):
        data = temp_storage[i]
        feature_index.append(data['f'])
        threshold.append(data['t'])
        left_child.append(data['l'])
        right_child.append(data['r'])
        value.append(data['v'])
        
    return {
        "feature_index": feature_index,
        "threshold": threshold,
        "left_child": left_child,
        "right_child": right_child,
        "value": value
    }

def _unflatten_arrays_to_tree(tree_struct: Dict[str, List]) -> Dict[str, Any]:
    """Reconstruye un árbol a partir de arrays (para carga desde JSON)."""
    f_idx = tree_struct['feature_index']
    thresh = tree_struct['threshold']
    l_child = tree_struct['left_child']
    r_child = tree_struct['right_child']
    vals = tree_struct['value']

    def build(i):
        if f_idx[i] == -1:
            # Hoja: Devolver dict para compatibilidad con visualizador,
            # o valor si se prefiere. El visualizador prefiere dicts.
            return {'index': -1, 'value': vals[i]}
        
        return {
            'index': f_idx[i],
            'value': thresh[i],
            'left': build(l_child[i]),
            'right': build(r_child[i])
        }
    
    if not f_idx: return None
    return build(0)

# -------------------------------------------
#  Utilidades de Datos

def check_dims(X, expected_dim):
    """Verifica dimensiones de entrada."""
    if not X: return False
    if len(X[0]) != expected_dim:
        return False
    return True

def impute_missing_values(dataset: List[List[Any]], strategy: str = 'mean') -> List[List[float]]:
    """
    Rellena valores None/NaN.
    
    Args:
        dataset: Lista de listas con datos crudos.
        strategy: 'mean' (única soportada actualmente, mantenida por compatibilidad de API).
        
    Returns:
        Nuevo dataset con floats limpios.
    """
    if not dataset: return []
    
    n_cols = len(dataset[0])
    # n_rows = len(dataset) # Unused
    col_sums = [0.0] * n_cols
    col_counts = [0] * n_cols
    
    # Calcular medias (ignorando Nones)
    for row in dataset:
        for i in range(n_cols):
            val = to_float_if_possible(row[i])
            if val is not None:
                col_sums[i] += val
                col_counts[i] += 1
    
    means = [ (s/c if c>0 else 0.0) for s,c in zip(col_sums, col_counts) ]
    
    # Reemplazar
    clean_data = []
    for row in dataset:
        new_row = []
        for i in range(n_cols):
            val = to_float_if_possible(row[i])
            new_row.append(val if val is not None else means[i])
        clean_data.append(new_row)
        
    return clean_data