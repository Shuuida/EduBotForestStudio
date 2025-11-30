"""
MiniML Core for EduBot

Objetivos:
 - MiniMatrixOps: operaciones de matriz/vectores ligeras (sin numpy)
 - DecisionTreeClassifier / DecisionTreeRegressor (CART-style) robustos
 - RandomForestClassifier / RandomForestRegressor
 - MiniLinearModel (Mini)
 - MiniSVM (lineal, perceptron-like / hinge update)
 - MiniNeuralNetwork: MLP con backprop que admite múltiples salidas
 - Utilities: accuracy_score, mse, mae, r2_score
 - Export helpers: to_arduino_code para generar código embebible
 - Sin dependencias externas (diseñado para firmware y prototipos)
"""

from __future__ import annotations
from typing import List, Any, Optional, Tuple, Union, Dict
import random
import math
from core.ml_compat import safe_compare_le, _flatten_tree_to_arrays

# ---------------------------
# MiniMatrixOps (sin numpy)
# ---------------------------
class MiniMatrixOps:
    """Operaciones básicas de vectores/matrices sin dependencias externas."""

    @staticmethod
    def dot(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            raise ValueError("Vectors must have same length for dot product")
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    @staticmethod
    def matvec(mat: List[List[float]], vec: List[float]) -> List[float]:
        return [MiniMatrixOps.dot(row, vec) for row in mat]

    @staticmethod
    def transpose(mat: List[List[float]]) -> List[List[float]]:
        if not mat:
            return []
        rows = len(mat)
        cols = len(mat[0])
        return [[mat[r][c] for r in range(rows)] for c in range(cols)]

    @staticmethod
    def outer(a: List[float], b: List[float]) -> List[List[float]]:
        return [[ai * bj for bj in b] for ai in a]

    @staticmethod
    def add_vec(a: List[float], b: List[float]) -> List[float]:
        if len(a) != len(b):
            raise ValueError("Vector length mismatch")
        return [a[i] + b[i] for i in range(len(a))]

    @staticmethod
    def scalar_mul_vec(s: float, v: List[float]) -> List[float]:
        return [s * x for x in v]

    @staticmethod
    def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Multiplicación de matrices A (m x n) * B (n x p) -> (m x p)."""
        if not A or not B:
            return []
        m = len(A)
        n = len(A[0])
        if any(len(row) != n for row in A):
            raise ValueError("Invalid matrix A")
        if any(len(row) != len(B[0]) for row in B):
            pass  # permitir filas B irregulares? validaremos correctamente a continuación
        p = len(B[0])
        # Validar forma B
        if any(len(row) != len(B[0]) for row in B):
            raise ValueError("Invalid matrix B")
        if len(B) != n:
            raise ValueError("Incompatible dimensions for matrix multiply")
        # compute
        BT = MiniMatrixOps.transpose(B)
        result = []
        for i in range(m):
            row_res = []
            for j in range(p):
                row_res.append(MiniMatrixOps.dot(A[i], BT[j]))
            result.append(row_res)
        return result

# ---------------------------
# Utilities
# ---------------------------
# Activation Utilities (global para MiniML)

def clip(value: float, min_val: float = -60.0, max_val: float = 60.0) -> float:
    """Protege activaciones (sigmoid overflow)."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value

def sigmoid(x: float) -> float:
    x = clip(x)
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0
    except Exception:
        return 0.5  # fallback seguro

def sigmoid_derivative(output: float) -> float:
    return output * (1.0 - output)

def relu(x: float) -> float:
    return x if x > 0 else 0.0

def relu_derivative(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def linear(x: float) -> float:
    return x

def linear_derivative(_: float) -> float:
    return 1.0

# ---------------------------
# Decision tree helpers (CART)
# ---------------------------
def split_dataset(dataset: List[List[Any]], feature_index: int, value: Any) -> Tuple[List[List[Any]], List[List[Any]]]:
    """
    Divide un conjunto de datos en dos grupos (izquierda y derecha) basándose en el valor de una característica.
    Esta función es robusta contra tipos de datos no numéricos para evitar errores de tipo en tiempo de ejecución.
    """
    left, right = [], []
    
    # Determinar si el valor de división es un número válido.
    is_value_numeric = isinstance(value, (int, float))

    for row in dataset:
        try:
            # Si el índice está fuera de rango, la fila no se puede dividir.
            if feature_index >= len(row):
                right.append(row)
                continue

            row_feature = row[feature_index]
            is_row_feature_numeric = isinstance(row_feature, (int, float))

            # Solo se puede realizar una comparación numérica si ambos valores son números.
            if is_value_numeric and is_row_feature_numeric:
                if row_feature <= value:
                    left.append(row)
                else:
                    right.append(row)
            # Si el valor de división o el de la fila no son numéricos, la comparación es inválida. 
            # Se asigna la fila a la derecha por defecto. Esto asegura que la división
            # no será "pura" y el algoritmo la descartará por tener una puntuación de impureza alta.
            else:
                right.append(row)
        
        except Exception:
            # Ante cualquier otro error inesperado, se asigna la fila a la derecha como medida de seguridad.
            right.append(row)
            
    return left, right

def gini_index(groups: Union[List[List[List[Any]]], Tuple[List[List[Any]], List[List[Any]]]], classes: List[Any]) -> float:
    n_instances = sum(len(g) for g in groups)
    if n_instances == 0:
        return 0.0
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = sum(1 for row in group if row[-1] == class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def mse_index(groups: Union[List[List[List[Any]]], Tuple[List[List[Any]], List[List[Any]]]]) -> float:
    n_instances = sum(len(g) for g in groups)
    if n_instances == 0:
        return 0.0
    mse = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        mean = sum(row[-1] for row in group) / size
        sq_error = sum((row[-1] - mean) ** 2 for row in group)
        mse += sq_error * (size / n_instances)
    return mse

def to_terminal_class(group: List[List[Any]]):
    outcomes = {}
    for row in group:
        label = row[-1]
        outcomes[label] = outcomes.get(label, 0) + 1
    # CORRECCIÓN
    if not outcomes:
        # Un grupo vacío no puede tener una clase 'max'.
        # Devolver None rompe la validación (un nodo hoja no puede ser None).
        # Devolvemos '0' (int) como un valor terminal de fallback seguro.
        # Esto es análogo a to_terminal_reg que devuelve 0.0.
        return 0
    return max(outcomes.items(), key=lambda x: x[1])[0]

def to_terminal_reg(group: List[List[Any]]):
    if not group:
        return 0.0
    vals = [row[-1] for row in group]
    return sum(vals) / len(vals)

def get_split_class(dataset: List[List[Any]], n_features: Optional[int] = None):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = None, None, float('inf'), None
    
    if not dataset or not dataset[0]:
        return {'index': b_index, 'value': b_value, 'groups': b_groups}
        
    features = list(range(len(dataset[0]) - 1))
    if n_features is not None and n_features > 0:
        features = random.sample(features, max(1, min(len(features), n_features)))
        
    for index in features:
        values = set(row[index] for row in dataset)
        
        # Filtrar valores para asegurar que solo los números se usen para las divisiones.
        # Esto previene que un 'dict' o 'str' sea usado como umbral numérico.
        numeric_values = {v for v in values if isinstance(v, (int, float))}
        
        for value in numeric_values:
            groups = split_dataset(dataset, index, value)
            gini = gini_index(groups, class_values)
            
            # Esta comparación ahora es segura, porque 'gini' y 'b_score' son floats garantizados.
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, value, gini, groups
                
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def get_split_regression(dataset: List[List[Any]], n_features: Optional[int] = None):
    b_index, b_value, b_score, b_groups = None, None, float('inf'), None

    if not dataset or not dataset[0]:
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    features = list(range(len(dataset[0]) - 1))
    if n_features is not None and n_features > 0:
        features = random.sample(features, max(1, min(len(features), n_features)))

    for index in features:
        values = set(row[index] for row in dataset)
        
        # Aplicar el mismo filtro para la regresión.
        numeric_values = {v for v in values if isinstance(v, (int, float))}
        
        for value in numeric_values:
            groups = split_dataset(dataset, index, value)
            score = mse_index(groups)

            # Esta comparación también es segura ahora.
            if score < b_score:
                b_index, b_value, b_score, b_groups = index, value, score, groups
                
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def build_tree_class(node: Dict[str, Any], max_depth: int, min_size: int, n_features: Optional[int]):
    groups = node.get('groups')
    if not groups or not isinstance(groups, (list, tuple)) or len(groups) != 2:
        # convertir a terminal
        node['left'] = to_terminal_class(node.get('groups') or [])
        node['right'] = node['left']
        node.pop('groups', None)
        return
    left, right = groups
    # terminal condition
    if not left or not right or max_depth <= 0:
        node['left'] = to_terminal_class(left)
        node['right'] = to_terminal_class(right)
        node.pop('groups', None)
        return
    # construcción recursiva
    left_child = get_split_class(left, n_features)
    node['left'] = left_child
    build_tree_class(left_child, max_depth - 1, min_size, n_features)
    right_child = get_split_class(right, n_features)
    node['right'] = right_child
    build_tree_class(right_child, max_depth - 1, min_size, n_features)
    node.pop('groups', None)

def build_tree_reg(node: Dict[str, Any], max_depth: int, min_size: int, n_features: Optional[int]):
    groups = node.get('groups')
    if not groups or not isinstance(groups, (list, tuple)) or len(groups) != 2:
        node['left'] = to_terminal_reg(node.get('groups') or [])
        node['right'] = node['left']
        node.pop('groups', None)
        return
    left, right = groups
    if not left or not right or max_depth <= 0:
        node['left'] = to_terminal_reg(left)
        node['right'] = to_terminal_reg(right)
        node.pop('groups', None)
        return
    left_child = get_split_regression(left, n_features)
    node['left'] = left_child
    build_tree_reg(left_child, max_depth - 1, min_size, n_features)
    right_child = get_split_regression(right, n_features)
    node['right'] = right_child
    build_tree_reg(right_child, max_depth - 1, min_size, n_features)
    node.pop('groups', None)

# ---------------------------
# Classifiers / Regressors
# ---------------------------
class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 10, min_size: int = 1, n_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.root: Optional[Dict[str, Any]] = None

    def fit(self, dataset: List[List[Any]]):
        if not dataset or not isinstance(dataset, list):
            raise ValueError("Dataset invalid for fit()")
        root = get_split_class(dataset, self.n_features)
        # si split no pudo encontrar grupos -> terminal
        if not root.get('groups'):
            self.root = {'index': None, 'value': None, 'left': to_terminal_class(dataset), 'right': to_terminal_class(dataset)}
            return
        # construir árbol completo
        build_tree_class(root, self.max_depth, self.min_size, self.n_features)
        self.root = root

    def _predict_row(self, node, row):
        # Si el nodo actual no es un diccionario, es un valor terminal (una predicción).
        if not isinstance(node, dict):
            return node

        index = node.get('index')
        value = node.get('value')

        # Si el nodo no tiene un índice de característica, es un nodo terminal.
        # Se devuelve el valor de la rama 'left' como predicción final.
        if index is None:
            # La predicción es el valor terminal, que no debería ser un dict.
            if 'left' in node and not isinstance(node['left'], dict):
                return node['left']
            # Fallback por si la estructura del nodo es inesperada.
            return node

        # El valor de un nodo ('value') puede ser a su vez un sub-árbol (dict). 
        # No podemos comparar un float con un dict.
        # Si 'value' es un diccionario, significa que debemos seguir recorriendo el árbol
        # a través de este sub-nodo.
        if isinstance(value, dict):
            return self._predict_row(value, row)

        # Si el valor de la característica en la fila actual no existe o no es un número,
        # no se puede hacer una comparación válida. Se sigue por la derecha como fallback.
        try:
            row_feature = row[index]
            if not isinstance(row_feature, (int, float)):
                return self._predict_row(node.get('right'), row)
        except (IndexError, KeyError):
            return self._predict_row(node.get('right'), row)

        # Ahora que sabemos que ambos son números, la comparación es segura.
        if safe_compare_le(row_feature, value):
            return self._predict_row(node.get('left'), row)
        else:
            return self._predict_row(node.get('right'), row)

    def predict(self, X: List[List[Any]]) -> List[Any]:
        if self.root is None:
            raise ValueError("Model not trained")
        preds = []
        for row in X:
            preds.append(self._predict_row(self.root, row))
        return preds

    # Función `to_arduino_code` corregido para evitar anidados de if/else que causen un desbordamiento de pila (Stack Overflow)
    def to_arduino_code(self, fn_name: str = "predict_row") -> str:
        """
        Genera código C iterativo (seguro para stack) usando arrays aplanados.
        """
        if self.root is None:
            return "// Error: El modelo no está entrenado (root is None)."
        
        try:
            flat_tree = _flatten_tree_to_arrays(self.root)
        except Exception as e:
            return f"// Error durante el aplanamiento del árbol: {e}"

        # Helper para formatear arrays de C
        def format_c_array(name, arr, dtype="int"):
            if not arr:
                return f"{dtype} {name}[1] = {{0}};" # Evitar arrays vacíos
            values = ", ".join(map(str, arr))
            return f"{dtype} {name}[{len(arr)}] = {{{values}}};"

        # Generar arrays C
        code = [
            "// MiniML: Decision Tree Classifier (Iterative C Export)",
            "// Exportación segura con uso de pila constante (O(1))",
            format_c_array("tree_feature_index", flat_tree['feature_index'], "int"),
            format_c_array("tree_threshold", flat_tree['threshold'], "float"),
            format_c_array("tree_left_child", flat_tree['left_child'], "int"),
            format_c_array("tree_right_child", flat_tree['right_child'], "int"),
            format_c_array("tree_value", flat_tree['value'], "int") + " // Predicciones (clases)",
            ""
        ]
        
        # Generar función C iterativa
        func = [
            f"int {fn_name}(float row[]) {{",
            "  int node_index = 0; // Iniciar en el nodo raíz (índice 0)",
            "",
            "  // Recorrer el árbol mientras no estemos en un nodo hoja",
            "  // Un nodo hoja se marca con feature_index == -1",
            "  while (tree_feature_index[node_index] != -1) {",
            "    if (row[tree_feature_index[node_index]] <= tree_threshold[node_index]) {",
            "      node_index = tree_left_child[node_index];",
            "    } else {",
            "      node_index = tree_right_child[node_index];",
            "    }",
            "  }",
            "",
            "  // Salimos del bucle, node_index apunta a una hoja.",
            "  // Devolver el valor (clase) de esa hoja.",
            "  return tree_value[node_index];",
            "}"
        ]
        code.extend(func)
        return "\n".join(code)

class DecisionTreeRegressor(DecisionTreeClassifier):
    def fit(self, dataset: List[List[Any]]):
        if not dataset or not isinstance(dataset, list):
            raise ValueError("Dataset invalid for fit()")
        root = get_split_regression(dataset, self.n_features)
        if not root.get('groups'):
            self.root = {'index': None, 'value': None, 'left': to_terminal_reg(dataset), 'right': to_terminal_reg(dataset)}
            return
        build_tree_reg(root, self.max_depth, self.min_size, self.n_features)
        self.root = root

    def predict(self, X: List[List[Any]]) -> List[float]:
        if self.root is None:
            raise ValueError("Model not trained")
        preds = []
        for row in X:
            p = self._predict_row(self.root, row)
            preds.append(float(p))
        return preds

    # Override de `to_arduino_code` para Regresor
    def to_arduino_code(self, fn_name: str = "predict_row") -> str:
        """
        Genera código C iterativo (seguro para stack) para regresión (devuelve float).
        """
        if self.root is None:
            return "// Error: El modelo no está entrenado (root is None)."
        
        try:
            flat_tree = _flatten_tree_to_arrays(self.root)
        except Exception as e:
            return f"// Error durante el aplanamiento del árbol: {e}"

        # Helper para formatear arrays de C
        def format_c_array(name, arr, dtype="int"):
            if not arr:
                return f"{dtype} {name}[1] = {{0}};"
            values = ", ".join(map(str, arr))
            return f"{dtype} {name}[{len(arr)}] = {{{values}}};"

        # Generar arrays C
        code = [
            "// MiniML: Decision Tree Regressor (Iterative C Export)",
            "// Exportación segura con uso de pila constante (O(1))",
            format_c_array("tree_feature_index", flat_tree['feature_index'], "int"),
            format_c_array("tree_threshold", flat_tree['threshold'], "float"),
            format_c_array("tree_left_child", flat_tree['left_child'], "int"),
            format_c_array("tree_right_child", flat_tree['right_child'], "int"),
            format_c_array("tree_value", flat_tree['value'], "float") + " // Predicciones (valores)",
            ""
        ]
        
        # Generar función C iterativa
        func = [
            f"float {fn_name}(float row[]) {{", # <-- Devuelve float
            "  int node_index = 0;",
            "  while (tree_feature_index[node_index] != -1) {",
            "    if (row[tree_feature_index[node_index]] <= tree_threshold[node_index]) {",
            "      node_index = tree_left_child[node_index];",
            "    } else {",
            "      node_index = tree_right_child[node_index];",
            "    }",
            "  }",
            "  return tree_value[node_index];",
            "}"
        ]
        code.extend(func)
        return "\n".join(code)

# ---------------------------
# Random Forest
# ---------------------------
class RandomForestClassifier:
    def __init__(self, n_trees: int = 5, max_depth: int = 10, min_size: int = 1,
                 sample_size: float = 1.0, n_features: Optional[int] = None, seed: Optional[int] = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_features = n_features
        self.seed = seed
        self.trees: List[DecisionTreeClassifier] = []

    def _subsample(self, dataset):
        n_sample = max(1, int(len(dataset) * self.sample_size))
        return [random.choice(dataset) for _ in range(n_sample)]

    def fit(self, dataset: List[List[Any]]):
        self.trees = []
        random.seed(self.seed)
        for i in range(self.n_trees):
            sample = self._subsample(dataset)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_size=self.min_size, n_features=self.n_features)
            tree.fit(sample)
            self.trees.append(tree)

    def predict(self, X: List[List[Any]]) -> List[Any]:
        if not self.trees:
            raise ValueError("Not trained")
        votes = []
        for row in X:
            row_votes = [t._predict_row(t.root, row) for t in self.trees]
            agg = {}
            for v in row_votes:
                agg[v] = agg.get(v, 0) + 1
            votes.append(max(agg.items(), key=lambda x: x[1])[0])
        return votes

    # Implementación de `to_arduino_code` para RandomForestClassifier
    def to_arduino_code(self, fn_name: str = "predict_row") -> str:
        """
        Genera código C iterativo (seguro para stack) para el bosque aleatorio completo.
        """
        if not self.trees:
            return "// Error: El modelo no está entrenado (no hay árboles)."

        # Helper para formatear arrays de C
        def format_c_array(name, arr, dtype="int"):
            if not arr:
                return f"{dtype} {name}[1] = {{0}};"
            values = ", ".join(map(str, arr))
            return f"{dtype} {name}[{len(arr)}] = {{{values}}};"

        code = [
            "// MiniML: Random Forest Classifier (Iterative C Export)",
            f"// {self.n_trees} árboles, exportación segura con uso de pila constante (O(1))",
            ""
        ]

        # 1. Generar los datos y funciones para CADA árbol
        tree_fn_names = []
        for i, tree in enumerate(self.trees):
            if tree.root is None:
                code.append(f"// Árbol {i} no está entrenado, se omite.")
                continue
            
            try:
                flat_tree = _flatten_tree_to_arrays(tree.root)
            except Exception as e:
                code.append(f"// Error aplanando árbol {i}: {e}")
                continue

            tree_name = f"tree{i}"
            tree_fn_name = f"predict_{tree_name}"
            tree_fn_names.append(tree_fn_name)

            code.append(f"// --- Datos del Árbol {i} ---")
            code.append(format_c_array(f"{tree_name}_feature_index", flat_tree['feature_index'], "int"))
            code.append(format_c_array(f"{tree_name}_threshold", flat_tree['threshold'], "float"))
            code.append(format_c_array(f"{tree_name}_left_child", flat_tree['left_child'], "int"))
            code.append(format_c_array(f"{tree_name}_right_child", flat_tree['right_child'], "int"))
            code.append(format_c_array(f"{tree_name}_value", flat_tree['value'], "int"))
            code.append("")

            func = [
                f"int {tree_fn_name}(float row[]) {{",
                "  int node_index = 0;",
                f"  while ({tree_name}_feature_index[node_index] != -1) {{",
                f"    if (row[{tree_name}_feature_index[node_index]] <= {tree_name}_threshold[node_index]) {{",
                f"      node_index = {tree_name}_left_child[node_index];",
                "    } else {",
                f"      node_index = {tree_name}_right_child[node_index];",
                "    }",
                "  }",
                f"  return {tree_name}_value[node_index];",
                "}",
                ""
            ]
            code.extend(func)

        # 2. Generar la función de Voto Mayoritario
        # (Implementación simple O(N^2), segura para N pequeño como n_trees)
        vote_helper = [
            "// Función auxiliar de voto mayoritario",
            f"int majority_vote(int votes[], int num_votes) {{",
            "  if (num_votes == 0) return 0;",
            "  int max_count = 0;",
            "  int max_vote = votes[0];",
            "  for (int i = 0; i < num_votes; i++) {",
            "    int current_count = 0;",
            "    for (int j = 0; j < num_votes; j++) {",
            "      if (votes[j] == votes[i]) {",
            "        current_count++;",
            "      }",
            "    }",
            "    if (current_count > max_count) {",
            "      max_count = current_count;",
            "      max_vote = votes[i];",
            "    }",
            "  }",
            "  return max_vote;",
            "}",
            ""
        ]
        code.extend(vote_helper)

        # 3. Generar la función principal
        main_func = [
            f"int {fn_name}(float row[]) {{",
            f"  int votes[{len(tree_fn_names)}];",
        ]
        # Recoger votos
        for i, fn in enumerate(tree_fn_names):
            main_func.append(f"  votes[{i}] = {fn}(row);")
        
        main_func.extend([
            f"  return majority_vote(votes, {len(tree_fn_names)});",
            "}"
        ])
        code.extend(main_func)
        return "\n".join(code)


class RandomForestRegressor(RandomForestClassifier):
    def fit(self, dataset: List[List[Any]]):
        self.trees = []
        random.seed(self.seed)
        for i in range(self.n_trees):
            sample = self._subsample(dataset)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_size=self.min_size, n_features=self.n_features)
            tree.fit(sample)
            self.trees.append(tree)

    def predict(self, X: List[List[Any]]) -> List[float]:
        if not self.trees:
            raise ValueError("Not trained")
        preds = []
        for row in X:
            row_preds = [t._predict_row(t.root, row) for t in self.trees]
            avg = sum(float(p) for p in row_preds) / len(row_preds)
            preds.append(avg)
        return preds

    # Implementación de `to_arduino_code` para RandomForestRegressor
    def to_arduino_code(self, fn_name: str = "predict_row") -> str:
        """
        Genera código C iterativo (seguro para stack) para el bosque aleatorio completo.
        """
        # Import local
        if not self.trees:
            return "// Error: El modelo no está entrenado (no hay árboles)."

        # Helper para formatear arrays de C
        def format_c_array(name, arr, dtype="int"):
            if not arr:
                return f"{dtype} {name}[1] = {{0}};"
            values = ", ".join(map(str, arr))
            return f"{dtype} {name}[{len(arr)}] = {{{values}}};"

        code = [
            "// MiniML: Random Forest Regressor (Iterative C Export)",
            f"// {self.n_trees} árboles, exportación segura con uso de pila constante (O(1))",
            ""
        ]

        # 1. Generar los datos y funciones para CADA árbol
        tree_fn_names = []
        for i, tree in enumerate(self.trees):
            if tree.root is None:
                code.append(f"// Árbol {i} no está entrenado, se omite.")
                continue
            
            try:
                flat_tree = _flatten_tree_to_arrays(tree.root)
            except Exception as e:
                code.append(f"// Error aplanando árbol {i}: {e}")
                continue

            tree_name = f"tree{i}"
            tree_fn_name = f"predict_{tree_name}"
            tree_fn_names.append(tree_fn_name)

            code.append(f"// --- Datos del Árbol {i} ---")
            code.append(format_c_array(f"{tree_name}_feature_index", flat_tree['feature_index'], "int"))
            code.append(format_c_array(f"{tree_name}_threshold", flat_tree['threshold'], "float"))
            code.append(format_c_array(f"{tree_name}_left_child", flat_tree['left_child'], "int"))
            code.append(format_c_array(f"{tree_name}_right_child", flat_tree['right_child'], "int"))
            code.append(format_c_array(f"{tree_name}_value", flat_tree['value'], "float")) # <-- float
            code.append("")

            func = [
                f"float {tree_fn_name}(float row[]) {{", # <-- float
                "  int node_index = 0;",
                f"  while ({tree_name}_feature_index[node_index] != -1) {{",
                f"    if (row[{tree_name}_feature_index[node_index]] <= {tree_name}_threshold[node_index]) {{",
                f"      node_index = {tree_name}_left_child[node_index];",
                "    } else {",
                f"      node_index = {tree_name}_right_child[node_index];",
                "    }",
                "  }",
                f"  return {tree_name}_value[node_index];",
                "}",
                ""
            ]
            code.extend(func)

        # 2. Generar la función principal (Promedio)
        main_func = [
            f"float {fn_name}(float row[]) {{", # <-- float
            f"  float predictions[{len(tree_fn_names)}];",
            "  float sum = 0.0;",
        ]
        # Recoger predicciones
        for i, fn in enumerate(tree_fn_names):
            main_func.append(f"  predictions[{i}] = {fn}(row);")
            main_func.append(f"  sum += predictions[{i}];")
        
        main_func.extend([
            f"  return sum / {len(tree_fn_names)}.0;",
            "}"
        ])
        code.extend(main_func)
        return "\n".join(code)

# ---------------------------
# Mini Linear Model
# ---------------------------
class MiniLinearModel:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.weights = None  # contendrá [w0, w1, ..., wn] donde w0..wn-1 features, wn = bias

    def _unpack(self, dataset):
        X = [row[:-1] for row in dataset]
        y = [row[-1] for row in dataset]
        return X, y

    def fit(self, dataset):
        X, y = self._unpack(dataset)
        if not X:
            raise ValueError("Empty dataset")
        n_samples = len(X)
        n_features = len(X[0])
        # weights iniciales + bias
        self.weights = [0.0] * n_features + [0.0]  # el último elemento es bias
        for epoch in range(self.epochs):
            grads = [0.0] * (n_features + 1)
            for xi, yi in zip(X, y):
                pred = sum(w * xv for w, xv in zip(self.weights[:-1], xi)) + self.weights[-1]
                err = pred - yi
                for j in range(n_features):
                    grads[j] += (2.0 / n_samples) * err * xi[j]
                grads[-1] += (2.0 / n_samples) * err  # bias grad
            # update
            for j in range(n_features + 1):
                self.weights[j] -= self.learning_rate * grads[j]

    def predict(self, X_list):
        if self.weights is None:
            raise ValueError("Model not trained")
        preds = []
        for xi in X_list:
            # xi puede ser una lista o un escalar -> normalizar
            if not isinstance(xi, (list, tuple)):
                xi = [xi]
            pred = sum(w * xv for w, xv in zip(self.weights[:-1], xi)) + self.weights[-1]
            preds.append(pred)
        return preds

    def to_arduino_code(self, fn_name="predict_row"):
        w = self.weights or []
        code = f"float weights[{len(w)}] = {{{', '.join(map(str, w))}}};\n"
        code += f"float {fn_name}(float row[]) {{\n"
        code += "  float s = 0.0;\n"
        for i in range(len(w)-1):
            code += f"  s += weights[{i}] * row[{i}];\n"
        if len(w) > 0:
            code += f"  s += weights[{len(w)-1}];\n" # Añadir bias
        code += "  return s;\n}\n"
        return code

# ---------------------------
# Mini SVM (simple linear)
# ---------------------------
class MiniSVM:
    """
    Simple linear SVM using stochastic sub-gradient (Pegasos-style).
    Dataset format: [x1, x2, ..., label] with label in {1, -1}.
    Constructor args:
      - learning_rate (float)
      - lambda_param (regularization, float)
      - n_iters (int): number of epochs / passes
    Methods:
      - fit(dataset)
      - predict(X_list) -> list of labels {1, -1}
    """
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = float(learning_rate)
        self.lambda_param = float(lambda_param)
        self.n_iters = int(n_iters)
        self.weights = None  # incluye bias como último elemento

    def fit(self, dataset):
        X = [row[:-1] for row in dataset]
        y = [row[-1] for row in dataset]
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        # weights de inicialización + bias
        self.weights = [0.0] * (n_features + 1)
        for it in range(self.n_iters):
            for xi, yi in zip(X, y):
                if not isinstance(yi, (int, float)):
                    raise ValueError("Labels must be numeric 1 or -1")
                yi = 1 if yi > 0 else -1
                # margen de predicción
                wx = sum(w * xv for w, xv in zip(self.weights[:-1], xi)) + self.weights[-1]
                if yi * wx < 1:
                    # actualización con subgradiente de pérdida de bias+ regularización
                    for j in range(n_features):
                        self.weights[j] = (1 - self.learning_rate * self.lambda_param) * self.weights[j] + self.learning_rate * yi * xi[j]
                    self.weights[-1] = (1 - self.learning_rate * self.lambda_param) * self.weights[-1] + self.learning_rate * yi  # bias
                else:
                    # solo regulariza
                    for j in range(n_features + 1):
                        self.weights[j] = (1 - self.learning_rate * self.lambda_param) * self.weights[j]

    def predict(self, X_list):
        if self.weights is None:
            raise ValueError("Model not trained")
        out = []
        for xi in X_list:
            if not isinstance(xi, (list, tuple)):
                xi = [xi]
            s = sum(w * xv for w, xv in zip(self.weights[:-1], xi)) + self.weights[-1]
            out.append(1 if s >= 0 else -1)
        return out

    # Implementación de `to_arduino_code` para MiniSVM
    def to_arduino_code(self, fn_name="predict_row"):
        w = self.weights or []
        code = f"// EduBot ML: MiniSVM (Linear) C Export\n"
        code += f"float svm_weights[{len(w)}] = {{{', '.join(map(str, w))}}};\n"
        code += f"int {fn_name}(float row[]) {{\n"
        code += "  float s = 0.0;\n"
        # Producto punto
        for i in range(len(w)-1):
            code += f"  s += svm_weights[{i}] * row[{i}];\n"
        # Añadir bias
        if len(w) > 0:
            code += f"  s += svm_weights[{len(w)-1}];\n"
        # Devolver 1 o -1
        code += "  return (s >= 0.0) ? 1 : -1;\n}\n"
        return code

# ---------------------------
# MiniNeuralNetwork (MLP) con backprop multisalida
# ---------------------------
class MiniNeuralNetwork:
    """
    Mini MLP with one hidden layer.
    Constructor:
      MiniNeuralNetwork(n_inputs, n_hidden, n_outputs, learning_rate=0.1, epochs=1000)
    Public attributes (useful para export):
      - W1: list of lists (n_hidden x n_inputs)
      - B1: list of lists (n_hidden x 1)
      - W2: list of lists (n_outputs x n_hidden)
      - B2: list of lists (n_outputs x 1)
    Methods:
      - fit(X, y)
      - predict(X)
      - sigmoid, sigmoid_deriv, clip
    """
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.1, epochs=1000, seed=None):
        self.n_inputs = int(n_inputs)
        self.n_hidden = int(n_hidden)
        self.n_outputs = int(n_outputs)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        # Activaciones
        self.hidden_activation = "sigmoid"
        self.output_activation = "sigmoid"
        if seed is not None:
            random.seed(seed)
        # pequeña inicialización de random
        def rand_matrix(rows, cols):
            return [[(random.random() - 0.5) * 0.2 for _ in range(cols)] for _ in range(rows)]
        self.W1 = rand_matrix(self.n_hidden, self.n_inputs)
        self.B1 = [[0.0] for _ in range(self.n_hidden)]
        self.W2 = rand_matrix(self.n_outputs, self.n_hidden)
        self.B2 = [[0.0] for _ in range(self.n_outputs)]

    # Compatibility wrappers (public)
    # Estos métodos existen solo para evitar AttributeError si el test llama al módulo
    # llaman model.sigmoid(...) o model.sigmoid_deriv(...)
    def clip(self, value: float, min_val: float = -60.0, max_val: float = 60.0) -> float:
        return clip(value, min_val, max_val)

    def sigmoid(self, x: float) -> float:
        return sigmoid(x)

    def sigmoid_deriv(self, out_val: float) -> float:
        return sigmoid_derivative(out_val)

    def relu(self, x: float) -> float:
        return relu(x)

    def relu_derivative(self, x: float) -> float:
        return relu_derivative(x)

    # abstracción de activación interna
    def _activate(self, x: float, act: str):
        """Devuelve activación según 'act' (usa utilidades centrales)."""
        if act == 'sigmoid':
            return sigmoid(x)
        if act == 'relu':
            return relu(x)
        if act == 'linear':
            return linear(x)
        # fallback
        return sigmoid(x)

    def _act_derivative(self, out_val: float, act: str, pre_x: Optional[float] = None):
        """Derivative w.r.t. output (o pre-activation si necesario para ReLU)."""
        if act == 'sigmoid':
            return sigmoid_derivative(out_val)
        if act == 'relu':
            return relu_derivative(pre_x if pre_x is not None else out_val)
        if act == 'linear':
            return linear_derivative(out_val)
        return sigmoid_derivative(out_val)

    def _forward(self, x_row):
        """
        Propagación hacia adelante con soporte de activaciones configurables.
        Calcula las salidas intermedias y finales usando los pesos y sesgos.
        """
        # Capa oculta
        z1, a1 = [], []
        for i in range(self.n_hidden):
            # Suma ponderada entrada -> capa oculta
            s = sum(self.W1[i][j] * x_row[j] for j in range(self.n_inputs)) + self.B1[i][0]
            si = self._activate(s, getattr(self, "hidden_activation", "sigmoid"))
            z1.append(s)
            a1.append(si)

        # Capa de salida
        z2, a2 = [], []
        for k in range(self.n_outputs):
            # Suma ponderada capa oculta -> salida
            s = sum(self.W2[k][i] * a1[i] for i in range(self.n_hidden)) + self.B2[k][0]
            si = self._activate(s, getattr(self, "output_activation", "sigmoid"))
            z2.append(s)
            a2.append(si)

        # Devuelve activaciones intermedias y finales (útil para backpropagation)
        return a1, a2

    def fit(self, X, y):
        # X: lista de listas de entrada; y: lista de listas de destino (longitud n_outputs o escalares)
        # normaliza la forma y
        y_formatted = []
        for yi in y:
            if isinstance(yi, (list, tuple)):
                y_formatted.append([float(v) for v in yi])
            else:
                y_formatted.append([float(yi)])
        for epoch in range(self.epochs):
            for xi, yi in zip(X, y_formatted):
                a1, a2 = self._forward(xi)
                # output error y delta
                delta2 = [0.0] * self.n_outputs
                for k in range(self.n_outputs):
                    err = a2[k] - yi[k]
                    delta2[k] = err * self._act_derivative(a2[k], self.output_activation)
                # hidden deltas
                delta1 = [0.0] * self.n_hidden
                for i in range(self.n_hidden):
                    s = 0.0
                    for k in range(self.n_outputs):
                        s += self.W2[k][i] * delta2[k]
                    delta1[i] = s * self._act_derivative(a1[i], self.hidden_activation)
                # update W2, B2
                for k in range(self.n_outputs):
                    for i in range(self.n_hidden):
                        self.W2[k][i] -= self.learning_rate * delta2[k] * a1[i]
                    self.B2[k][0] -= self.learning_rate * delta2[k]
                # update W1, B1
                for i in range(self.n_hidden):
                    for j in range(self.n_inputs):
                        self.W1[i][j] -= self.learning_rate * delta1[i] * xi[j]
                    self.B1[i][0] -= self.learning_rate * delta1[i]

    def predict(self, X_list):
        preds = []
        for xi in X_list:
            _, a2 = self._forward(xi)
            if self.n_outputs == 1:
                preds.append([a2[0]])
            else:
                preds.append(a2[:])
        return preds

    def to_arduino_code(self, fn_name: str = "nn_predict"):
        # mantiene compatibilidad con código anterior, usa W1/W2/B1/B2
        lines = []
        # convierte weights/biases a matrices para la generación de código (two-layer case)
        lines.append(f"// Auto-generated NN ({self.n_inputs} -> {self.n_hidden} -> {self.n_outputs})")
        # layer 1
        rows1 = len(self.W1)
        cols1 = len(self.W1[0]) if rows1 > 0 else 0
        lines.append(f"float W1[{rows1}][{cols1}] = {{{', '.join('{{' + ','.join(map(str, r)) + '}}' for r in self.W1)}}};")
        lines.append(f"float b1[{rows1}] = {{{', '.join(str(b[0]) for b in self.B1)}}};")
        # layer 2
        rows2 = len(self.W2)
        cols2 = len(self.W2[0]) if rows2 > 0 else 0
        lines.append(f"float W2[{rows2}][{cols2}] = {{{', '.join('{{' + ','.join(map(str, r)) + '}}' for r in self.W2)}}};")
        lines.append(f"float b2[{rows2}] = {{{', '.join(str(b[0]) for b in self.B2)}}};")

        lines.append(f"void {fn_name}(float row[], float out[]) {{")
        # hidden layer
        lines.append(f"  float a1[{rows1}];")
        for i in range(rows1):
            summ = " + ".join(f"W1[{i}][{j}]*row[{j}]" for j in range(cols1))
            lines.append(f"  float s1_{i} = {summ} + b1[{i}];")
            if self.hidden_activation == 'sigmoid':
                lines.append(f"  if (s1_{i} > 60) a1[{i}] = 1.0; else if (s1_{i} < -60) a1[{i}] = 0.0; else a1[{i}] = 1.0/(1.0+exp(-s1_{i}));")
            elif self.hidden_activation == 'relu':
                lines.append(f"  a1[{i}] = s1_{i} > 0 ? s1_{i} : 0.0;")
            else:
                lines.append(f"  a1[{i}] = s1_{i};")
        # output layer
        lines.append(f"  float a2[{rows2}];")
        for k in range(rows2):
            summ = " + ".join(f"W2[{k}][{j}]*a1[{j}]" for j in range(cols2))
            lines.append(f"  float s2_{k} = {summ} + b2[{k}];")
            if self.output_activation == 'sigmoid':
                lines.append(f"  if (s2_{k} > 60) a2[{k}] = 1.0; else if (s2_{k} < -60) a2[{k}] = 0.0; else a2[{k}] = 1.0/(1.0+exp(-s2_{k}));")
            elif self.output_activation == 'relu':
                lines.append(f"  a2[{k}] = s2_{k} > 0 ? s2_{k} : 0.0;")
            else:
                lines.append(f"  a2[{k}] = s2_{k};")
        for i in range(rows2):
            lines.append(f"  out[{i}] = a2[{i}];")
        lines.append("}")
        return "\n".join(lines)

# ---------------------------
# Evaluation metrics
# ---------------------------
def accuracy_score(y_true: List[Any], y_pred: List[Any]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

def mse(y_true: List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n

def mae(y_true: List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n

def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    mean_y = sum(y_true) / len(y_true) if y_true else 0.0
    ss_tot = sum((yi - mean_y) ** 2 for yi in y_true)
    ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y_true, y_pred))
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

# ---------------------------
# Metadata attach helper
# ---------------------------
def attach_metadata(model_obj, metadata: Dict[str, Any]):
    try:
        if not hasattr(model_obj, "metadata"):
            model_obj.metadata = {}
        model_obj.metadata.update(metadata)
    except Exception:
        pass

# ---------------------------
# Self-test (manual)
# ---------------------------
if __name__ == "__main__":
    data_cls = [[2.7, 2.5, 0], [1.3, 3.5, 0], [3.5, 1.4, 1], [3.9, 4.0, 1]]
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(data_cls)
    print("DT predict:", dt.predict([[2.5,2.3],[3.7,3.9]]))

    data_reg = [[1.0,2.0,2.1],[2.0,3.0,3.9],[3.0,4.0,6.1],[4.0,5.0,8.2]]
    dr = DecisionTreeRegressor(max_depth=3)
    dr.fit(data_reg)
    print("DTR predict:", dr.predict([[2.5,3.5],[3.5,4.5]]))

    nn = MiniNeuralNetwork([2,4,1], activations=['relu','linear'], lr=0.01)
    ds = [[1.0,2.0,3.0], [2.0,3.0,5.0], [3.0,4.0,7.0]]
    nn.fit(ds, n_iter=10)
    print("NN preds:", nn.predict([[1,2],[2,3]]))