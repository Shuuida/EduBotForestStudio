"""
MiniML Core for EduBot
======================
Motor de Machine Learning ligero y optimizado para educación y sistemas embebidos.

Estructura del Módulo:
1. Operaciones Matemáticas (MiniMatrixOps)
2. Utilidades Globales de Árboles (Splits, Gini, MSE)
3. Modelos de Árboles (Clasificación y Regresión)
4. Modelos de Bosques (Random Forest)
5. K-Nearest Neighbors (KNN)
6. Modelos Lineales (Linear & SVM)
7. Red Neuronal (MLP)
8. Métricas y Utilidades

Versión: 1.3 (Stable & Structured)
"""

from __future__ import annotations
from typing import List, Any, Dict
import random
import math
from core.ml_compat import safe_compare_le, _flatten_tree_to_arrays

# ---------------------------------------------------------
# MOTOR MATEMÁTICO (MiniMatrixOps)
class MiniMatrixOps:
    """Operaciones de álgebra lineal optimizadas para listas nativas."""

    @staticmethod
    def dot(a: List[float], b: List[float]) -> float:
        # Fallback seguro para longitudes desiguales
        limit = min(len(a), len(b))
        s = 0.0
        for i in range(limit):
            s += a[i] * b[i]
        return s

    @staticmethod
    def matvec(mat: List[List[float]], vec: List[float]) -> List[float]:
        return [MiniMatrixOps.dot(row, vec) for row in mat]

    @staticmethod
    def transpose(mat: List[List[float]]) -> List[List[float]]:
        if not mat: return []
        return [list(row) for row in zip(*mat)]

    @staticmethod
    def add(a: List[float], b: List[float]) -> List[float]:
        return [x + y for x, y in zip(a, b)]

    @staticmethod
    def sub(a: List[float], b: List[float]) -> List[float]:
        return [x - y for x, y in zip(a, b)]

    @staticmethod
    def scalar_mul(vec: List[float], s: float) -> List[float]:
        return [x * s for x in vec]
    
    @staticmethod
    def sigmoid(x: float) -> float:
        if x < -700: return 0.0
        if x > 700: return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(output: float) -> float:
        return output * (1.0 - output)

# ------------------------------------------------------------------------
# 2. UTILIDADES GLOBALES DE ÁRBOLES (Shared Logic)
# Estas funciones están fuera de las clases para ser usadas tanto por
# Clasificación como por Regresión sin duplicar código.

def test_split(index, value, dataset):
    """Divide un dataset en dos grupos."""
    left, right = [], []
    for row in dataset:
        if safe_compare_le(row[index], value):
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, classes):
    """Cálculo de impureza Gini (para Clasificación)."""
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0: continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def mse_metric(groups):
    """Cálculo de Mean Squared Error (para Regresión)."""
    total_error = 0.0
    n_instances = float(sum([len(group) for group in groups]))
    for group in groups:
        size = float(len(group))
        if size == 0: continue
        outcomes = [row[-1] for row in group]
        mean_val = sum(outcomes) / size
        error = sum([(x - mean_val)**2 for x in outcomes])
        total_error += error
    return total_error / n_instances if n_instances > 0 else 0.0

def to_terminal_classifier(group):
    """Retorna la clase más común (Moda)."""
    outcomes = [row[-1] for row in group]
    if not outcomes: return 0
    return max(set(outcomes), key=outcomes.count)

def to_terminal_regressor(group):
    """Retorna el promedio de valores (Media)."""
    outcomes = [row[-1] for row in group]
    if not outcomes: return 0.0
    return sum(outcomes) / len(outcomes)

def _generate_tree_c_code(root, fn_name):
    """Generador genérico de código C para cualquier árbol."""
    flat = _flatten_tree_to_arrays(root)
    code = [f"// DT Export: {fn_name}"]
    
    def arr(n, d, t="int"):
        return f"const {t} {n}[] PROGMEM = {{ {', '.join(map(str, d))} }};"

    code.append(arr(f"{fn_name}_idx", flat['feature_index']))
    code.append(arr(f"{fn_name}_val", flat['threshold'], "float"))
    code.append(arr(f"{fn_name}_L", flat['left_child']))
    code.append(arr(f"{fn_name}_R", flat['right_child']))
    code.append(arr(f"{fn_name}_out", flat['value'], "float"))
    
    code.append(f"\nfloat {fn_name}(float* input) {{")
    code.append("  int curr = 0;")
    code.append("  while(true) {")
    code.append(f"    int idx = pgm_read_word_near({fn_name}_idx + curr);")
    code.append(f"    if (idx == -1) return pgm_read_float_near({fn_name}_out + curr);")
    code.append(f"    float thresh = pgm_read_float_near({fn_name}_val + curr);")
    code.append(f"    if (input[idx] <= thresh) curr = pgm_read_word_near({fn_name}_L + curr);")
    code.append(f"    else curr = pgm_read_word_near({fn_name}_R + curr);")
    code.append("  }\n}")
    return "\n".join(code)

# ------------------------------------------------------
# MODELOS DE ÁRBOLES

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_size=1, n_features=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.root = None

    def fit(self, dataset):
        self.root = self._build_tree(dataset, self.max_depth)

    def predict(self, X):
        return [self._predict_row(self.root, row) for row in X]

    def _build_tree(self, train, depth):
        if not train: return None
        # Separación simple
        y = [row[-1] for row in train]
        if not y: return 0
        if len(set(y)) == 1: return to_terminal_classifier(train)
        if depth == 0 or len(train) < self.min_size: return to_terminal_classifier(train)

        best = self._get_best_split(train)
        if not best: return to_terminal_classifier(train)
        
        l, r = best['groups']
        del best['groups']
        if not l or not r: return to_terminal_classifier(l + r)
        
        best['left'] = self._build_tree(l, depth - 1)
        best['right'] = self._build_tree(r, depth - 1)
        return best

    def _get_best_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_idx, b_val, b_score, b_groups = 999, 999, 999, None
        n_features = len(dataset[0]) - 1
        features = list(range(n_features))
        
        if self.n_features and self.n_features < n_features:
            features = random.sample(features, self.n_features)

        for index in features:
            for row in dataset:
                groups = test_split(index, row[index], dataset)
                gini = gini_index(groups, class_values)
                if gini < b_score:
                    b_idx, b_val, b_score, b_groups = index, row[index], gini, groups
        
        if b_score == 999: return None
        return {'index': b_idx, 'value': b_val, 'groups': b_groups}

    def _predict_row(self, node, row):
        if not isinstance(node, dict): return node
        if node.get('index') == -1 or 'left' not in node: return node.get('value', 0)
        
        if safe_compare_le(row[node['index']], node['value']):
            return self._predict_row(node['left'], row)
        else:
            return self._predict_row(node['right'], row)

    def to_arduino_code(self, fn_name="tree_predict"):
        return _generate_tree_c_code(self.root, fn_name)


class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_size=1, n_features=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.root = None

    def fit(self, dataset):
        self.root = self._build_tree(dataset, self.max_depth)

    def predict(self, X):
        return [self._predict_row(self.root, row) for row in X]

    def _build_tree(self, train, depth):
        if not train: return None
        if depth == 0 or len(train) < self.min_size: return to_terminal_regressor(train)

        best = self._get_best_split(train)
        if not best: return to_terminal_regressor(train)
        
        l, r = best['groups']
        del best['groups']
        if not l or not r: return to_terminal_regressor(l + r)
        
        best['left'] = self._build_tree(l, depth - 1)
        best['right'] = self._build_tree(r, depth - 1)
        return best

    def _get_best_split(self, dataset):
        b_idx, b_val, b_score, b_groups = 999, 999, float('inf'), None
        n_features = len(dataset[0]) - 1
        features = list(range(n_features))
        if self.n_features and self.n_features < n_features:
            features = random.sample(features, self.n_features)

        for index in features:
            for row in dataset:
                groups = test_split(index, row[index], dataset)
                mse = mse_metric(groups)
                if mse < b_score:
                    b_idx, b_val, b_score, b_groups = index, row[index], mse, groups
        
        if b_score == float('inf'): return None
        return {'index': b_idx, 'value': b_val, 'groups': b_groups}

    def _predict_row(self, node, row):
        if not isinstance(node, dict): return node
        if node.get('index') == -1: return node.get('value', 0.0)
        
        if safe_compare_le(row[node['index']], node['value']):
            return self._predict_row(node['left'], row)
        else:
            return self._predict_row(node['right'], row)

    def to_arduino_code(self, fn_name="tree_reg_predict"):
        return _generate_tree_c_code(self.root, fn_name)

# ---------------------------------------------------------------------
# MODELOS DE BOSQUES (Random Forest)

def _subsample(dataset, ratio):
    """Función helper para bagging."""
    sample = []
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = random.randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def _generate_rf_c_code(trees, fn_name, task="classification"):
    """Generador genérico para Random Forest."""
    code = []
    names = []
    for i, t in enumerate(trees):
        nm = f"{fn_name}_t{i}"
        names.append(nm)
        code.append(t.to_arduino_code(nm))
        code.append("")
    
    code.append(f"float {fn_name}(float* input) {{")
    code.append(f"  float votes[{len(trees)}];")
    for i, nm in enumerate(names):
        code.append(f"  votes[{i}] = {nm}(input);")
    
    if task == "regression":
        code.append("  // Average logic")
        code.append("  float sum = 0;")
        code.append(f"  for(int i=0; i<{len(trees)}; i++) sum += votes[i];")
        code.append(f"  return sum / {len(trees)};")
    else:
        code.append("  // Majority Vote logic")
        code.append("  int c0=0, c1=0;")
        code.append(f"  for(int i=0; i<{len(trees)}; i++) (votes[i]<0.5)? c0++ : c1++;")
        code.append("  return (c1 > c0) ? 1.0 : 0.0;")
    
    code.append("}")
    return "\n".join(code)

class RandomForestClassifier:
    def __init__(self, n_trees=5, max_depth=5, min_size=1, sample_size=1.0, n_features=None, seed=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_features = n_features
        self.trees = []
        if seed: random.seed(seed)

    def fit(self, dataset):
        self.trees = []
        for _ in range(self.n_trees):
            sample = _subsample(dataset, self.sample_size)
            tree = DecisionTreeClassifier(self.max_depth, self.min_size, self.n_features)
            tree.fit(sample)
            self.trees.append(tree)

    def predict(self, X):
        preds = [tree.predict(X) for tree in self.trees]
        per_row = list(zip(*preds))
        return [max(set(row), key=row.count) for row in per_row]

    def to_arduino_code(self, fn_name="rf_predict"):
        return _generate_rf_c_code(self.trees, fn_name, task="classification")


class RandomForestRegressor:
    def __init__(self, n_trees=5, max_depth=5, min_size=1, sample_size=1.0, n_features=None, seed=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_features = n_features
        self.trees = []
        if seed: random.seed(seed)

    def fit(self, dataset):
        self.trees = []
        for _ in range(self.n_trees):
            sample = _subsample(dataset, self.sample_size)
            tree = DecisionTreeRegressor(self.max_depth, self.min_size, self.n_features)
            tree.fit(sample)
            self.trees.append(tree)

    def predict(self, X):
        preds = [tree.predict(X) for tree in self.trees]
        per_row = list(zip(*preds))
        return [sum(row)/len(row) for row in per_row]

    def to_arduino_code(self, fn_name="rf_reg_predict"):
        return _generate_rf_c_code(self.trees, fn_name, task="regression")

# -------------------------------------------------
# K-NEAREST NEIGHBORS (KNN)

class KNearestNeighbors:
    def __init__(self, k=3, task='classification'):
        self.k = k
        self.task = task
        self.X_train = []
        self.y_train = []
        self.n_features_trained = 0

    def fit(self, dataset):
        self.X_train = [row[:-1] for row in dataset]
        self.y_train = [row[-1] for row in dataset]
        self.n_features_trained = len(self.X_train[0]) if self.X_train else 0

    def predict(self, X):
        preds = []
        for row in X:
            dists = []
            for i, tr in enumerate(self.X_train):
                # Distancia Euclidiana
                d = math.sqrt(sum((row[j]-tr[j])**2 for j in range(len(row))))
                dists.append((self.y_train[i], d))
            
            # Ordenar por distancia
            dists.sort(key=lambda x: x[1])
            neighbors = [n[0] for n in dists[:self.k]]
            
            if self.task == 'regression':
                preds.append(sum(neighbors)/max(len(neighbors), 1))
            else:
                preds.append(max(set(neighbors), key=neighbors.count))
        return preds

    def to_arduino_code(self, fn_name="knn_predict"):
        # Almacena datos en Flash y busca vecinos
        flat_X = [x for r in self.X_train for x in r]
        code = [f"// KNN: {fn_name}"]
        code.append(f"const float {fn_name}_X[] PROGMEM = {{ {', '.join(f'{x:.4f}' for x in flat_X)} }};")
        code.append(f"const float {fn_name}_y[] PROGMEM = {{ {', '.join(f'{x:.4f}' for x in self.y_train)} }};")
        
        code.append(f"\nfloat {fn_name}(float* input) {{")
        code.append(f"  int k={self.k}, n={len(self.y_train)}, d={self.n_features_trained};")
        code.append("  float bdists[10]; float bvals[10];") # Limite K=10 hardcoded para C array
        code.append("  for(int i=0;i<k;i++) bdists[i]=999999.0;")
        code.append("  for(int i=0; i<n; i++) {")
        code.append("    float dist=0; for(int j=0; j<d; j++) {")
        code.append(f"      float v = pgm_read_float_near({fn_name}_X + i*d + j);")
        code.append("      dist += (input[j]-v)*(input[j]-v); }")
        code.append("    dist = sqrt(dist);")
        code.append("    // Insertion sort")
        code.append("    for(int m=0; m<k; m++) if(dist < bdists[m]) {")
        code.append("      for(int x=k-1; x>m; x--) { bdists[x]=bdists[x-1]; bvals[x]=bvals[x-1]; }")
        code.append(f"      bdists[m]=dist; bvals[m]=pgm_read_float_near({fn_name}_y + i); break;")
        code.append("    }")
        code.append("  }")
        code.append("  float s=0; for(int i=0;i<k;i++) s+=bvals[i];")
        code.append("  float mean = s/k;")
        if self.task == 'regression': return "\n".join(code) + " return mean; }"
        return "\n".join(code) + " return (mean>=0.5)? 1.0 : 0.0; }"

# --------------------------------------------------------
# MODELOS LINEALES (SGD)

class MiniLinearModel:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr, self.epochs = learning_rate, epochs
        self.weights, self.bias = [], 0.0

    def fit(self, dataset):
        n_features = len(dataset[0]) - 1
        self.weights, self.bias = [0.0] * n_features, 0.0
        for _ in range(self.epochs):
            for row in dataset:
                # Usamos _predict_single para evitar duplicidad de lógica
                pred = self._predict_single(row[:-1])
                err = pred - row[-1]
                self.bias -= self.lr * err
                for i in range(n_features): self.weights[i] -= self.lr * err * row[i]

    def _predict_single(self, row):
        return MiniMatrixOps.dot(row, self.weights) + self.bias

    def predict(self, X):
        return [self._predict_single(r) for r in X]

    def to_arduino_code(self, fn_name="linear_predict"):
        """
        Genera código C para inferencia lineal (Regresión).
        y = (w * x) + b
        """
        code = []
        code.append(f"// Linear Model Export: {fn_name}")
        
        # Pesos en memoria Flash (PROGMEM)
        w_str = ", ".join(f"{w:.5f}" for w in self.weights)
        code.append(f"const float {fn_name}_w[] PROGMEM = {{ {w_str} }};")
        code.append(f"const float {fn_name}_b = {self.bias:.5f};")
        code.append(f"const int {fn_name}_dim = {len(self.weights)};")
        
        # Función de inferencia
        code.append(f"\nfloat {fn_name}(float* input) {{")
        code.append("  float result = 0.0;")
        code.append(f"  for(int i=0; i<{fn_name}_dim; i++) {{")
        code.append(f"    result += input[i] * pgm_read_float_near({fn_name}_w + i);")
        code.append("  }")
        code.append(f"  return result + {fn_name}_b;")
        code.append("}")
        return "\n".join(code)

class MiniSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr, self.lambda_param, self.n_iters = learning_rate, lambda_param, n_iters
        self.weights, self.bias = [], 0.0

    def fit(self, dataset):
        n_feats = len(dataset[0]) - 1
        self.weights, self.bias = [0.0] * n_feats, 0.0
        formatted = [(r[:-1], 1 if r[-1]>0.5 else -1) for r in dataset]
        
        for _ in range(self.n_iters):
            for X, y in formatted:
                cond = y * (MiniMatrixOps.dot(X, self.weights) - self.bias) >= 1
                if cond:
                    for i in range(n_feats): self.weights[i] -= self.lr * (2*self.lambda_param*self.weights[i])
                else:
                    for i in range(n_feats): self.weights[i] -= self.lr * (2*self.lambda_param*self.weights[i] - y*X[i])
                    self.bias -= self.lr * y

    def predict(self, X):
        return [1.0 if (MiniMatrixOps.dot(r, self.weights) - self.bias) >= 0 else 0.0 for r in X]
    
    def to_arduino_code(self, fn_name="svm_predict"):
        w_s = ", ".join(f"{w:.5f}" for w in self.weights)
        code = [f"const float {fn_name}_w[] PROGMEM = {{ {w_s} }};"]
        code.append(f"const float {fn_name}_b = {self.bias:.5f};")
        code.append(f"\nfloat {fn_name}(float* in) {{")
        code.append(f"  float d=0; for(int i=0;i<{len(self.weights)};i++) d+=in[i]*pgm_read_float_near({fn_name}_w+i);")
        code.append(f"  return (d - {fn_name}_b >= 0) ? 1.0 : 0.0; }}")
        return "\n".join(code)

# ----------------------------------------------------------
# RED NEURONAL (MLP)

class MiniNeuralNetwork:
    def __init__(self, n_inputs=2, n_hidden=4, n_outputs=1, epochs=1000, learning_rate=0.1):
        try:
            self.n_in = int(n_inputs)
            self.n_hid = int(n_hidden)
            self.n_out = int(n_outputs)
            self.epochs = int(epochs)
            self.lr = float(learning_rate)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Parámetros de Red Neuronal inválidos: {e}. Asegúrese de usar números.")
        self.W1, self.B1, self.W2, self.B2 = [], [], [], []
        self._init_weights()
        self.quantized = False
        self.act_scales = {}

    def _init_weights(self):
        random.seed(42)
        l1 = math.sqrt(6/(self.n_in + self.n_hid))
        self.W1 = [[random.uniform(-l1, l1) for _ in range(self.n_in)] for _ in range(self.n_hid)]
        self.B1 = [0.0] * self.n_hid
        l2 = math.sqrt(6/(self.n_hid + self.n_out))
        self.W2 = [[random.uniform(-l2, l2) for _ in range(self.n_hid)] for _ in range(self.n_out)]
        self.B2 = [0.0] * self.n_out

    def forward(self, x):
        self.z1 = MiniMatrixOps.add(MiniMatrixOps.matvec(self.W1, x), self.B1)
        self.a1 = [MiniMatrixOps.sigmoid(v) for v in self.z1]
        self.z2 = MiniMatrixOps.add(MiniMatrixOps.matvec(self.W2, self.a1), self.B2)
        self.a2 = [MiniMatrixOps.sigmoid(v) for v in self.z2]
        return self.a2

    def fit(self, dataset):
        X = [row[:-1] for row in dataset]
        y = [row[-1] for row in dataset]
        if not isinstance(y[0], list): y = [[val] for val in y]

        for _ in range(self.epochs):
            for i, row in enumerate(X):
                target = y[i]
                out = self.forward(row)
                
                # Backprop
                err = MiniMatrixOps.sub(out, target)
                d2 = [e * MiniMatrixOps.sigmoid_derivative(o) for e, o in zip(err, out)]
                
                W2_T = MiniMatrixOps.transpose(self.W2)
                her = MiniMatrixOps.matvec(W2_T, d2)
                d1 = [h * MiniMatrixOps.sigmoid_derivative(a) for h, a in zip(her, self.a1)]
                
                for r in range(self.n_out):
                    self.B2[r] -= self.lr * d2[r]
                    for c in range(self.n_hid): self.W2[r][c] -= self.lr * d2[r] * self.a1[c]
                for r in range(self.n_hid):
                    self.B1[r] -= self.lr * d1[r]
                    for c in range(self.n_in): self.W1[r][c] -= self.lr * d1[r] * row[c]

    def calibrate(self, dataset): 
        """Stub para compatibilidad."""
        pass
    
    def quantize(self): 
        """Stub para compatibilidad."""
        self.quantized = True

    def predict(self, X):
        preds = [self.forward(r) for r in X]
        # Corrección crítica para evitar List[List] en salida escalar
        if self.n_out == 1: return [p[0] for p in preds]
        return preds

    def to_arduino_code(self, fn_name="nn_predict"):
        code = [f"// NN: {fn_name}"]
        # Macros para exportar matrices
        def m2c(n, m): return f"const float {n}[] PROGMEM = {{ {', '.join(f'{x:.4f}' for x in [v for r in m for v in r])} }};"
        def v2c(n, v): return f"const float {n}[] PROGMEM = {{ {', '.join(f'{x:.4f}' for x in v)} }};"
        
        code.append(m2c(f"{fn_name}_W1", self.W1))
        code.append(v2c(f"{fn_name}_B1", self.B1))
        code.append(m2c(f"{fn_name}_W2", self.W2))
        code.append(v2c(f"{fn_name}_B2", self.B2))
        
        code.append(f"\nvoid {fn_name}(float* in, float* out) {{")
        code.append(f"  float h[{self.n_hid}];")
        code.append(f"  for(int i=0; i<{self.n_hid}; i++) {{ float s=pgm_read_float_near({fn_name}_B1+i);")
        code.append(f"    for(int j=0; j<{self.n_in}; j++) s+=in[j]*pgm_read_float_near({fn_name}_W1 + i*{self.n_in}+j);")
        code.append(f"    h[i] = 1.0/(1.0+exp(-s)); }}")
        code.append(f"  for(int i=0; i<{self.n_out}; i++) {{ float s=pgm_read_float_near({fn_name}_B2+i);")
        code.append(f"    for(int j=0; j<{self.n_hid}; j++) s+=h[j]*pgm_read_float_near({fn_name}_W2 + i*{self.n_hid}+j);")
        code.append(f"    out[i] = 1.0/(1.0+exp(-s)); }}")
        code.append("}")
        return "\n".join(code)

class MiniScaler:
    """StandardScaler optimizado (z-score normalization)."""
    def __init__(self):
        self.means = []
        self.stds = []
        self.fitted = False

    def fit(self, dataset: List[List[float]]) -> None:
        """Calcula media y desviación estándar de las columnas."""
        if not dataset: return
        n_rows = len(dataset)
        n_cols = len(dataset[0])
        
        self.means = [0.0] * n_cols
        self.stds = [0.0] * n_cols
        
        for i in range(n_cols):
            col_values = [row[i] for row in dataset]
            mean = sum(col_values) / n_rows
            variance = sum((x - mean) ** 2 for x in col_values) / n_rows
            std = math.sqrt(variance)
            
            self.means[i] = mean
            # Evitar división por cero si la columna es constante
            self.stds[i] = std if std > 1e-9 else 1.0
        
        self.fitted = True

    def transform(self, dataset: List[List[float]]) -> List[List[float]]:
        """Aplica el escalado: (x - u) / s."""
        if not self.fitted: return dataset
        
        scaled_data = []
        for row in dataset:
            new_row = []
            for i, val in enumerate(row):
                # Proteger contra dimensiones incorrectas
                if i < len(self.means):
                    new_val = (val - self.means[i]) / self.stds[i]
                    new_row.append(new_val)
                else:
                    new_row.append(val)
            scaled_data.append(new_row)
        return scaled_data

    def fit_transform(self, dataset: List[List[float]]) -> List[List[float]]:
        self.fit(dataset)
        return self.transform(dataset)

# -----------------------------------------------
# MÉTRICAS Y UTILIDADES

def accuracy_score(y_true, y_pred):
    if not y_true: return 0.0
    # Manejo robusto float vs int
    correct = sum(1 for t, p in zip(y_true, y_pred) if (1 if t>0.5 else 0) == (1 if p>0.5 else 0))
    return correct / len(y_true)

def mse_score(y_true, y_pred):
    if not y_true: return 0.0
    return sum((t-p)**2 for t, p in zip(y_true, y_pred)) / len(y_true)

def mae_score(y_true, y_pred):
    if not y_true: return 0.0
    return sum(abs(t-p) for t, p in zip(y_true, y_pred)) / len(y_true)

def r2_score(y_true, y_pred):
    if not y_true: return 0.0
    mean_y = sum(y_true)/len(y_true)
    ss_tot = sum((y-mean_y)**2 for y in y_true)
    ss_res = sum((t-p)**2 for t,p in zip(y_true, y_pred))
    return 1 - (ss_res/ss_tot) if ss_tot > 1e-9 else 0.0

def attach_metadata(model_obj, metadata: Dict[str, Any]):
    try:
        if not hasattr(model_obj, "metadata"):
            model_obj.metadata = {}
        model_obj.metadata.update(metadata)
    except Exception:
        pass