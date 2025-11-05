"""
ML Manager — Modo Dual Intelligent
- Detecta scikit-learn y usa sklearn cuando esté disponible.
- Si sklearn no está, usa core.ml_runtime (MiniML).
- Proporciona API unificada: train, predict, save, load, export_for_embedded.
"""

from typing import Any, Dict, List, Optional
import importlib
import time
import json
import os
#from statistics import mean
#import math

# Preferir el tiempo de ejecución interno
from core import ml_runtime

# Intenta importar sklearn y joblib opcionalmente.
_sklearn = None
_joblib = None
try:
    _sklearn = importlib.import_module("sklearn.ensemble")
    _joblib = importlib.import_module("joblib")
    _sklearn_available = True
except Exception:
    _sklearn_available = False

# Registro en memoria para modelos (ambos modos)
_MODEL_REGISTRY: Dict[str, Any] = {}


# -------------------------
# Helpers

def available_mode() -> str:
    return "sklearn" if _sklearn_available else "mini"

def clear_registry():
    _MODEL_REGISTRY.clear()

def list_models() -> List[str]:
    """Lista todos los modelos actualmente registrados en memoria."""
    return list(_MODEL_REGISTRY.keys())

def get_model(name: str) -> Any:
    """Obtiene un modelo registrado por nombre."""
    return _MODEL_REGISTRY.get(name, {}).get("model")

def _is_regression_dataset(dataset):
    """
    Determina si el dataset parece ser de regresión (valores continuos) o clasificación.
    Se considera regresión si:
    - Hay floats no enteros en la última columna (targets continuos), o
    - Hay más de 10 valores únicos en el target.
    """
    if not dataset or not isinstance(dataset, list):
        return False
    try:
        y = [row[-1] for row in dataset]
        # Si el conjunto tiene más de 10 valores únicos o valores no enteros, se considera regresión
        unique = set(y)
        if any(isinstance(v, float) and not v.is_integer() for v in y):
            return True
        return len(unique) > 10  # heurística simple
    except Exception:
        return False

# -------------------------
# Train wrappers

def train_random_forest(name: Optional[str] = None, dataset: Optional[List[List[Any]]] = None, *,
                        model_name: Optional[str] = None,
                        n_trees: int = 5, max_depth: int = 10,
                        min_size: int = 1, sample_size: float = 1.0,
                        n_features: Optional[int] = None, seed: Optional[int] = None
                        ) -> Dict[str, Any]:
    """
    Entrena un modelo RandomForest (clasificador o regresor) según los datos.
    """
    name = model_name or name
    if name is None:
        raise ValueError("Debe especificarse un nombre de modelo (name o model_name).")

    if dataset is None:
        raise ValueError("El dataset no puede ser None.")

    start = time.time()
    mode = available_mode()
    is_regression = _is_regression_dataset(dataset)
    meta = {'mode': mode, 'type': 'regression' if is_regression else 'classification'}

    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]

    if mode == "sklearn":
        try:
            if is_regression:
                from sklearn.ensemble import RandomForestRegressor as SKForest
            else:
                from sklearn.ensemble import RandomForestClassifier as SKForest

            model = SKForest(
                n_estimators=n_trees,
                max_depth=max_depth,
                random_state=seed,
                n_jobs=-1
            )
            model.fit(X, y)
            _MODEL_REGISTRY[name] = {'mode': 'sklearn', 'model': model, 'type': meta['type']}
        except Exception as e:
            raise RuntimeError(f"Error entrenando RandomForest sklearn: {e}")
    else:
        try:
            rf = ml_runtime.RandomForestClassifier(
                n_trees=n_trees, max_depth=max_depth,
                min_size=min_size, sample_size=sample_size,
                n_features=n_features, seed=seed
            )
            rf.fit(dataset)
            _MODEL_REGISTRY[name] = {'mode': 'mini', 'model': rf, 'type': meta['type']}
        except Exception as e:
            raise RuntimeError(f"Error entrenando RandomForest interno: {e}")

    meta['time_seconds'] = time.time() - start
    model = _MODEL_REGISTRY[name]['model']
    return {'meta': meta, 'model': model, 'model_name': name}


# -------------------------
# TRAIN LINEAR MODEL

def train_linear_model(model_name: str, dataset: list, *, learning_rate=0.01, epochs=1000):
    """
    Entrena un modelo lineal (MiniML o sklearn).
    """
    is_regression = _is_regression_dataset(dataset)

    try:
        model = ml_runtime.MiniLinearModel(learning_rate=learning_rate, epochs=epochs)
        model.fit(dataset)
        _MODEL_REGISTRY[model_name] = {
            "mode": "mini",
            "model": model,
            "type": "linear_regression" if is_regression else "linear_classification"
        }
        meta = {"mode": "mini", "framework": "MiniML", "type": "linear"}
        model = _MODEL_REGISTRY[model_name]['model']
        return {'meta': meta, 'model': model, 'model_name': model_name}
    except Exception as e:
        raise RuntimeError(f"Error entrenando modelo lineal MiniML: {e}")

#-------------------------
# TRAIN SVM

def train_svm(model_name: str, dataset: list, *, learning_rate=0.001, lambda_param=0.01, epochs=1000):
    """Entrena un clasificador SVM simple."""
    try:
        model = ml_runtime.MiniSVM(learning_rate=learning_rate, lambda_param=lambda_param, epochs=epochs)
        model.fit(dataset)
        _MODEL_REGISTRY[model_name] = {'mode': 'mini', 'model': model, 'type': 'svm'}
        meta = {'mode': 'mini', 'type': 'svm', 'framework': 'MiniML'}
        model = _MODEL_REGISTRY[model_name]['model']
        return {'meta': meta, 'model': model, 'model_name': model_name}
    except Exception as e:
        raise RuntimeError(f"Error entrenando SVM: {e}")

# -------------------------
# TRAIN NEURAL NETWORK

def train_neural_network(model_name: str, dataset: list, *, hidden_size=4, epochs=1000, lr=0.01):
    """Entrena una pequeña red neuronal (MiniML)."""
    import numpy as np
    input_size = len(dataset[0]) - 1
    try:
        model = ml_runtime.MiniNeuralNetwork(input_size=input_size, hidden_size=hidden_size, lr=lr, epochs=epochs)
        model.fit(dataset)
        _MODEL_REGISTRY[model_name] = {'mode': 'mini', 'model': model, 'type': 'neural_net'}
        meta = {'mode': 'mini', 'type': 'neural_net', 'framework': 'MiniML'}
        model = _MODEL_REGISTRY[model_name]['model']
        return {'meta': meta, 'model': model, 'model_name': model_name}
    except Exception as e:
        raise RuntimeError(f"Error entrenando red neuronal: {e}")

# -------------------------
def predict(name: Optional[str] = None, model: Optional[Any] = None, X: Optional[List[List[Any]]] = None) -> List[Any]:
    """
    Predict wrapper flexible:
      - Puede llamarse como predict(name="model_name", X=...)
      - O como predict(model=model_object, X=...)
      - O como predict("model_name", X) (posicional)
    Devuelve lista de predicciones.
    """
    # Normalizar argumentos posicionales si se pasó name como lista accidentalmente
    if X is None and isinstance(model, list) and X is None:
        # llamado predict(name, X) con dos posicionales puede terminar aquí, pero es raro
        pass

    # Determinar modelo real a usar
    model_obj = None
    if model is not None:
        # Si model es un string (nombre), buscar en registry
        if isinstance(model, str):
            if model not in _MODEL_REGISTRY:
                raise KeyError(f"Model '{model}' not found")
            model_obj = _MODEL_REGISTRY[model]['model']
        else:
            # Asumimos que es un objeto modelo ya instanciado (sklearn o mini)
            model_obj = model
    elif name is not None:
        # name dado: buscar en el registro
        if name not in _MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' not found")
        model_obj = _MODEL_REGISTRY[name]['model']
    else:
        raise ValueError("predict requires either 'name' (registered model name) or 'model' (model object)")

    if X is None:
        raise ValueError("predict requires 'X' input (list of samples)")

    # Ejecutar predicción según tipo de modelo
    # Si el modelo es sklearn (posible), usa model.predict(X)
    try:
        # Algunos modelos sklearn devuelven numpy arrays; forzamos lista
        preds = model_obj.predict(X)
        # Convertir a lista de tipos básicos
        try:
            return list(preds)
        except Exception:
            return preds
    except AttributeError:
        # Fallback: si tiene método 'predict' no encontrado, lanzar error controlado
        raise RuntimeError("Model object does not support 'predict' method")

# -------------------------
# Persistencia de modelos

def save_model(name: str, path: str) -> None:
    """
    Guardar el modelo en el disco. Si está en modo sklearn, use joblib; 
    de lo contrario, use la serialización personalizada JSON.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found")
    entry = _MODEL_REGISTRY[name]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if entry['mode'] == 'sklearn' and _joblib is not None:
        _joblib.dump(entry['model'], path)
    else:
        # MiniML: serializar la estructura del árbol(es) en JSON
        # Para RandomForest, cada árbol se almacena como entrada (dict).
        model = entry['model']
        if hasattr(model, 'trees'):
            # MiniML no tiene un atributo de árboles directo, pero podemos intentar acceder a `.trees` o a la lista de `.trees`.
            try:
                trees = getattr(model, 'trees', None)
                if trees is None and hasattr(model, 'trees'):
                    trees = model.trees
            except Exception:
                trees = model.trees if hasattr(model, 'trees') else None
        else:
            trees = getattr(model, 'trees', None)
        # Para DecisionTreeClassifier (Árbol único) almacenamos el diccionario raíz
        serial = {'type': 'mini_model', 'repr': repr(model)}
        # Volcado básico: depende de que el modelo tenga .root o .trees; dejamos que ml_runtime proporcione la exportación si es necesario
        if hasattr(model, 'root'):
            serial['root'] = model.root
        if hasattr(model, 'trees'):
            serial['trees'] = [getattr(t, 'root', None) for t in model.trees]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serial, f, indent=2)


def load_model(name: str, path: str) -> None:
    """Carga el modelo desde el disco y lo registra bajo 'name'."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # Prueba primero con sklearn.
    if _sklearn_available and _joblib is not None:
        try:
            model = _joblib.load(path)
            _MODEL_REGISTRY[name] = {'mode': 'sklearn', 'model': model}
            return
        except Exception:
            # fallback al minicargador de abajo
            pass
    # fallback para el minicargador
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data.get('type') == 'mini_model':
        # naive loader: Reconstruir DecisionTree/RandomForest a partir de las raíces almacenadas
        if 'trees' in data and data['trees'] is not None:
            # Construir RandomForest solo con árboles: creamos un RandomForest vacío y establecemos las raíces.
            rf = ml_runtime.RandomForestClassifier(n_trees=len(data['trees']))
            # Construir objetos DecisionTreeClassifier ficticios y establecer la raíz
            rf.trees = []
            for root in data['trees']:
                dt = ml_runtime.DecisionTreeClassifier()
                dt.root = root
                rf.trees.append(dt)
            _MODEL_REGISTRY[name] = {'mode': 'mini', 'model': rf}
            return
        if 'root' in data:
            dt = ml_runtime.DecisionTreeClassifier()
            dt.root = data['root']
            _MODEL_REGISTRY[name] = {'mode': 'mini', 'model': dt}
            return
    raise ValueError("Unknown model format or failed to load.")


# -------------------------
# Exportador a código embebido

def _export_tree_to_ifelse(root: dict, fn_name: str = "predict_row", indent: str = "    ") -> str:
    """
    Convierte recursivamente la raíz de un árbol (un diccionario de nodos con 'index','value','left','right' o terminal) en una función if/else.
    Se espera una estructura de nodo compatible con la salida de ml_runtime.get_split/build_tree.
    """
    def node_to_code(node, level=1):
        pad = indent * level
        if not isinstance(node, dict):
            # terminal
            return pad + f"return {repr(node)}\n"
        idx = node.get('index')
        val = node.get('value')
        code = pad + f"if row[{idx}] <= {repr(val)}:\n"
        code += node_to_code(node.get('left'), level + 1)
        code += pad + "else:\n"
        code += node_to_code(node.get('right'), level + 1)
        return code

    func = f"def {fn_name}(row):\n"
    func += node_to_code(root, 1)
    return func

def export_model_to_python_function(name: str, out_path: str, fn_name: str = "predict_row") -> str:
    """
    Exportar un modelo registrado (mini mode o sklearn DecisionTree mediante conversión)
    a un archivo Python independiente que define la función `fn_name(row)`.
    Devuelve la ruta al archivo.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found")
    entry = _MODEL_REGISTRY[name]
    if entry['mode'] == 'mini':
        model = entry['model']
        # Si se trata de RandomForest, creamos un llamador de conjunto que vota
        if isinstance(model, ml_runtime.RandomForestClassifier):
            parts = []
            parts.append("def majority_vote(votes):")
            parts.append("    agg = {}")
            parts.append("    for v in votes:")
            parts.append("        agg[v] = agg.get(v, 0) + 1")
            parts.append("    return max(agg.items(), key=lambda x: x[1])[0]\n")
            # genera funciones para cada árbol
            for i, t in enumerate(model.trees):
                root = getattr(t, 'root', None)
                if root is None:
                    raise ValueError("Tree has no root")
                parts.append(_export_tree_to_ifelse(root, fn_name=f"tree_{i}_predict"))
            # función de conjunto
            parts.append(f"def {fn_name}(row):")
            parts.append("    votes = []")
            for i in range(len(model.trees)):
                parts.append(f"    votes.append(tree_{i}_predict(row))")
            parts.append("    return majority_vote(votes)\n")
            code = "\n".join(parts)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
            return out_path
        else:
            # DecisionTree único
            root = getattr(model, 'root', None)
            if root is None:
                raise ValueError("Decision tree model has no root")
            code = _export_tree_to_ifelse(root, fn_name=fn_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
            return out_path
    else:
        # sklearn: intenta convertir un único árbol de decisión en el bosque o lo aumenta
        model = entry['model']
        # Si es un RandomForestClassifier, extrae árboles mediante estimators_
        try:
            estimators = getattr(model, "estimators_", None)
            if not estimators:
                raise ValueError("sklearn model has no estimators_ (train must be called first)")
            parts = []
            parts.append("def majority_vote(votes):")
            parts.append("    agg = {}")
            parts.append("    for v in votes:")
            parts.append("        agg[v] = agg.get(v, 0) + 1")
            parts.append("    return max(agg.items(), key=lambda x: x[1])[0]\n")
            # Convierte cada árbol de sklearn en condicionales anidados usando `tree_.feature` y `threshold`; ten cuidado con los ID de los nodos.
            for i, est in enumerate(estimators):
                tree = est.tree_
                # Construye un asistente de recursión para traducir nodos; este es un exportador simple
                def sklearn_node_code(node_id, level=1):
                    pad = "    " * level
                    if tree.children_left[node_id] == tree.children_right[node_id]:
                        # leaf
                        # predecir clase: argmax del valor[node_id][0]
                        vals = tree.value[node_id][0]
                        max_idx = int(vals.argmax()) if hasattr(vals, 'argmax') else int(max(range(len(vals)), key=lambda k: vals[k]))
                        return pad + f"return {repr(max_idx)}\n"
                    feat = int(tree.feature[node_id])
                    thr = float(tree.threshold[node_id])
                    s = pad + f"if row[{feat}] <= {repr(thr)}:\n"
                    s += sklearn_node_code(tree.children_left[node_id], level + 1)
                    s += pad + "else:\n"
                    s += sklearn_node_code(tree.children_right[node_id], level + 1)
                    return s
                # función de construcción del árbol
                func_code = f"def tree_{i}_predict(row):\n"
                func_code += sklearn_node_code(0, 1)
                parts.append(func_code)
            parts.append(f"def {fn_name}(row):")
            parts.append("    votes = []")
            for i in range(len(estimators)):
                parts.append(f"    votes.append(tree_{i}_predict(row))")
            parts.append("    return majority_vote(votes)\n")
            code = "\n".join(parts)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
            return out_path
        except Exception as e:
            raise NotImplementedError("Export of sklearn forest to pure Python is experimental: " + str(e))


# -------------------------
# Comodidad: evaluar

def evaluate_accuracy(name: str, X: List[List[Any]], y_true: List[Any]) -> float:
    y_pred = predict(name, X)
    return ml_runtime.accuracy_score(y_true, y_pred)

# -------------------------
# Árbol de decisión (DecisionTree)

def train_decision_tree(name: Optional[str] = None, dataset: Optional[List[List[Any]]] = None, *,
                        model_name: Optional[str] = None,
                        max_depth: int = 10, min_size: int = 1,
                        n_features: Optional[int] = None, seed: Optional[int] = None
                        ) -> Dict[str, Any]:
    """
    Entrena un DecisionTree inteligente que puede ser de clasificación o regresión.
    Detecta automáticamente el tipo de salida del dataset.
    """
    # Soporte retrocompatible
    name = model_name or name
    if name is None:
        raise ValueError("Debe especificarse un nombre de modelo (name o model_name).")

    if dataset is None:
        raise ValueError("El dataset no puede ser None.")

    start = time.time()
    mode = available_mode()
    is_regression = _is_regression_dataset(dataset)
    meta = {'mode': mode, 'type': 'regression' if is_regression else 'classification'}

    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]

    if mode == "sklearn":
        try:
            if is_regression:
                from sklearn.tree import DecisionTreeRegressor as SKDecisionTree
            else:
                from sklearn.tree import DecisionTreeClassifier as SKDecisionTree

            model = SKDecisionTree(max_depth=max_depth, random_state=seed)
            model.fit(X, y)
            _MODEL_REGISTRY[name] = {'mode': 'sklearn', 'model': model, 'type': meta['type']}
        except Exception as e:
            raise RuntimeError(f"Error entrenando árbol sklearn: {e}")
    else:
        try:
            # Selección automática del modelo según el tipo detectado
            if is_regression:
                model = ml_runtime.DecisionTreeRegressor(max_depth=max_depth, min_size=min_size)
                meta["type"] = "regression"
            else:
                model = ml_runtime.DecisionTreeClassifier(max_depth=max_depth, min_size=min_size, n_features=n_features)
                meta["type"] = "classification"

            # Entrenamiento
            model.fit(dataset)

            # Registro en memoria
            _MODEL_REGISTRY[name] = {
                'mode': 'mini',
                'model': model,
                'type': meta['type']
            }
        except Exception as e:
            raise RuntimeError(f"Error entrenando árbol interno: {e}")

    meta['time_seconds'] = time.time() - start
    model = _MODEL_REGISTRY[name]['model']
    return {'meta': meta, 'model': model, 'model_name': name}


def save_registry(file='registry.json'):
    serializable = {k: {'mode': v['mode'], 'type': v['type']} for k, v in _MODEL_REGISTRY.items()}  # No serializa models, solo meta
    with open(file, 'w') as f:
        json.dump(serializable, f)

def load_registry(file='registry.json'):
    with open(file, 'r') as f:
        data = json.load(f)
    for k, v in data.items():
        _MODEL_REGISTRY[k] = v  # Models necesitan re-train

# -------------------------
# Evaluación general

def evaluate(y_true=None, y_pred=None, output: Optional[str] = None, metric: str = "accuracy") -> Dict[str, Any]:
    """
    Evalúa resultados de ML de forma flexible.
    Compatible con nodos ML ('ml_eval') y ml_adapter.
    - y_true: lista o variable de verdad (ground truth)
    - y_pred: lista o variable de predicciones
    - output: nombre del campo de salida (opcional)
    - metric: tipo de métrica ('accuracy', etc.)
    Devuelve un diccionario con los resultados.
    """
    try:
        if y_true is None or y_pred is None:
            raise ValueError("evaluate() requires both y_true and y_pred.")

        # Si los valores vienen como strings (nombres de variables), intenta resolverlos del registro
        if isinstance(y_true, str) and y_true in _MODEL_REGISTRY:
            y_true = _MODEL_REGISTRY[y_true]
        if isinstance(y_pred, str) and y_pred in _MODEL_REGISTRY:
            y_pred = _MODEL_REGISTRY[y_pred]

        # Convertir a listas planas si son numpy arrays
        try:
            import numpy as np
            if isinstance(y_true, np.ndarray):
                y_true = y_true.tolist()
            if isinstance(y_pred, np.ndarray):
                y_pred = y_pred.tolist()
        except ImportError:
            pass

        # Escoge métrica (por ahora solo accuracy)
        if metric == "accuracy":
            score = ml_runtime.accuracy_score(y_true, y_pred)
        else:
            raise NotImplementedError(f"Métrica '{metric}' no implementada")

        # Resultado
        result = {"metric": metric, "score": score}
        if output:
            result["output_var"] = output

        return result

    except Exception as e:
        raise RuntimeError(f"Error evaluating results: {e}")

# =========================================================
# Evaluador Inteligente: clasificación / regresión

def evaluate_ext(y_true=None, y_pred=None, metrics=None, output=None, detailed=False):
    """
    Evaluación extendida unificada para clasificación y regresión.
    - metrics: lista de métricas a calcular ["accuracy", "precision", "recall", "f1", "mae", "mse", "r2"]
    - y_true: lista real
    - y_pred: lista de predicciones (puede ser nombre de variable o directamente lista)
    - output: variable de salida opcional (solo para logging)
    - detailed: si True, devuelve diccionario detallado de métricas
    """
    import numpy as np

    # Si y_pred es una string (nombre de variable), intenta resolverla desde el registro si existe
    if isinstance(y_pred, str):
        # Busca en el registro interno de modelos o variables previas (si lo maneja)
        try:
            y_pred = globals().get(y_pred, [])
        except Exception:
            y_pred = []

    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise ValueError("y_true and y_pred must be lists or tuples.")

    if metrics is None:
        metrics = ["accuracy"]

    results = {}

    # Clasificación
    if any(m in metrics for m in ["accuracy", "precision", "recall", "f1"]):
        acc = ml_runtime.accuracy_score(y_true, y_pred)
        results["accuracy"] = acc
        try:
            import sklearn.metrics as skm
            if "precision" in metrics:
                results["precision"] = skm.precision_score(y_true, y_pred, average="macro", zero_division=0)
            if "recall" in metrics:
                results["recall"] = skm.recall_score(y_true, y_pred, average="macro", zero_division=0)
            if "f1" in metrics:
                results["f1"] = skm.f1_score(y_true, y_pred, average="macro", zero_division=0)
        except Exception:
            pass  # sklearn opcional

    # Regresión
    if any(m in metrics for m in ["mae", "mse", "r2"]):
        try:
            import sklearn.metrics as skm
            y_true_np = np.array(y_true, dtype=float)
            y_pred_np = np.array(y_pred, dtype=float)
            if "mae" in metrics:
                results["mae"] = float(skm.mean_absolute_error(y_true_np, y_pred_np))
            if "mse" in metrics:
                results["mse"] = float(skm.mean_squared_error(y_true_np, y_pred_np))
            if "r2" in metrics:
                results["r2"] = float(skm.r2_score(y_true_np, y_pred_np))
        except Exception:
            pass

    # Retorno final
    if detailed:
        return results

    # Devuelve una sola métrica si no se solicita modo detallado
    main_metric = metrics[0]
    return results.get(main_metric, None)