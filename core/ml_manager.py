"""
ML Manager — Modo Dual Intelligent
- Detecta scikit-learn y usa sklearn cuando esté disponible.
- Si sklearn no está, usa core.ml_runtime (MiniML).
- Proporciona API unificada: train, predict, save, load, export_for_embedded.

CORRECCIONES APLICADAS:
1. evaluate_ext(): Ahora devuelve dict cuando se solicitan múltiples métricas
2. Agregada variable epsilon en el scope correcto para evitar NameError
"""

from typing import Any, Dict, List, Optional
import importlib
import time
import json
import os

# Preferir el tiempo de ejecución interno
from core import ml_runtime
from core.ml_compat import normalize_tree

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
    """Devuelve el modo de ML actualmente disponible ('sklearn' o 'mini')."""
    return "sklearn" if _sklearn_available else "mini"

def clear_registry():
    """Limpia el registro de modelos en memoria."""
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
        if any(isinstance(v, float) and not v.is_integer() for v in y):
            return True
        return len(set(y)) > 10  # heurística simple
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

    Args:
        name (str, optional): Nombre para registrar el modelo. Defaults to None.
        dataset (List[List[Any]], optional): Datos de entrenamiento. Defaults to None.
        model_name (str, optional): Alias para `name`. Defaults to None.
        n_trees (int, optional): Número de árboles en el bosque. Defaults to 5.
        max_depth (int, optional): Profundidad máxima de cada árbol. Defaults to 10.
        min_size (int, optional): Tamaño mínimo de un nodo para dividir. Defaults to 1.
        sample_size (float, optional): Tamaño de la muestra para cada árbol (fracción de datos). Defaults to 1.0.
        n_features (Optional[int], optional): Número de características a considerar en cada división. Defaults to None (usa todas).
        seed (Optional[int], optional): Semilla para la generación de números aleatorios. Defaults to None.

    Returns:
        Dict[str, Any]: Un diccionario con metadatos del entrenamiento y el modelo entrenado.

    Raises:
        ValueError: Si `name` o `dataset` son None.
        RuntimeError: Si ocurre un error durante el entrenamiento del modelo.
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
                SKForest = _sklearn.RandomForestRegressor
            else:
                SKForest = _sklearn.RandomForestClassifier

            model = SKForest(
                n_estimators=n_trees,
                max_depth=max_depth,
                random_state=seed,
                n_jobs=-1  # Usa todos los núcleos disponibles
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
            
            # Validación de nodos de árbol (si el modelo tiene la estructura esperada)
            if hasattr(rf, 'trees'):
                for tree in rf.trees:
                    validate_tree_node(getattr(tree, 'root', None))
            elif hasattr(rf, 'root'):
                validate_tree_node(rf.root)

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
    Entrena un modelo lineal (MiniML).

    Args:
        model_name (str): Nombre para registrar el modelo.
        dataset (list): Datos de entrenamiento.
        learning_rate (float, optional): Tasa de aprendizaje. Defaults to 0.01.
        epochs (int, optional): Número de épocas de entrenamiento. Defaults to 1000.

    Returns:
        Dict[str, Any]: Metadatos del entrenamiento y el modelo.

    Raises:
        RuntimeError: Si ocurre un error durante el entrenamiento.
    """
    is_regression = _is_regression_dataset(dataset)

    try:
        # Actualmente solo soporta MiniML para modelos lineales
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
    """
    Entrena un clasificador SVM simple usando MiniML.

    Args:
        model_name (str): Nombre para registrar el modelo.
        dataset (list): Datos de entrenamiento.
        learning_rate (float, optional): Tasa de aprendizaje. Defaults to 0.001.
        lambda_param (float, optional): Parámetro de regularización lambda. Defaults to 0.01.
        epochs (int, optional): Número de épocas de entrenamiento. Defaults to 1000.

    Returns:
        Dict[str, Any]: Metadatos del entrenamiento y el modelo.

    Raises:
        RuntimeError: Si ocurre un error durante el entrenamiento.
    """
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
    """
    Entrena una pequeña red neuronal (MiniML).

    Args:
        model_name (str): Nombre para registrar el modelo.
        dataset (list): Datos de entrenamiento.
        hidden_size (int, optional): Tamaño de la capa oculta. Defaults to 4.
        epochs (int, optional): Número de épocas de entrenamiento. Defaults to 1000.
        lr (float, optional): Tasa de aprendizaje. Defaults to 0.01.

    Returns:
        Dict[str, Any]: Metadatos del entrenamiento y el modelo.

    Raises:
        RuntimeError: Si ocurre un error durante el entrenamiento.
    """
    
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
    Función genérica para realizar predicciones.

    Puede ser llamada de varias maneras:
      - `predict(name="nombre_modelo", X=datos)`
      - `predict(model=objeto_modelo, X=datos)`
      - `predict("nombre_modelo", datos)` (argumentos posicionales, menos recomendado)

    Args:
        name (Optional[str]): Nombre del modelo registrado a usar.
        model (Optional[Any]): Objeto del modelo a usar (si no se usa `name`).
        X (Optional[List[List[Any]]]): Datos de entrada para la predicción.

    Returns:
        List[Any]: Una lista de predicciones.

    Raises:
        KeyError: Si el `name` del modelo no se encuentra en el registro.
        ValueError: Si no se proporcionan ni `name` ni `model`, o si `X` es None.
        RuntimeError: Si el objeto del modelo no soporta el método `predict`.
    """
    # Normalizar argumentos posicionales si se pasó name como lista accidentalmente
    if X is None and isinstance(model, list) and name is None:
        # Posible llamada predict(name, X) donde name es una lista
        # Esto es inusual y probablemente un error del usuario
        pass

    # Determinar modelo real a usar
    model_obj = None
    model_entry = None # Para obtener el modo del modelo

    if model is not None:
        # Si model es un string (nombre), buscar en registry
        if isinstance(model, str):
            if model not in _MODEL_REGISTRY:
                raise KeyError(f"Model '{model}' not found")
            model_entry = _MODEL_REGISTRY[model]
            model_obj = model_entry['model']
        else:
            # Asumimos que es un objeto modelo ya instanciado (sklearn o mini)
            model_obj = model
            # Intentar inferir el modo si es posible (difícil sin info explícita)
            # Si no, asumiremos que el usuario sabe lo que hace
            if hasattr(model, 'trees') or hasattr(model, 'root'): # Heurística simple para MiniML
                model_entry = {'mode': 'mini'}
            else:
                model_entry = {'mode': 'sklearn'} # Asunción
    elif name is not None:
        # name dado: buscar en el registro
        if name not in _MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' not found")
            model_entry = _MODEL_REGISTRY[name]
            model_obj = model_entry['model']
        else:
            raise ValueError("predict requires either 'name' (registered model name) or 'model' (model object)")

    if X is None:
        raise ValueError("predict requires 'X' input (list of samples)")

    # Validación de estructura del árbol para modelos MiniML antes de predecir
    if model_entry and model_entry.get('mode') == 'mini':
        if hasattr(model_obj, 'trees'):
            for tree in model_obj.trees:
                validate_tree_node(getattr(tree, 'root', None))
        elif hasattr(model_obj, 'root'):
            validate_tree_node(model_obj.root)
        # Si es un RandomForest simple de MiniML, validamos la estructura raíz
        elif hasattr(model_obj, 'root') and isinstance(model_obj.root, dict):
            validate_tree_node(model_obj.root)
        # Si no, asumimos que es otro tipo de modelo MiniML que no requiere validación explícita de árbol aquí

    # Ejecutar predicción según tipo de modelo
    try:
        # Si el modelo es sklearn (posible), usa model.predict(X)
        # Si es un modelo MiniML, también debería tener un método predict.
        preds = model_obj.predict(X)
        
        # Convertir a lista de tipos básicos si es necesario (e.g., numpy arrays)
        try:
            if hasattr(preds, 'tolist') and callable(preds.tolist):
                return preds.tolist()
            return list(preds) # Intenta convertir a lista si no es array numpy
        except Exception:
            return preds # Devuelve tal cual si no se puede convertir
    except AttributeError:
        # Fallback: si tiene método 'predict' no encontrado, lanzar error controlado
        raise RuntimeError("Model object does not support 'predict' method")

# -------------------------
# Persistencia de modelos

def save_model(name: str, path: str) -> None:
    """
    Guarda el modelo en el disco.
    Si el modelo está en modo 'sklearn' y `joblib` está disponible, usa `joblib.dump`.
    De lo contrario, para modelos 'mini', usa una serialización personalizada basada en JSON.

    Args:
        name (str): Nombre del modelo a guardar (debe estar registrado).
        path (str): Ruta del archivo donde guardar el modelo.

    Raises:
        KeyError: Si el modelo con el nombre especificado no se encuentra registrado.
        FileNotFoundError: Si la ruta del directorio padre no existe.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found")
    entry = _MODEL_REGISTRY[name]
    
    # Asegurarse de que el directorio de destino exista
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            raise FileNotFoundError(f"Could not create directory '{dir_path}': {e}")
            
    if entry['mode'] == 'sklearn' and _joblib is not None:
        # Usar joblib para modelos sklearn
        _joblib.dump(entry['model'], path)
    else:
        # MiniML: serializar la estructura del árbol(es) en JSON
        # Para RandomForest, cada árbol se almacena como entrada (dict).
        model = entry['model']
        serial_data = {'type': 'mini_model', 'mode': entry['mode'], 'model_type': entry['type']}
        
        # Intenta obtener la representación serializable del modelo MiniML
        if hasattr(model, 'to_serializable'):
            serial_data['data'] = model.to_serializable()
        elif hasattr(model, 'root'):
            # Para árboles simples, guarda la raíz
            serial_data['root'] = model.root
        elif hasattr(model, 'trees'):
            # Para RandomForest, guarda las raíces de todos los árboles
            serial_data['trees'] = [getattr(t, 'root', None) for t in model.trees]
        else:
            # Si no hay una forma clara de serializar, guarda una representación de string (menos robusto)
            try:
                serial_data['repr'] = repr(model)
            except Exception as e:
                print(f"Warning: Could not get a good representation for model {name}: {e}")
                
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serial_data, f, indent=2)


def load_model(name: str, path: str) -> None:
    """
    Carga un modelo desde el disco y lo registra bajo el nombre `name`.
    Intenta cargar primero como un modelo sklearn usando joblib. Si falla,
    intenta cargar como un modelo MiniML usando la serialización JSON.

    Args:
        name (str): Nombre bajo el cual registrar el modelo cargado.
        path (str): Ruta del archivo del modelo a cargar.

    Raises:
        FileNotFoundError: Si el archivo del modelo no existe.
        ValueError: Si el formato del archivo del modelo es desconocido o no se puede cargar.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at '{path}'")
        
    loaded = False
    # Prueba primero con sklearn.
    if _sklearn_available and _joblib is not None:
        try:
            model = _joblib.load(path)
            # Intenta inferir el tipo
            model_type = 'unknown'
            if hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'tree_'): # RandomForest
                model_type = 'random_forest'
            elif hasattr(model, 'tree_'): # DecisionTree
                model_type = 'decision_tree'
            elif hasattr(model, '__module__') and 'sklearn' in model.__module__: # Generic sklearn
                model_type = model.__module__.split('.')[-1] # Intentar inferir de su módulo

            _MODEL_REGISTRY[name] = {'mode': 'sklearn', 'model': model, 'type': model_type}
            print(f"Model '{name}' loaded as sklearn.")
            loaded = True
        except Exception as e:
            # Si falla la carga con joblib, puede ser un modelo MiniML. Continuar con el fallback.
            print(f"Info: Failed to load '{path}' as sklearn model, attempting MiniML load. Error: {e}")

    if not loaded:
        # Fallback para el minicargador
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Normalizar la estructura de datos si es necesario (especialmente para árboles)
            data = normalize_tree(data)

            if data.get('type') == 'mini_model':
                mode = data.get('mode', 'mini') # Asume 'mini' si no se especifica
                model_type = data.get('model_type', 'unknown')
                
                if 'data' in data:
                    # Si el modelo tiene un método to_serializable, úsalo
                    if mode == 'mini':
                        # Crea una instancia genérica y llama a un método de carga
                        if model_type == 'neural_net':
                            # Supone que MiniNeuralNetwork tiene un método `from_serializable`
                            if hasattr(ml_runtime.MiniNeuralNetwork, 'from_serializable'):
                                model = ml_runtime.MiniNeuralNetwork.from_serializable(data['data'])
                            else:
                                raise NotImplementedError("MiniNeuralNetwork.from_serializable not implemented.")
                        elif model_type == 'linear':
                            if hasattr(ml_runtime.MiniLinearModel, 'from_serializable'):
                                model = ml_runtime.MiniLinearModel.from_serializable(data['data'])
                            else:
                                raise NotImplementedError("MiniLinearModel.from_serializable not implemented.")
                        elif model_type == 'svm':
                             if hasattr(ml_runtime.MiniSVM, 'from_serializable'):
                                model = ml_runtime.MiniSVM.from_serializable(data['data'])
                             else:
                                raise NotImplementedError("MiniSVM.from_serializable not implemented.")
                        else:
                             raise ValueError(f"Unsupported MiniML model type for loading from 'data': {model_type}")
                    else: # Para otros modos si se añaden en el futuro
                        raise NotImplementedError(f"Loading for mode '{mode}' not implemented yet.")
                elif 'root' in data:
                    # Reconstruir un DecisionTree simple
                    if mode == 'mini':
                        dt = ml_runtime.DecisionTreeClassifier() # o Regressor, debería inferirse del tipo guardado
                        dt.root = data['root']
                        model = dt
                    else:
                        raise NotImplementedError(f"Loading for mode '{mode}' with 'root' not implemented yet.")
                elif 'trees' in data and data['trees'] is not None:
                    # Reconstruir un RandomForest simple
                    if mode == 'mini':
                        rf = ml_runtime.RandomForestClassifier(n_trees=len(data['trees'])) # n_trees es una suposición aquí
                        rf.trees = []
                        for root_data in data['trees']:
                            dt = ml_runtime.DecisionTreeClassifier() # o Regressor
                            dt.root = root_data
                            rf.trees.append(dt)
                        model = rf
                    else:
                        raise NotImplementedError(f"Loading for mode '{mode}' with 'trees' not implemented yet.")
                else:
                    raise ValueError("MiniML model data is missing 'data', 'root', or 'trees' key.")
                
                # Registrar el modelo MiniML cargado
                _MODEL_REGISTRY[name] = {'mode': mode, 'model': model, 'type': model_type}
                print(f"Model '{name}' loaded as MiniML ({model_type}).")
                loaded = True

                # Validación adicional de nodos de árbol si el modelo es visible
                if hasattr(model, 'trees'):
                    for tree in model.trees:
                        validate_tree_node(getattr(tree, 'root', None))
                elif hasattr(model, 'root'):
                    validate_tree_node(model.root)

            else:
                raise ValueError("Unknown model format: Expected 'type': 'mini_model'")
                
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from '{path}'. Is it a valid JSON file?")
        except Exception as e:
            raise ValueError(f"Failed to load MiniML model from '{path}'. Error: {e}")

    if not loaded:
        raise ValueError("Model could not be loaded using either sklearn (joblib) or MiniML (JSON) methods.")


# -------------------------
# Exportador a código embebido

def _export_tree_to_ifelse(root: dict, fn_name: str = "predict_row", indent: str = "    ") -> str:
    """
    Convierte recursivamente la raíz de un árbol (un diccionario de nodos con 'index','value','left','right' o terminal) en una función if/else Python.
    Se espera una estructura de nodo compatible con la salida de ml_runtime.get_split/build_tree.

    Args:
        root (dict): El nodo raíz del árbol a exportar.
        fn_name (str, optional): Nombre de la función Python a generar. Defaults to "predict_row".
        indent (str, optional): Cadena de indentación a usar. Defaults to "    ".

    Returns:
        str: Código Python que define la función de predicción.

    Raises:
        ValueError: Si el nodo de hoja tiene un tipo no soportado, o si un nodo interno está malformado.
    """
    def node_to_code(node, level=1):
        pad = indent * level
        if not isinstance(node, dict):
            # Terminal node (leaf)
            # Aseguramos que el valor sea representable como literal Python
            if isinstance(node, (int, float, str, bool)):
                return pad + f"return {repr(node)}\n\n"
            else:
                raise ValueError(f"Leaf node has unsupported type {type(node).__name__}. Expected int, float, or str.")
        
        # Internal node
        if 'index' not in node or not isinstance(node['index'], int):
            raise ValueError(f"Internal node is missing 'index' or 'index' is not an integer.")
        if 'value' not in node:
            raise ValueError("Internal node is missing 'value'.")
        if 'left' not in node or 'right' not in node:
            raise ValueError("Internal node is missing 'left' or 'right' child.")
            
        idx = node['index']
        val = node['value']
        
        # Handle cases where value might not be directly comparable or needs special formatting
        # For simplicity, assuming basic types that work with <=
        
        code = pad + f"if row[{idx}] <= {repr(val)}:\n"
        code += node_to_code(node['left'], level + 1)
        code += pad + "else:\n"
        code += node_to_code(node['right'], level + 1)
        return code

    func = f"def {fn_name}(row):\n"
    func += node_to_code(root, 1)
    return func

def export_model_to_python_function(name: str, out_path: str, fn_name: str = "predict_row") -> str:
    """
    Exporta un modelo registrado a un archivo Python independiente que define una función de predicción.
    Actualmente soporta modelos de modo 'mini' (árboles de decisión y Random Forests).
    Para sklearn, la exportación es experimental y puede no soportar todos los tipos de modelos o complejidades.

    Args:
        name (str): Nombre del modelo registrado a exportar.
        out_path (str): Ruta del archivo Python de salida.
        fn_name (str, optional): Nombre de la función de predicción a generar en el archivo de salida. Defaults to "predict_row".

    Returns:
        str: La ruta al archivo Python generado.

    Raises:
        KeyError: Si el modelo con el nombre especificado no se encuentra registrado.
        ValueError: Si el modelo no es de un tipo soportado para exportación o si la estructura del árbol es inválida.
        NotImplementedError: Si la exportación para el tipo de modelo o modo particular no está implementada.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found")
    entry = _MODEL_REGISTRY[name]
    
    code_parts = []

    if entry['mode'] == 'mini':
        model = entry['model']
        # Si se trata de RandomForest, creamos un llamador de conjunto que vota
        if hasattr(model, 'trees') and isinstance(model, (ml_runtime.RandomForestClassifier, ml_runtime.RandomForestRegressor)):
            code_parts.append("def majority_vote(votes):")
            code_parts.append("    agg = {}")
            code_parts.append("    for v in votes:")
            code_parts.append("        agg[v] = agg.get(v, 0) + 1")
            code_parts.append("    return max(agg.items(), key=lambda x: x[1])[0]\n")
            
            # genera funciones para cada árbol
            for i, t in enumerate(model.trees):
                root = getattr(t, 'root', None)
                if root is None:
                    raise ValueError(f"Tree {i} in RandomForest '{name}' has no root node.")
                code_parts.append(_export_tree_to_ifelse(root, fn_name=f"tree_{i}_predict"))
            
            # función de conjunto
            code_parts.append(f"def {fn_name}(row):")
            code_parts.append("    votes = []")
            for i in range(len(model.trees)):
                code_parts.append(f"    votes.append(tree_{i}_predict(row))")
            code_parts.append("    return majority_vote(votes)\n")
        
        elif hasattr(model, 'root') and isinstance(model, (ml_runtime.DecisionTreeClassifier, ml_runtime.DecisionTreeRegressor)):
            # DecisionTree único
            root = getattr(model, 'root', None)
            if root is None:
                raise ValueError(f"Decision tree model '{name}' has no root node.")
            code_parts.append(_export_tree_to_ifelse(root, fn_name=fn_name))
        else:
            raise NotImplementedError(f"Exporting MiniML model of type {type(model).__name__} is not supported yet.")

    elif entry['mode'] == 'sklearn':
        try:
            # sklearn: intenta convertir un único árbol de decisión en el bosque o lo aumenta
            model = entry['model']
            
            # Si es un RandomForestClassifier/Regressor, extrae árboles mediante estimators_
            if hasattr(model, 'estimators_'):
                estimators = model.estimators_
                if not estimators:
                    raise ValueError("Sklearn RandomForest model has no 'estimators_' attribute (training might not be complete).")
                
                code_parts.append("def majority_vote(votes):")
                code_parts.append("    agg = {}")
                code_parts.append("    for v in votes:")
                code_parts.append("        agg[v] = agg.get(v, 0) + 1")
                code_parts.append("    return max(agg.items(), key=lambda x: x[1])[0]\n")
                
                # Convierte cada árbol de sklearn en condicionales anidados usando `tree_.feature` y `threshold`; ten cuidado con los ID de los nodos.
                for i, est in enumerate(estimators):
                    tree = est.tree_
                    # Construye un asistente de recursión para traducir nodos; este es un exportador simple
                    def sklearn_node_code(node_id, level=1):
                        pad = "    " * level
                        try:
                            # Check if it's a leaf node
                            if tree.children_left[node_id] == tree.children_right[node_id]:
                                # Leaf node: return the predicted class (argmax of tree.value[node_id][0])
                                vals = tree.value[node_id][0]
                                # Ensure vals is iterable and has an argmax method or can be processed
                                # Prioritize argmax if available (e.g., numpy arrays)
                                if hasattr(vals, 'argmax'):
                                    max_idx = int(vals.argmax())
                                else:
                                    # Fallback for lists/tuples
                                    max_idx = int(max(range(len(vals)), key=lambda k: vals[k]))
                                return pad + f"return {repr(max_idx)}\n"
                            
                            # Internal node
                            feat = int(tree.feature[node_id])
                            thr = float(tree.threshold[node_id])
                            
                            s = pad + f"if row[{feat}] <= {repr(thr)}:\n"
                            s += sklearn_node_code(tree.children_left[node_id], level + 1)
                            s += pad + "else:\n"
                            s += sklearn_node_code(tree.children_right[node_id], level + 1)
                            return s
                        except Exception as e:
                            raise ValueError(f"Could not process node {node_id}: {e}")
                    
                    # función de construcción del árbol
                    func_code = f"def tree_{i}_predict(row):\n"
                    try:
                        func_code += sklearn_node_code(0, 1) # Start from root node (ID 0)
                        code_parts.append(func_code)
                    except ValueError as e:
                        raise ValueError(f"Error exporting sklearn tree {i}: {e}")

                # función de conjunto para RandomForest
                code_parts.append(f"def {fn_name}(row):")
                code_parts.append("    votes = []")
                for i in range(len(estimators)):
                    code_parts.append(f"    votes.append(tree_{i}_predict(row))")
                code_parts.append("    return majority_vote(votes)\n")
            
            # Si es un único DecisionTreeClassifier/Regressor
            elif hasattr(model, 'tree_'):
                tree = model.tree_
                def sklearn_node_code(node_id, level=1):
                    pad = "    " * level
                    try:
                        if tree.children_left[node_id] == tree.children_right[node_id]:
                            vals = tree.value[node_id][0]
                            if hasattr(vals, 'argmax'):
                                max_idx = int(vals.argmax())
                            else:
                                max_idx = int(max(range(len(vals)), key=lambda k: vals[k]))
                            return pad + f"return {repr(max_idx)}\n"

                        feat = int(tree.feature[node_id])
                        thr = float(tree.threshold[node_id])
                        
                        s = pad + f"if row[{feat}] <= {repr(thr)}:\n"
                        s += sklearn_node_code(tree.children_left[node_id], level + 1)
                        s += pad + "else:\n"
                        s += sklearn_node_code(tree.children_right[node_id], level + 1)
                        return s
                    except Exception as e:
                        raise ValueError(f"Could not process node {node_id}: {e}")
                
                code = f"def {fn_name}(row):\n"
                try:
                    code += sklearn_node_code(0, 1)
                    code_parts.append(code)
                except ValueError as e:
                    raise ValueError(f"Error exporting sklearn DecisionTree: {e}")
            else:
                raise NotImplementedError("Exporting this type of sklearn model to pure Python is not supported yet.")
        except Exception as e:
            raise RuntimeError(f"Export for sklearn model failed: {e}")

    # Escribir el código al archivo de salida
    full_code = "\n".join(code_parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_code)
        
    return out_path

# -------------------------
# Comodidad: evaluar

def evaluate_accuracy(name: str, X: List[List[Any]], y_true: List[Any]) -> float:
    """
    Evalúa la precisión de un modelo registrado dado un conjunto de datos.

    Args:
        name (str): Nombre del modelo registrado a evaluar.
        X (List[List[Any]]): Datos de entrada.
        y_true (List[Any]): Valores verdaderos (ground truth).

    Returns:
        float: La puntuación de precisión del modelo.

    Raises:
        KeyError: Si el modelo con el nombre especificado no se encuentra registrado.
    """
    y_pred = predict(name, X=X)
    # Usar la implementación interna de ML Runtime para precisión
    return ml_runtime.accuracy_score(y_true, y_pred)

# -------------------------
# Árbol de decisión (DecisionTree)

def train_decision_tree(name: Optional[str] = None, dataset: Optional[List[List[Any]]] = None, *,
                        model_name: Optional[str] = None,
                        max_depth: int = 10, min_size: int = 1,
                        n_features: Optional[int] = None, seed: Optional[int] = None
                        ) -> Dict[str, Any]:
    """
    Entrena un árbol de decisión inteligente que puede ser de clasificación o regresión.
    Detecta automáticamente el tipo de salida del dataset.

    Args:
        name (str, optional): Nombre para registrar el modelo. Defaults to None.
        dataset (List[List[Any]], optional): Datos de entrenamiento. Defaults to None.
        model_name (str, optional): Alias para `name`. Defaults to None.
        max_depth (int, optional): Profundidad máxima del árbol. Defaults to 10.
        min_size (int, optional): Tamaño mínimo de un nodo para dividir. Defaults to 1.
        n_features (Optional[int], optional): Número de características a considerar en cada división. Defaults to None.
        seed (Optional[int], optional): Semilla para la generación de números aleatorios. Defaults to None.

    Returns:
        Dict[str, Any]: Metadatos del entrenamiento y el modelo.

    Raises:
        ValueError: Si `name` o `dataset` son None.
        RuntimeError: Si ocurre un error durante el entrenamiento del modelo.
    """
    # Soporte retrocompatible para `model_name`
    name = model_name or name
    if name is None:
        raise ValueError("Debe especificarse un nombre de modelo (name o model_name).")

    if dataset is None:
        raise ValueError("El dataset no puede ser None.")

    start = time.time()
    mode = available_mode()
    is_regression = _is_regression_dataset(dataset)
    meta = {'mode': mode, 'type': 'regression' if is_regression else 'classification'}

    X = [row[:-1] for row in dataset] # Características
    y = [row[-1] for row in dataset]  # Target

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

            # Validación de la estructura del árbol
            if hasattr(model, 'trees'): # RandomForest
                for tree in model.trees:
                    validate_tree_node(getattr(tree, 'root', None))
            elif hasattr(model, 'root'): # DecisionTree
                validate_tree_node(model.root)

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


# -------------------------
def save_registry(file='registry.json'):
    """
    Guarda metadatos de los modelos registrados en un archivo JSON.
    No guarda los objetos de los modelos, solo su nombre, modo y tipo.

    Args:
        file (str, optional): Ruta del archivo JSON a crear. Defaults to 'registry.json'.
    """
    # Serializable solo contiene metadatos (nombre, modo, tipo)
    serializable = {k: {'mode': v['mode'], 'type': v.get('type', 'unknown')} for k, v in _MODEL_REGISTRY.items()}
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)

def load_registry(file='registry.json'):
    """
    Carga metadatos de modelos registrados desde un archivo JSON.
    Los modelos cargados no contendrán el objeto del modelo entrenado;
    deberán ser re-entrenados o cargados individualmente.

    Args:
        file (str, optional): Ruta del archivo JSON a leer. Defaults to 'registry.json'.
    """
    if not os.path.exists(file):
        print(f"Warning: Registry file '{file}' not found. Skipping load.")
        return
        
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for k, v in data.items():
        # Aquí solo guardamos metadatos. El objeto real del modelo debe ser cargado o re-entrenado.
        _MODEL_REGISTRY[k] = v
    print(f"Loaded registry from '{file}'. Models: {list_models()}.")


# -------------------------
def validate_tree_node(node: Any) -> None:
    """
    Valida recursivamente la estructura de un nodo de árbol (dict) para asegurar que sea correcta.
    Se utiliza para verificar la integridad de los árboles de decisión y random forests.

    - Nodos internos: deben tener 'index' (int), 'value' (int/float), 'left' y 'right' (nodos válidos).
    - Nodos hoja: no dict, debe ser int/float/str (clase o valor de regresión).

    Args:
        node (Any): El nodo a validar.

    Raises:
        ValueError: Si la estructura del nodo es inválida o los tipos de datos no son los esperados.
    """
    if not isinstance(node, dict):
        # Hoja: permite valores simples (clases o predicciones)
        if not isinstance(node, (int, float, str, bool)): # Permitir bool como clase también
            raise ValueError(f"Invalid leaf node: type {type(node).__name__} not supported (expected int, float, str, or bool).")
        return
    
    # Nodo interno
    if 'index' not in node or not isinstance(node['index'], int):
        raise ValueError("Invalid internal node: missing 'index' or 'index' is not an integer.")
    
    if 'value' not in node or not isinstance(node['value'], (int, float)):
        raise ValueError(f"Invalid internal node: 'value' must be int or float, but found {type(node['value']).__name__}.")
    
    if 'left' not in node or 'right' not in node:
        raise ValueError("Invalid internal node: missing 'left' or 'right' child.")
    
    # Validación recursiva de hijos
    validate_tree_node(node['left'])
    validate_tree_node(node['right'])

# -------------------------
# Evaluación general

def evaluate(y_true=None, y_pred=None, output: Optional[str] = None, metrics: str = "accuracy") -> Dict[str, Any]:
    """
    Evalúa resultados de ML de forma flexible. Compatible con nodos ML ('ml_eval') y ml_adapter.
    Puede aceptar `y_true` y `y_pred` directamente o como nombres de variables registradas.

    Args:
        y_true (list | str, optional): Valores reales o nombre de variable registrada. Defaults to None.
        y_pred (list | str, optional): Valores predichos o nombre de variable registrada. Defaults to None.
        output (Optional[str], optional): Nombre del campo de salida en el diccionario de resultados. Defaults to None.
        metrics (str, optional): Tipo de métrica ('accuracy', etc.). Defaults to "accuracy".

    Returns:
        Dict[str, Any]: Un diccionario con los resultados de la evaluación.

    Raises:
        ValueError: Si faltan `y_true` o `y_pred`, o si la métrica no está implementada.
        RuntimeError: Si ocurre un error durante la evaluación.
    """
    try:
        if y_true is None or y_pred is None:
            raise ValueError("evaluate() requires both y_true and y_pred.")

        # Si los valores vienen como strings (nombres de variables), intenta resolverlos del registro
        if isinstance(y_true, str):
            if y_true in _MODEL_REGISTRY:
                # Si es un nombre de modelo registrado, asumimos que queremos sus predicciones (esto es ambiguo)
                # Para simplicidad, aquí asumimos que y_true es la verdad, no un modelo.
                # Si se quisiera usar las predicciones de un modelo guardado, se haría predict() primero.
                pass # No hacemos nada si es un nombre de registro, esperamos que sea el valor literal
            else:
                 # Intentar buscar la variable en el scope global (esto es un poco arriesgado)
                 try:
                     y_true = globals()[y_true]
                 except KeyError:
                     raise ValueError(f"Variable '{y_true}' for y_true not found in global scope.")

        if isinstance(y_pred, str):
            if y_pred in _MODEL_REGISTRY:
                # Si es un nombre de modelo, se espera que se use predict() antes.
                # Si no, se intenta resolverlo como variable global.
                try:
                    y_pred = globals()[y_pred]
                except KeyError:
                    raise ValueError(f"Variable '{y_pred}' for y_pred not found in global scope.")
            else:
                 try:
                     y_pred = globals()[y_pred]
                 except KeyError:
                     raise ValueError(f"Variable '{y_pred}' for y_pred not found in global scope.")


        # Conversión universal de arrays a listas (sin dependencias externas)
        # Si se detecta un objeto con tolist(), se asume que es compatible (e.g. numpy)
        if hasattr(y_true, "tolist") and callable(y_true.tolist):
            y_true = y_true.tolist()
        if hasattr(y_pred, "tolist") and callable(y_pred.tolist):
            y_pred = y_pred.tolist()

        # Asegurarse de que ambos sean listas simples para el procesamiento
        if not isinstance(y_true, list): y_true = list(y_true)
        if not isinstance(y_pred, list): y_pred = list(y_pred)

        # Manejo flexible de metrics (alineado con evaluate_ext)
        # Si metrics es una lista de strings, tomamos el primer elemento.
        # evaluate_ext es para múltiples métricas.
        if isinstance(metrics, (list, tuple)):
            if len(metrics) > 1:
                print("Warning: evaluate() only supports one metric. Using the first one provided.")
            metrics = metrics[0] if metrics else "accuracy"
        elif isinstance(metrics, str) and metrics.startswith("[") and metrics.endswith("]"):
            # Parsear si es una cadena que representa una lista (e.g., "['accuracy']")
            try:
                import ast
                parsed = ast.literal_eval(metrics)
                if isinstance(parsed, list) and len(parsed) == 1:
                    metrics = parsed[0]
                else:
                    metrics = "accuracy"  # Fallback
            except Exception:
                metrics = "accuracy"  # Fallback si parse falla

        # Escoge métrica (ahora metrics es siempre un string simple)
        if metrics == "accuracy":
            score = ml_runtime.accuracy_score(y_true, y_pred)
        elif metrics == "mse":
            score = ml_runtime.mse(y_true, y_pred)
        elif metrics == "mae":
            score = ml_runtime.mae(y_true, y_pred)
        elif metrics == "r2":
            score = ml_runtime.r2_score(y_true, y_pred)
        else:
            raise NotImplementedError(f"Metric '{metrics}' not implemented in evaluate(). Use evaluate_ext for more options.")
        
        # Resultado
        result = {"metrics": metrics, "score": score}
        if output:
            result["output_var"] = output

        return result

    except Exception as e:
        raise RuntimeError(f"Error evaluating results: {e}")

# -------------------------
# Evaluador Inteligente: clasificación / regresión

def evaluate_ext(y_true=None, y_pred=None, metrics=None, output=None, detailed=False):
    """
    Evaluación extendida unificada para clasificación y regresión (sin dependencias externas).
    Compatible con entornos embebidos (firmware tipo MegaPi / Arduino).
    Puede aceptar `y_pred` como lista o como nombre de variable registrada.

    CORRECCIÓN APLICADA:
    - Si se solicitan múltiples métricas (len(metrics) > 1), SIEMPRE devuelve un diccionario
    - Si detailed=True, siempre devuelve un diccionario (incluso con una sola métrica)
    - Si una sola métrica y detailed=False, devuelve solo el valor float

    Args:
        y_true (list): Valores reales.
        y_pred (list | str): Valores predichos o nombre de variable registrada.
        metrics (list | str, optional): Lista de métricas a calcular ("accuracy", "precision", "recall", "f1", "mae", "mse", "r2").
                                        Puede ser un string simple o una lista. Defaults to ["accuracy"].
        output (str, optional): Variable de salida opcional (para logs). Defaults to None.
        detailed (bool, optional): Si True, devuelve un diccionario completo con todas las métricas calculadas.
                                    Si False, devuelve solo el valor de la primera métrica solicitada (si es una sola métrica).

    Returns:
        float | dict: Resultado principal (si detailed=False y una sola métrica) o diccionario de métricas.

    Raises:
        ValueError: Si `y_true` o `y_pred` no son listas/tuplas, o si `y_pred` es un nombre de variable
                    que no se puede resolver.
        NotImplementedError: Si se solicitan métricas no implementadas.
    """
    # CORRECCIÓN: Definir epsilon al inicio de la función
    epsilon = 1e-10
    
    # Resolución de nombre si y_pred es un string
    if isinstance(y_pred, str):
        try:
            # Intentar buscar la variable en el scope global
            y_pred = globals()[y_pred]
        except KeyError:
            # Si no se encuentra en globals, intentar buscar en el scope local de esta función (menos probable)
            # O simplemente lanzar un error si no es una variable conocida.
            raise ValueError(f"Variable '{y_pred}' for y_pred not found in global scope.")

    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise ValueError("y_true and y_pred must be lists or tuples.")
        
    # Asegurarse de que ambos sean listas simples para el procesamiento
    if not isinstance(y_true, list): y_true = list(y_true)
    if not isinstance(y_pred, list): y_pred = list(y_pred)

    # Normalización de la entrada de `metrics`
    if metrics is None:
        metrics = ["accuracy"]
    elif isinstance(metrics, str):
        metrics = [metrics] # Convertir a lista si es un solo string
    elif isinstance(metrics, (list, tuple)) and len(metrics) == 1:
        # Si es una lista con un solo elemento que es otra lista/tupla o string con formato de lista
        if isinstance(metrics[0], (list, tuple)):
            metrics = list(metrics[0])
        elif isinstance(metrics[0], str) and metrics[0].startswith("["):
            try:
                import ast
                parsed = ast.literal_eval(metrics[0])
                if isinstance(parsed, list):
                    metrics = parsed
                else:
                    metrics = ["accuracy"] # Fallback
            except Exception:
                metrics = ["accuracy"] # Fallback si parse falla
    
    # Lista para almacenar resultados
    results = {}

    # Cálculo de Métricas

    # Métricas de Clasificación
    classification_metrics = ["accuracy", "precision", "recall", "f1"]
    if any(m in metrics for m in classification_metrics):
        acc = ml_runtime.accuracy_score(y_true, y_pred)
        results["accuracy"] = acc

        # Calcular TP, FP, FN, TN para métricas de clasificación binaria (asumiendo clases 0 y 1)
        tp = fp = fn = tn = 0
        # Para manejar clases arbitrarias, se necesitaría un enfoque más complejo
        # Aquí se asume un escenario binario simple para este ejemplo.
        # Si las clases no son 0/1, estas métricas podrían no ser adecuadas.
        try:
            for yt, yp in zip(y_true, y_pred):
                # Asumiendo clases binarias {0, 1}
                if yt == 1 and yp == 1: tp += 1
                elif yt == 0 and yp == 1: fp += 1
                elif yt == 1 and yp == 0: fn += 1
                elif yt == 0 and yp == 0: tn += 1
            
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)

            if "precision" in metrics: results["precision"] = precision
            if "recall" in metrics: results["recall"] = recall
            if "f1" in metrics: results["f1"] = f1
        except Exception as e:
            print(f"Warning: Could not calculate classification metrics (precision, recall, f1). Ensure binary classes (0/1) or adapt logic: {e}")

    # Métricas de Regresión
    regression_metrics = ["mae", "mse", "r2"]
    if any(m in metrics for m in regression_metrics):
        n = len(y_true)
        if n == 0:
            print("Warning: Cannot calculate regression metrics for empty dataset.")
            if "mae" in metrics: results["mae"] = 0.0
            if "mse" in metrics: results["mse"] = 0.0
            if "r2" in metrics: results["r2"] = 0.0
        else:
            try:
                # Convertir a float para asegurar compatibilidad
                y_true_f = [float(yt) for yt in y_true]
                y_pred_f = [float(yp) for yp in y_pred]

                diffs = [(yt - yp) for yt, yp in zip(y_true_f, y_pred_f)]
                abs_err = [abs(d) for d in diffs]
                sq_err = [d * d for d in diffs]

                mae = sum(abs_err) / n
                mse = sum(sq_err) / n
                
                # Cálculo de R2
                mean_true = sum(y_true_f) / n
                ss_tot = sum((yt - mean_true) ** 2 for yt in y_true_f)
                ss_res = sum(sq_err) # sum of squared residuals is sum of squared errors

                # Evitar división por cero para ss_tot
                r2 = 1 - (ss_res / (ss_tot + epsilon)) if ss_tot > epsilon else 1.0

                if "mae" in metrics: results["mae"] = mae
                if "mse" in metrics: results["mse"] = mse
                if "r2" in metrics: results["r2"] = r2
            except ValueError as e:
                raise ValueError(f"Non-numeric values found in y_true or y_pred for regression metrics: {e}")
            except Exception as e:
                print(f"Warning: Error calculating regression metrics: {e}")

    # CORRECCIÓN: Si se solicitan múltiples métricas, devolver siempre un diccionario
    if len(metrics) > 1 or detailed:
        # Devolver todas las métricas calculadas que fueron solicitadas
        final_results = {}
        for m in metrics:
            if m in results:
                final_results[m] = results[m]
            else:
                # Si una métrica solicitada no se pudo calcular o no está en results
                final_results[m] = None
        return final_results
    else:
        # Devolver solo la primera métrica solicitada (comportamiento original para una sola métrica)
        main_metric = metrics[0]
        return results.get(main_metric, None)