"""
Adaptador universal para la ejecución de estructuras ML y puente de traducción.
Conserva compatibilidad total con módulos de clasificación y regresión ya probados.
"""

from typing import Any, Dict, List, Callable
import json
import traceback
#import time

# Dependencias del ecosistema ML
from core import ml_manager

# Registro local de modelos
_models: Dict[str, Any] = {}

# =====================================================
# Helpers internos

def _load_csv_to_list(path: str) -> List[List[Any]]:
    import csv
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [list(map(_coerce_num, row)) for row in reader]

def _coerce_num(x: str):
    try:
        return float(x)
    except Exception:
        return x

def _ensure_dataset(struct: Dict[str, Any], context: Dict[str, Any]):
    """Registra datasets en el contexto actual."""
    name = struct.get('name', 'data')
    src = struct.get('source', 'inline')
    data = struct.get('data')
    path = struct.get('path')
    if src == 'file' and path:
        data = _load_csv_to_list(path)
    if not data:
        raise ValueError(f"Dataset vacío o no válido en {name}")
    context.setdefault('datasets', {})[name] = data
    return data

def _coerce_json_if_str(value):
    """Convierte JSON en lista si viene como string."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value

# =====================================================
# Handlers de acciones ML

def _train_dt(struct: Dict[str, Any], context: Dict[str, Any]):
    """Entrena Decision Tree con soporte dual (clasificación y regresión)."""
    ds_name = struct.get('dataset', 'data')
    if 'datasets' not in context or ds_name not in context['datasets']:
        raise ValueError(f"Dataset '{ds_name}' no encontrado")

    dataset = context['datasets'][ds_name]
    model_name = struct.get('model_name', 'dt_model')
    max_depth = struct.get('max_depth', 10)
    min_size = struct.get('min_size', 1)
    n_features = struct.get('n_features', None)

    # Mantiene compatibilidad total con ml_manager
    meta = ml_manager.train_decision_tree(
        name=model_name,
        dataset=dataset,
        max_depth=max_depth,
        min_size=min_size,
        n_features=n_features
    )

    # Sincroniza modelo local para predict posteriores
    try:
        entry = ml_manager._MODEL_REGISTRY.get(model_name)
        if entry:
            _models[model_name] = entry.get('model')
    except Exception:
        _models[model_name] = None

    return {'status': 'trained', 'model': model_name, 'meta': meta}


def _train_rf(struct: Dict[str, Any], context: Dict[str, Any]):
    """Entrena Random Forest con soporte dual."""
    ds_name = struct.get('dataset', 'data')
    if 'datasets' not in context or ds_name not in context['datasets']:
        raise ValueError(f"Dataset '{ds_name}' no encontrado")

    dataset = context['datasets'][ds_name]
    model_name = struct.get('model_name', 'rf_model')
    n_trees = struct.get('n_trees', 5)
    max_depth = struct.get('max_depth', 10)
    min_size = struct.get('min_size', 1)
    sample_size = struct.get('sample_size', 1.0)
    n_features = struct.get('n_features', None)
    seed = struct.get('seed', None)

    meta = ml_manager.train_random_forest(
        name=model_name,
        dataset=dataset,
        n_trees=n_trees,
        max_depth=max_depth,
        min_size=min_size,
        sample_size=sample_size,
        n_features=n_features,
        seed=seed
    )

    try:
        entry = ml_manager._MODEL_REGISTRY.get(model_name)
        if entry:
            _models[model_name] = entry.get('model')
    except Exception:
        _models[model_name] = None

    return {'status': 'trained', 'model': model_name, 'meta': meta}


def _predict(struct: Dict[str, Any], context: Dict[str, Any]):
    """Ejecuta predicción con modelo ya entrenado."""
    model_name = struct.get('model', 'model')
    model = _models.get(model_name)

    if model is None:
        try:
            entry = ml_manager._MODEL_REGISTRY.get(model_name)
            if entry:
                model = entry.get('model')
                _models[model_name] = model
        except Exception:
            pass

    if model is None:
        raise ValueError(f"Modelo '{model_name}' no encontrado para predicción")

    X = _coerce_json_if_str(struct.get('X'))
    preds = model.predict(X)
    output_var = struct.get('output', 'preds')
    context.setdefault('outputs', {})[output_var] = preds
    return {'status': 'predicted', 'model': model_name, 'output_var': output_var, 'preds': preds}


def _eval(struct: Dict[str, Any], context: Dict[str, Any]):
    """Evalúa métricas del modelo (usa ml_manager.evaluate_ext si está disponible)."""
    y_true = _coerce_json_if_str(struct.get('y_true'))
    y_pred = struct.get('y_pred')
    if isinstance(y_pred, str) and y_pred in context.get('outputs', {}):
        y_pred = context['outputs'][y_pred]

    metrics = struct.get('metrics', ['accuracy'])
    output_var = struct.get('output', 'results')

    # Compatibilidad extendida: usa evaluate_ext si existe
    if hasattr(ml_manager, 'evaluate_ext'):
        result = ml_manager.evaluate_ext(
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
            output=output_var,
            detailed=False
        )
    else:
        result = ml_manager.evaluate(y_true=y_true, y_pred=y_pred, output=output_var)

    context.setdefault('outputs', {})[output_var] = result
    return {'status': 'evaluated', 'metrics': metrics, 'output_var': output_var, 'result': result}

def _train_linear(struct: Dict[str, Any], context: Dict[str, Any]):
    ds_name = struct.get('dataset', 'data')
    if 'datasets' not in context or ds_name not in context['datasets']:
        raise ValueError(f"Dataset '{ds_name}' no encontrado")

    dataset = context['datasets'][ds_name]
    model_name = struct.get('model_name', 'linear_model')
    lr = struct.get('learning_rate', 0.01)
    epochs = struct.get('epochs', 1000)

    meta = ml_manager.train_linear_model(
        model_name=model_name,
        dataset=dataset,
        learning_rate=lr,
        epochs=epochs
    )

    try:
        entry = ml_manager._MODEL_REGISTRY.get(model_name)
        if entry:
            _models[model_name] = entry.get('model')
    except Exception:
        _models[model_name] = None

    return {'status': 'trained', 'model': model_name, 'meta': meta}

def _train_svm(struct: Dict[str, Any], context: Dict[str, Any]):
    ds_name = struct.get('dataset', 'data')
    if 'datasets' not in context or ds_name not in context['datasets']:
        raise ValueError(f"Dataset '{ds_name}' no encontrado")

    dataset = context['datasets'][ds_name]
    model_name = struct.get('model_name', 'svm_model')
    lr = struct.get('learning_rate', 0.001)
    lambda_param = struct.get('lambda_param', 0.01)
    epochs = struct.get('epochs', 1000)

    meta = ml_manager.train_svm(
        model_name=model_name,
        dataset=dataset,
        learning_rate=lr,
        lambda_param=lambda_param,
        epochs=epochs
    )

    try:
        entry = ml_manager._MODEL_REGISTRY.get(model_name)
        if entry:
            _models[model_name] = entry.get('model')
    except Exception:
        _models[model_name] = None

    return {'status': 'trained', 'model': model_name, 'meta': meta}

def _train_nn(struct: Dict[str, Any], context: Dict[str, Any]):
    ds_name = struct.get('dataset', 'data')
    if 'datasets' not in context or ds_name not in context['datasets']:
        raise ValueError(f"Dataset '{ds_name}' no encontrado")

    dataset = context['datasets'][ds_name]
    model_name = struct.get('model_name', 'nn_model')
    hidden_size = struct.get('hidden_size', 4)
    epochs = struct.get('epochs', 1000)
    lr = struct.get('lr', 0.01)

    meta = ml_manager.train_neural_network(
        model_name=model_name,
        dataset=dataset,
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr
    )

    try:
        entry = ml_manager._MODEL_REGISTRY.get(model_name)
        if entry:
            _models[model_name] = entry.get('model')
    except Exception:
        _models[model_name] = None

    return {'status': 'trained', 'model': model_name, 'meta': meta}

# =====================================================
# Diccionario de acciones ML soportadas

_ACTION_HANDLERS: Dict[str, Callable] = {
    'dataset': _ensure_dataset,
    'train_dt': _train_dt,
    'train_rf': _train_rf,
    'predict': _predict,
    'eval': _eval
}

_ACTION_HANDLERS['train_linear'] = _train_linear
_ACTION_HANDLERS['train_svm'] = _train_svm
_ACTION_HANDLERS['train_nn'] = _train_nn

# =====================================================
# Ejecutor de estructuras ML

def execute_structs(structs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ejecuta una lista de estructuras ML en orden."""
    context: Dict[str, Any] = {}
    results = []
    try:
        for struct in structs:
            action = struct.get('action')
            handler = _ACTION_HANDLERS.get(action)
            if handler is None:
                raise ValueError(f"Acción desconocida o no soportada: {action}")
            result = handler(struct, context)
            results.append({'action': action, 'result': result})
        return {
            'success': True,
            'results': results,
            'outputs': context.get('outputs', {}),
            'models': list(_models.keys())
        }
    except Exception as e:
        return {
            'success': False,
            'results': results,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# =====================================================
# struct_to_action — compatibilidad con pipelines externos

def struct_to_action(struct: Dict[str, Any]) -> tuple[str, Callable]:
    """Convierte una estructura ML en una acción ejecutable."""
    action = struct.get("action")
    if action == "dataset":
        return ("dataset", lambda: struct.get("data"))

    if action == "train_dt":
        return (
            "train_dt",
            lambda: ml_manager.train_decision_tree(
                dataset=struct.get("dataset"),
                model_name=struct.get("model_name"),
                max_depth=struct.get("max_depth"),
                min_size=struct.get("min_size"),
                n_features=struct.get("n_features"),
            ),
        )

    if action == "train_rf":
        return (
            "train_rf",
            lambda: ml_manager.train_random_forest(
                dataset=struct.get("dataset"),
                model_name=struct.get("model_name"),
                n_trees=struct.get("n_trees"),
                max_depth=struct.get("max_depth"),
                min_size=struct.get("min_size"),
                sample_size=struct.get("sample_size"),
                n_features=struct.get("n_features"),
                seed=struct.get("seed"),
            ),
        )

    if action == "predict":
        return (
            "predict",
            lambda: ml_manager.predict(
                name=struct.get("model"),
                X=struct.get("X"),
            ),
        )

    if action in ("eval", "eval_ext"):
        return (
            "eval",
            lambda: ml_manager.evaluate_ext(
                y_true=struct.get("y_true"),
                y_pred=struct.get("y_pred"),
                metrics=struct.get("metrics", ["accuracy"]),
                output=struct.get("output"),
                detailed=False,
            ),
        )

    if callable(action):
        return action

    raise TypeError(f"Acción inválida: {type(action).__name__}")


# =====================================================
# Translator Wrapper — integración con file_handler

try:
    from core.translator import Translator as CoreTranslator
except Exception:
    CoreTranslator = None


class Translator:
    """
    Wrapper de traducción universal compatible con file_handler.
    Permite traducir entre código Python y estructuras de bloques.
    """
    def __init__(self):
        # Inicializa el traductor real del núcleo si está disponible
        if CoreTranslator is not None:
            self._backend = CoreTranslator()
        else:
            self._backend = None

    def translate_to_blocks(self, code: str):
        if self._backend is None:
            raise RuntimeError("Backend de traducción no disponible (core.translator faltante)")
        return self._backend.translate_to_blocks(code)

    def translate_to_python(self, blocks):
        if self._backend is None:
            raise RuntimeError("Backend de traducción no disponible (core.translator faltante)")

        # Aseguramos lista, aunque se reciba un solo bloque
        if isinstance(blocks, dict):
            blocks = [blocks]
        elif not isinstance(blocks, list):
            raise ValueError(f"Formato de bloques inválido: {type(blocks)}")

        return self._backend.translate_to_python(blocks)

# =====================================================
SUPPORTED_MODELS = [
    "DecisionTree",
    "RandomForest",
    "SVM",
    "LinearModel",
    "NeuralNetwork"
]