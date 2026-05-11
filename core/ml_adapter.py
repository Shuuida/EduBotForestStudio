"""
Adaptador universal para la ejecución de estructuras ML y puente de traducción.
Actualizado para arquitectura unificada (train_pipeline) y soporte completo de modelos (KNN, SVM, NN).
"""

from typing import Any, Dict, List, Callable
import json
import traceback
import os

# Dependencias del ecosistema ML
from core import ml_manager

# Registro local de modelos (caché para ejecución en lote)
_models: Dict[str, Any] = {}

# ---------------------------------------
# Helpers internos

def _load_csv_to_list(path: str) -> List[List[Any]]:
    import csv
    
    # Lógica de resolución de rutas inteligente
    target_path = path
    if not os.path.exists(target_path):
        # Si no está en raíz, buscar en ./datasets
        candidate = os.path.join("datasets", path)
        if os.path.exists(candidate):
            target_path = candidate
    
    if not os.path.exists(target_path):
         raise FileNotFoundError(f"No se encuentra el archivo: {path} (buscado en raíz y /datasets)")

    with open(target_path, 'r', encoding='utf-8') as f:
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
    
    # Aceptar tanto 'file' como 'csv' como indicadores de archivo
    if (src == 'file' or src == 'csv') and path:
        data = _load_csv_to_list(path)
        
    if not data:
        raise ValueError(f"Dataset vacío o no válido en '{name}'. Verifique la ruta del CSV.")
        
    context.setdefault('datasets', {})[name] = data
    return data

def _coerce_json_if_str(value):
    if isinstance(value, str):
        try: return json.loads(value)
        except Exception: return value
    return value

# -------------------------------------------------
# Mappers de Parámetros (Bloque -> Factory)

def _dt_params(struct, dataset):
    return {
        'max_depth': struct.get('max_depth', 10),
        'min_size': struct.get('min_size', 1),
        'n_features': struct.get('n_features', None)
    }

def _rf_params(struct, dataset):
    return {
        'n_trees': struct.get('n_trees', 5),
        'max_depth': struct.get('max_depth', 10),
        'min_size': struct.get('min_size', 1),
        'sample_size': struct.get('sample_size', 1.0),
        'n_features': struct.get('n_features', None),
        'seed': struct.get('seed', None)
    }

def _svm_params(struct, dataset):
    return {
        'learning_rate': struct.get('learning_rate', 0.001),
        'lambda_param': struct.get('lambda_param', 0.01),
        'n_iters': struct.get('epochs', 1000) # Mapeo epochs -> n_iters
    }

def _linear_params(struct, dataset):
    return {
        'learning_rate': struct.get('learning_rate', 0.01),
        'epochs': struct.get('epochs', 1000)
    }

def _nn_params(struct, dataset):
    # Inferencia automática de dimensiones
    n_inputs = struct.get('n_inputs')
    if n_inputs is None and dataset and len(dataset) > 0:
        n_inputs = len(dataset[0]) - 1 # Asumiendo última columna target
    
    return {
        'n_inputs': n_inputs or 1,
        'n_hidden': struct.get('hidden_size', 4),
        'n_outputs': struct.get('n_outputs', 1),
        'epochs': struct.get('epochs', 1000),
        'learning_rate': struct.get('lr', 0.01)
    }

def _knn_params(struct, dataset):
    return {
        'k': struct.get('k', 3),
        'task': struct.get('task', 'classification')
    }

# --------------------------------------------------
# Handlers de acciones ML (Unified Pipeline)

def _train_model(struct: Dict[str, Any], context: Dict[str, Any], 
                         model_type: str, param_mapper: Callable):
    """
    Handler genérico que conecta los bloques con ml_manager.train_pipeline.
    """
    ds_name = struct.get('dataset', 'data')
    if 'datasets' not in context or ds_name not in context['datasets']:
        raise ValueError(f"Dataset '{ds_name}' no encontrado")

    dataset = context['datasets'][ds_name]
    model_name = struct.get('model_name', f'{model_type.lower()}_model')
    
    # Construir parámetros específicos del modelo
    params = param_mapper(struct, dataset)
    
    # Detectar escalado (si el bloque lo soporta en el futuro)
    scaling = struct.get('scaling', None)

    # Ejecutar Pipeline Unificado
    result = ml_manager.train_pipeline(
        model_name=model_name,
        dataset=dataset,
        model_type=model_type,
        params=params,
        scaling=scaling
    )

    # Sincronizar caché local
    if result.get('model'):
        _models[model_name] = result['model']

    return {'status': 'trained', 'model': model_name, 'meta': result.get('meta')}

# Wrappers específicos
def _train_dt(struct, context): return _train_model(struct, context, 'DecisionTree', _dt_params)
def _train_rf(struct, context): return _train_model(struct, context, 'RandomForest', _rf_params)
def _train_svm(struct, context): return _train_model(struct, context, 'MiniSVM', _svm_params)
def _train_linear(struct, context): return _train_model(struct, context, 'MiniLinearModel', _linear_params)
def _train_nn(struct, context): return _train_model(struct, context, 'MiniNeuralNetwork', _nn_params)
def _train_knn(struct, context): return _train_model(struct, context, 'KNearestNeighbors', _knn_params)


def _predict(struct: Dict[str, Any], context: Dict[str, Any]):
    """Ejecuta predicción con modelo ya entrenado."""
    model_name = struct.get('model', 'model')
    
    # Buscar primero en caché local, luego en registro global
    model = _models.get(model_name)
    if model is None:
        try:
            model = ml_manager.get_model(model_name)
            if model:
                _models[model_name] = model
        except Exception:
            pass

    if model is None:
        raise ValueError(f"Modelo '{model_name}' no encontrado para predicción")

    X = _coerce_json_if_str(struct.get('X'))
    
    # Usar predict de ml_manager que maneja escalado automático
    preds = ml_manager.predict(model, X)
    
    output_var = struct.get('output', 'preds')
    context.setdefault('outputs', {})[output_var] = preds
    return {'status': 'predicted', 'model': model_name, 'output_var': output_var, 'preds': preds}


def _eval(struct: Dict[str, Any], context: Dict[str, Any]):
    """Evalúa métricas del modelo (usa ml_manager.evaluate_ext)."""
    y_true = _coerce_json_if_str(struct.get('y_true'))
    y_pred = struct.get('y_pred')
    
    # Resolver referencia a variable de salida
    if isinstance(y_pred, str) and y_pred in context.get('outputs', {}):
        y_pred = context['outputs'][y_pred]

    metrics = struct.get('metrics', ['accuracy'])
    output_var = struct.get('output', 'results')
    detailed = struct.get('detailed', False)

    # Delegar a ml_manager
    if hasattr(ml_manager, 'evaluate_ext'):
         result = ml_manager.evaluate_ext(
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
            output=output_var,
            detailed=detailed
        )
    else:
        result = ml_manager.evaluate(y_true=y_true, y_pred=y_pred, output=output_var)

    context.setdefault('outputs', {})[output_var] = result
    return {'status': 'evaluated', 'metrics': metrics, 'output_var': output_var, 'result': result}


# --------------------------------------------
# Diccionario de acciones ML soportadas

_ACTION_HANDLERS: Dict[str, Callable] = {
    'dataset': _ensure_dataset,
    'train_dt': _train_dt,
    'train_rf': _train_rf,
    'train_svm': _train_svm,
    'train_linear': _train_linear,
    'train_nn': _train_nn,
    'train_knn': _train_knn,
    'predict': _predict,
    'eval': _eval,
    'eval_ext': _eval # Alias para compatibilidad
}

# -------------------------------------------------
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
                # Intenta ignorar acciones desconocidas o reportar error suave
                raise ValueError(f"Acción desconocida o no soportada por adaptador: {action}")
            
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


# -----------------------------------------------------
# Translator Wrapper — integración con file_handler

try:
    from core.translator import Translator as CoreTranslator
except Exception:
    CoreTranslator = None


class Translator:
    """
    Wrapper de traducción universal compatible con file_handler.
    """
    def __init__(self):
        if CoreTranslator is not None:
            self._backend = CoreTranslator()
        else:
            self._backend = None

    def translate_to_blocks(self, code: str):
        if self._backend is None:
            raise RuntimeError("Backend de traducción no disponible")
        return self._backend.translate_to_blocks(code)

    def translate_to_python(self, blocks):
        if self._backend is None:
            raise RuntimeError("Backend de traducción no disponible")
        if isinstance(blocks, dict):
            blocks = [blocks]
        return self._backend.translate_to_python(blocks)

# ------------------------------------------
SUPPORTED_MODELS = [
    "DecisionTree",
    "RandomForest",
    "SVM",
    "LinearModel",
    "NeuralNetwork",
    "KNearestNeighbors"
]