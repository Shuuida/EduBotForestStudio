"""
Normalizadores / reglas para bloques ML.
Cada función toma un bloque (dict desde frontend) y devuelve una estructura
({'action': ..., ...}) estandarizada que el ml_adapter entienda.
"""

from typing import Dict, Any

# ============================================================
# BLOQUES -> ESTRUCTURA

def ml_dataset_block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza block 'ml_dataset' a estructura estándar."""
    return {
        'action': 'dataset',
        'name': block.get('name', 'data'),
        'source': block.get('source', 'inline'),
        'data': block.get('data', None),
        'path': block.get('path', None),
    }

def ml_train_dt_block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'action': 'train_dt',
        'dataset': block.get('dataset', 'data'),
        'model_name': block.get('model_name', 'dt_model'),
        'max_depth': int(block.get('max_depth', 10)),
        'min_size': int(block.get('min_size', 1)),
        'n_features': None if block.get('n_features') in (None, 'None', '') else int(block.get('n_features')),
    }

def ml_train_rf_block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'action': 'train_rf',
        'dataset': block.get('dataset', 'data'),
        'model_name': block.get('model_name', 'rf_model'),
        'n_trees': int(block.get('n_trees', 5)),
        'max_depth': int(block.get('max_depth', 10)),
        'min_size': int(block.get('min_size', 1)),
        'sample_size': float(block.get('sample_size', 1.0)),
        'n_features': None if block.get('n_features') in (None, 'None', '') else int(block.get('n_features')),
        'seed': None if block.get('seed') in (None, 'None', '') else int(block.get('seed')),
    }

def ml_predict_block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'action': 'predict',
        'model': block.get('model', 'model'),
        'X': block.get('X', None),
        'output': block.get('output', 'preds'),
    }

def ml_eval_block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'action': 'eval',
        'y_true': block.get('y_true', None),
        'y_pred': block.get('y_pred', None),
        'output': block.get('output', 'metric'),
    }

def ml_eval_ext_block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    """Soporte extendido para métricas múltiples (accuracy, precision, recall, f1)."""
    return {
        'action': 'eval_ext',
        'y_true': block.get('y_true', None),
        'y_pred': block.get('y_pred', None),
        'metrics': block.get('metrics', ['accuracy']),
        'output': block.get('output', 'results')
    }


# ============================================================
# REGISTRO Y CONVERSIÓN

BLOCK_TO_STRUCT = {
    'ml_dataset': ml_dataset_block_to_struct,
    'ml_train_dt': ml_train_dt_block_to_struct,
    'ml_train_rf': ml_train_rf_block_to_struct,
    'ml_predict': ml_predict_block_to_struct,
    'ml_eval': ml_eval_block_to_struct,
    'ml_eval_ext': ml_eval_ext_block_to_struct,
}

def block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte un bloque visual a estructura ML normalizada."""
    t = block.get('type')
    if not t:
        return {'action': 'unsupported', 'raw': block}
    fn = BLOCK_TO_STRUCT.get(t)
    if fn:
        try:
            return fn(block)
        except Exception as e:
            return {'action': 'error', 'error': str(e), 'raw': block}
    return {'action': 'unsupported', 'raw': block}

MODEL_TYPE_MAP = {
    "decisiontree": "ml_decision_tree",
    "randomforest": "ml_random_forest",
    "svm": "ml_svm",
    "linear": "ml_linear_model",
    "neural": "ml_neural_network",
}

def detect_block_type_from_struct(struct: dict) -> str:
    """
    Detecta el tipo de bloque visual adecuado a partir de una estructura ML.
    Si no encuentra coincidencias conocidas, retorna 'ml_model_generic'.
    """
    model_type = (struct.get("model_type") or struct.get("type") or "").lower()
    for key, block_name in MODEL_TYPE_MAP.items():
        if key in model_type:
            return block_name
    return "ml_model_generic"


# ============================================================
# ESTRUCTURA -> BLOQUES (CONVERSIÓN INVERSA)

def struct_to_visual_block(struct: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte una estructura ML en bloque visual (para exportación inversa).
    Soporta modelos MiniML, sklearn y genéricos, así como datasets y evaluadores.
    """
    action = struct.get("action")
    block_type = detect_block_type_from_struct(struct)

    # Dataset
    if action == "dataset":
        return {
            "type": "ml_dataset",
            "name": struct.get("name"),
            "source": struct.get("source", "inline"),
            "data": struct.get("data"),
            "path": struct.get("path"),
        }

    # Entrenamiento de Árbol de Decisión
    elif action == "train_dt":
        return {
            "type": "ml_train_dt",
            "dataset": struct.get("dataset"),
            "model_name": struct.get("model_name"),
            "max_depth": struct.get("max_depth"),
            "min_size": struct.get("min_size"),
            "n_features": struct.get("n_features"),
        }

    # Entrenamiento de Bosque Aleatorio
    elif action == "train_rf":
        return {
            "type": "ml_train_rf",
            "dataset": struct.get("dataset"),
            "model_name": struct.get("model_name"),
            "n_trees": struct.get("n_trees"),
            "max_depth": struct.get("max_depth"),
            "min_size": struct.get("min_size"),
            "sample_size": struct.get("sample_size"),
            "n_features": struct.get("n_features"),
            "seed": struct.get("seed"),
        }

    # Predicción
    elif action == "predict":
        return {
            "type": "ml_predict",
            "model": struct.get("model"),
            "X": struct.get("X"),
            "output": struct.get("output"),
        }

    # Evaluación simple
    elif action == "eval":
        return {
            "type": "ml_eval",
            "y_true": struct.get("y_true"),
            "y_pred": struct.get("y_pred"),
            "output": struct.get("output"),
        }

    # Evaluación extendida
    elif action == "eval_ext":
        return {
            "type": "ml_eval_ext",
            "y_true": struct.get("y_true"),
            "y_pred": struct.get("y_pred"),
            "metrics": struct.get("metrics", ["accuracy"]),
            "output": struct.get("output"),
        }

    # Bloque de modelo genérico 
    elif block_type == "ml_model_generic":
        block = {
            "type": "ml_model_generic",
            "model_name": struct.get("model_name", "UnknownModel"),
            "framework": struct.get("framework", "unknown"),
            "parameters": struct.get("parameters", {}),
            "raw": struct,
        }

        # Si el modelo tiene marca temporal → incluir como metadato
        if "saved_at" in struct:
            block["metadata"] = {"trained_at": struct["saved_at"]}

        return block

    # Bloques reconocidos (DecisionTree, RandomForest, etc.)
    elif block_type in MODEL_TYPE_MAP.values():
        block = {
            "type": block_type,
            "framework": struct.get("framework", "MiniML"),
            "parameters": struct.get("parameters", {}),
            "raw": struct,
        }

        if "saved_at" in struct:
            block["metadata"] = {"trained_at": struct["saved_at"]}

        if block_type == "ml_svm":
                block["kernel"] = struct.get("kernel", "linear")
                block["bias"] = struct.get("bias", 0.0)
        elif block_type == "ml_linear_model":
            block["intercept"] = struct.get("intercept", 0.0)
        elif block_type == "ml_neural_network":
            block["activation"] = struct.get("activation", "relu")
            block["layers"] = struct.get("layers", [])

        return block

    # Fallback de seguridad
    else:
        return {
            "type": "ml_model_generic",
            "framework": struct.get("framework", "unknown"),
            "parameters": struct.get("parameters", {}),
            "raw": struct,
        }



# ============================================================
# VALIDADOR

def validate_struct(struct: Dict[str, Any]) -> bool:
    """Valida que la estructura ML contenga campos mínimos requeridos según su
    tipo de acción o modelo.
    Incluye soporte para modelos genéricos ('ml_model_generic') y estructuras
    híbridas provenientes de MiniML o sklearn.
    """
    required_fields = {
        'dataset': ['action', 'data'],
        'train_dt': ['action', 'dataset', 'model_name'],
        'train_rf': ['action', 'dataset', 'model_name'],
        'predict': ['action', 'model', 'X'],
        'eval': ['action', 'y_true', 'y_pred'],
        'eval_ext': ['action', 'y_true', 'y_pred', 'metrics'],
    }
    # Detección principal
    action = struct.get('action')
    model_type = (struct.get('type') or '').lower()

    # Estructura ML tradicional
    if action in required_fields:
        return all(field in struct for field in required_fields[action])

    # Modelo genérico
    if model_type == 'ml_model_generic' or 'framework' in struct:
        # Verifica existencia mínima de nombre o framework
        return any(k in struct for k in ('framework', 'parameters', 'model_name', 'type'))

    # Estructuras sin action pero con árbol o bosque
    if any(k in struct for k in ('root', 'trees')):
        return True  # válido como modelo ML MiniML o sklearn

    # Fallback
    return False