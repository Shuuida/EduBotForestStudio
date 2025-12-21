"""
core/ml_struct_rules.py
-----------------------
Reglas de normalización de estructuras ML para EduBot.
Gestiona la conversión bidireccional entre Bloques Visuales (Frontend) y Estructuras de Ejecución (Backend).

ACTUALIZADO:
- Soporte completo para MiniML Engine v1.0.2 (KNN, SVM, Linear, NN Avanzada).
- Reconstrucción inteligente de topologías neuronales para visualización.
- Extracción automática de bias/intercept desde pesos crudos.
"""

from typing import Dict, Any

# ----------------------------------------------------------------
# BLOQUES VISUALES -> ESTRUCTURA INTERNA (Normalización)

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

def ml_train_knn_block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'action': 'train_knn',
        'dataset': block.get('dataset', 'data'),
        'model_name': block.get('model_name', 'knn_model'),
        'k': int(block.get('k', 3)),
        'task': block.get('task', 'classification') 
    }

def ml_train_svm_block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'action': 'train_svm',
        'dataset': block.get('dataset', 'data'),
        'model_name': block.get('model_name', 'svm_model'),
        'learning_rate': float(block.get('learning_rate', 0.001)),
        'lambda_param': float(block.get('lambda_param', 0.01)),
        'epochs': int(block.get('epochs', 1000)), # El runtime usa 'n_iters', el adaptador lo mapeará
    }

def ml_train_linear_block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'action': 'train_linear',
        'dataset': block.get('dataset', 'data'),
        'model_name': block.get('model_name', 'linear_model'),
        'learning_rate': float(block.get('learning_rate', 0.01)),
        'epochs': int(block.get('epochs', 1000)),
    }

def ml_train_nn_block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'action': 'train_nn',
        'dataset': block.get('dataset', 'data'),
        'model_name': block.get('model_name', 'nn_model'),
        'hidden_size': int(block.get('hidden_size', 4)),
        'epochs': int(block.get('epochs', 1000)),
        'lr': float(block.get('lr', 0.01)),
        'n_inputs': block.get('n_inputs') # Opcional, puede ser None (auto-detect)
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
    """Soporte extendido para métricas múltiples."""
    return {
        'action': 'eval_ext',
        'y_true': block.get('y_true', None),
        'y_pred': block.get('y_pred', None),
        'metrics': block.get('metrics', ['accuracy']),
        'output': block.get('output', 'results')
    }

# Registro de convertidores
BLOCK_TO_STRUCT = {
    'ml_dataset': ml_dataset_block_to_struct,
    'ml_train_dt': ml_train_dt_block_to_struct,
    'ml_train_rf': ml_train_rf_block_to_struct,
    'ml_train_knn': ml_train_knn_block_to_struct,
    'ml_train_svm': ml_train_svm_block_to_struct,
    'ml_train_linear': ml_train_linear_block_to_struct,
    'ml_train_nn': ml_train_nn_block_to_struct,
    'ml_predict': ml_predict_block_to_struct,
    'ml_eval': ml_eval_block_to_struct,
    'ml_eval_ext': ml_eval_ext_block_to_struct,
}

def block_to_struct(block: Dict[str, Any]) -> Dict[str, Any]:
    """Punto de entrada: Convierte un bloque visual a estructura ML normalizada."""
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


# ----------------------------------------------------------------------
# ESTRUCTURA INTERNA -> BLOQUES VISUALES (Traducción Inversa)

MODEL_TYPE_MAP = {
    "decisiontree": "ml_decision_tree",
    "randomforest": "ml_random_forest",
    "svm": "ml_svm",
    "minisvm": "ml_svm", # Alias para el nombre de clase real
    "linear": "ml_linear_model",
    "minilinearmodel": "ml_linear_model", # Alias
    "neural": "ml_neural_network",
    "minineuralnetwork": "ml_neural_network", # Alias
    "knn": "ml_train_knn", # Para KNN usamos el bloque de entrenamiento como visualizador por ahora
    "knearestneighbors": "ml_train_knn"
}

def detect_block_type_from_struct(struct: dict) -> str:
    """Detecta el tipo de bloque visual a partir de metadata o nombre de clase."""
    # Intentar detectar por campo 'type' explícito
    model_type = (struct.get("model_type") or struct.get("type") or "").lower()
    
    # Búsqueda difusa en el mapa
    for key, block_name in MODEL_TYPE_MAP.items():
        if key in model_type:
            return block_name
            
    return "ml_model_generic"

def struct_to_visual_block(struct: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte una estructura ML (guardada o en runtime) en un bloque visual.
    CORREGIDO: Maneja atributos reales de MiniML (weights, config) en lugar de imaginarios.
    """
    action = struct.get("action")
    block_type = detect_block_type_from_struct(struct)

    # Bloques de Acción (Pipeline)
    
    if action == "dataset":
        return {
            "type": "ml_dataset",
            "name": struct.get("name"),
            "source": struct.get("source", "inline"),
            "data": struct.get("data"),
            "path": struct.get("path"),
        }
    elif action == "train_dt":
        return {
            "type": "ml_train_dt",
            "dataset": struct.get("dataset"),
            "model_name": struct.get("model_name"),
            "max_depth": struct.get("max_depth"),
            "min_size": struct.get("min_size"),
            "n_features": struct.get("n_features"),
        }
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
    elif action == "train_knn":
        return {
            "type": "ml_train_knn",
            "dataset": struct.get("dataset"),
            "model_name": struct.get("model_name"),
            "k": struct.get("k", 3),
            "task": struct.get("task", "classification"),
        }
    elif action == "train_svm":
        return {
            "type": "ml_train_svm",
            "dataset": struct.get("dataset"),
            "model_name": struct.get("model_name"),
            "learning_rate": struct.get("learning_rate"),
            "epochs": struct.get("epochs") or struct.get("n_iters"),
            "lambda_param": struct.get("lambda_param")
        }
    elif action == "train_linear":
        return {
            "type": "ml_train_linear",
            "dataset": struct.get("dataset"),
            "model_name": struct.get("model_name"),
            "learning_rate": struct.get("learning_rate"),
            "epochs": struct.get("epochs")
        }
    elif action == "train_nn":
        return {
            "type": "ml_train_nn",
            "dataset": struct.get("dataset"),
            "model_name": struct.get("model_name"),
            "hidden_size": struct.get("hidden_size"),
            "epochs": struct.get("epochs"),
            "lr": struct.get("lr")
        }
    elif action == "predict":
        return {
            "type": "ml_predict",
            "model": struct.get("model"),
            "X": struct.get("X"),
            "output": struct.get("output"),
        }
    elif action == "eval":
        return {
            "type": "ml_eval",
            "y_true": struct.get("y_true"),
            "y_pred": struct.get("y_pred"),
            "output": struct.get("output"),
        }
    elif action == "eval_ext":
        return {
            "type": "ml_eval_ext",
            "y_true": struct.get("y_true"),
            "y_pred": struct.get("y_pred"),
            "metrics": struct.get("metrics", ["accuracy"]),
            "output": struct.get("output"),
        }

    # Bloques de Modelos Entrenados (Visualización)
    
    # Modelo Genérico (Fallback)
    if block_type == "ml_model_generic":
        block = {
            "type": "ml_model_generic",
            "model_name": struct.get("model_name", "UnknownModel"),
            "framework": struct.get("framework", "unknown"),
            "parameters": struct.get("parameters", {}),
            "raw": struct,
        }
        if "saved_at" in struct:
            block["metadata"] = {"trained_at": struct["saved_at"]}
        return block

    # Modelos Específicos
    elif block_type in MODEL_TYPE_MAP.values():
        block = {
            "type": block_type,
            "framework": struct.get("framework", "MiniML"),
            "parameters": struct.get("parameters", {}),
            # "raw": struct, # Opcional, para debug
        }

        # Metadata universal
        if "saved_at" in struct:
            block["metadata"] = {"trained_at": struct["saved_at"]}

        # Lógica Específica por Modelo
        
        if block_type == "ml_svm":
            # Extraer bias de los pesos si no existe explícitamente
            weights = struct.get("weights", [])
            bias = struct.get("bias", 0.0)
            if not bias and weights and len(weights) > 0:
                bias = weights[-1] # En MiniML, bias es el último peso
            
            block["kernel"] = struct.get("kernel", "linear")
            block["bias"] = bias
            block["epochs"] = struct.get("n_iters", 1000)

        elif block_type == "ml_linear_model":
            weights = struct.get("weights", [])
            intercept = struct.get("intercept", 0.0)
            # Intentar recuperar intercept de weights[-1] si no viene explícito
            if not intercept and weights and len(weights) > 0:
                intercept = weights[-1]
                
            block["intercept"] = intercept
            block["epochs"] = struct.get("epochs", 1000)

        elif block_type == "ml_neural_network":
            # Reconstruir topología visual desde config o inferencia
            config = struct.get("config", {})
            n_in = struct.get("n_inputs") or config.get("n_inputs", 1)
            n_hid = struct.get("n_hidden") or config.get("n_hidden", 4)
            n_out = struct.get("n_outputs") or config.get("n_outputs", 1)
            
            # El frontend espera 'layers' para dibujar la red
            block["layers"] = [n_in, n_hid, n_out]
            
            # Activaciones
            block["activation"] = struct.get("hidden_activation") or config.get("hidden_activation", "sigmoid")

        elif block_type == "ml_train_knn":
            block["k"] = struct.get("k", 3)
            block["task"] = struct.get("task", "classification")

        return block

    # Fallback final
    return {
        "type": "ml_model_generic",
        "raw": struct
    }


# -----------------------------------------------
# VALIDADOR DE ESTRUCTURAS

def validate_struct(struct: Dict[str, Any]) -> bool:
    """
    Valida que la estructura ML contenga campos mínimos requeridos.
    ACTUALIZADO: Incluye todos los modelos soportados (SVM, Linear, NN, KNN).
    """
    required_fields = {
        'dataset': ['action', 'data'],
        'train_dt': ['action', 'dataset', 'model_name'],
        'train_rf': ['action', 'dataset', 'model_name'],
        # Nuevos modelos
        'train_knn': ['action', 'dataset', 'model_name', 'k'],
        'train_svm': ['action', 'dataset', 'model_name'],
        'train_linear': ['action', 'dataset', 'model_name'],
        'train_nn': ['action', 'dataset', 'model_name'],
        # Ejecución
        'predict': ['action', 'model', 'X'],
        'eval': ['action', 'y_true', 'y_pred'],
        'eval_ext': ['action', 'y_true', 'y_pred', 'metrics'],
    }
    
    # Validación por Acción
    action = struct.get('action')
    if action in required_fields:
        return all(field in struct for field in required_fields[action])

    # Validación por Tipo (Modelos ya entrenados/guardados)
    model_type = (struct.get('type') or '').lower()
    
    # Modelos Genéricos o MiniML Serializados
    if model_type == 'ml_model_generic' or 'framework' in struct:
        return True # Asumimos válido si tiene framework

    # Estructuras de Árbol (para compatibilidad legacy)
    if any(k in struct for k in ('root', 'trees', 'weights', 'W1')):
        return True

    return False