"""
==============================
Gestor de exportación para pipelines ML en el ecosistema MiniML/Sklearn de EduBot.
ACTUALIZADO: Soporte nativo para inspección de objetos MiniML (v2) y reconstrucción de bloques.

Provee herramientas para:
- Exportar estructuras ML (clasificación/regresión) a JSON.
- Inspeccionar modelos en memoria para generar representaciones visuales.
- Puente entre el Runtime (Python Objects) y el Frontend (JSON Blocks).
"""

from typing import List, Dict, Any, Optional
import json
import time

# Dependencias del Núcleo
from core import ml_struct_rules
from core import ml_runtime
from core import ml_factory
from core.ml_compat import _flatten_tree_to_arrays, _unflatten_arrays_to_tree

# Registro de eventos internos
_EXPORT_LOG: List[str] = []

# -------------------------
# UTILIDADES INTERNAS

def _log(msg: str):
    """Registra un mensaje de depuración o advertencia en el buffer interno."""
    ts = time.strftime("[%H:%M:%S]")
    _EXPORT_LOG.append(f"{ts} {msg}")

def get_export_log(limit: int = 25) -> List[str]:
    """Devuelve las últimas líneas del registro interno de exportación."""
    return _EXPORT_LOG[-limit:]

# -----------------------------------------------------------
# SERIALIZACIÓN (MODELO -> JSON) - Para Guardar en Disco

def serialize_model(model_obj: Any) -> Dict[str, Any]:
    """
    Convierte un objeto modelo en memoria a un diccionario serializable para disco.
    Usa la lógica de extracción unificada.
    """
    try:
        data = extract_model_structure(model_obj)
        
        # Agregamos metadatos específicos de persistencia si faltan
        if "model_module" not in data:
            data["model_module"] = model_obj.__class__.__module__
        if "model_class" not in data:
            data["model_class"] = model_obj.__class__.__name__
            
        # Casos especiales que necesitan datos extra para ser re-entrenables o funcionales
        # KNN: Necesita guardar los datos de entrenamiento (Lazy Learning)
        if isinstance(model_obj, ml_runtime.KNearestNeighbors):
            data["X_train"] = getattr(model_obj, "X_train", [])
            data["y_train"] = getattr(model_obj, "y_train", [])
            
        # NN: Necesita pesos exactos (no solo config)
        if isinstance(model_obj, ml_runtime.MiniNeuralNetwork):
            data["weights"] = {
                "W1": getattr(model_obj, "W1", []),
                "B1": getattr(model_obj, "B1", []),
                "W2": getattr(model_obj, "W2", []),
                "B2": getattr(model_obj, "B2", [])
            }
            
        return data
        
    except Exception as e:
        _log(f"Error serializing model: {e}")
        raise RuntimeError(f"Fallo en serialización: {e}")

# -----------------------------------------------------------
# DESERIALIZACIÓN (JSON -> MODELO) - Para Cargar de Disco

def deserialize_model(data: Dict[str, Any]) -> Any:
    """
    Reconstruye un objeto modelo en memoria a partir de su diccionario JSON.
    """
    try:
        framework = data.get("framework", "unknown")
        model_type = data.get("type", "").lower()
        
        if framework == "MiniML":
            # Reconstrucción MiniML
            
            # K-Nearest Neighbors
            if "knn" in model_type:
                model = ml_factory.create_model("knn", {
                    "k": data.get("k", 3),
                    "task": data.get("task", "classification")
                })
                # Restaurar estado interno
                if "X_train" in data: model.X_train = data["X_train"]
                if "y_train" in data: model.y_train = data["y_train"]
                model.n_features_trained = len(model.X_train[0]) if model.X_train else 0
                return model

            # SVM
            elif "svm" in model_type:
                model = ml_factory.create_model("svm", {
                    "learning_rate": data.get("learning_rate", 0.001),
                    "lambda_param": data.get("lambda_param", 0.01),
                    "n_iters": data.get("n_iters", 1000)
                })
                if "weights" in data: model.weights = data["weights"]
                if "bias" in data: model.bias = data["bias"]
                return model

            # Linear Model
            elif "linear" in model_type:
                model = ml_factory.create_model("linear", {
                    "learning_rate": data.get("learning_rate", 0.01),
                    "epochs": data.get("epochs", 1000)
                })
                if "weights" in data: model.weights = data["weights"]
                if "intercept" in data: model.bias = data["intercept"]
                return model

            # Neural Network
            elif "neural" in model_type:
                config = data.get("config", {})
                params = {
                    "n_inputs": config.get("n_inputs", 2),
                    "n_hidden": config.get("n_hidden", 4),
                    "n_outputs": config.get("n_outputs", 1),
                    "epochs": data.get("epochs", 1000),
                    "learning_rate": config.get("learning_rate", 0.1)
                }
                model = ml_factory.create_model("neural", params)
                
                # Restaurar pesos si existen
                w_data = data.get("weights", {})
                if w_data:
                    model.W1 = w_data.get("W1", [])
                    model.B1 = w_data.get("B1", [])
                    model.W2 = w_data.get("W2", [])
                    model.B2 = w_data.get("B2", [])
                
                if "quantized" in data: model.quantized = data["quantized"]
                return model

            # Árboles (DecisionTree / RandomForest)
            elif "decisiontree" in model_type:
                is_reg = "regressor" in data.get("model_class", "").lower()
                # Usa factory para decidir Regressor/Classifier
                model = ml_factory.create_model("decisiontree_regressor" if is_reg else "decisiontree", {
                    "max_depth": data.get("max_depth", 5),
                    "min_size": data.get("min_size", 1)
                })
                # Reconstruir árbol desde struct aplanada
                if "struct" in data:
                    model.root = _unflatten_arrays_to_tree(data["struct"])
                return model

            elif "randomforest" in model_type:
                is_reg = "regressor" in data.get("model_class", "").lower()
                model = ml_factory.create_model("randomforest_regressor" if is_reg else "randomforest", {
                    "n_trees": data.get("n_trees", 5),
                    "max_depth": data.get("max_depth", 5)
                })
                # Nota: Restaurar un RF completo requeriría serializar cada árbol de la lista 'trees'.
                # Por simplicidad en MVP, a veces solo se guarda la config.
                # Si se necesita persistencia total de RF, se debería iterar data['trees'] aquí.
                return model

        # Fallback
        _log(f"Framework desconocido o no soportado para deserialización: {framework}")
        return None

    except Exception as e:
        _log(f"Error deserializing model: {e}")
        raise RuntimeError(f"Fallo en deserialización: {e}")

# -----------------------------------------------------------
# EXTRACCIÓN ESTRUCTURAL (Para Visualización / Bloques)

def extract_model_structure(model_obj: Any) -> Dict[str, Any]:
    """
    Analiza un objeto modelo y extrae su estructura estandarizada.
    """
    struct = {}
    try:
        # K-Nearest Neighbors
        if isinstance(model_obj, ml_runtime.KNearestNeighbors):
            struct = {
                "framework": "MiniML",
                "type": "knn", 
                "k": getattr(model_obj, "k", 3),
                "task": getattr(model_obj, "task", "classification"),
            }
            return struct

        # Mini SVM
        if isinstance(model_obj, ml_runtime.MiniSVM):
            weights = getattr(model_obj, "weights", [])
            bias = getattr(model_obj, "bias", 0.0)
            struct = {
                "framework": "MiniML",
                "type": "svm",
                "learning_rate": getattr(model_obj, "lr", 0.001),
                "lambda_param": getattr(model_obj, "lambda_param", 0.01),
                "n_iters": getattr(model_obj, "n_iters", 1000),
                "weights": weights,
                "bias": bias
            }
            return struct

        # Mini Linear Model
        if isinstance(model_obj, ml_runtime.MiniLinearModel):
            weights = getattr(model_obj, "weights", [])
            bias = getattr(model_obj, "bias", 0.0)
            struct = {
                "framework": "MiniML",
                "type": "linear",
                "learning_rate": getattr(model_obj, "lr", 0.01),
                "epochs": getattr(model_obj, "epochs", 1000),
                "weights": weights,
                "intercept": bias # Mapeo para visualizador
            }
            return struct

        # Mini Neural Network
        if isinstance(model_obj, ml_runtime.MiniNeuralNetwork):
            struct = {
                "framework": "MiniML",
                "type": "neural",
                "config": {
                    "n_inputs": getattr(model_obj, "n_in", 2),
                    "n_hidden": getattr(model_obj, "n_hid", 4),
                    "n_outputs": getattr(model_obj, "n_out", 1),
                    "learning_rate": getattr(model_obj, "lr", 0.1),
                },
                "epochs": getattr(model_obj, "epochs", 1000),
                "trained": hasattr(model_obj, "W1") and len(model_obj.W1) > 0
            }
            if hasattr(model_obj, "quantized"):
                struct["quantized"] = model_obj.quantized
            return struct

        # Árboles y Bosques
        # Detección por atributos para ser agnóstico a la clase exacta
        if hasattr(model_obj, "root") and hasattr(model_obj, "max_depth"): 
            struct = {
                "framework": "MiniML",
                "type": "DecisionTree",
                "max_depth": getattr(model_obj, "max_depth", 10),
                "min_size": getattr(model_obj, "min_size", 1)
            }
            # Aplanar estructura para guardado eficiente
            if getattr(model_obj, "root", None):
                try:
                    struct["struct"] = _flatten_tree_to_arrays(model_obj.root)
                except Exception: pass
            return struct

        if hasattr(model_obj, "trees") and isinstance(model_obj.trees, list):
            struct = {
                "framework": "MiniML",
                "type": "RandomForest",
                "n_trees": len(model_obj.trees),
                "max_depth": getattr(model_obj, "max_depth", 10),
            }
            return struct

        # Fallback genérico
        return {"framework": "unknown", "repr": repr(model_obj), "type": "UnknownModel"}

    except Exception as e:
        _log(f"Error extracting model structure: {e}")
        return {"framework": "error", "error": str(e)}

# -----------------------------------------------------------
# RECONSTRUCCIÓN VISUAL (Objeto -> Bloque)

def export_model_to_block_structure(model_obj: Any, model_name: str = "exported_model") -> Dict[str, Any]:
    """
    Convierte un objeto modelo en memoria directamente a un bloque visual JSON.
    """
    internal_struct = extract_model_structure(model_obj)
    internal_struct["model_name"] = model_name
    internal_struct["action"] = "visualize" 
    
    # Delegar a ml_struct_rules para formato visual
    return ml_struct_rules.struct_to_visual_block(internal_struct)