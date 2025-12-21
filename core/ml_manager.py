"""
ML Manager
=========================
Gestor central de ciclo de vida de modelos para EduBot.
Orquesta: Entrenamiento, Registro, Predicción y Evaluación.

ACTUALIZADO:
- Agregadas funciones getters/setters (register_model, get_model) para file_handler.
- train_pipeline unificado con manejo de imputación y factory.
- evaluate() robusto que devuelve diccionarios para frontend/tests.
"""

from typing import Any, Dict, List, Optional, Union
import time

# Dependencias del Núcleo
from core import ml_runtime
from core import ml_factory
from core.ml_compat import impute_missing_values

# ---------------------------------------------------------
# REGISTRO GLOBAL DE MODELOS (Memoria RAM)
# Estructura: {'nombre_modelo': {'model': objeto, 'meta': dict, 'mode': 'mini'}}
_MODEL_REGISTRY: Dict[str, Any] = {}

# ---------------------------------------------------------
# GESTIÓN DEL REGISTRO (API Pública para file_handler)

def register_model(name: str, model_obj: Any, metadata: Dict = None) -> None:
    """
    Registra un modelo instanciado en la memoria global.
    Vital para cargar modelos desde disco.
    """
    _MODEL_REGISTRY[name] = {
        'model': model_obj,
        'mode': 'mini', # Por defecto en v2.x
        'meta': metadata or {},
        'created_at': time.time()
    }

def get_model(name: str) -> Any:
    """
    Recupera el objeto modelo crudo desde el registro.
    Retorna None si no existe.
    """
    entry = _MODEL_REGISTRY.get(name)
    if entry:
        return entry['model']
    return None

def list_models() -> List[str]:
    """Lista los nombres de modelos en memoria."""
    return list(_MODEL_REGISTRY.keys())

def clear_registry():
    """Limpia todos los modelos de la memoria."""
    _MODEL_REGISTRY.clear()

# ---------------------------------------------------------
# PIPELINE DE ENTRENAMIENTO

def train_pipeline(model_name: str, dataset: List[List[float]], model_type: str, params: Dict[str, Any], scaling: Optional[str] = None) -> Dict[str, Any]:
    print(f"--- Iniciando Pipeline para '{model_name}' ({model_type}) ---")
    
    # Limpieza
    print(" Ejecutando imputación de datos...")
    clean_dataset = impute_missing_values(dataset, strategy='mean')
    
    # Escalado Inteligente
    scaler = None
    if scaling == 'standard':
        print(" Aplicando escalado Standard (Z-Score)...")
        # Separamos Features (X) y Target (y) para no escalar el target (clase)
        # Asumimos que la última columna es el target
        X = [row[:-1] for row in clean_dataset]
        y = [row[-1] for row in clean_dataset]
        
        scaler = ml_runtime.MiniScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reconstruimos el dataset
        clean_dataset = [x + [target] for x, target in zip(X_scaled, y)]
    else:
        print(" Escalado omitido.")

    # Creación del Modelo
    try:
        model = ml_factory.create_model(model_type, params)
    except ValueError as e:
        raise ValueError(f"Error en Factory: {e}")

    # Inyectar Scaler en el Modelo (Para que persista)
    if scaler:
        model.scaler = scaler

    # Entrenamiento
    print(f" Entrenando modelo {type(model).__name__}...")
    start_time = time.time()
    model.fit(clean_dataset)
    duration = time.time() - start_time
    
    # Registro
    register_model(model_name, model, metadata={
        "type": model_type,
        "params": params,
        "duration": duration,
        "scaling": scaling # Guardamos info del escalado
    })
    
    print(f"--- Pipeline finalizado exitosamente en {duration:.4f}s ---")
    
    return {"status": "success", "model": model, "name": model_name}

# ---------------------------------------------------------
# INFERENCIA Y EVALUACIÓN

def predict(name_or_model: Union[str, Any], X: List[Any]) -> List[Any]:
    """
    Ejecuta predicciones aplicando escalado automático si el modelo lo tiene.
    """
    # Resolver modelo
    if isinstance(name_or_model, str):
        model = get_model(name_or_model)
        if not model: raise ValueError(f"Modelo '{name_or_model}' no encontrado.")
    else:
        model = name_or_model

    # Auto-corrección de Dimensiones (Input 1D -> 2D)
    if isinstance(X, list) and len(X) > 0 and not isinstance(X[0], list):
        X = [X]

    # Aplicar Escalado Automático (Si el modelo fue entrenado con él)
    if hasattr(model, 'scaler') and model.scaler:
        # print(" [Debug] Aplicando escalado a entrada de predicción...")
        X = model.scaler.transform(X)

    # 4. Ejecutar predicción
    if hasattr(model, 'predict'):
        return model.predict(X)
    else:
        raise AttributeError(f"El objeto {type(model)} no tiene método 'predict'.")

def evaluate(y_true: List[float], y_pred: List[float], output: str = None) -> Union[float, Dict[str, float]]:
    """
    Calcula métricas de rendimiento.
    Retorna un diccionario completo para robustez del frontend.
    """
    if len(y_true) != len(y_pred):
        return {"error": f"Dimension mismatch: {len(y_true)} vs {len(y_pred)}"}

    # Calcular todas las métricas disponibles (Eficiencia: O(N))
    acc = ml_runtime.accuracy_score(y_true, y_pred)
    mse = ml_runtime.mse_score(y_true, y_pred)
    mae = ml_runtime.mae_score(y_true, y_pred)
    r2 = ml_runtime.r2_score(y_true, y_pred)

    results = {
        "accuracy": acc,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "count": len(y_true)
    }

    # Si se pide una específica, se retorna (legacy support)
    if output and output in results:
        return results[output]
    
    # Por defecto retornar todo el reporte
    return results

# Alias para compatibilidad con ml_adapter
evaluate_ext = evaluate