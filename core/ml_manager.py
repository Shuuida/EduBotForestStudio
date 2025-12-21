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
    """
    Orquesta el flujo completo: Limpieza -> Instanciación -> Entrenamiento -> Registro.
    """
    print(f"--- Iniciando Pipeline para '{model_name}' ({model_type}) ---")
    
    # Imputación de Datos (Limpieza)
    print(" Ejecutando imputación de datos...")
    # Usamos strategy='mean' explícito compatible con ml_compat v2.4
    clean_dataset = impute_missing_values(dataset, strategy='mean')
    
    # Escalado (Opcional - Placeholder para futura implementación)
    if scaling:
        print(f" Aplicando escalado: {scaling} (Simulado)")
        # Aquí iría la lógica de StandardScaler si se implementa en runtime
    else:
        print(" Escalado omitido.")

    # Creación del Modelo (Factory)
    try:
        model = ml_factory.create_model(model_type, params)
    except ValueError as e:
        raise ValueError(f"Error en Factory: {e}")

    # Entrenamiento
    print(f" Entrenando modelo {type(model).__name__}...")
    start_time = time.time()
    model.fit(clean_dataset)
    duration = time.time() - start_time
    
    # Registro automático
    register_model(model_name, model, metadata={
        "type": model_type,
        "params": params,
        "duration": duration
    })
    
    print(f"--- Pipeline finalizado exitosamente en {duration:.4f}s ---")
    
    return {
        "status": "success",
        "model": model,
        "name": model_name,
        "duration": duration
    }

# ---------------------------------------------------------
# INFERENCIA Y EVALUACIÓN

def predict(name_or_model: Union[str, Any], X: List[Any]) -> List[Any]:
    """
    Ejecuta predicciones. Acepta nombre del registro u objeto directo.
    CORRECCIÓN: Maneja inputs 1D (un solo ejemplo) envolviéndolos automáticamente.
    """
    # Resolver modelo
    if isinstance(name_or_model, str):
        model = get_model(name_or_model)
        if not model:
            raise ValueError(f"Modelo '{name_or_model}' no encontrado.")
    else:
        model = name_or_model

    # Auto-corrección de Dimensiones (UX Friendly)
    # Si X es [1, 2, 3] (lista de números), lo convertimos a [[1, 2, 3]]
    if isinstance(X, list) and len(X) > 0:
        # Verificamos si el primer elemento NO es una lista (es decir, es un número)
        if not isinstance(X[0], list):
            # Asumimos que es una sola muestra y la envolvemos
            X = [X]

    # Ejecutar predicción
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