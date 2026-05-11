"""
Memory Estimator for EduBot / Arduino Uno
=========================================
Estima el consumo de memoria Flash (Programa) y SRAM (Variables)
para modelos MiniML exportados a C/C++ (Arduino).

Base de cálculo (Arduino Uno - ATmega328P):
- Flash Total: 32,256 bytes (32KB - 0.5KB bootloader)
- SRAM Total: 2,048 bytes (2KB)
- Float size: 4 bytes
- Pointer/Int size: 2 bytes
"""

from typing import Dict, Any, Union
from core import ml_runtime

# Constantes de Hardware (Arduino Uno)
UNO_FLASH_LIMIT = 32256
UNO_SRAM_LIMIT = 2048
FLOAT_SIZE = 4
INT_SIZE = 2

def estimate_memory(model: Any, quantized: bool = False, target_flash: int = UNO_FLASH_LIMIT, target_sram: int = UNO_SRAM_LIMIT) -> Dict[str, Any]:
    """
    Calcula el uso de memoria estimado del modelo.
    """
    flash_bytes = 0
    sram_bytes = 0
    overhead_code = 0  # Peso base de la función de inferencia en C

    model_type = type(model).__name__

    # RED NEURONAL (MiniNeuralNetwork)
    if isinstance(model, ml_runtime.MiniNeuralNetwork):
        # Overhead: Código de forward pass, sigmoide, loops (~1.5KB)
        overhead_code = 1500
        
        # Pesos (Matrices W1, B1, W2, B2)
        # Se guardan en PROGMEM (Flash) como floats
        w1_count = len(model.W1) * len(model.W1[0])
        b1_count = len(model.B1)
        w2_count = len(model.W2) * len(model.W2[0])
        b2_count = len(model.B2)
        
        total_params = w1_count + b1_count + w2_count + b2_count
        flash_bytes = (total_params * FLOAT_SIZE) + overhead_code
        
        # SRAM: Buffers de activación (no se usa malloc, pero se usa stack)
        # Se necesita espacio para inputs, hidden layer output, output layer output
        # Arrays temporales en la función 'nn_predict'
        sram_bytes = (model.n_in + model.n_hid + model.n_out) * FLOAT_SIZE

    # ÁRBOLES DE DECISIÓN (DecisionTree)
    elif isinstance(model, (ml_runtime.DecisionTreeClassifier, ml_runtime.DecisionTreeRegressor)):
        # Overhead: Lógica de recorrido de árbol (~800B)
        overhead_code = 800
        
        # Contar nodos totales
        n_nodes = _count_tree_nodes(model.root)
        
        # Estructura aplanada en C: 
        # 5 arrays paralelos (idx, val, left, right, out)
        # idx(2B) + left(2B) + right(2B) + val(4B) + out(4B) = ~14 bytes por nodo
        # (Aunque optimizado podría ser menos, estimamos conservadoramente)
        flash_bytes = (n_nodes * 14) + overhead_code
        
        # SRAM: Mínima (variables de recorrido)
        sram_bytes = 50 

    # RANDOM FOREST
    elif isinstance(model, (ml_runtime.RandomForestClassifier, ml_runtime.RandomForestRegressor)):
        overhead_code = 1200 # Lógica de votación + árboles
        
        total_nodes = 0
        for tree in model.trees:
            total_nodes += _count_tree_nodes(tree.root)
            
        # Cada árbol suma peso en Flash
        flash_bytes = (total_nodes * 14) + overhead_code
        
        # SRAM: Array de votos (un float por árbol)
        sram_bytes = (len(model.trees) * FLOAT_SIZE) + 100

    # KNN
    elif isinstance(model, ml_runtime.KNearestNeighbors):
        overhead_code = 1000
        
        # KNN guarda todo el dataset en Flash
        n_samples = len(model.y_train)
        n_features = model.n_features_trained
        
        # Matriz X (n_samples * n_features * 4) + Vector y (n_samples * 4)
        dataset_size = (n_samples * n_features * FLOAT_SIZE) + (n_samples * FLOAT_SIZE)
        flash_bytes = dataset_size + overhead_code
        
        # SRAM: Buffer para calcular distancias
        sram_bytes = 100

    # MODELOS LINEALES / SVM
    elif isinstance(model, (ml_runtime.MiniLinearModel, ml_runtime.MiniSVM)):
        overhead_code = 600
        n_weights = len(model.weights)
        
        # Pesos + Bias
        flash_bytes = (n_weights * FLOAT_SIZE) + FLOAT_SIZE + overhead_code
        sram_bytes = 40

    else:
        # Fallback para modelos desconocidos
        return {
            "flash_percent": 0,
            "sram_percent": 0,
            "error": f"Estimador no implementado para {model_type}"
        }

    # CÁLCULO DE PORCENTAJES
    flash_pct = round((flash_bytes / target_flash) * 100, 2)
    sram_pct = round((sram_bytes / target_sram) * 100, 2)

    return {
        "model_type": model_type,
        "flash_bytes": int(flash_bytes),
        "flash_total": target_flash,
        "flash_percent": flash_pct,
        "sram_bytes": int(sram_bytes),
        "sram_total": target_sram,
        "sram_percent": sram_pct
    }

def _count_tree_nodes(node):
    """Cuenta nodos recursivamente."""
    if not isinstance(node, dict): 
        return 1 # Es una hoja escalar
    if node.get('index') == -1: 
        return 1 # Es una hoja dict
    # Es un nodo de decisión
    return 1 + _count_tree_nodes(node.get('left')) + _count_tree_nodes(node.get('right'))