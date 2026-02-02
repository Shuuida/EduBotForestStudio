"""
EduBot - Tree Tracing & Diagnostics
-------------------------------------------------
Diagnóstico profundo de árboles de decisión.
Actualizado para soportar Hojas Escalares (Scalar Leaves) del Runtime v2.3+.
"""

import json
import time
import traceback
from core import ml_runtime, ml_manager

def trace_prediction(node, row, depth=0):
    """
    Traza el recorrido de una predicción a través del árbol.
    Soporta nodos diccionario y hojas escalares optimizadas.
    """
    indent = "  " * depth
    if node is None:
        return f"{indent}❌ Nodo nulo — predicción abortada.\n"

    if isinstance(node, (int, float)):
        return f"{indent}🌿 Hoja alcanzada (Escalar) → valor: {node}\n"
    # ---------------------------------------------------------------

    # Detección de hoja formato antiguo (Diccionario)
    if isinstance(node, dict):
        # Si no tiene 'left' o 'index' es -1, es una hoja legacy o wrapper
        if node.get("index", -1) == -1 or "left" not in node:
            val = node.get("value", "N/A")
            return f"{indent}🌿 Hoja alcanzada (Dict) → valor: {val}\n"
        
        # Es un nodo de decisión
        index = node.get("index")
        value = node.get("value") # Umbral
        
        # Validación de integridad
        if index is None or value is None:
             return f"{indent}❓ Nodo corrupto (sin index/value): {node}\n"

        # Validación de datos de entrada
        if index >= len(row):
            return f"{indent}❌ Índice fuera de rango (idx={index}, len={len(row)})\n"

        # Lógica de recorrido
        cond = row[index] <= value # Usamos <= para coincidir con runtime
        branch = "izquierda" if cond else "derecha"
        log = f"{indent}🔎 Nodo (idx={index}, umbral={value:.2f}) → val={row[index]} → {branch}\n"

        # Descenso recursivo
        next_node = node["left"] if cond else node["right"]
        return log + trace_prediction(next_node, row, depth + 1)

    return f"{indent}❓ Formato de nodo desconocido: {type(node)}\n"


def run_pipeline(mode="classification"):
    print(f"\n🚀 Iniciando diagnóstico de pipeline: {mode.upper()}")
    
    try:
        # Datos sintéticos
        if mode == "classification":
            # Dataset OR gate: x1=0,x2=1 -> 1
            data = [[0,0,0], [0,1,1], [1,0,1], [1,1,1]]
            model = ml_runtime.DecisionTreeClassifier(max_depth=3)
            test_row = [0, 1] # Esperado: 1
        else:
            # Dataset escalón
            data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
            model = ml_runtime.DecisionTreeRegressor(max_depth=3)
            test_row = [1.5] # Esperado: ~1.0 o ~2.0

        model.fit(data)
        print("✅ Modelo entrenado en memoria.")
        
        # Probar trazado
        print("📜 Trazado de predicción:")
        if hasattr(model, 'root') and model.root is not None:
            print(trace_prediction(model.root, test_row))
            
            # Predicción real para confirmar
            pred = model.predict([test_row])[0]
            print(f"✅ Predicción ejecutada: {pred}")
        else:
            print("⚠️ El modelo no tiene estructura 'root' accesible.")
            
    except Exception as e:
        print(f"❌ Error en pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline("classification")
    run_pipeline("regression")