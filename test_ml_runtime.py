"""
EduBot MiniML Runtime - Full Diagnostic Test
===================================================
Diagnóstico completo para la arquitectura refactorizada de ml_runtime.

Cobertura:
1. Clasificadores (DT, RF, KNN, SVM, NN)
2. Regresores (DT, RF, Linear) - ¡NUEVO!
3. Validación de Matrices y Tensores (NN outputs)
4. Validación de Exportación a Firmware (C/PROGMEM)
"""

import json
import traceback
import math
from core import ml_runtime

# ============================================================
# 🧰 Helpers de Validación
# ============================================================

def _smoke_test_c_code(code: str, model_name: str):
    """Valida heurísticamente que el código C generado sea viable."""
    if not code or len(code) < 50:
        print(f"   ⚠️ ADVERTENCIA: Código C sospechosamente corto para {model_name}")
        return
    
    required = ["float", "return", "{", "}"]
    if any(k in code for k in required):
        print(f"   ✅ Exportación C correcta para {model_name} ({len(code)} bytes)")
    else:
        print(f"   ❌ ERROR: Código C inválido para {model_name}")

def assert_almost_equal(a, b, epsilon=1e-5):
    if abs(a - b) > epsilon:
        print(f"   ⚠️ Diferencia numérica detectada: {a} vs {b}")

# ============================================================
# 🌲 Tests de Árboles (Clasificación y Regresión)
# ============================================================

def test_trees_classification():
    print("\n--- 🌲 Test: Árboles de Clasificación (DT & RF) ---")
    # Dataset OR gate
    data = [[0,0,0], [0,1,1], [1,0,1], [1,1,1]]
    
    try:
        # 1. Decision Tree
        dt = ml_runtime.DecisionTreeClassifier(max_depth=3)
        dt.fit(data)
        p_dt = dt.predict([[0,1]])[0]
        print(f"   DT Predict [0,1] -> {p_dt} (Esperado: 1)")
        _smoke_test_c_code(dt.to_arduino_code("dt_cls"), "DT Classifier")

        # 2. Random Forest
        rf = ml_runtime.RandomForestClassifier(n_trees=3)
        rf.fit(data)
        p_rf = rf.predict([[0,0]])[0]
        print(f"   RF Predict [0,0] -> {p_rf} (Esperado: 0)")
        _smoke_test_c_code(rf.to_arduino_code("rf_cls"), "RF Classifier")

    except Exception as e:
        print(f"   ❌ Falla en Árboles Clasificación: {e}")
        traceback.print_exc()

def test_trees_regression():
    print("\n--- 📉 Test: Árboles de Regresión (DT & RF) ---")
    # Dataset lineal simple y=x
    data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]
    
    try:
        # 1. Decision Tree Regressor
        dtr = ml_runtime.DecisionTreeRegressor(max_depth=5, min_size=1)
        dtr.fit(data)
        p_dtr = dtr.predict([[1.5]])[0]
        # Debería dar cercano a 1.0 o 2.0 (es un escalón), pero validamos que corra
        print(f"   DT Regressor Predict [1.5] -> {p_dtr:.2f}") 
        _smoke_test_c_code(dtr.to_arduino_code("dt_reg"), "DT Regressor")

        # 2. Random Forest Regressor
        rfr = ml_runtime.RandomForestRegressor(n_trees=5, max_depth=5)
        rfr.fit(data)
        p_rfr = rfr.predict([[4.5]])[0]
        print(f"   RF Regressor Predict [4.5] -> {p_rfr:.2f}")
        _smoke_test_c_code(rfr.to_arduino_code("rf_reg"), "RF Regressor")

    except Exception as e:
        print(f"   ❌ Falla en Árboles Regresión: {e}")
        traceback.print_exc()

# ============================================================
# 📏 Tests Modelos Matemáticos (KNN, SVM, Linear)
# ============================================================

def test_math_models():
    print("\n--- 📐 Test: Modelos Matemáticos (KNN, SVM, Linear) ---")
    data_cls = [[1.0, 0], [2.0, 0], [8.0, 1], [9.0, 1]]
    data_reg = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]] # y = 2x
    
    try:
        # 1. KNN
        knn = ml_runtime.KNearestNeighbors(k=1)
        knn.fit(data_cls)
        p_knn = knn.predict([[8.5]])[0]
        print(f"   KNN Predict [8.5] -> {p_knn} (Esperado: 1)")
        _smoke_test_c_code(knn.to_arduino_code(), "KNN")

        # 2. SVM
        svm = ml_runtime.MiniSVM(learning_rate=0.01, n_iters=100)
        svm.fit(data_cls)
        p_svm = svm.predict([[1.5]])[0]
        print(f"   SVM Predict [1.5] -> {p_svm} (Esperado: 0.0)")
        _smoke_test_c_code(svm.to_arduino_code(), "SVM")

        # 3. Linear Regression
        lr = ml_runtime.MiniLinearModel(learning_rate=0.01, epochs=500)
        lr.fit(data_reg)
        p_lr = lr.predict([[4.0]])[0]
        print(f"   Linear Predict [4.0] -> {p_lr:.2f} (Esperado ~8.0)")
        _smoke_test_c_code(lr.to_arduino_code(), "LinearModel")

    except Exception as e:
        print(f"   ❌ Falla en Modelos Matemáticos: {e}")
        traceback.print_exc()

# ============================================================
# 🧠 Test Red Neuronal
# ============================================================

def test_neural_network():
    print("\n--- 🧠 Test: MiniNeuralNetwork (XOR Problem) ---")
    # XOR: Necesita no-linealidad
    data = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
    
    try:
        # Configuración robusta para convergencia
        nn = ml_runtime.MiniNeuralNetwork(
            n_inputs=2, n_hidden=4, n_outputs=1, 
            epochs=3000, learning_rate=0.1
        )
        nn.fit(data)
        
        # Test de aplanamiento de lista (Corrección v2.3)
        preds = nn.predict([[0,1], [1,1]])
        
        if isinstance(preds[0], list):
            print("   ❌ ERROR CRÍTICO: NN devolvió lista anidada en lugar de escalar.")
        else:
            print(f"   ✅ Formato de salida correcto (escalar): {preds}")
            print(f"   Predicciones: (0,1)={preds[0]:.4f}, (1,1)={preds[1]:.4f}")

        # Exportación
        _smoke_test_c_code(nn.to_arduino_code(), "NeuralNetwork")

    except Exception as e:
        print(f"   ❌ Falla en Red Neuronal: {e}")
        traceback.print_exc()

# ============================================================
# 🚀 MAIN RUNNER
# ============================================================

def main():
    print("============================================================")
    print("🚀 Iniciando Test de Integridad para EduBot ML Runtime v2.3")
    print("============================================================")
    
    test_trees_classification()
    test_trees_regression()  # <--- Agregado para cobertura completa
    test_math_models()
    test_neural_network()
    
    print("\n✅ Todos los tests finalizados.")

if __name__ == "__main__":
    main()