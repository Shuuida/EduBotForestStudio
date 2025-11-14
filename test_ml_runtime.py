"""
EduBot MiniML Runtime - Full Diagnostic Test (CON VALIDACIÓN DE EXPORTACIÓN A C)
==================================================================================

🔍 Propósito:
- Validar todas las clases de ml_runtime (DecisionTree, RandomForest, LinearRegression, SVM, NeuralNetwork)
- Comprobar estabilidad numérica y funcionamiento del backpropagation multicapa
- Validar la generación de código `to_arduino_code` para todos los modelos exportables.
- Exportar pesos y matrices de la red neuronal en formato JSON legible para firmware C / Arduino / MegaPi

🧩 Dependencias: Ninguna (sin pytest, sin numpy)

CORRECCIÓN APLICADA: Se ha hecho más inteligente `_smoke_test_c_code` para
validar la intención (ej. 'while' para árboles, '[]' para arrays) y
no fallar en funciones 'float' o 'void' que no contengan 'int'.
"""

import json
import traceback
from core import ml_runtime


# ============================================================
# 🧰 Función auxiliar para exportar a firmware
# ============================================================

def export_to_firmware_json(model, filename="firmware_export.json"):
    """Exporta pesos, biases y topología a formato JSON para firmware."""
    try:
        data = {
            "model_type": model.__class__.__name__,
            "weights": {
                "W1": getattr(model, "W1", []),
                "W2": getattr(model, "W2", []),
            },
            "biases": {
                "B1": getattr(model, "B1", []),
                "B2": getattr(model, "B2", []),
            },
            "meta": {
                "n_inputs": getattr(model, "n_inputs", None),
                "n_hidden": getattr(model, "n_hidden", None),
                "n_outputs": getattr(model, "n_outputs", None),
                "learning_rate": getattr(model, "learning_rate", None),
                "epochs": getattr(model, "epochs", None),
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Exportado correctamente a {filename}")
    except Exception as e:
        print(f"❌ Error exportando modelo a firmware JSON: {e}")
        traceback.print_exc()


# ============================================================
# 🧰 <--- TEST DE HUMO C CORREGIDO --->
# ============================================================

def _smoke_test_c_code(c_code: str, model_name: str):
    """
    Realiza un 'smoke test' inteligente en el código C generado.
    """
    print(f"\n--- ⚡ VALIDANDO EXPORTACIÓN C: {model_name} ---")
    if not isinstance(c_code, str) or len(c_code) < 40:
        raise ValueError(f"Test de Humo C FALLIDO: El código C para {model_name} es demasiado corto o inválido.")

    # --- INICIO DE LA LÓGICA CORREGIDA ---
    
    # 1. Palabras clave base que esperamos ver
    if "float" not in c_code:
        raise ValueError(f"Test de Humo C FALLIDO: Código de {model_name} no contiene 'float'.")
    if "(" not in c_code or ")" not in c_code:
        raise ValueError(f"Test de Humo C FALLIDO: Código de {model_name} no parece tener una función '()'.")

    # 2. O es una función que retorna (return) o es void
    if "return" not in c_code and "void" not in c_code:
         raise ValueError(f"Test de Humo C FALLIDO: Código de {model_name} no contiene 'return' ni 'void'.")

    # 3. Los árboles (DecisionTree, RandomForest) DEBEN ser iterativos (usar 'while')
    if ("Tree" in model_name or "Forest" in model_name) and "while" not in c_code:
        raise ValueError(f"Test de Humo C FALLIDO: El código de {model_name} NO es iterativo (no se encontró 'while').")

    # 4. Los modelos lineales/SVM/NN DEBEN definir un array
    if ("Linear" in model_name or "SVM" in model_name or "Neural" in model_name) and "]" not in c_code:
        raise ValueError(f"Test de Humo C FALLIDO: El código de {model_name} no parece definir un array '[]'.")
        
    # --- FIN DE LA LÓGICA CORREGIDA ---
    
    print(f"✅ Validación C de {model_name} superada.")
    print("-------------------------------------------------")
    # Imprime un fragmento del código
    print("\n".join(c_code.splitlines()[:10]) + "\n[...]")
    print("-------------------------------------------------")


# ============================================================
# 🌳 Árboles de decisión y bosques
# ============================================================

def test_decision_tree():
    print("\n--- 🌳 TEST: DecisionTreeClassifier ---")
    dataset = [
        [2.7, 2.5, 0],
        [1.3, 3.5, 0],
        [3.5, 1.4, 1],
        [3.9, 4.0, 1],
    ]
    try:
        model = ml_runtime.DecisionTreeClassifier(max_depth=3, min_size=1)
        model.fit(dataset)
        preds = model.predict([[2.5, 2.3], [3.7, 3.9]])
        print("Predicciones Python:", preds)
        
        # Test de exportación a C
        c_code = model.to_arduino_code()
        _smoke_test_c_code(c_code, "DecisionTreeClassifier")
        
    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()


def test_random_forest():
    print("\n--- 🌲 TEST: RandomForestClassifier ---")
    dataset = [
        [2.7, 2.5, 0],
        [1.3, 3.5, 0],
        [3.5, 1.4, 1],
        [3.9, 4.0, 1],
    ]
    try:
        model = ml_runtime.RandomForestClassifier(n_trees=3, max_depth=3, min_size=1, seed=42)
        model.fit(dataset)
        preds = model.predict([[2.5, 2.3], [3.7, 3.9]])
        print("Predicciones Python:", preds)
        
        # Test de exportación a C
        c_code = model.to_arduino_code()
        _smoke_test_c_code(c_code, "RandomForestClassifier")
        
    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()


# ============================================================
# 📈 Regresión lineal
# ============================================================

def test_linear_regression():
    print("\n--- 📈 TEST: LinearRegression (MiniLinearModel) ---")
    dataset = [
        [1.0, 2.0],
        [2.0, 3.9],
        [3.0, 6.1],
        [4.0, 8.2],
    ]
    try:
        model = ml_runtime.MiniLinearModel()
        model.fit(dataset)
        preds = model.predict([[5.0], [6.0]])
        print("Predicciones Python:", preds)
        print("Coeficientes:", getattr(model, 'weights', None))
        
        # Test de exportación a C
        c_code = model.to_arduino_code()
        _smoke_test_c_code(c_code, "MiniLinearModel")
        
    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()


# ============================================================
# ⚙️ SVM
# ============================================================

def test_svm():
    print("\n--- ⚙️ TEST: MiniSVM ---")
    dataset = [
        [2.7, 2.5, 1],
        [1.3, 3.5, -1],
        [3.5, 1.4, 1],
        [3.9, 4.0, -1],
    ]
    try:
        model = ml_runtime.MiniSVM(learning_rate=0.01, lambda_param=0.01, n_iters=500)
        model.fit(dataset)
        preds = model.predict([[2.5, 2.3], [3.7, 3.9]])
        print("Predicciones Python:", preds)
        print("Vector de pesos:", getattr(model, 'weights', None))
        
        # Test de exportación a C
        c_code = model.to_arduino_code()
        _smoke_test_c_code(c_code, "MiniSVM")
        
    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()


# ============================================================
# 🧠 Red neuronal (XOR + diagnóstico extendido)
# ============================================================

def test_neural_network():
    print("\n--- 🧠 TEST: MiniNeuralNetwork (XOR + Exportación Firmware) ---")
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]  # XOR clásico

    try:
        model = ml_runtime.MiniNeuralNetwork(
            n_inputs=2,
            n_hidden=4,
            n_outputs=1,
            learning_rate=0.1,
            epochs=3000,
            seed=42
        )
        model.fit(X, y)
        preds = model.predict(X)
        print("Predicciones XOR (Python):", preds)

        # Diagnóstico extendido
        print("\n🔍 Diagnóstico de Backpropagation:")
        # Imprime solo una muestra para no saturar la consola
        print("W1 (muestra):", [row[:2] for row in model.W1[:2]])
        print("W2 (muestra):", [row[:2] for row in model.W2[:2]])

        # Verificación de control de overflow sigmoid
        test_vals = [-100, -10, 0, 10, 100]
        results = [model.sigmoid(v) for v in test_vals]
        print("\n🧮 Verificación de overflow en sigmoid:")
        for val, r in zip(test_vals, results):
            print(f"sigmoid({val}) = {r:.6f}")

        # Cálculo del MSE
        mse = sum((preds[i][0] - y[i][0]) ** 2 for i in range(len(y))) / len(y)
        print(f"\nMSE promedio: {mse:.6f}")

        # Exportar pesos a firmware JSON (para NN, esto es diferente de C)
        export_to_firmware_json(model, "firmware_export.json")
        
        # Test de exportación a C
        c_code = model.to_arduino_code()
        _smoke_test_c_code(c_code, "MiniNeuralNetwork")

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()


# ============================================================
# 🚀 MAIN
# ============================================================

def main():
    print("============================================================")
    print("🧩 Iniciando test extendido de ml_runtime.py (FULL MODE)")
    print("   (Incluye validación de exportación a C)")
    print("============================================================")

    test_decision_tree()
    test_random_forest()
    test_linear_regression()
    test_svm()
    test_neural_network()

    print("\n============================================================")
    print("✅ TEST COMPLETO FINALIZADO (INCLUYENDO EXPORTACIÓN A C)")
    print("============================================================")


if __name__ == "__main__":
    main()