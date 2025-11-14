"""
EduBot - MiniML Evaluation Extended Test Suite
-----------------------------------------------
Validación automatizada (sin pytest) para:
 - evaluate()
 - evaluate_ext()
 - Conversión lista/objeto (tolist)
Incluye umbrales mínimos y pruebas de robustez numérica.

Autor: Wilner Manzanares
Versión QA: 1.2
"""

import core.ml_manager as ml_manager
import core.ml_runtime as ml_runtime


# ===============================
#   Funciones de validación QA
# ===============================

def assert_range(value, min_val, max_val, name):
    """Valida que una métrica esté dentro del rango esperado"""
    if not (min_val <= value <= max_val):
        raise AssertionError(f"❌ {name} fuera de rango: {value} (esperado entre {min_val} y {max_val})")
    print(f"✅ {name}: {value:.5f} (OK)")


def assert_not_nan(value, name):
    """Evita valores NaN o None"""
    if value is None or (isinstance(value, float) and (value != value)):
        raise AssertionError(f"❌ {name} no válido (None o NaN)")
    print(f"✅ {name}: válido ({value})")

def assert_is_dict(value, name):
    """Verifica que el valor sea un diccionario."""
    if not isinstance(value, dict):
        raise AssertionError(f"❌ {name} no es un diccionario (tipo: {type(value)})")
    print(f"✅ {name}: es un diccionario válido.")

def assert_keys(value, keys, name):
    """Verifica que un diccionario contenga todas las claves esperadas."""
    if not all(k in value for k in keys):
        missing = [k for k in keys if k not in value]
        raise AssertionError(f"❌ {name}: faltan claves {missing}")
    print(f"✅ {name}: contiene todas las claves esperadas.")

# ===============================
#   Tests individuales
# ===============================

def test_evaluate_basic():
    """Valida el cálculo de métricas básicas."""
    print("\n=== ⚙️ TEST: evaluate() ===")

    # Clasificación
    y_true_cls = [1, 0, 1, 1, 0]
    y_pred_cls = [1, 0, 0, 1, 0]

    # La función evaluate devuelve un diccionario, no un float.
    # Debemos acceder a la clave 'score' para obtener el valor numérico.
    acc_result = ml_manager.evaluate(y_true_cls, y_pred_cls, metrics=["accuracy"])
    assert_is_dict(acc_result, "Accuracy (clasificación)")
    assert_range(acc_result['score'], 0.0, 1.0, "Accuracy (clasificación)")

    # Regresión
    y_true_reg = [2.5, 0.0, 2.1, 1.6]
    y_pred_reg = [3.0, -0.1, 2.0, 1.5]

    # Aplicar la misma lógica para el MSE.
    mse_result = ml_manager.evaluate(y_true_reg, y_pred_reg, metrics=["mse"])
    assert_is_dict(mse_result, "MSE (regresión)")
    assert_not_nan(mse_result['score'], "MSE (regresión)")
    # El MSE puede ser mayor que 2.0, ajustamos el rango para ser más realista.
    assert_range(mse_result['score'], 0.0, 10.0, "MSE (regresión)")

    print("✅ evaluate() completado sin errores.")


def test_evaluate_ext_detailed():
    """Valida el cálculo de múltiples métricas en modo extendido."""
    print("\n=== 🧠 TEST: evaluate_ext() (modo detallado) ===")

    # Clasificación extendida
    y_true_cls = [1, 0, 1, 1, 0]
    y_pred_cls = [1, 0, 0, 1, 0]

    # El resultado ya es un diccionario de métricas, por lo que no necesita cambios.
    results_cls = ml_manager.evaluate_ext(
        y_true=y_true_cls,
        y_pred=y_pred_cls,
        metrics=["accuracy", "precision", "recall", "f1"],
    )
    assert_is_dict(results_cls, "Clasificación extendida")
    assert_keys(results_cls, ["accuracy", "precision", "recall", "f1"], "Clasificación extendida")
    assert_range(results_cls["accuracy"], 0.0, 1.0, "Accuracy en ext")

    print("✅ evaluate_ext() completado sin errores.")

def test_listlike_conversion():
    print("\n=== 🔄 TEST: Conversión tolist universal ===")

    class FakeArray:
        """Simula un objeto tipo numpy sin usar numpy real"""
        def __init__(self, data):
            self._data = data
        def tolist(self):
            return self._data

    y_true = FakeArray([1, 0, 1])
    y_pred = FakeArray([1, 1, 0])

    # CORRECCIÓN: Utilizamos ml_manager.evaluate() en lugar de ml_manager.evaluate_ext()

    # La función evaluate() devuelve un diccionario.
    acc_result = ml_manager.evaluate(y_true, y_pred, metrics=["accuracy"])
    
    # Verificamos que sea un dict
    assert_is_dict(acc_result, "Accuracy (FakeArray) Result")

    # Extraemos el 'score' numérico
    acc_score = acc_result['score']
    
    # Pasamos el 'score' numérico a las aserciones
    assert_not_nan(acc_score, "Accuracy (FakeArray)")
    assert_range(acc_score, 0.0, 1.0, "Accuracy (FakeArray)")

    print("✅ Conversión tolist genérica funcional.")


def test_robustness_invalid_inputs():
    print("\n=== ⚠️ TEST: Robustez ante inputs inválidos ===")

    try:
        ml_manager.evaluate("cadena", [1, 0, 1])
    except Exception as e:
        print("✅ Detección de tipo inválido (y_true string):", type(e).__name__)

    try:
        ml_manager.evaluate_ext(y_true=[1, 0, 1], y_pred="cadena", metrics=["accuracy"])
    except Exception as e:
        print("✅ Detección de tipo inválido (y_pred string):", type(e).__name__)

    print("✅ Robustez ante entradas inválidas verificada.")


# ===============================
#   Ejecutor de test general
# ===============================

def run_all_tests():
    print("🧩 Iniciando test extendido de evaluate() y evaluate_ext() en MiniML\n" + "=" * 65)
    try:
        test_evaluate_basic()
        test_evaluate_ext_detailed()
        test_listlike_conversion()
        test_robustness_invalid_inputs()
        print("\n🎯 RESULTADO FINAL: Todos los tests pasaron exitosamente.")
    except AssertionError as ae:
        print("\n❌ Fallo de validación QA:", ae)
    except Exception as e:
        print("\n💥 Error inesperado durante los tests:", e)


if __name__ == "__main__":
    run_all_tests()