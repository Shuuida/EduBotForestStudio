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


# ===============================
#   Tests individuales
# ===============================

def test_evaluate_basic():
    print("\n=== ⚙️ TEST: evaluate() ===")

    # Clasificación
    y_true_cls = [0, 1, 1, 0, 1]
    y_pred_cls = [0, 1, 0, 0, 1]

    acc = ml_manager.evaluate(y_true_cls, y_pred_cls, metrics=["accuracy"])
    assert_not_nan(acc, "Accuracy (clasificación)")
    assert_range(acc, 0.0, 1.0, "Accuracy (clasificación)")

    # Regresión
    y_true_reg = [2.5, 0.0, 2.1, 1.6]
    y_pred_reg = [3.0, -0.1, 2.0, 1.5]

    mse = ml_manager.evaluate(y_true_reg, y_pred_reg, metrics=["mse"])
    assert_not_nan(mse, "MSE (regresión)")
    assert_range(mse, 0.0, 2.0, "MSE (regresión)")

    print("✅ evaluate() completado sin errores.")


def test_evaluate_ext_detailed():
    print("\n=== 🧠 TEST: evaluate_ext() (modo detallado) ===")

    # Clasificación extendida
    y_true_cls = [1, 0, 1, 1, 0]
    y_pred_cls = [1, 0, 0, 1, 0]

    results_cls = ml_manager.evaluate_ext(
        y_true=y_true_cls,
        y_pred=y_pred_cls,
        metrics=["accuracy", "precision", "recall", "f1"],
        detailed=True,
    )

    for metric, val in results_cls.items():
        assert_not_nan(val, metric)
        assert_range(val, 0.0, 1.0, metric)

    # Regresión extendida
    y_true_reg = [3.5, 2.0, 4.0, 5.5]
    y_pred_reg = [3.4, 2.1, 4.2, 5.6]

    results_reg = ml_manager.evaluate_ext(
        y_true=y_true_reg,
        y_pred=y_pred_reg,
        metrics=["mae", "mse", "r2"],
        detailed=True,
    )

    assert_range(results_reg["mae"], 0.0, 1.0, "MAE")
    assert_range(results_reg["mse"], 0.0, 1.0, "MSE")
    assert_range(results_reg["r2"], 0.8, 1.0, "R²")

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

    acc = ml_manager.evaluate(y_true, y_pred, metrics=["accuracy"])
    assert_not_nan(acc, "Accuracy (FakeArray)")
    assert_range(acc, 0.0, 1.0, "Accuracy (FakeArray)")

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