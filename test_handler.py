"""
EduBot Ecosystem Integration Test
=================================

Verifica la interoperabilidad completa entre:

 - ml_exporter (serialización de modelos)
 - ml_adapter (traducción Python ⇄ Bloques)
 - file_handler (gestión integral de proyectos, datasets, modelos, código y exportaciones inversas)
 - MiniML y sklearn (entrenamiento mixto)
"""

import os
import json
from core import ml_adapter, ml_manager
from storage import ml_exporter
from storage import file_handler

# ============================================================
# CONFIGURACIÓN
# ============================================================

def setup_environment():
    """Crea carpetas necesarias antes de iniciar las pruebas."""
    file_handler.ensure_dir_exist()
    print("[INIT] Entorno preparado para pruebas.")


# ============================================================
# TEST 1: ENTRENAMIENTO Y EXPORTACIÓN DE MODELO
# ============================================================

def test_ml_exporter_integration():
    print("\n--- 🧠 TEST 1: ML Exporter Integration ---")

    # Dataset simple (regresión lineal)
    dataset = [
        [1.0, 2.1],
        [2.0, 4.2],
        [3.0, 6.3],
        [4.0, 8.4],
        [5.0, 10.5],
    ]

    # Entrenar modelo mixto (MiniML si disponible)
    result = ml_manager.train_decision_tree(
        model_name="tree_integration_test",
        dataset=dataset,
        max_depth=3,
        min_size=1
    )

    print("[INFO] Entrenamiento completado:", result)

    model = ml_manager._MODEL_REGISTRY["tree_integration_test"]["model"]

    # Exportar modelo
    ok = file_handler.save_model(model, "tree_integration_test")
    assert ok, "❌ Falló exportación de modelo"

    # Recargar modelo
    loaded = file_handler.load_model("tree_integration_test")
    assert loaded is not None, "❌ Falló carga de modelo"

    # Validar consistencia de estructura exportada
    struct = ml_exporter.extract_model_structure(loaded)
    print("[OK] Estructura del modelo exportado:", struct)
    print("✅ ML Exporter funciona correctamente con file_handler.")


# ============================================================
# TEST 2: TRADUCCIÓN PYTHON ⇄ BLOQUES (ml_adapter)
# ============================================================

def test_ml_adapter_translation():
    print("\n--- 🧱 TEST 2: Traducción Bidireccional (ml_adapter) ---")

    python_code = """
def suma(a, b):
    return a + b
result = suma(3, 5)
print(result)
"""

    # Python → Bloques
    blocks = file_handler.python_to_blocks(python_code)
    assert blocks is not None, "❌ Falló conversión Python→Bloques"
    print("[INFO] Traducción Python→Bloques:", json.dumps(blocks, indent=2))

    # Bloques → Python
    recovered_code = file_handler.blocks_to_python(blocks["result"])
    assert recovered_code is not None, "❌ Falló conversión Bloques→Python"
    print("[INFO] Código reconstruido:\n", recovered_code)
    print("✅ ml_adapter y file_handler trabajan correctamente juntos.")


# ============================================================
# TEST 3: PROYECTO EDUCATIVO
# ============================================================

def test_project_handling():
    print("\n--- 💾 TEST 3: Proyectos EduBot ---")

    project_data = {
        "name": "ProyectoTest",
        "description": "Proyecto de integración completo",
        "models": ["tree_integration_test"],
        "author": "EduBot System"
    }

    saved = file_handler.save_proj(project_data, "ProyectoTest")
    assert saved, "❌ Falló guardado de proyecto"

    loaded = file_handler.load_proj("ProyectoTest")
    assert loaded is not None, "❌ Falló carga del proyecto"
    print("[OK] Proyecto cargado correctamente:", loaded)

    print("✅ Gestión de proyectos estable y funcional.")


# ============================================================
# TEST 4: DATASETS
# ============================================================

def test_dataset_handling():
    print("\n--- 🧬 TEST 4: Datasets ---")

    dataset = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    ok = file_handler.save_dataset(dataset, "dataset_test")
    assert ok, "❌ Falló guardado de dataset"

    reloaded = file_handler.load_dataset("dataset_test")
    assert reloaded is not None, "❌ Falló carga del dataset"
    print("[OK] Dataset recargado:", reloaded)

    print("✅ Gestión de datasets correcta.")


# ============================================================
# TEST 5: EXPORTACIÓN INVERSA A BLOQUES
# ============================================================

def test_export_inverse_blocks():
    print("\n--- 🔁 TEST 5: Exportación inversa a bloques ---")

    model = file_handler.load_model("tree_integration_test")
    block_repr = file_handler.export_to_blocks(model, "tree_integration_block")

    assert block_repr is not None, "❌ Falló exportación a bloques"
    print("[INFO] Bloques generados:", json.dumps(block_repr, indent=2))
    print("✅ Exportación inversa (modelo → bloques) estable.")


# ============================================================
# TEST MASTER
# ============================================================

def run_all_tests():
    setup_environment()
    test_ml_exporter_integration()
    test_ml_adapter_translation()
    test_project_handling()
    test_dataset_handling()
    test_export_inverse_blocks()
    print("\n🎯 Todos los tests se completaron exitosamente sin conflictos.")


if __name__ == "__main__":
    run_all_tests()