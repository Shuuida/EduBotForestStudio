"""
EduBot Ecosystem Integration Test
========================================
Verifica la interoperabilidad completa con la API actualizada.
- Adapta exportación inversa a la nueva función export_model_to_block_structure.
- Verifica serialización de nuevos modelos (KNN).
"""

import os
import json
from core import ml_adapter, ml_manager
from storage import ml_exporter
from storage import file_handler

def setup_environment():
    file_handler.ensure_dir_exist()
    print("[INIT] Entorno preparado.")

# ============================================================
# TEST 1: ENTRENAMIENTO Y PERSISTENCIA
# ============================================================

def test_integration_flow():
    print("\n--- 🧠 TEST 1: Flujo Completo (Train -> Save -> Load) ---")

    dataset = [[1.0, 2.1], [2.0, 4.2], [3.0, 6.3], [4.0, 8.4]]
    
    # Entrenar modelo simple
    print("[1] Entrenando modelo...")
    ml_manager.train_pipeline(
        model_name="integration_test_model",
        dataset=dataset,
        model_type="DecisionTree",
        params={"max_depth": 3},
        scaling=None
    )
    
    # Guardar usando file_handler
    print("[2] Guardando modelo...")
    saved = file_handler.save_model("integration_test_model", "test_model_v2")
    assert saved, "❌ Falló guardado de modelo"
    
    # Cargar
    print("[3] Cargando modelo...")
    model = file_handler.load_model("test_model_v2")
    assert model is not None, "❌ Falló carga de modelo"
    print("✅ Integración básica OK.")

# ============================================================
# TEST 2: EXPORTACIÓN INVERSA A BLOQUES (Refactorizado)
# ============================================================

def test_export_inverse_blocks():
    print("\n--- 🔁 TEST 2: Exportación inversa a bloques (Nueva API) ---")

    # Crear un KNN manual para probar
    from core.ml_runtime import KNearestNeighbors
    knn = KNearestNeighbors(k=3)
    knn.X_train = [[1,1], [2,2]]
    knn.y_train = [0, 1]
    knn.n_features_trained = 2
    
    # Probamos la nueva función del exporter
    try:
        # Nota: La función ahora se llama export_model_to_block_structure
        block_struct = ml_exporter.export_model_to_block_structure(knn, "knn_visual_test")
        
        assert block_struct is not None
        assert block_struct['type'] == 'ml_train_knn'
        assert block_struct['k'] == 3
        
        print("[INFO] Bloque generado:", json.dumps(block_struct, indent=2))
        print("✅ Exportación inversa (KNN -> Bloque) estable.")
    except Exception as e:
        print(f"❌ Error en exportación inversa: {e}")
        raise e

# ============================================================
# TEST MASTER
# ============================================================

def run_all_tests():
    setup_environment()
    test_integration_flow()
    test_export_inverse_blocks()

if __name__ == "__main__":
    run_all_tests()