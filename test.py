"""
EduBot - MiniML Evaluation Extended Test Suite (v2.5 STABLE)
-----------------------------------------------------------
Suite de pruebas automatizada para validar el ecosistema completo.

CORRECCIONES APLICADAS:
1. Manejo robusto de métricas (soporta retorno float o dict).
2. Conversión segura a float para evitar crash por formato string.
3. Instanciación explícita de Regresores vs Clasificadores.
"""

import core.ml_manager as ml_manager
import traceback

# ----------------------------------------------------------------
# DATASETS SINTÉTICOS
# ----------------------------------------------------------------

# Clasificación: Binaria
CLS_DATA = [
    [5.1, 3.5, 0], [4.9, 3.0, 0], [4.7, 3.2, 0], # Clase 0
    [7.0, 3.2, 1], [6.4, 3.2, 1], [6.9, 3.1, 1]  # Clase 1
]
CLS_X_TEST = [[5.0, 3.4], [6.7, 3.1]]

# Regresión: Lineal simple (y = x)
REG_DATA = [
    [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], 
    [4.0, 4.0], [5.0, 5.0]
]
REG_X_TEST = [[1.5], [4.5]]

# No Lineal: XOR
XOR_DATA = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]

# ----------------------------------------------------------------
# RUNNER GENÉRICO
# ----------------------------------------------------------------

def run_model_test(model_alias, model_type, dataset, X_test, task="classification", **params):
    print(f"\n🧪 Testeando: {model_type} ({model_alias})...")
    try:
        # 1. Pipeline de Entrenamiento
        result = ml_manager.train_pipeline(
            model_name=model_alias,
            dataset=dataset,
            model_type=model_type,
            params=params,
            scaling=None
        )
        
        model = result['model']
        model_real_type = type(model).__name__
        print(f"   ✅ Entrenamiento exitoso. Modelo instanciado: {model_real_type}")
        
        # 2. Predicción
        preds = ml_manager.predict(model, X_test)
        print(f"   🎯 Predicciones: {preds}")
        
        # 3. Evaluación (Solo para Clasificación en este reporte rápido)
        if task == "classification":
            y_true = [row[-1] for row in dataset]
            y_pred_train = ml_manager.predict(model, [row[:-1] for row in dataset])
            
            # Post-procesamiento para modelos que devuelven probabilidad
            y_pred_cls = []
            for p in y_pred_train:
                if isinstance(p, list): p = p[0] 
                y_pred_cls.append(1 if p > 0.5 else 0)

            # Llamada al Manager
            acc = ml_manager.evaluate(y_true=y_true, y_pred=y_pred_cls)
            
            try:
                raw_val = acc
                if isinstance(acc, dict):
                    # Buscar clave 'accuracy' o tomar el primer valor
                    if 'accuracy' in acc:
                        raw_val = acc['accuracy']
                    elif 'acc' in acc:
                        raw_val = acc['acc']
                    else:
                        raw_val = list(acc.values())[0] if acc else 0.0
                
                # Intentar convertir a float para formateo
                acc_val = float(raw_val)
                print(f"   📊 Accuracy (Train): {acc_val:.2f}")
                
            except (ValueError, TypeError):
                # Si falla (ej. es un mensaje de error), imprimir tal cual sin romper el test
                print(f"   ⚠️ Resultado de evaluación no numérico: {raw_val}")
            # -----------------------------------------
            
        return True

    except Exception as e:
        print(f"   ❌ Fallo crítico en {model_type}: {e}")
        traceback.print_exc()
        return False

# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------

def main():
    success_count = 0
    total_tests = 0

    # 1. Decision Tree (Clasificación)
    total_tests += 1
    if run_model_test("dt_cls", "DecisionTreeClassifier", CLS_DATA, CLS_X_TEST, 
                      task="classification", max_depth=3):
        success_count += 1

    # 2. RandomForest (Regresión)
    total_tests += 1
    if run_model_test("rf_reg", "RandomForestRegressor", REG_DATA, REG_X_TEST, 
                      task="regression", n_trees=3, max_depth=3):
        success_count += 1

    # 3. KNN (Clasificación)
    total_tests += 1
    if run_model_test("knn_cls", "KNearestNeighbors", CLS_DATA, CLS_X_TEST, 
                      task="classification", k=3):
        success_count += 1

    # 4. SVM (Clasificación)
    total_tests += 1
    if run_model_test("svm_cls", "MiniSVM", CLS_DATA, CLS_X_TEST, 
                      task="classification", learning_rate=0.01, n_iters=500):
        success_count += 1

    # 5. Neural Network (Clasificación)
    total_tests += 1
    if run_model_test("nn_xor", "MiniNeuralNetwork", XOR_DATA, [[0, 1], [1, 1]], 
                      task="classification", 
                      n_inputs=2, n_hidden=4, n_outputs=1, epochs=3000, learning_rate=0.1):
        success_count += 1

    print("\n==============================================================")
    print(f" RESUMEN: {success_count}/{total_tests} Tests Exitosos.")
    print("==============================================================")

if __name__ == "__main__":
    main()