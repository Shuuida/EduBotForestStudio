"""
Reglas de traducción Bloques -> Código Python (Machine Learning).
ACTUALIZADO: Sincronizado con las firmas del nuevo ml_runtime (MiniML Engine).
"""

from typing import Dict, Any

def ml_dataset_block_to_code(block: Dict[str, Any]) -> str:
    """Convierte un bloque de dataset en código Python seguro."""
    try:
        name = block.get("name", "data")
        source = block.get("source", "inline")

        if source == "csv":
            path = block.get("path", "")
            return (
                "import csv\n"
                f"{name} = []\n"
                f"with open(r'{path}', 'r', encoding='utf-8') as _f:\n"
                f"    _r = csv.reader(_f)\n"
                f"    for _row in _r:\n"
                f"        try:\n"
                f"            _row_vals = [float(x) for x in _row]\n"
                f"        except Exception:\n"
                f"            _row_vals = _row\n"
                f"        {name}.append(_row_vals)\n"
            )
        else:
            data_literal = block.get("data", "[]")
            return f"{name} = {data_literal}\n"

    except Exception as e:
        return f"# Error en ml_dataset_block_to_code: {e}\n"


def ml_train_dt_block_to_code(block: Dict[str, Any]) -> str:
    try:
        ds = block.get("dataset", "data")
        model = block.get("model_name", "dt_model")
        max_depth = block.get("max_depth", 10)
        min_size = block.get("min_size", 1)
        n_features = block.get("n_features", "None")
        return (
            f"from core.ml_runtime import DecisionTreeClassifier\n"
            f"{model} = DecisionTreeClassifier(max_depth={max_depth}, min_size={min_size}, n_features={n_features})\n"
            f"{model}.fit({ds})\n"
        )
    except Exception as e:
        return f"# Error en ml_train_dt_block_to_code: {e}\n"


def ml_train_rf_block_to_code(block: Dict[str, Any]) -> str:
    try:
        ds = block.get("dataset", "data")
        model = block.get("model_name", "rf_model")
        n_trees = block.get("n_trees", 5)
        max_depth = block.get("max_depth", 10)
        min_size = block.get("min_size", 1)
        sample_size = block.get("sample_size", 1.0)
        n_features = block.get("n_features", "None")
        seed = block.get("seed", "None")
        return (
            f"from core.ml_runtime import RandomForestClassifier\n"
            f"{model} = RandomForestClassifier(n_trees={n_trees}, max_depth={max_depth}, min_size={min_size}, "
            f"sample_size={sample_size}, n_features={n_features}, seed={seed})\n"
            f"{model}.fit({ds})\n"
        )
    except Exception as e:
        return f"# Error en ml_train_rf_block_to_code: {e}\n"


def ml_train_svm_block_to_code(block: Dict[str, Any]) -> str:
    """Entrenamiento de SVM (Actualizado: epochs -> n_iters)."""
    try:
        ds = block.get("dataset", "data")
        model = block.get("model_name", "svm_model")
        lr = block.get("learning_rate", 0.001)
        lambda_param = block.get("lambda_param", 0.01)
        epochs = block.get("epochs", 1000)
        
        # CORRECCIÓN: ml_runtime.MiniSVM usa 'n_iters', no 'epochs'
        return (
            f"from core.ml_runtime import MiniSVM\n"
            f"{model} = MiniSVM(learning_rate={lr}, lambda_param={lambda_param}, n_iters={epochs})\n"
            f"{model}.fit({ds})\n"
        )
    except Exception as e:
        return f"# Error en ml_train_svm_block_to_code: {e}\n"


def ml_train_linear_block_to_code(block: Dict[str, Any]) -> str:
    try:
        ds = block.get("dataset", "data")
        model = block.get("model_name", "linear_model")
        lr = block.get("learning_rate", 0.01)
        epochs = block.get("epochs", 1000)
        return (
            f"from core.ml_runtime import MiniLinearModel\n"
            f"{model} = MiniLinearModel(learning_rate={lr}, epochs={epochs})\n"
            f"{model}.fit({ds})\n"
        )
    except Exception as e:
        return f"# Error en ml_train_linear_block_to_code: {e}\n"


def ml_train_nn_block_to_code(block: Dict[str, Any]) -> str:
    """Entrenamiento de Neural Network (Actualizado: input_size -> n_inputs)."""
    try:
        ds = block.get("dataset", "data")
        model = block.get("model_name", "nn_model")
        hidden_size = block.get("hidden_size", 4)
        epochs = block.get("epochs", 1000)
        lr = block.get("lr", 0.01)
        
        # Lógica para inferir n_inputs desde el código generado es difícil sin ejecutarlo.
        # Asumimos que el usuario o el adaptador proveen 'n_inputs'.
        # Si no, usamos len(data[0])-1 dinámicamente en el código generado.
        
        code = f"from core.ml_runtime import MiniNeuralNetwork\n"
        
        # Generamos un código que calcula n_inputs dinámicamente si no se provee
        n_inputs_val = block.get("n_inputs")
        
        if n_inputs_val:
             code += f"{model} = MiniNeuralNetwork(n_inputs={n_inputs_val}, n_hidden={hidden_size}, n_outputs=1, epochs={epochs}, learning_rate={lr})\n"
        else:
             # Generación inteligente: Calcular inputs basado en el dataset en tiempo de ejecución
             code += f"# Auto-detect n_inputs from {ds}\n"
             code += f"_n_in = len({ds}[0]) - 1 if {ds} and len({ds}) > 0 else 1\n"
             code += f"{model} = MiniNeuralNetwork(n_inputs=_n_in, n_hidden={hidden_size}, n_outputs=1, epochs={epochs}, learning_rate={lr})\n"

        code += f"{model}.fit({ds})\n"
        return code
    except Exception as e:
        return f"# Error en ml_train_nn_block_to_code: {e}\n"

def ml_train_knn_block_to_code(block: Dict[str, Any]) -> str:
    try:
        dataset = block.get("dataset", "data")
        model_name = block.get("model_name", "knn_model")
        k = block.get("k", 3)
        task = block.get("task", "'classification'")
        
        return (
            f"from core.ml_runtime import KNearestNeighbors\n"
            f"{model_name} = KNearestNeighbors(k={k}, task={task})\n"
            f"{model_name}.fit({dataset})\n"
        )
    except Exception as e:
        return f"# Error en ml_train_knn_block_to_code: {e}\n"

def ml_predict_block_to_code(block: Dict[str, Any]) -> str:
    try:
        model = block.get("model", "model")
        X = block.get("X", "X")
        output_var = block.get("output", "preds")
        return f"{output_var} = {model}.predict({X})\n"
    except Exception as e:
        return f"# Error en ml_predict_block_to_code: {e}\n"


def ml_eval_block_to_code(block: Dict[str, Any]) -> str:
    try:
        y_true = block.get("y_true", "y_true")
        y_pred = block.get("y_pred", "y_pred")
        out = block.get("output", "acc")
        return (
            f"from core.ml_runtime import accuracy_score\n"
            f"{out} = accuracy_score({y_true}, {y_pred})\n"
        )
    except Exception as e:
        return f"# Error en ml_eval_block_to_code: {e}\n"

def ml_validator_block_to_code(block: Dict[str, Any]) -> str:
    try:
        challenge_id = block.get('challenge_id', 'reto_ml_1')
        target = block.get('target', 'acc')
        expected = block.get('expected', '0.8')
        operator = block.get('operator', '>=')
        
        return (
            f"# --- VALIDACIÓN ML: {challenge_id} ---\n"
            f"if float({target}) {operator} float({expected}):\n"
            f"    print('EDUBOT_VAL_PASS|{challenge_id}')\n"
            f"    print('✅ [Validador ML] ¡Métrica de modelo alcanzada!')\n"
            f"else:\n"
            f"    print('EDUBOT_VAL_FAIL|{challenge_id}')\n"
            f"    print('❌ [Validador ML] El modelo no alcanza el rendimiento esperado.')\n"
        )
    except Exception as e:
        return f"# Error en ml_validator_block_to_code: {e}\n"

# Registro Maestro
BLOCK_TO_CODE = {
    'ml_dataset': ml_dataset_block_to_code,
    'ml_train_dt': ml_train_dt_block_to_code,
    'ml_train_rf': ml_train_rf_block_to_code,
    'ml_train_svm': ml_train_svm_block_to_code,
    'ml_train_linear': ml_train_linear_block_to_code,
    'ml_train_nn': ml_train_nn_block_to_code,
    'ml_train_knn': ml_train_knn_block_to_code,
    'ml_predict': ml_predict_block_to_code,
    'ml_eval': ml_eval_block_to_code,
    'ml_eval_ext': ml_eval_block_to_code, # Alias
    'ml_validator': ml_validator_block_to_code,
}

def get_ml_code(block: Dict[str, Any]) -> str:
    t = block.get('type')
    fn = BLOCK_TO_CODE.get(t)
    if fn:
        return fn(block)
    return f"# Unsupported ML block: {t}\n"