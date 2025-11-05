"""
Reglas de traducción Bloques -> Código Python (Machine Learning).
Contiene conversores seguros con docstrings y try/except internos.
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
    """Entrenamiento de Decision Tree en código Python."""
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
    """Entrenamiento de Random Forest en código Python."""
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
    """Entrenamiento de SVM en código Python."""
    try:
        ds = block.get("dataset", "data")
        model = block.get("model_name", "svm_model")
        lr = block.get("learning_rate", 0.001)
        lambda_param = block.get("lambda_param", 0.01)
        epochs = block.get("epochs", 1000)
        return (
            f"from core.ml_runtime import MiniSVM\n"
            f"{model} = MiniSVM(learning_rate={lr}, lambda_param={lambda_param}, epochs={epochs})\n"
            f"{model}.fit({ds})\n"
        )
    except Exception as e:
        return f"# Error en ml_train_svm_block_to_code: {e}\n"


def ml_train_linear_block_to_code(block: Dict[str, Any]) -> str:
    """Entrenamiento de Linear Model en código Python."""
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
    """Entrenamiento de Neural Network en código Python."""
    try:
        ds = block.get("dataset", "data")
        model = block.get("model_name", "nn_model")
        hidden_size = block.get("hidden_size", 4)
        epochs = block.get("epochs", 1000)
        lr = block.get("lr", 0.01)
        input_size = len(block.get("data", [[]])[0]) - 1 if block.get("data") else "input_size"  # Dinámico si posible
        return (
            f"from core.ml_runtime import MiniNeuralNetwork\n"
            f"{model} = MiniNeuralNetwork(input_size={input_size}, hidden_size={hidden_size}, epochs={epochs}, lr={lr})\n"
            f"{model}.fit({ds})\n"
        )
    except Exception as e:
        return f"# Error en ml_train_nn_block_to_code: {e}\n"


def ml_predict_block_to_code(block: Dict[str, Any]) -> str:
    """Genera código para predicciones con un modelo entrenado."""
    try:
        model = block.get("model", "model")
        X = block.get("X", "X")
        output_var = block.get("output", "preds")
        return f"{output_var} = {model}.predict({X})\n"
    except Exception as e:
        return f"# Error en ml_predict_block_to_code: {e}\n"


def ml_eval_block_to_code(block: Dict[str, Any]) -> str:
    """Genera código Python para evaluación de modelo."""
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

# Nuevo: Dict dinámico para extensión futura
BLOCK_TO_CODE = {
    'ml_dataset': ml_dataset_block_to_code,
    'ml_train_dt': ml_train_dt_block_to_code,
    'ml_train_rf': ml_train_rf_block_to_code,
    'ml_train_svm': ml_train_svm_block_to_code,
    'ml_train_linear': ml_train_linear_block_to_code,
    'ml_train_nn': ml_train_nn_block_to_code,
    'ml_predict': ml_predict_block_to_code,
    'ml_eval': ml_eval_block_to_code,
}

def get_ml_code(block: Dict[str, Any]) -> str:
    t = block.get('type')
    fn = BLOCK_TO_CODE.get(t)
    if fn:
        return fn(block)
    return f"# Unsupported ML block: {t}\n"