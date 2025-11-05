# EduBot – Developer Technical Manual
**Version:** 1.0  
**Author:** Michego Takoro  
**Developed by:** EduBot Development Team  
**Date:** 2025  

---

## 🧩 1. Overview

EduBot’s architecture consists of a modular and lightweight design for embedded machine learning systems.  
It bridges Python ML training and C firmware execution through efficient translation and matrix operations.

---

## ⚙️ 2. Core Components

| File | Purpose |
|------|----------|
| `ml_runtime.py` | Main ML core: includes DecisionTree, RandomForest, SVM, LinearRegression, and MiniNeuralNetwork with backpropagation. |
| `ml_exporter.py` | Converts models into firmware-ready JSON/YAML. |
| `ml_manager.py` | Manages training sessions and registry persistence. |
| `ml_struct_rules.py` | Defines structural blocks for training and model visualization. |
| `ml_rules.py` | Translates visual blocks into executable Python ML code. |
| `translator.py` | Bidirectional translation between code and block structures. |
| `file_handler.py` | Manages backups and file integrity. |
| `ml_adapter.py` | Middleware linking models and execution handlers. |
| `executor.py` | Safe runtime execution with thread-based timeouts. |
| `server.py` | REST API backend and hardware deployment interface. |

---

## 🔧 3. Firmware Integration

Matrix operations are exported as simple nested loops for compatibility:

```c
for (i = 0; i < ROWS_A; i++) {
    for (j = 0; j < COLS_B; j++) {
        result[i][j] = 0;
        for (k = 0; k < COLS_A; k++) {
            result[i][j] += A[i][k] * B[k][j];
        }
    }
}
This allows execution on any microcontroller with C standard libraries.
```

## 🧠 4. Neural Network Architecture
The MiniNeuralNetwork supports:

- One or more hidden layers

- Sigmoid, ReLU, and Linear activations

- Gradient descent learning with manual overflow protection

- Firmware export with direct matrix mapping

## 🧩 5. Developer Guidelines

- Maintain minimal floating-point operations for firmware stability.

- Test with test_ml_runtime.py after modifications.

- Keep model names consistent (MiniSVM, MiniNN, etc.).

- Avoid recursion in runtime logic for embedded execution.

## 📚 6. Security & Execution

- executor.py prevents infinite loops and unsafe imports.

- server.py provides API key validation and request limiting.

- All components follow strict exception handling to ensure runtime stability.


## 🧾 7. Licensing
EduBot ML Runtime © 2025 – Michego Takoro & EduBot Team
Registered under Educational Embedded AI Framework.

