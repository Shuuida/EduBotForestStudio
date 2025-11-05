# EduBot – Manual Técnico para Desarrolladores
**Versión:** 1.0  
**Autor:** Michego Takoro  
**Desarrollado por:** EduBot Team
**Fecha:** 2025  

---

## 📚 Índice Técnico

1. Arquitectura General  
2. Núcleo ML Runtime  
3. Módulos Complementarios  
4. Compatibilidad con Firmware C  
5. Reglas de Traducción y Exportación  
6. Adaptadores y Estructuras  
7. Seguridad del Servidor  
8. Ejecución Segura  
9. Buenas Prácticas de Desarrollo  

---

## 🧩 1. Arquitectura General

EduBot se compone de módulos altamente desacoplados y compatibles entre sí.  
Cada uno maneja una parte del ciclo de vida del modelo:

Dataset → Manager → Runtime → Exporter → Firmware

Los módulos principales son:

| Archivo             |        Función     |
|----------|----------|
| `ml_runtime.py` | Núcleo ML: modelos, entrenamiento y backpropagation. |
| `ml_exporter.py` | Exportación de modelos a C y YAML. |
| `ml_manager.py` | Gestión y persistencia de modelos. |
| `ml_struct_rules.py` | Definición de bloques y parámetros. |
| `ml_rules.py` | Traducción entre bloques y código ML. |
| `translator.py` / `translation_rules.py` | Conversión entre código Python ↔ bloques visuales. |
| `file_handler.py` | Guardado, carga y backups. |
| `ml_adapter.py` | Integración de modelos y ejecución de acciones. |
| `executor.py` | Ejecución segura con timeouts. |
| `server.py` | API backend y compatibilidad hardware. |

---

## ⚙️ 2. Núcleo ML Runtime

### Modelos Implementados
- **DecisionTreeClassifier**
- **RandomForestClassifier**
- **LinearRegression**
- **MiniSVM**
- **MiniNeuralNetwork**

### Backpropagation
La clase `MiniNeuralNetwork` aplica retropropagación usando `MiniMatrixOps`, una clase interna que maneja operaciones matriciales sin librerías externas.  
Las funciones `_activate` y `_act_derivative` manejan las activaciones (`sigmoid`, `relu`, `linear`) de forma flexible.

### Multiplicación de Matrices en Firmware
Para la exportación al microcontrolador, las matrices se representan como bucles anidados en C:

```c
for (i = 0; i < ROWS_A; i++) {
    for (j = 0; j < COLS_B; j++) {
        result[i][j] = 0;
        for (k = 0; k < COLS_A; k++) {
            result[i][j] += A[i][k] * B[k][j];
        }
    }
}
Esto asegura compatibilidad con firmware C ANSI sin librerías externas.

```

## 🧩 3. Módulos Complementarios
ml_manager.py
Gestiona la persistencia (save_registry, load_registry) y mantiene compatibilidad entre entrenamiento, exportación y ejecución.

ml_exporter.py
Convierte modelos a JSON o YAML, aptos para firmware. Usa extract_model_structure para aislar pesos, bias y topología.

ml_rules.py
Traduce estructuras de alto nivel a bloques ML visuales (train_svm, train_nn, etc.).

translator.py
Asegura la traducción Python ↔ visual blocks, corrigiendo NoneType y estructuras incompletas.

## 🧮 4. Integración con Firmware C
- Entrenar modelo en EduBot.

- Exportar modelo (export_to_firmware).

- Compilar el código en un entorno Arduino IDE.

- Subir a la placa y conectar sensores según el modelo (ej. red neuronal XOR → sensores digitales).

## 🔒 5. Seguridad y Estabilidad
- Todos los módulos usan control de errores explícito.

- executor.py limita ejecución infinita mediante threading y queue.

- server.py añade autenticación por clave API y limitador de peticiones.

## 🧱 6. Buenas Prácticas
- Evitar estructuras de datos de gran tamaño.

- No usar floats con más de 6 dígitos decimales en firmware.

- Testear con test_ml_runtime.py antes de desplegar.

- Mantener coherencia entre nombres de modelo (MiniSVM, MiniNeuralNetwork, etc.).



## 📑 Derechos de Autor
EduBot ML System © 2025 – Michego Takoro & EduBot Team
Registrado como sistema de IA embebido educativo.

