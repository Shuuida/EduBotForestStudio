# 🛠️ EduBot v1.0 Stable - Manual del Desarrollador

---

## 1. Visión General de la Arquitectura

EduBot v1.0 es un entorno educativo de Machine Learning diseñado para ejecutarse localmente y exportar modelos a hardware limitado (Arduino/AVR). Su núcleo evita dependencias pesadas (como NumPy o Scikit-Learn en el runtime) para garantizar la portabilidad y transparencia educativa.

**Flujo de Datos**

* **Frontend (UI):** Bloques visuales envían JSON al Backend.

* **Server/Main:** Recibe la petición vía HTTP/Eel.

* **Adapter:** Normaliza el JSON y orquesta la ejecución.

* **Manager:** Instancia modelos, gestiona el entrenamiento y persistencia.

* **Runtime:** Ejecuta la matemática pura (Python listas).

---

## 2. Referencia de Módulos (Core)

**A. Capa de Ejecución y Matemáticas**

* **ml_runtime.py:** El corazón del sistema. Contiene las implementaciones "Pure Python" de los algoritmos de ML.

* **MiniMatrixOps:** Clase estática que reimplementa álgebra lineal (dot, transpose, add) usando listas nativas de Python para evitar NumPy.

* **Modelos:** Implementación completa de DecisionTreeClassifier, RandomForestClassifier, MiniNeuralNetwork, MiniSVM, etc..

* **ml_compat.py:** Utilidades de compatibilidad y estructuras de datos.

* **_flatten_tree_to_arrays:** Convierte árboles recursivos en arrays lineales para exportación a C.

* **impute_missing_values:** Manejo básico de datos faltantes (estrategia de media).

**B. Capa de Gestión y Orquestación**

* **ml_manager.py:** Singleton funcional que gestiona el ciclo de vida.

* **train_pipeline:** Función maestra que limpia datos, escala, instancia el modelo (vía Factory) y entrena.

* **_MODEL_REGISTRY:** Diccionario en memoria RAM que almacena los modelos entrenados.

* **ml_factory.py:** Patrón de diseño "Factory".

    Desacopla la creación de objetos. Recibe un string (ej. "NeuralNetwork") y devuelve la instancia configurada.

* **ml_adapter.py:** El puente entre el Frontend y el Backend.

* **_ACTION_HANDLERS:** Diccionario que mapea "acciones" del JSON (ej. train_rf) a funciones Python.

    Mapea los parámetros visuales (ej. epochs) a los parámetros del constructor de la clase.

**C. Capa de Traducción y Reglas**

* **ml_rules.py:** Generador de Código (Python).

    Convierte bloques JSON en strings de código Python ejecutable (ej. ml_train_rf_block_to_code).

* **ml_struct_rules.py:** Normalizador de Estructuras.

    Valida y sanea el JSON entrante antes de que llegue al Adapter. Asegura que existan campos como action o dataset.

* **translator.py & translation_rules.py:** Motor AST.

    Realiza ingeniería inversa: parsea código Python escrito por el usuario y lo convierte en bloques visuales usando el módulo ast.

**D. Infraestructura**

* **server.py & main.py:** Puntos de entrada.

    * main.py inicia la UI con Eel y expone la API al navegador.

    * server.py provee endpoints REST para depuración y operaciones de archivo.

* **executor.py:** Sandbox de ejecución.

* **execute_user_code:** Ejecuta código de estudiante en un hilo separado, capturando stdout y bloqueando imports peligrosos (os, sys).

---

## 3. Optimización y Métodos Internos

En la versión v1.0, la optimización se centra en la **viabilidad en microcontroladores** sin usar librerías externas.

**A. Aplanado de Árboles (Tree Flattening)**

Para ejecutar un Árbol de Decisión o Random Forest en C sin recursión profunda (que desborda la pila en Arduino):

* **Método: ml_compat._flatten_tree_to_arrays(root).**

* **Lógica: Recorre el árbol recursivo (nodos con punteros left/right) y lo convierte en 4 arrays paralelos: feature_index, threshold, left_child_idx, right_child_idx.**

* **Resultado: La inferencia en C se convierte en un simple bucle while saltando índices de array, extremadamente rápido y eficiente en memoria.**

**B. Refactorización a C (Exportación PROGMEM)**

Para que los modelos quepan en los 2KB de RAM del Arduino Uno:

* **Método: to_arduino_code(fn_name) presente en cada clase de modelo en ml_runtime.py.**

* **Técnica: Todos los pesos (Weights) y estructuras de árboles se declaran con el modificador PROGMEM (Program Memory).**

    * Ejemplo (NN): const float nn_W1[ ] PROGMEM = { ... };.

* **Acceso: El código C generado usa pgm_read_float_near() para leer los pesos directamente desde la Flash, dejando la RAM libre para variables dinámicas.**

**C. Cuantización Simulada (Neural Networks)**

En v1.0, la cuantización es una preparación arquitectónica.

* **Implementación:** La clase MiniNeuralNetwork tiene métodos quantize() y calibrate(), aunque en v1.0 actúan principalmente como "stubs" (marcadores de posición) o banderas booleanas (self.quantized = True).

+ **Propósito:** Permite que el flujo educativo enseñe el concepto de cuantización, aunque el hardware final siga operando en float (32-bit) en esta versión estable.

---

## 4. Limitaciones de la Versión v1.0 Stable

* **Rendimiento en Entrenamiento:** Al usar listas de Python, el entrenamiento con datasets grandes (>5000 filas) será lento.

* **Matemáticas:** La MiniNeuralNetwork usa Backpropagation fijo (no es un grafo dinámico Autograd). Solo soporta topologías secuenciales simples. (Se piensa arreglar en la v1.5+)

* **Exportación:** La exportación a C genera código float. No soporta operaciones int8 reales en el microcontrolador (esto está reservado para v1.5+).

---

## 5. Guía del Desarrollador: Cómo Extender EduBot

**Añadir un Nuevo Modelo de IA**

Si deseas agregar un nuevo algoritmo (ej. Naive Bayes), sigue estos pasos estrictos para mantener la estabilidad (Se recomienda tener experiencia en sistemas embebidos e ingeniería de datos, sobre todo si se tiene intención de hacer un modelo de IA desde cero sin numpy, usando MiniMatrixOps):

1. **Runtime (ml_runtime.py):**

* **Crea la clase MiniNaiveBayes** o el que tengas en mente.

* Implementa fit(dataset) usando solo listas Python.

* Implementa predict(X).

* Implementa to_arduino_code() generando C válido por medio de PROGMEM para usar la memoria flash y no la RAM del microcontrolador.

2. **Factory (ml_factory.py):**

* Añade el caso elif "bayes" in model_type: en create_model.

3. **Adapter (ml_adapter.py):**

* Crea el mapper de parámetros: def _bayes_params(struct, dataset).

* Registra el handler: def _train_bayes(...).

* Añade a _ACTION_HANDLERS.

4. **Rules (ml_rules.py y ml_struct_rules.py):**

* Define cómo se genera el código Python (ml_train_bayes_block_to_code).

* Define la estructura JSON válida (ml_train_bayes_block_to_struct).

### Adaptación al Código Fuente

* **No uses NumPy en ml_runtime:** Romperá la compatibilidad con la filosofía "Zero-Dependency" del núcleo v1.0. Usa el NumPy en algún módulo a parte, ml_runtime se debe mantener Python Pure.

* **Sigue el patrón _safe_str:** En el traductor, nunca asumas que un campo existe. Usa métodos seguros de extracción.

---
