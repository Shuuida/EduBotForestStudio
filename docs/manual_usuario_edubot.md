# EduBot – Manual del Usuario
**Versión:** 1.0  
**Desarrollado por:** EduBot Team  
**Autor principal:** Michego Takoro  
**Fecha:** 2025  

---

## 🧩 Introducción

EduBot es una plataforma educativa de aprendizaje orientado a objetos y aprendizaje automático (Machine Learning) embebido, diseñada para enseñar principios de inteligencia artificial y robótica en entornos de bajo costo como Arduino o MegaPi.

El núcleo de EduBot permite **entrenar modelos de ML directamente en Python**, exportarlos a **firmware en C**, y ejecutarlos en microcontroladores reales para aplicaciones educativas (seguimiento de línea, clasificación de colores, predicciones sensoriales, etc.).

---

## ⚙️ Funcionalidades Principales

- **Aprendizaje Automático Local:** entrenar Decision Trees, Random Forests, SVMs, Redes Neuronales y Regresión Lineal sin necesidad de librerías externas como NumPy o TensorFlow.  
- **Exportación a Firmware C:** genera código C optimizado y portable para integrarse en placas Arduino o MegaPi.  
- **Compatibilidad Cross-Firmware:** el modelo se traduce a operaciones de matriz simples (como `matrix_multiply`) que cualquier microcontrolador puede ejecutar.  
- **Entorno Educativo Interactivo:** permite visualizar, editar y probar los modelos desde una interfaz visual.

---

## 🧠 Módulos de Aprendizaje Soportados

| Módulo | Descripción | Uso |
|--------|--------------|-----|
| **DecisionTreeClassifier** | Clasificador jerárquico basado en condiciones. | Ideal para detección binaria (sí/no). |
| **RandomForestClassifier** | Conjunto de árboles de decisión. | Mejora la precisión de modelos ruidosos. |
| **LinearRegression** | Modelo predictivo de regresión. | Predicciones numéricas o sensoriales. |
| **MiniSVM** | Máquina de Vectores de Soporte ligera. | Clasificación de datos en límites lineales. |
| **MiniNeuralNetwork** | Red neuronal multicapa con retropropagación. | Ideal para proyectos con sensores o cámaras. |

---

## 🔧 Requisitos

- Python 3.10 o superior  
- Librerías estándar (no requiere NumPy ni TensorFlow)  
- Firmware C compatible con **operaciones básicas de matrices**

---

## 📲 Flujo de Trabajo

1. **Cargar Dataset:**  
   Desde el entorno EduBot, selecciona o importa tus datos (CSV o JSON).  
2. **Seleccionar Modelo:**  
   Elige entre Árbol de Decisión, Bosque Aleatorio, SVM, Regresión Lineal o Red Neuronal.  
3. **Entrenar:**  
   Presiona “Entrenar modelo”. El sistema procesará los datos internamente y mostrará resultados.  
4. **Probar Predicciones:**  
   Ingresa nuevos valores de entrada y observa las predicciones.  
5. **Exportar a Firmware:**  
   El modelo se traduce automáticamente a C con funciones como `matrix_multiply` y se guarda como `firmware_export.json` o `.c`.  

---

## 🧩 Ejemplo de Uso – Red Neuronal

```python
from core import ml_runtime

model = ml_runtime.MiniNeuralNetwork(
    n_inputs=2,
    n_hidden=4,
    n_outputs=1,
    learning_rate=0.5,
    epochs=3000
)

# Entrenamiento XOR
X = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [1], [1], [0]]

model.fit(X, y)
preds = [model.predict(x) for x in X]
print("Predicciones:", preds)
```

##  ⚠️ Advertencias y Limitaciones
- Los modelos están optimizados para firmware educativo, no para producción industrial.

- El tamaño de las redes neuronales debe mantenerse pequeño (idealmente 1 capa oculta).

- Evita datasets con más de 200 registros en hardware limitado.

- Para Arduino, la exportación usa tipos float y cálculos iterativos, evitando double o librerías externas.


## 🧾 Licencia y Derechos
EduBot ML Runtime © 2025 por Michego Takoro y EduBot Team.
Todos los derechos reservados.
Uso permitido únicamente para fines educativos y de investigación.

