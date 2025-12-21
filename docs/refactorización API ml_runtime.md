# **📘 Documentación Técnica: EduBot MiniML Engine v2.3**

| Metadatos | Detalles |
| :---- | :---- |
| **Módulo** | core/ml\_runtime.py |
| **Versión** | 2.3 (Stable Release) |
| **Fecha** | 18 de Diciembre, 2025 |
| **Arquitectura** | Pure Python (No NumPy/Pandas) |
| **Target** | Educación & TinyML (Arduino AVR) |

## **1\. Resumen Ejecutivo**

La versión 2.3 representa una **refactorización estructural completa** del motor de inferencia de EduBot. El objetivo principal ha sido transformar una colección de scripts experimentales en un **Micro-Framework mantenible y escalable**.

Se han eliminado redundancias lógicas (DRY), se ha estabilizado la aritmética de exportación para microcontroladores de 8 bits y se ha unificado la API para garantizar que todos los modelos (Clasificación y Regresión) sean intercambiables.

## **2\. Cambios Arquitectónicos Críticos**

### **2.1. Estructura Jerárquica ("De Átomos a Organismos")**

El código ha sido reorganizado siguiendo una dependencia lógica lineal para facilitar la lectura y el mantenimiento:

1. **MiniMatrixOps (El Átomo):** Operaciones matemáticas base.  
2. **Utilidades Globales:** Lógica compartida de árboles (Splits, Gini, MSE).  
3. **Modelos de Árboles:** DecisionTree (Classifier/Regressor).  
4. **Modelos de Bosques:** RandomForest (usa Árboles).  
5. **Modelos Matemáticos:** KNN, Linear, SVM.  
6. **Redes Neuronales:** MLP (usa todo lo anterior).

### **2.2. Globalización de Utilidades (Principio DRY)**

Antes: Cada clase de árbol (DecisionTree, RandomForest) tenía su propia copia privada de la función para dividir datos (test\_split) y calcular impurezas.  
Ahora: Se han extraído test\_split, gini\_index y mse\_metric como funciones globales puras.

* **Beneficio:** Cualquier mejora en la eficiencia de test\_split optimiza automáticamente a 4 modelos (DT Classifier, DT Regressor, RF Classifier, RF Regressor).

### **2.3. Single Source of Truth en Modelos Lineales**

Se implementó el método privado \_predict\_single(row) en MiniLinearModel.

* **Justificación:** Garantiza que la fórmula matemática usada durante el entrenamiento (fit para calcular gradientes) sea **idéntica** a la usada en inferencia (predict). Esto elimina errores sutiles de divergencia matemática.

## **3\. Nueva Estrategia de Cuantización (TinyML)**

Uno de los cambios más importantes es cómo manejamos la exportación a hardware limitado (Arduino Uno \- 2KB RAM).

### **El Problema Anterior (Pure Int8)**

Intentar simular aritmética de enteros de 8 bits (int8) en Python puro para luego exportarla a C era inestable. Arduino AVR no tiene instrucciones DSP para acelerar esto, y la gestión de *overflows* en C puro hacía el código ilegible para estudiantes.

### **La Solución v2.3 (Híbrido: Float Compute / Flash Storage)**

Hemos adoptado una estrategia pragmática:

1. **Almacenamiento (Flash):** Todos los pesos del modelo (matrices grandes) se guardan en la memoria de programa (PROGMEM) usando const float ... PROGMEM. Esto libera la RAM casi por completo.  
2. **Cálculo (Float):** La inferencia se realiza usando float estándar.  
   * *Ventaja:* Precisión matemática idéntica a la de Python.  
   * *Ventaja:* Código C generado legible y fácil de depurar.  
   * *Nota:* Los métodos quantize() y calibrate() se mantienen en la API por compatibilidad, pero internamente preparan el modelo para esta exportación híbrida.

## **4\. Mejoras Específicas por Modelo**

### **🧠 MiniNeuralNetwork (MLP)**

* **Fix de Dimensionalidad:** Se corrigió un bug donde predict() devolvía una lista de listas (ej. \[\[0.8\]\]) incluso para salidas escalares. Ahora, si n\_outputs=1, devuelve una lista plana \[0.8\], asegurando compatibilidad con scikit-learn y métricas de evaluación.  
* **Estabilidad:** Se añadieron protecciones contra *overflow* en la función sigmoid (clipping entre \-700 y 700).

### **🌲 Árboles y Bosques**

* **Restauración de Regresión:** Se reintrodujeron DecisionTreeRegressor y RandomForestRegressor, que habían sido omitidos en versiones intermedias. Ahora EduBot soporta predicción de valores continuos.  
* **Exportación Plana:** Los árboles se exportan a C como **Arrays Planos** (navegación por índices) en lugar de punteros struct. Esto evita la fragmentación de memoria en el Arduino.

### **📍 K-Nearest Neighbors (KNN)**

* **Nuevo Modelo:** Implementación completa de KNN (Lazy Learning).  
* **Exportación "Hard":** El método to\_arduino\_code escribe **todo el dataset de entrenamiento** en la memoria Flash del Arduino.  
  * *Limitación:* Solo apto para datasets pequeños (\<500 muestras en Arduino Uno) debido al límite de 32KB de Flash.

### **📏 MiniLinearModel**

* **Fix de Exportación:** Se añadió el método to\_arduino\_code que faltaba, permitiendo exportar regresiones lineales simples a firmware.

## **5\. Capacidades y Limitaciones**

### **✅ Potencialidades**

* **Independencia Total:** No requiere numpy, pandas ni scikit-learn. Corre en cualquier entorno Python estándar.  
* **Determinismo:** Al controlar la semilla aleatoria y la matemática matricial, los resultados son reproducibles.  
* **Ready for Arduino:** El código C generado incluye automáticamente las directivas avr/pgmspace.h necesarias para funcionar en microcontroladores AVR.

### **⚠️ Limitaciones Conocidas**

* **Velocidad de Entrenamiento:** Al ser Python puro, entrenar con datasets masivos (\>10,000 filas) será lento comparado con C++/NumPy. *Uso recomendado: Datasets educativos (\<1,000 filas).*  
* **Consumo de Flash (KNN):** El modelo KNN consume memoria linealmente con el tamaño del dataset.  
* **Energía:** La inferencia en float consume más ciclos de CPU (y batería) que la inferencia en int8 puro. Para fines educativos no es crítico, pero sí para productos comerciales a batería.

## **6\. Guía Rápida para Desarrolladores**

### **Instanciación y Uso**

from core import ml\_runtime

\# 1\. Crear Modelo  
model \= ml\_runtime.RandomForestClassifier(n\_trees=10)

\# 2\. Entrenar (Datos: Lista de Listas, última col es target)  
dataset \= \[\[1.2, 0.5, 0\], \[3.4, 1.1, 1\]\]   
model.fit(dataset)

\# 3\. Predecir  
pred \= model.predict(\[\[1.5, 0.6\]\])  
print(pred) \# \-\> \[0\]

### **Exportación a Firmware**

\# Genera código C++ listo para copiar al IDE de Arduino  
code\_c \= model.to\_arduino\_code(fn\_name="mi\_modelo")  
print(code\_c)

Conclusión del Equipo de Ingeniería:  
La versión 2.3 de ml\_runtime establece una base sólida para el futuro de EduBot. La arquitectura modular permite añadir nuevos algoritmos (como Naive Bayes o Gradient Boosting) sin riesgo de romper la funcionalidad existente.