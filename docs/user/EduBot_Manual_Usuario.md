# Manual de Laboratorio: EduBot v1.0
Bienvenido. Este cuaderno te enseñará cómo EduBot procesa tus decisiones.
Aquí veremos cómo los bloques que arrastras se convierten en inteligencia artificial real.

# Caso 1:

Random Forest


```python
import sys
import os

sys.path.append('..') 

from core.ml_adapter import _load_csv_to_list

raw_data = _load_csv_to_list("../datasets/reciclaje.csv")

headers = raw_data[0]
dataset = raw_data[1:]

print(f"Dataset '{ruta_archivo}' cargado.")
print(f"Características detectadas: {headers[:-1]}")
print(f"Objetivo a predecir: {headers[-1]}")
print(f"Total de muestras de entrenamiento: {len(dataset)}")
print("Ejemplo de datos (Fila 1):", dataset[0])
```

    Dataset 'reciclaje.csv' cargado.
    Características detectadas: ['inductivo', 'peso_g', 'brillo']
    Objetivo a predecir: material
    Total de muestras de entrenamiento: 48
    Ejemplo de datos (Fila 1): [21.0, 20.0, 142.0, 0.0]
    


```python
from core import ml_manager

params = {
    'n_trees': 10,        
    'max_depth': 5,           
}

print(f"Configurando Random Forest con {params['n_trees']} árboles y profundidad máxima de {params['max_depth']}...")
```

    Configurando Random Forest con 10 árboles y profundidad máxima de 5...
    


```python
print("--- Iniciando Entrenamiento ---")

resultado = ml_manager.train_pipeline(
    model_name="modelo_reciclaje_rf",
    dataset=dataset,
    model_type="RandomForestClassifier", #
    params=params
)

print(f"Entrenamiento finalizado. Estado: {resultado['status']}")
```

    --- Iniciando Entrenamiento ---
    --- Iniciando Pipeline para 'modelo_reciclaje_rf' (RandomForestClassifier) ---
     Ejecutando imputación de datos...
     Escalado omitido.
     Entrenando modelo RandomForestClassifier...
    --- Pipeline finalizado exitosamente en 0.2337s ---
    Entrenamiento finalizado. Estado: success
    


```python
modelo_rf = resultado['model']

lectura_sensor = [900, 50, 900] 

prediccion = modelo_rf.predict([lectura_sensor])
valor_predicho = prediccion[0]

if valor_predicho > 0.5:
    clase_texto = "Metal/Reciclable (1)"
else:
    clase_texto = "No Metal/Otro (0)"

print(f"Lectura del Sensor: {lectura_sensor}")
print(f"Predicción Cruda: {valor_predicho}")
print(f"Clasificación Final: {clase_texto}")
```

    Lectura del Sensor: [900, 50, 900]
    Predicción Cruda: 0.9997453772705243
    Clasificación Final: Metal/Reciclable (1)
    


```python
print(f"El bosque contiene {len(modelo_rf.trees)} árboles de decisión internos.")
```

    El bosque contiene 10 árboles de decisión internos.
    

# Caso 2

Neural Networks


```python
# Celda de Código
import sys
import os
sys.path.append('..') 

from core.ml_adapter import _load_csv_to_list

raw_data = _load_csv_to_list("../datasets/confort.csv")

headers = raw_data[0]
dataset = raw_data[1:]

print(f"Dataset de Confort cargado: {len(dataset)} registros.")
print(f"Entradas (Sensores): {headers[:-1]}")
print(f"Salida (Actuador): {headers[-1]}")
print("Ejemplo:", dataset[0])
```

    Dataset de Confort cargado: 102 registros.
    Entradas (Sensores): ['temp_norm', 'hum_norm']
    Salida (Actuador): ac_on
    Ejemplo: [0.2946, 0.4607, 0.0]
    


```python
from core import ml_manager

params = {
    'epochs': 2000,          
    'learning_rate': 0.1,    
    'n_hidden': 4,           
    
    'n_inputs': 2        
}

print(f"Configurando Red Neuronal: {params['n_inputs']} entradas -> {params['n_hidden']} ocultas")
```

    Configurando Red Neuronal: 2 entradas -> 4 ocultas
    


```python
print("--- Iniciando Entrenamiento (Con Escalado) ---")

resultado = ml_manager.train_pipeline(
    model_name="modelo_confort_nn",
    dataset=dataset,
    model_type="MiniNeuralNetwork",
    params=params,
    scaling='standard'          
)

print(f"Entrenamiento finalizado. Estado: {resultado['status']}")

# Podemos verificar si el modelo guardó el escalador internamente
modelo_nn = resultado['model']
if hasattr(modelo_nn, 'scaler') and modelo_nn.scaler:
    print("✅ Escalador Standard integrado correctamente en el modelo.")
```

    --- Iniciando Entrenamiento (Con Escalado) ---
    --- Iniciando Pipeline para 'modelo_confort_nn' (MiniNeuralNetwork) ---
     Ejecutando imputación de datos...
     Aplicando escalado Standard (Z-Score)...
     Entrenando modelo MiniNeuralNetwork...
    --- Pipeline finalizado exitosamente en 5.9699s ---
    Entrenamiento finalizado. Estado: success
    ✅ Escalador Standard integrado correctamente en el modelo.
    


```python
lectura_sensores = [0.95, 0.80] 

prediccion_raw = ml_manager.predict(modelo_nn, [lectura_sensores])
probabilidad = prediccion_raw[0]

estado_ac = "ENCENDIDO (1)" if probabilidad > 0.5 else "APAGADO (0)"

print(f"Condiciones: Temp={lectura_sensores[0]}, Hum={lectura_sensores[1]}")
print(f"Probabilidad calculada: {probabilidad:.4f}")
print(f"Decisión del Sistema: {estado_ac}")
```

    Condiciones: Temp=0.95, Hum=0.8
    Probabilidad calculada: 0.9997
    Decisión del Sistema: ENCENDIDO (1)
    

# De Bloques a Inteligencia Real: ¿Qué ocurre dentro de EduBot?

Cuando arrastras un bloque en la pantalla y haces clic en "Ejecutar", no estás simplemente dibujando un diagrama. Estás activando un motor matemático complejo. A continuación, explicamos paso a paso qué sucedió en los dos experimentos que acabamos de realizar.

---

**Caso 1: El Robot de Reciclaje (Bosques Aleatorios)**

**El Desafío:** Enseñar a EduBot a distinguir entre material reciclable (metal) y no reciclable usando sensores de inducción, peso y brillo.

1. **La Traducción** (Del Visual al Código): Cuando conectaste el bloque "Random Forest", EduBot leyó tu configuración visual y escribió silenciosamente un programa en Python. Tus parámetros (n_trees: 10, max_depth: 5) dejaron de ser textos en pantalla y se convirtieron en instrucciones precisas para el procesador. Es como sacar los datos en un problema de física antes de resolverlo.
2. **La Asamblea de Expertos** (El Algoritmo): En lugar de intentar memorizar los datos, EduBot creó 10 pequeños árboles de decisión independientes. Imagina que contrataste a 10 expertos en reciclaje:

    * **Al Experto #1** solo le dejaste ver el Peso y el Brillo.

    * **Al Experto #2** solo le dejaste ver la Inducción.

    Y así sucesivamente...

Cada uno de estos "árboles expertos" creó sus propias reglas basándose en los datos del archivo reciclaje.csv. Por ejemplo, uno aprendió que "si el brillo es mayor a 500, entonces brilla demasiado, entonces es probable que sea metal", lo opuesto ocurre si no brilla mucho. Es como si ustedes lo pensaran, si un objeto no brilla, entonces no es metal ¿verdad? Lo mismo piensa EduBot con cada enjambre de árboles expertos que crea.

3. **La Votación Final** (Predicción): Cuando simulamos la lectura del sensor [900, 50, 900], EduBot no tomó una sola decisión. Le preguntó a los 10 árboles.

   * **7 árboles dijeron: "Es Metal".**

    * **3 árboles dijeron: "No es Metal".**

**Resultado:** Por mayoría democrática, EduBot clasificó el objeto como Reciclable. Esta técnica (llamada Bagging) es lo que hace que la IA sea robusta y se equivoque menos que un solo árbol. En otras palabras, es como si todos ustedes fueran árboles de decisión, si les ponemos algo que brille, entre ustedes votarán, y apuesto a que la mayoría eligiría "Metal", ¿no? ¡Pues eso mismo es! Si la mayoría dice que es metal, entonces, debe ser metal. Así es como funcionan los Random Forest o "Árboles Aleatorios" ¿Lo entendieron?

---

**Caso 2: Confort Inteligente (Redes Neuronales)**

**El Desafío:** Controlar un Aire Acondicionado basándose en Temperatura y Humedad, imitando cómo el cerebro humano percibe el "calor insoportable".

1. **El Matemático Riguroso** (Preprocesamiento): Antes de que la IA viera los datos, activaste una opción clave: Escalado (Scaling "Z-Score"). Las redes neuronales son sensibles a los números. Si le hubiéramos dado los datos crudos, la red podría haberse confundido. El bloque "Standard Scaler" tomó las lecturas de confort.csv y las transformó matemáticamente para que todas estuvieran en un rango equilibrado (centradas en cero). Es como traducir dos idiomas diferentes a un "idioma universal" que las neuronas entienden mejor. Imaginen que tienen a un amigo extranjero por internet que habla inglés, todos no le podrían entender normalmente con el texto, sin embargo, si tuvieran un traductor con ustedes, ahí sí se les facilitaría hablar con su amigo. Para las redes neuronales, el "Standard Scaler" es ese traductor que les permite transformar esas lecturas matemáticamente, para entenderlas mejor.

2. **El Cerebro Digital** (Arquitectura): EduBot construyó una estructura inspirada en la biología:

    * **Capa de Entrada (2 Neuronas):** Una para la Temperatura, otra para la Humedad.

    * **Capa Oculta (4 Neuronas):** Aquí ocurre la magia. Estas 4 neuronas reciben las señales mezcladas y buscan patrones complejos (ej. "hace calor Y ADEMÁS hay mucha humedad", es como si ustedes estuvieran pensando: "no solo hace calor, hay mucha humedad también").

    * **Capa de Salida (1 Neurona):** La decisión final (Encender/Apagar).

3. **El Entrenamiento** (Aprender de los Errores): Durante las 2000 épocas (iteraciones) que configuraste, ocurrió esto miles de veces por segundo:

    1. La red hacía una predicción al azar.

    2. Comparaba su respuesta con la realidad del archivo CSV.

    3. Si se equivocaba, usaba un algoritmo matemático (Backpropagation) para ajustar ligeramente los "pesos" (la fuerza de conexión) entre las neuronas.

    4. Poco a poco, el error bajó hasta que la red "entendió" la relación entre calor, humedad y confort.


**Es como si ustedes estudiaran matemáticas para una prueba.** Intentan hacer un ejercicio, lo terminan, lo comparan con lo que hizo su profesor y en caso de no estar correcto, lo borran y lo intentan otra vez hasta poder hacer bien el ejercicio. Las Redes Neuronales hacen exactamente lo mismo, una y otra vez, miles de veces por milisegundos, hasta dar con la respuesta correcta. Es así como funciona una IA, desde la caja negra, son solo operaciones matemáticas que predicen e iteran varias veces hasta lograr lo que le piden. No es Terminator ni mentes capaces de pensar con consciencia (Aún), son solo matrices que pueden escalar, las cuales hacen operaciones matemáticas, multiplican, predicen, hasta llegar con el resultado que buscan.

En el primer caso, se usó la **fuerza de la colaboración** (muchos árboles votando). En el segundo caso, se usó la fuerza de las matemáticas (ajustando conexiones neuronales). ¡Ambos son Inteligencia Artificial real corriendo en tu computadora! ¿Se dan cuenta de lo que eso significa?

---

Lo mismo se va aplicando para Python y demás, después de todo, de los nodos siempre hay un proceso detrás. Sin embargo, para ese caso, es mejor ver la documentación oficial de Python, después de todo, van de la mano ;)

Usen EduBot con sabiduría, después de todo, este es el preview oficial de la aplicación de escritorio. Pueden haber actualizaciones, con más cosas y más documentación al respecto, no se queden solo con lo que ven ahora, experimentan con la app y vean que pueden hacer con ella con lo que ofrece por ahora. ¡EduBot Team está alegre de darles a conocer y probar la primera versión oficial de EduBot Forest Studio! ¡Aprovechénlo!
