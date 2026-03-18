# Manual de Laboratorio: EduBot v1.0 (Modo Python)

Bienvenido. Este cuaderno te enseñará cómo EduBot procesa tus decisiones lógicas.
Aquí veremos cómo los bloques que arrastras se convierten en sintaxis real de Python y cómo la computadora interpreta tus algoritmos paso a paso. 
---

# Caso 1: La Calculadora de Promedios (Lógica Secuencial)

**El Desafío:** Enseñar a EduBot a pedir dos calificaciones, sumarlas, dividirlas y mostrar el resultado final en la terminal.

### El Código Generado por tus Bloques:
```python
# 1. Entrada de datos (Nodos Input)
nota_1 = input('Ingresa la primera nota: ')
nota_2 = input('Ingresa la segunda nota: ')

# 2. Conversión y Matemáticas (Nodos Math)
suma = float(nota_1) + float(nota_2)
promedio = suma / 2

# 3. Salida (Nodo Print)
print("El promedio final es:")
print(promedio)
```

### ¿Qué ocurrió aquí?
Cuando arrastras los bloques y los conectas de izquierda a derecha, estás creando una **Secuencia**. 
Imagina que EduBot es un chef de cocina muy obediente, pero que no sabe hacer nada si no le das la receta exacta. 
1. **Las Variables (`nota_1`, `nota_2`):** Son como "cajas vacías" donde el chef guarda los ingredientes que le pasas.
2. **Las Matemáticas (`suma`, `promedio`):** Es la licuadora. EduBot toma el contenido de las cajas, lo procesa usando los operadores matemáticos (`+`, `/`) y guarda el resultado en una caja nueva.
3. **El Print:** Es el chef sirviendo el plato terminado en la mesa (tu Terminal). 

EduBot lee el bosque de nodos **estrictamente de arriba hacia abajo y de izquierda a derecha**. Si pones el bloque `Print` antes de que el bloque `Math` haga la división, el programa fallará porque el chef intentará servir un plato que aún no ha cocinado. El orden importa.

---

# Caso 2: El Bucle del Adivino (Control de Flujo y Condicionales)

**El Desafío:** Crear un pequeño juego donde el usuario debe adivinar una contraseña. Si se equivoca, el sistema se lo vuelve a preguntar infinitamente hasta que acierte. 
### El Código Generado por tus Bloques:
```python
# 1. Definimos la variable secreta
password_correcta = "1234"

# 2. Bucle infinito (Nodo While)
while True:
    intento = input('Adivina la contraseña: ')
    
    # 3. Condicional (Nodo If / Compare)
    if intento == password_correcta:
        print("¡Acceso Concedido!")
        break  # (Nodo Control)
    else:
        print("Contraseña incorrecta, intenta de nuevo.")
```

### ¿Qué ocurrió aquí?
Aquí es donde la programación cobra vida. Ya no es una simple calculadora, ahora el programa **toma decisiones**.
1. **El Bucle `While`:** Es una trampa de tiempo. Le dice a EduBot: *"Repite todo lo que está conectado a mi derecha una y otra vez"*. 
2. **El Condicional `If`:** Es el guardia de seguridad. Revisa la variable `intento`. Si es idéntica a `1234`, te deja pasar. Si no, te manda por el camino del `else` (imprimir error) y el bucle te obliga a intentarlo de nuevo.
3. **El Nodo Control (`break`):** Es la llave de salida de emergencia. Sin este bloque, estarías atrapado en el bucle para siempre, incluso si adivinas la contraseña. 

---

# De Bloques a Código Real: ¿Qué ocurre dentro de EduBot?

Cuando arrastras un bloque en la pantalla y haces clic en "Ejecutar", no estás simplemente dibujando un diagrama. Estás activando un motor de traducción complejo. A continuación, explicamos paso a paso qué sucedió en los experimentos anteriores.

1. **El Traductor (De Visual a Texto):**
   A diferencia del lenguaje humano, las computadoras no entienden "dibujos". Cuando presionas Ejecutar, EduBot escanea tu lienzo visual. Toma cada bloque (como un `py_if` o `py_print`) y busca en su diccionario interno de reglas. Luego, inyecta la sintaxis exacta de Python, cuidando los espacios en blanco (indentación) que Python exige para funcionar. Pasa de ser un archivo `.edubotproj` a un código en memoria puro y duro.

2. **El Sandbox (La Zona Segura):**
   Una vez que el código está traducido, EduBot no lo ejecuta directamente en tu computadora. Lo mete en un *Sandbox* (Caja de Arena). Esta es una zona aislada y segura. 
   ¿Por qué? Porque si un estudiante comete un error y crea un "Bucle Infinito" (un `While True` sin un `break`), una computadora normal se colgaría o se quedaría sin memoria. El *Sandbox* de EduBot vigila el código: si detecta que tarda demasiado, lo detiene automáticamente (Timeout) y te devuelve un mensaje de error amigable para que corrijas tu lógica sin dañar el equipo.

3. **El Interceptor de Notas (Para Profesores):**
   Si eres estudiante y pasaste por un nodo "Caja Negra" (Validador), EduBot hizo un trabajo silencioso. Al terminar tu código, comprobó si tu variable final coincidía con la respuesta que el profesor esperaba. Si acertaste, EduBot no te lo dice a gritos; envía silenciosamente un mensaje (`EDUBOT_VAL_PASS`) a la base de datos local de tu profesor. ¡Así es como funciona la evaluación automática!

---

Usen EduBot con sabiduría, después de todo, este es el preview oficial de la aplicación de escritorio. Pueden haber actualizaciones, con más cosas y más documentación al respecto, no se queden solo con lo que ven ahora, experimenten con la app y vean qué pueden hacer con ella con la lógica de programación pura. 

¡EduBot Team está alegre de darles a conocer y probar la primera versión oficial de **EduBot Forest Studio**! ¡Aprovéchenlo!