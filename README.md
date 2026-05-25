# 🌳 EduBot Forest Studio - Offline Suite

**v1.0 Stable • Python Fundamentals & OOP**

**EduBot Forest Studio** es un Entorno de Desarrollo Integrado (IDE) visual y local, diseñado específicamente como software EdTech (Tecnología Educativa). Su misión principal es cerrar la brecha cognitiva entre la programación visual (tipo bloques) y la sintaxis de texto puro, permitiendo a los estudiantes de secundaria y nivel universitario aprender **Fundamentos de Python y Programación Orientada a Objetos (POO)** de manera intuitiva, segura y sin necesidad de conexión a internet.

---

## 📖 Tabla de Contenidos

1. [¿Qué es EduBot Forest Studio?](https://www.google.com/search?q=%23-qu%C3%A9-es-edubot-forest-studio)
2. [El Paradigma: Programación en Bosques](https://www.google.com/search?q=%23-el-paradigma-programaci%C3%B3n-en-bosques)
3. [Público Objetivo y Casos de Uso](https://www.google.com/search?q=%23-p%C3%BAblico-objetivo)
4. [Módulos Principales (EdTech)](https://www.google.com/search?q=%23-m%C3%B3dulos-principales-edtech)
5. [Arquitectura Técnica del Sistema](https://www.google.com/search?q=%23-arquitectura-t%C3%A9cnica)
6. [Guía de Instalación y Despliegue](https://www.google.com/search?q=%23-gu%C3%ADa-de-instalaci%C3%B3n)
7. [Créditos y Licencia](https://www.google.com/search?q=%23-cr%C3%A9ditos-y-licencia)

---

## 💡 ¿Qué es EduBot Forest Studio?

En la enseñanza tradicional de la informática, los estudiantes suelen enfrentarse a una curva de aprendizaje pronunciada al saltar de herramientas de arrastrar y soltar (como Scratch) a entornos de desarrollo profesionales basados en texto, donde un simple error de indentación o tipografía puede causar frustración severa.

**EduBot** resuelve este problema al ofrecer un entorno híbrido. El estudiante construye la lógica de su programa orquestando un "Bosque" de nodos interconectados. En tiempo real, el sistema lee esta orquestación visual y genera código Python 100% puro y estructurado. El entorno permite ejecutar ese código en un *Sandbox* (caja de arena) seguro, leer los resultados en una terminal interactiva y recibir correcciones de sintaxis asistidas.

EduBot está diseñado con una premisa fundamental: **Accesibilidad Total Offline**. Reconociendo las limitaciones de infraestructura en muchas instituciones latinoamericanas, el software empaqueta su propio motor de Chromium, base de datos encriptada y entorno de ejecución, garantizando que un laboratorio de computación entero pueda funcionar sin un solo byte de conexión a internet.

---

## 🌿 El Paradigma: Programación en Bosques

EduBot introduce el concepto pedagógico de **Programación en Bosques** (Forest Programming Paradigm), alejándose del concepto de "bloques de rompecabezas" tradicionales.

En lugar de apilar bloques verticalmente de forma rígida, el estudiante planta "Semillas" (Nodos) en un lienzo infinito bidimensional.

* **Las Raíces (Conexiones de Entrada):** Reciben datos de otras estructuras.
* **Las Ramas (Conexiones de Salida):** Definen el flujo de ejecución o envían datos procesados.
* **La Topología del Bosque:** Define el alcance (*Scope*). A diferencia del código escrito donde la indentación es abstracta, en EduBot un ciclo `For` o una clase `Class` envuelve visualmente a sus "hojas", haciendo que conceptos complejos como la herencia, la instanciación de objetos, el parámetro `self` y la encapsulación sean tangibles y observables geométricamente.

Además, la composición y estructura de la lógica dentro de la propuesta de la programación visual en bosques, cambia a como está planteado en la programación en bloques tradicional, para poder adaptar fielmente la abstracción compleja de la Programación Orientada a Objetos y hacerlo visualmente intuitivo y comprensible para quienes estén empezando. 

Existen dos orientaciones dentro del lienzo para la construcción y orden de la lógica en EduBot al orquestar visualmente el código:

* **Eje Horizontal (Izquierda a Derecha) para la Lógica de Construcción**: Este eje permite al estudiante modelar gráficamente la composición y jerarquía de los objetos. Al enlazar un nodo principal o padre (como una Clase) con nodos adyacentes o también llamados hijos (Como una función o variable) hacia la derecha, se definen los atributos, propiedades y métodos encapsulados, estableciendo visualmente la estructura fundamental del componente.

* **Eje Vertical (De Arriba hacia Abajo) para el Orden de Ejecución**: Este eje gestiona la dimensión temporal y el flujo del algoritmo. Las conexiones dispuestas verticalmente determinan la secuencia cronológica en la que el intérprete ejecutará las instrucciones, facilitando la comprensión de anidamientos, estructuras de control y llamadas a funciones dentro de una misma instancia.

Este paradigma y enfoque en la progrmación visual permite que el cerebro del estudiante asimile la lógica algorítmica (diagramas de flujo) y la Programación Orientada a Objetos simultáneamente, viendo cómo interactúan las clases y los métodos antes de preocuparse por si olvidaron colocar dos puntos (`:`) al final de una línea.

---

## 🎯 Público Objetivo

EduBot Forest Studio está estructurado para atender a dos perfiles principales en el ecosistema educativo:

### 1. Instituciones y Profesores de Informática

* **Gestión Centralizada:** Ideal para docentes que necesitan llevar un control estricto del progreso de sus alumnos. EduBot incluye un panel de control oculto bajo llave maestra (root).
* **Auditoría Offline:** Permite registrar, editar y auditar calificaciones localmente en las computadoras del laboratorio, vinculadas a retos específicos ("Pruebas o Exámenes").

### 2. Estudiantes (Secundaria Avanzada y Universidad)

* **Principiantes Absolutos:** Para quienes escriben su primera línea de código y necesitan entender las estructuras de control (Condicionales, Ciclos).
* **Estudiantes Intermedios:** Aquellos que están haciendo la transición a la **Programación Orientada a Objetos (POO)** en Python y necesitan visualizar cómo se construye un Constructor (`__init__`), cómo interactúan los atributos y cómo se manejan los errores (`Try/Except`).

---

## 📦 Módulos Principales (EdTech)

EduBot no es solo un lienzo; es una suite educativa compuesta por múltiples módulos interactuando en tiempo real:

### 🎨 1. El Lienzo de Bosque (Visual Editor)

Un entorno interactivo construido sobre React Flow. Ofrece un catálogo completo de nodos clasificados:

* **Fundamentos:** Variables, Inputs, Prints, Castings (Int/Float).
* **Matemáticas y Lógica:** Operaciones aritméticas y comparadores.
* **Control de Flujo:** If, Elif, Else, For, While, Break/Continue.
* **POO (Objetos y Funciones):** Definición de Clases, Herencia, Constructores, Métodos, Atributos de instancia (`self`), Returns y llamadas a funciones.
* **Avanzado:** Manejo de excepciones (Try/Except) y un placeholder de importación de módulos (`import`) para futuro manejo de librerías en caso de hacer una Suite Online.

### 🧠 2. Intérprete AST (Traductor Bidireccional)

El corazón del sistema. Un motor de traducción personalizado (basado en Abstract Syntax Trees) que es capaz de:

* Leer la orquestación topológica de los nodos e inyectar el código Python exacto con su correcta indentación.
* Tomar código escrito en Python e intentar realizar ingeniería inversa para generar los nodos visuales correspondientes.

### 🛡️ 3. Sandbox de Ejecución y Escudos Anti-Errores

La plataforma evita que la computadora del estudiante colapse ante errores lógicos comunes:

* **Analizador Estático (Anti-Bucles Infinitos):** Antes de enviar el código al procesador, un algoritmo de frontend escanea los nodos `While` y `For`. Si detecta que la condición de salida jamás se cumplirá, aborta la ejecución preventivamente, colorea el nodo en rojo y advierte al estudiante.
* **Lector Tipográfico Recursivo:** Revisa el árbol de variables. Si el estudiante escribe "numer" en lugar de "numero", el sistema detecta el error de sintaxis antes de la ejecución y sugiere correcciones.
* **Cola de Ejecución Desechable (Multiprocessing):** Cada ejecución ocurre en un proceso hijo aislado que se comunica a través de tuberías RAM. Si el código falla gravemente, el profesor o estudiante puede usar el botón de *SIGKILL* ("Detener") para asesinar el hilo sin provocar un *Deadlock* en la aplicación.

### 📊 4. Dashboard Docente y Seguridad

* Sistemas de Login independientes para Estudiantes y Profesores.
* Archivos de base de datos (`users.json`, `students.json`, `grades.json`) fuertemente encriptados con el algoritmo militar **AES (Fernet)**. Ningún estudiante puede alterar las notas abriendo los archivos desde el explorador de Windows/Linux.
* Sistema integral CRUD (Crear, Leer, Actualizar, Borrar) para gestionar aulas.

### ♿ 5. Módulo de Accesibilidad Cognitiva

Diseñado para la inclusión:

* Modo de Fuente para Dislexia (cambia las tipografías para facilitar la lectura de caracteres confusos).
* Reducción de Movimiento visual (desactiva las animaciones de flujo para estudiantes con hiperactividad o sensibilidad visual).
* Escalado de UI y Editor de Código adaptable a proyectores o pantallas pequeñas.

---

## ⚙️ Arquitectura Técnica

EduBot sigue una arquitectura **Frontend Reactivo + Backend Híbrido** empaquetada como aplicación de escritorio nativa:

* **Interfaz de Usuario (Frontend):** * **React.js 18** (Renderizado).
* **React Flow** (Motor matemático de renderizado de nodos y conexiones de curvas de Bézier).
* **Tailwind CSS** (Motor de estilos por utilidad).
* **Prism.js** (Resaltado de sintaxis de código en tiempo real).


* **Núcleo de Sistema (Backend):**
* **Python 3.8+** (Lenguaje anfitrión).
* **Eel** (Puente de comunicación asíncrona WebSocket entre Python y Javascript sin levantar un servidor web pesado tradicional).
* **Gevent & Multiprocessing** (Gestión de concurrencia y ejecución de hilos).
* **Cryptography** (Manejo de claves y encriptación local).


* **Motor de Renderizado:**
* El sistema levanta una instancia local incrustada de **Chromium Portable**, asegurando que la interfaz gráfica luzca y funcione exactamente igual en cualquier PC, sin importar si tienen un navegador desactualizado.



---

## 🚀 Guía de Instalación

EduBot está preparado para su distribución offline. Si vas a compilar o ejecutar el proyecto desde el código fuente, sigue estos pasos:

### Prerrequisitos

* Tener instalado **Python 3.8** o superior en el sistema.

### Instalación en Windows

1. Clona o descarga el código fuente del repositorio.
2. Haz doble clic en el archivo `install_dependencies.bat`.
3. El instalador verificará Python, creará un Entorno Virtual aislado (`venv`), instalará las librerías exactas (`Eel`, `cryptography`, etc.) y limpiará el caché de la máquina.
4. Para abrir el entorno, simplemente entra a la carpeta, activa el entorno y ejecuta:

```bash
venv\Scripts\activate
python main.py

```

### Instalación en Linux / Mac

1. Clona el repositorio y abre una terminal en la ruta.
2. Da permisos de ejecución e instala las dependencias:

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh

```

3. Activa y ejecuta:

```bash
source venv/bin/activate
python main.py

```

### En caso de tener el empaquetado en Linux

1. Extrae el .zip de EduBotForestStudio al descargar el paquete dentro de las releases en GitHub.
2. Accede a una terminal desde la carpeta de EduBot al terminar de extraer (Click derecho -> Abrir en terminal)
3. Escribe en la terminal ./install.sh para instalarlo localmente
4. Prueba el entorno desde el menú de Linux, en la sección de programación

### Compilar a Ejecutable (.exe)

Para distribuir EduBot a colegios sin necesidad de instalar Python, puedes compilarlo usando PyInstaller. Asegúrate de incluir la carpeta de Chromium portable para Windows:

```cmd
pyinstaller --noconfirm --windowed --icon="web/favicon.ico" --name "EduBot" --add-data "web;web" --add-data "chromium;chromium" --add-data "docs/user/Manual de Usuario EduBot.pdf;." main.py

```

---

## 📄 Créditos y Licencias

**Desarrollado y conceptualizado por:**

* Wilner Manzanares (Project Manager)
* Wilfredo García
* Dannielys Sandoval
* Rodrigo Moreno

**Universidad Politécnica Territorial de Falcón "Alonso Gamero" (UPTAG) - PNF en Informática.**

Este proyecto y sus manuales de usuario se distribuyen bajo la licencia **Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional (CC BY-NC-SA 4.0)**.
Eres libre de copiar, distribuir y adaptar este material para propósitos educativos y no comerciales, siempre y cuando otorgues el crédito adecuado al equipo de desarrollo y mantengas esta misma licencia en tus modificaciones.

El código del proyecto se comparte en GitHub bajo la licencia **AGPLv3.0**. 
Eres libre de tomar el código, copiar y distrubuirlo con tus propias versiones readaptadas del código fuente original, siempre y cuando mantengas esta misma licencia en tus modificaciones y compartas dicho código públicamente a favor de la comunidad Open-Source, además de abstenerce de usarlo con fines monetarios o comerciales sin consentimiento del equipo desarrollador.

> *"La computadora no es inteligente, es obediente. Nuestra labor es enseñarle a las nuevas generaciones cómo darle las instrucciones correctas."* — **EduBot Team**