# Manual Técnico: ChatBot Pro - Basado en TensorFlow

## **1. Introducción**
### Propósito
El modelo de Python con TensorFlow implementado en el archivo ```model.py``` es un chatbot basado en tópicos que utiliza aprendizaje profundo y procesamiento del lenguaje natural.
### Alcance
El chatbot es útil para aplicaciones como:
- Sistemas de soporte.
- Entrenamiento en lenguajes naturales.
- Automatización de interacciones repetitivas.

---

## **2. Requisitos del Sistema**

### Software
- **Sistema Operativo**: Windows, macOS o Linux.
- **Python**: Versión 3.8 o superior.
- **REACT**
- Dependencias:
    - TensorFlow
    - Keras
    - NLTK
    - NumPy
    - scikit-learn

### Configuración Previa
1. Instalar Python.
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

---

## **3. Descripción General**
### Arquitectura del Modelo
El modelo utilizado en el **ChatBot Pro** está diseñado para procesar texto de manera eficiente y proporcionar respuestas relevantes basadas en temas específicos. A continuación, se describe su funcionamiento y estructura:

#### 1. **Entrenamiento y Configuración en Python**
El modelo es inicialmente entrenado en Python utilizando la biblioteca **TensorFlow**. Los principales componentes de esta fase son:

- **Preprocesamiento de texto:** 
  - Se utiliza **NLTK** para tokenizar y limpiar el texto.
  - Se normalizan caracteres especiales (como acentos y puntuación) y se eliminan palabras irrelevantes.
  
- **Arquitectura del Modelo:**
  - **Embedding Layer:** Representa las palabras en un espacio vectorial.
  - **Conv1D:** Extrae patrones locales de las palabras para capturar relaciones entre ellas.
  - **GlobalMaxPooling1D:** Reduce la dimensionalidad del resultado y resalta características clave.
  - **Dense Layers:** Procesan la información para clasificarla en las categorías configuradas.
  - **Softmax:** Genera probabilidades para cada clase de respuesta.

- **Entrenamiento:**
  - Utiliza **categorical_crossentropy** como función de pérdida y el optimizador Adam.
  - Se divide el conjunto de datos en entrenamiento y validación para mejorar el rendimiento.

- **Guardado del Modelo:**
  - El modelo entrenado se guarda como un archivo H5, junto con el tokenizador y los datos de tópicos.

#### 2. **Exportación e Implementación en JavaScript**
El modelo exportado en Python se convierte para su uso en JavaScript mediante **TensorFlow.js**, adaptándose al entorno de ejecución del cliente. 

- **Cargado del Modelo:**
  - Se utilizan los archivos `model.json` (topología del modelo) y `tokenizer.json` (información del tokenizador).
  - Las capas se ajustan para mantener la compatibilidad con TensorFlow.js.

- **Procesamiento de Entrada:**
  - **Preprocesamiento en JS:** Realiza limpieza y tokenización similares a las del modelo en Python.
  - **Padding:** Se asegura de que las secuencias tengan una longitud fija (20 en este caso).

- **Identificación de Tópicos:**
  - Combina palabras clave extraídas del texto con un sistema de puntuación basado en relaciones entre tópicos y contexto histórico de la conversación.

- **Ajuste Basado en Tópicos:**
  - Si un tópico es identificado, las predicciones relacionadas se ajustan para aumentar su confianza.

#### 3. **Limitaciones de JavaScript**
Aunque se implementa el modelo en TensorFlow.js, su funcionalidad en el navegador tiene algunas limitaciones en comparación con Python:

- **Rendimiento:**
  - El procesamiento en el navegador puede ser más lento, especialmente en dispositivos de baja potencia.
  
- **Preprocesamiento:**
  - La tokenización y normalización de texto son menos sofisticadas en JavaScript.

- **Capacidades del Modelo:**
  - Algunos ajustes avanzados (como regularizaciones complejas) no son directamente compatibles en TensorFlow.js.

#### 4. **Funcionalidades Clave**
El modelo está diseñado para:
- Responder preguntas relacionadas con **programación, SQL y Python**.
- Manejar temas casuales como chistes, actividades, y citas de películas.
- Operar tanto en **español** como en **inglés**, adaptándose al idioma del usuario.

#### 5. **Interacción entre Python y JavaScript**
- **Python:** Responsable del entrenamiento y ajustes complejos del modelo.
- **JavaScript:** Implementa el modelo para aplicaciones web, permitiendo que los usuarios interactúen directamente desde el navegador.

### Tecnologías Utilizadas
- **TensorFlow/Keras**: Para la creación y entrenamiento del modelo.
- **NLTK**: Para tokenización y extracción de palabras clave.
- **scikit-learn**: Para dividir los datos de entrenamiento y validación.

### Arquitectura de la solución final
![model.png](docu-images%2Fmodel.png)

### Detalles del Modelo y Funcionalidades

#### **Datos y Entrenamiento**
- Los datos de entrenamiento provienen del archivo `data.json` que contiene pares de texto y etiquetas. Cada etiqueta representa una categoría de respuesta.
- Los datos son preprocesados para:
    - Normalizar el texto.
    - Eliminar caracteres especiales.
    - Extraer palabras clave.
- El modelo se entrena usando:
    - **`categorical_crossentropy`** como función de pérdida.

---

#### **Predicción y Respuesta**
- El modelo predice la etiqueta del texto ingresado por el usuario.
- Las respuestas correspondientes a cada etiqueta están definidas en el archivo `response.json`.
- El modelo ajusta la predicción en función del contexto y el historial de la conversación.

---

#### **Manejo de Tópicos**
- El chatbot utiliza un sistema de tópicos almacenado en `topic_data.json` para identificar y asociar temas relevantes al texto.
- Este sistema ayuda a:
    - Mejorar la precisión de las respuestas.
    - Mantener la coherencia en conversaciones prolongadas.

---

#### **Funciones Clave**
1. **Entrenamiento**:
    - Entrena el modelo con datos preprocesados y lo guarda en formato `.h5`.
2. **Carga del Modelo**:
    - Permite reutilizar un modelo entrenado previamente.
3. **Historial de Conversación**:
    - Utiliza una cola (`deque`) para mantener un historial limitado de interacciones.
4. **Extracción de Palabras Clave**:
    - Identifica palabras importantes del texto ingresado.


---

## **4. Configuración e Instalación**
1. Clonar el repositorio del proyecto:
```bash
git clone <URL-del-repositorio>
cd <nombre-del-directorio>
```
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```
3. Crear las carpetas necesarias para guardar modelos y datos:
```bash
mkdir -p model public
```
4. Verificar la instalación ejecutando:
```bash
python model.py
```

---

## **5. Estructura del Código**
### Archivos Principales
- **`model.py`**: Contiene la clase principal `TopicAwareChatbot` y su funcionalidad.
- **`data.json`**: Datos de entrenamiento con texto y etiquetas.
- **`response.json`**: Respuestas predefinidas para cada etiqueta.
- **`tokenizer.json`**: Tokenizador entrenado para convertir texto a secuencias.

### Componentes Clave
- **`TopicAwareChatbot`**: Clase que gestiona el modelo, preprocesamiento y generación de respuestas.
- **`train_model()`**: Funcionalidad para entrenar el modelo.
- **`get_response(text)`**: Genera una respuesta basada en el texto de entrada.

---

## **6. Uso del Sistema**
### Entrenamiento del Modelo
1. Cargar los datos de entrenamiento desde `data.json`.
2. Ejecutar el script principal:
```bash
python model.py
```
3. El modelo entrenado se guarda en `model/chatbot_model.h5`.

### Interacción con el Chatbot
1. Iniciar el chatbot ejecutando:
```bash
python model.py
```
2. Ingresar texto en la consola.
3. El chatbot responde basado en su entrenamiento y contexto.

---

## **7. Mantenimiento**
### Actualización de Datos
- **Agregar nuevas respuestas**: Editar `response.json`.
- **Ampliar datos de entrenamiento**: Añadir entradas en `data.json` y volver a entrenar.

### Depuración
- Verificar errores en consola durante la ejecución.
- Usar `print` o herramientas como TensorBoard para monitorear.

---

## **8. Ejemplo de Uso**
Entrada: `¿Qué es un ciclo?`

Salida:
```
Bot: Un ciclo permite ejecutar un bloque de código repetidamente mientras una condición sea verdadera.
Confianza: 0.92
```

---

## **9. Preguntas Frecuentes (FAQ)**
1. **¿El modelo necesita GPU?**
   No, pero mejora el rendimiento de entrenamiento.
2. **¿Cómo extiendo el vocabulario?**
   Actualizar `data.json` con nuevas entradas y volver a entrenar.

---

## **10. Anexos**
### Glosario
- **Tokenización**: Proceso de dividir texto en palabras o frases pequeñas.
- **Embedding**: Representación vectorial de palabras para aprendizaje.

### Referencias
- Documentación de [TensorFlow](https://www.tensorflow.org/).
- Guía de [Keras](https://keras.io/).


