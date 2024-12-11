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
1. Entrada: Texto proporcionado por el usuario.
2. Preprocesamiento: Normalización, tokenización y conversión a secuencias.
3. Modelo: Modelo Secuencial - Define las capas del modelo de manera lineal.
   - **_Embedding Layer_**: Convierte las palabras (o tokens) en representaciones vectoriales densas..
   - **_Capa convolucional_** (Conv1D): Extrae características locales (n-gramas) del texto para encontrar patrones relevantes.
   - **_Max Pooling Global_** (GlobalMaxPooling1D): Reduce dimensionalidad seleccionando las características más significativas. 
   - **_Capas densas para clasificación_:** Realizan la clasificación basada en las características extraídas.
   - **_Softmax_:** Genera probabilidades para cada clase, permitiendo asignar la categoría más probable.
4. Salida: Respuesta seleccionada de un archivo JSON basado en la predicción del modelo.

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


