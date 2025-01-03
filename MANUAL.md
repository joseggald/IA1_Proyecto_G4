# Manual de Usuario del **ChatBot Pro**

Bienvenido al manual de usuario del **ChatBot Pro**. Este chatbot ha sido diseñado como una herramienta avanzada de asistencia que responde preguntas y guía a los usuarios en diversas consultas utilizando inteligencia artificial.

Para inicializar la app y ejecutarla:

```
pip install -r requirements.txt
python chatbot_gui.py
```

## Capacidades del ChatBot Pro

El **ChatBot Pro** puede interactuar y responder dudas relacionadas con una amplia gama de temas, incluyendo:

- **Conceptos de programación**: desde fundamentos como variables, condicionales y ciclos, hasta temas avanzados como programación orientada a objetos (OOP), herencia, polimorfismo, abstracción, etc
- **Python y Javascript**: preguntas sobre sus usos, instalación, ejecución de scripts, estructuras de control, bucles y otros conceptos clave del lenguaje, como generación de codigo.

## Idiomas

El chatbot tiene la capacidad de manejar consultas tanto en **español** como en **inglés**, permitiendo una experiencia bilingüe fluida para los usuarios.

## Información sobre el ChatBot Pro

Además de responder preguntas, el **ChatBot Pro** puede proporcionar detalles sobre su desarrollo, como las tecnologías utilizadas para su implementación. Está construido con:

- **Inteligencia artificial** basada en modelos LSTM.
- **Tecnologías clave**: TensorFlow, Transformers, sklearn,  Python y Tkinter.

## Uso del Manual

Este documento describe los pasos necesarios para interactuar con el chatbot y detalla sus funciones principales. Encontrarás guías prácticas, ejemplos de preguntas y descripciones de las respuestas esperadas para aprovechar al máximo esta herramienta.

## **Descripción de la Interfaz**

![1735884860192](images/MANUAL/1735884860192.png)


| **#** | **Elemento**                              | **Descripción**                                                                                                                                                   |
| ----- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | **Campo de Respuesta del Chatbot**        | Muestra las respuestas generadas por el chatbot basadas en la entrada proporcionada por el usuario.                                                                |
| 2     | **Campo de Entrada del Chatbot**          | Mensajes que el usuario envío al chatbot, historial.                                                                                                              |
| 3     | **Cuadro de texto - Entrada del Usuario** | Área donde el usuario introduce sus mensajes o preguntas para interactuar con el chatbot.                                                                         |
| 4     | **Botón de Enviar**                      | Botón para enviar el mensaje ingresado por el usuario. Al hacer clic, el texto en el campo de entrada se envía al chatbot para procesar y generar una respuesta. |
| 5     | **Botón de Crear Nuevo Chat**            | Botón que permite crear nuevos chats.                                                                                                                             |

## **Guía de Uso**

### **Iniciar una Conversación**

1. Escribe tu mensaje o consulta en el **Campo de Entrada del Usuario** (Elemento 2).
2. Haz clic en el **Botón de Enviar** (Elemento 4) para enviar tu mensaje.
3. Observa la respuesta generada por el chatbot en el **Campo de Respuesta del Chatbot** (Elemento 3).
4. Si deseas comenzar una nueva conversación, haz clic en el **de cre Conversación** (Elemento 5).

## **Ejemplo de Interacción**

### Escenario 1: Saludo

- **Entrada del Usuario**: `Hola`.
- **Respuesta del Chatbot**: `¡Hola! ¿Qué tal?`.

### Escenario 2: Pregunta Técnica

- **Entrada del Usuario**: `¿Qué es python?`.
- **Respuesta del Chatbot**: `Python es ideal para principiantes y expertos, siendo utilizado en desarrollo web, análisis de datos, inteligencia artificial y más. ¿Quieres empezar por instalarlo?`.

![1735885336935](images/MANUAL/1735885336935.png)

## **Resolución de Problemas**

### **Problemas Comunes y Soluciones**

1. **El chatbot no responde**:
   - Verifica tu conexión a internet.
   - Reinicia la aplicación.
2. **Las respuestas no son claras o no aplican a la pregunta**:
   - Reformula tu pregunta o utiliza un lenguaje más específico.
3. **Error al enviar un mensaje**:
   - Asegúrate de que el campo de entrada no esté vacío.
