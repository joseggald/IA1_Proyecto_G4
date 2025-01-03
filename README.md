# Manual Técnico: ChatBot Pro - Sistema de Asistente Virtual Bilingüe para Programación

## 1. Introducción

### 1.1 Propósito
ChatBot Pro es un sistema de asistente virtual bilingüe especializado en programación, diseñado para proporcionar respuestas precisas y contextuales en español e inglés sobre temas de programación, con enfoque particular en Python y JavaScript.

### 1.2 Alcance
El sistema está diseñado para:
- Proporcionar asistencia en temas de programación básica y algoritmos
- Ofrecer soporte bilingüe (español e inglés)
- Mantener conversaciones contextuales coherentes
- Proporcionar ejemplos de código y explicaciones técnicas
- Gestionar un historial de conversaciones persistente

## 2. Arquitectura del Sistema

### 2.1 Componentes Principales
El sistema está compuesto por tres componentes principales:

1. **Backend del Modelo (model.py)**
   - Implementación del modelo de procesamiento de lenguaje natural
   - Sistema de gestión de tópicos
   - Motor de procesamiento de conversaciones

2. **Interfaz Gráfica (chatbot_gui.py)**
   - Interfaz de usuario moderna construida con tkinter y ttkbootstrap
   - Sistema de gestión de historial de chat
   - Manejo de múltiples conversaciones

3. **Sistema de Datos**
   - Gestión de respuestas (response.json)
   - Gestión de tópicos (topic_data.json)
   - Datos de entrenamiento (data.json)

### 2.2 Tecnologías Utilizadas
- **Python 3.8+**
- **TensorFlow/Keras** para el modelo de procesamiento de lenguaje natural
- **NLTK** para procesamiento de texto
- **sentence-transformers** para embeddings de texto
- **scikit-learn** para utilidades de machine learning
- **tkinter/ttkbootstrap** para la interfaz gráfica
- **JSON** para almacenamiento de datos

## 3. Componentes del Sistema en Detalle

### 3.1 Modelo de Procesamiento (EnhancedTopicAwareChatbot)

#### 3.1.1 Inicialización
```python
class EnhancedTopicAwareChatbot:
    def __init__(self):
        self.vocab_size = 20000
        self.max_length = 20
        self.embedding_dim = 128
        self.memory_size = 10
        self.topic_threshold = 0.6
        self.batch_size = 32
        self.epochs = 150
```

#### 3.1.2 Detección de Idioma
El sistema utiliza un conjunto de palabras en inglés comunes para detectar el idioma:
```python
def detect_language(self, text: str) -> str:
    words = set(text.lower().split())
    english_count = len(words.intersection(self.english_words))
    return 'en' if english_count / max(len(words), 1) > 0.2 else 'es'
```

#### 3.1.3 Sistema de Tópicos
El sistema implementa un sofisticado manejo de tópicos que incluye:
- Tópicos primarios y secundarios
- Palabras clave asociadas
- Relaciones entre tópicos
- Gestión de subtópicos

### 3.2 Interfaz Gráfica (ModernChatbotGUI)

#### 3.2.1 Componentes de la GUI
La interfaz está construida con los siguientes elementos principales:
- Panel lateral para historial de conversaciones
- Área principal de chat
- Campo de entrada de mensajes
- Sistema de temas visuales

#### 3.2.2 Gestión de Conversaciones
```python
def save_current_chat(self):
    chat_data = {
        'id': self.current_chat_id,
        'timestamp': datetime.now().isoformat(),
        'messages': self.get_messages_from_content(content)
    }
```

### 3.3 Sistema de Datos

#### 3.3.1 Estructura de Tópicos
```json
{
  "topics": {
    "topic_id": {
      "primary_keywords": [],
      "keywords": [],
      "labels": [],
      "subtopics": []
    }
  }
}
```

#### 3.3.2 Estructura de Respuestas
```json
{
  "label": [
    "respuesta1",
    "respuesta2",
    "respuesta3"
  ]
}
```

## 4. Flujo de Procesamiento

### 4.1 Procesamiento de Entrada
1. Recepción del texto del usuario
2. Detección de idioma
3. Preprocesamiento del texto
4. Identificación de tópicos
5. Generación de embeddings

### 4.2 Generación de Respuestas
1. Predicción de la clase de respuesta
2. Ajuste basado en tópicos
3. Selección de respuesta
4. Actualización del historial

## 5. Sistema de Persistencia

### 5.1 Almacenamiento de Conversaciones
- Las conversaciones se guardan en archivos JSON
- Cada conversación tiene un UUID único
- Se mantiene un registro de timestamp
- Se almacena el historial completo de mensajes

### 5.2 Estructura del Almacenamiento
```
chat_history/
  ├── chat_{uuid1}.json
  ├── chat_{uuid2}.json
  └── chat_{uuid3}.json
```

## 6. Características Bilingües

### 6.1 Gestión de Idiomas
- Detección automática de idioma
- Tópicos específicos por idioma
- Respuestas en el idioma correspondiente
- Mantenimiento de contexto bilingüe

### 6.2 Estructura de Tópicos Bilingües
```json
{
  "topic_es": {
    "keywords": ["palabras", "clave", "español"]
  },
  "topic_en": {
    "keywords": ["keywords", "in", "english"]
  }
}
```

## 7. Mantenimiento y Extensión

### 7.1 Agregar Nuevos Tópicos
1. Añadir entrada en topic_data.json
2. Definir palabras clave primarias y secundarias
3. Establecer relaciones con otros tópicos
4. Agregar respuestas correspondientes

### 7.2 Modificar Respuestas
1. Editar response.json
2. Mantener coherencia con los tópicos
3. Asegurar respuestas en ambos idiomas

## 8. Requisitos del Sistema

### 8.1 Requisitos de Software
- Python 3.8 o superior
- TensorFlow 2.x
- NLTK
- sentence-transformers
- scikit-learn
- ttkbootstrap

### 8.2 Requisitos de Hardware
- Mínimo 4GB RAM
- Procesador de 2 núcleos o superior
- 500MB espacio en disco

## 9. Instalación y Configuración

### 9.1 Preparación del Entorno
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 9.2 Configuración Inicial
```bash
python -m nltk.downloader punkt
python model.py --train  # Para entrenar el modelo inicial
```

## 10. Solución de Problemas

### 10.1 Problemas Comunes
1. **Errores de memoria**
   - Reducir batch_size
   - Disminuir vocab_size

2. **Problemas de rendimiento**
   - Ajustar memory_size
   - Optimizar topic_threshold

### 10.2 Mejores Prácticas
- Mantener respuestas concisas y coherentes
- Actualizar regularmente los tópicos
- Monitorear el uso de memoria
- Realizar copias de seguridad del historial

## 11. Anexos

### 11.1 Estructura de Archivos
```
project/
  ├── model.py
  ├── chatbot_gui.py
  ├── data.json
  ├── response.json
  ├── topic_data.json
  ├── requirements.txt
  └── chat_history/
```

### 11.2 Dependencias
```
tensorflow>=2.0.0
nltk>=3.6.0
scikit-learn>=0.24.0
sentence-transformers>=2.0.0
ttkbootstrap>=1.0.0
numpy>=1.19.0
```

## 12. Referencias

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [NLTK Documentation](https://www.nltk.org/)
- [sentence-transformers Documentation](https://www.sbert.net/)
- [ttkbootstrap Documentation](https://ttkbootstrap.readthedocs.io/)