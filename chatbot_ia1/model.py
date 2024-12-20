import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D, Dropout, Conv1D
import tensorflow as tf
from collections import deque
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from tensorflow.keras.regularizers import l2
from nltk.tokenize import word_tokenize
import nltk
import re


class TopicAwareChatbot:
    def __init__(self):
        # Parámetros del modelo
        self.vocab_size = 2000
        self.max_length = 20
        self.embedding_dim = 128
        self.memory_size = 10

        # Rutas de archivos
        self.model_dir = 'saved_model'
        self.keras_model_path = os.path.join('model', 'chatbot_model.h5')
        self.tokenizer_path = os.path.join('public', 'tokenizer.json')
        self.responses_path = 'response.json'
        self.topic_data_path = os.path.join('public', 'topic_data.json')
        
        # Inicialización de componentes
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = None
        self.responses = None
        self.topic_data = self.initialize_topic_data()
        self.conversation_history = deque(maxlen=self.memory_size)

        # Parámetros de entrenamiento
        self.batch_size = 32
        self.epochs = 150
        self.validation_split = 0.20
        
        # Inicializar NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Cargar datos
        self.load_responses()

    def initialize_topic_data(self):
        """Inicializa o carga los datos de tópicos"""
        try:
            with open(self.topic_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'topics': {},
                'keywords': {},
                'relations': {}
            }

    def load_responses(self):
        """Carga el archivo de respuestas"""
        try:
            with open(self.responses_path, 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
            print("Respuestas cargadas exitosamente")
        except Exception as e:
            print(f"Error al cargar respuestas: {str(e)}")
            self.responses = {}

    def save_model(self):
        """Guarda el modelo y datos relacionados"""
        try:
            # Crear directorios
            os.makedirs(os.path.dirname(self.keras_model_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.tokenizer_path), exist_ok=True)

            # Guardar modelo
            print("Guardando modelo...")
            self.model.save(self.keras_model_path)
            
            # Guardar tokenizer
            print("Guardando tokenizer...")
            tokenizer_json = self.tokenizer.to_json()
            with open(self.tokenizer_path, 'w', encoding='utf-8') as f:
                f.write(tokenizer_json)
            
            # Guardar topic data
            print("Guardando datos de tópicos...")
            with open(self.topic_data_path, 'w', encoding='utf-8') as f:
                json.dump(self.topic_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error al guardar: {str(e)}")
            return False

    def load_model(self):
        """Carga el modelo y datos relacionados"""
        try:
            if os.path.exists(self.keras_model_path) and os.path.exists(self.tokenizer_path):
                # Cargar modelo
                self.model = tf.keras.models.load_model(self.keras_model_path)
                
                # Cargar tokenizer
                with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
                    tokenizer_json = f.read()
                    self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
                
                print("Modelo y datos cargados exitosamente")
                return True
            return False
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            return False

    def build_model(self, num_classes):
        """Modelo más simple y efectivo"""
        model = Sequential([
            Embedding(self.vocab_size, 64, input_length=self.max_length),
            Conv1D(64, 3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def extract_keywords(self, text):
        """Extrae palabras clave del texto"""
        words = word_tokenize(text.lower())
        keywords = [w for w in words if len(w) > 2 and not w.isdigit()]
        return keywords

    def identify_topic(self, text, keywords):
        """Identifica el tópico del texto"""
        if not self.topic_data['topics']:
            return None

        topic_scores = {}
        for topic, data in self.topic_data['topics'].items():
            score = 0
            # Coincidencia de keywords
            if 'keywords' in data:
                matches = set(keywords) & set(data['keywords'])
                score += len(matches) * 2

            # Contexto de conversación
            if self.conversation_history:
                last_topic = self.conversation_history[-1].get('topic')
                if last_topic == topic:
                    score += 1
                elif topic in self.topic_data['relations'].get(last_topic, []):
                    score += 0.5

            topic_scores[topic] = score

        best_topic = max(topic_scores.items(), key=lambda x: x[1])
        return best_topic[0] if best_topic[1] > 0 else None

    def preprocess_text(self, text):
        """Preprocesa el texto de entrada"""
        # Normalización básica
        text = text.lower().strip()
        
        # Normalizar caracteres especiales
        chars_to_replace = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ü': 'u', 'ñ': 'n', '¿': '', '¡': ''
        }
        for old, new in chars_to_replace.items():
            text = text.replace(old, new)
        
        # Limpieza de texto
        text = ' '.join(text.split())
        
        # Extraer información adicional
        keywords = self.extract_keywords(text)
        topic = self.identify_topic(text, keywords)
        
        return {
            'processed_text': text,
            'keywords': keywords,
            'topic': topic
        }

    def train_model(self):
        """Entrena el modelo del chatbot"""
        print("Cargando datos de entrenamiento...")
        
        # Cargar datos
        with open('data.json', 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        # Procesar datos
        processed_data = []
        for item in training_data:
            proc_result = self.preprocess_text(item['input'])
            processed_data.append({
                'text': proc_result['processed_text'],
                'label': item['label'],
                'topic': proc_result['topic'],
                'keywords': proc_result['keywords']
            })

        # Actualizar topic_data
        for item in processed_data:
            if item['topic'] and item['label']:
                topic = item['topic']
                if topic not in self.topic_data['topics']:
                    self.topic_data['topics'][topic] = {
                        'keywords': [],
                        'labels': set()
                    }
                self.topic_data['topics'][topic]['labels'].add(item['label'])
                self.topic_data['topics'][topic]['keywords'].extend(item['keywords'])

        # Preparar datos para entrenamiento
        texts = [item['text'] for item in processed_data]
        labels = [item['label'] for item in processed_data]
        
        # Tokenización
        print("Realizando tokenización...")
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Preparar labels
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        categorical_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        
        # Split de datos
        X_train, X_val, y_train, y_val = train_test_split(
            padded, categorical_labels,
            test_size=self.validation_split,
            random_state=42,
            stratify=labels
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        print("Construyendo modelo...")
        self.model = self.build_model(num_classes)
        
        print("Iniciando entrenamiento...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Guardar modelo y datos
        print("Guardando modelo y datos...")
        self.save_model()
        
        return history

    def get_response(self, text):
        """Genera una respuesta para el texto de entrada"""
        # Preprocesar texto
        proc_result = self.preprocess_text(text)
        processed_text = proc_result['processed_text']
        current_topic = proc_result['topic']
        
        # Convertir a secuencia
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
        # Predicción
        prediction = self.model.predict(padded, verbose=0)[0]
        top_3_classes = np.argsort(prediction)[-3:][::-1]
        
        # Seleccionar mejor respuesta
        best_response = None
        best_confidence = 0
        
        for pred_class in top_3_classes:
            confidence = float(prediction[pred_class])
            
            if str(pred_class) in self.responses:
                # Ajustar confianza basada en tópico
                if current_topic:
                    topic_labels = self.topic_data['topics'].get(current_topic, {}).get('labels', set())
                    if str(pred_class) in topic_labels:
                        confidence += 0.2
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_response = random.choice(self.responses[str(pred_class)])
        
        if not best_response:
            best_response = "Lo siento, no estoy seguro de cómo responder a eso."
            best_confidence = 0.0
        
        # Actualizar historial
        self.conversation_history.append({
            'input': text,
            'processed_text': processed_text,
            'topic': current_topic,
            'keywords': proc_result['keywords'],
            'response': best_response,
            'confidence': best_confidence
        })
        
        return best_response, best_confidence

def main():
    """Función principal para ejecutar el chatbot"""
    print("Iniciando chatbot...")
    chatbot = TopicAwareChatbot()
    
    # Cargar o entrenar modelo
    if chatbot.load_model():
        print("Modelo cargado exitosamente")
    else:
        print("Entrenando nuevo modelo...")
        chatbot.train_model()
    
    print("\nListo para chatear! (escribe 'salir' para terminar)")
    
    while True:
        text = input("\nTú: ")
        
        if text.lower() == 'salir':
            break
            
        response, confidence = chatbot.get_response(text)
        print(f"Bot: {response}")
        print(f"(Confianza: {confidence:.2f})")

if __name__ == "__main__":
    main()