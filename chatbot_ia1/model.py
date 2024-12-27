import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D, Dropout, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import deque
import random
from sklearn.model_selection import train_test_split
import os
import nltk
from nltk.tokenize import word_tokenize

@dataclass
class Topic:
    id: str
    primary_keywords: List[str] 
    keywords: List[str]
    labels: List[str]
    subtopics: List[str]
    language: str

class EnhancedTopicAwareChatbot:
    def __init__(self):
        # Configuración básica
        self.vocab_size = 2000
        self.max_length = 20
        self.embedding_dim = 128
        self.memory_size = 10
        self.topic_threshold = 0.6
        self.batch_size = 32
        self.epochs = 150
        self.validation_split = 0.2

        # Palabras clave para detección de idioma
        self.english_words = set(['is', 'are', 'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'you', 'he', 'with', 'on', 'do', 'say', 'this', 'they', 'at', 'but', 'we', 'his', 'from', 'that', 'not', 'by', 'she', 'or', 'as', 'what', 'go', 'their', 'can', 'who', 'get', 'if', 'would', 'her', 'all', 'my', 'make', 'about', 'know', 'will', 'as', 'up', 'one', 'time', 'there', 'year', 'so', 'think', 'when', 'which', 'them', 'some', 'me', 'people', 'take', 'out', 'into', 'just', 'see', 'him', 'your', 'come', 'could', 'now', 'than', 'like', 'other', 'how', 'then', 'its', 'our', 'two', 'more', 'these', 'want', 'way', 'look', 'first', 'also', 'new', 'because', 'day', 'more', 'use', 'no', 'man', 'find', 'here', 'thing', 'give', 'many', 'well', 'what'])

        # Inicializar componentes
        self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased')
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = None
        self.conversation_history = deque(maxlen=self.memory_size)

        # Estructura para temas
        self.topics: Dict[str, Topic] = {}
        self.topic_embeddings: Dict[str, np.ndarray] = {}
        self.language_specific_topics: Dict[str, List[str]] = {
            'es': [],
            'en': []
        }

        # Cargar datos
        self.responses = self.load_responses()
        self.initialize_topics()

        # Inicializar NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def detect_language(self, text: str) -> str:
        """Simple language detection based on common words"""
        words = set(text.lower().split())
        english_count = len(words.intersection(self.english_words))
        
        return 'en' if english_count / max(len(words), 1) > 0.2 else 'es'

    def load_responses(self) -> Dict:
        try:
            with open('response.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error al cargar respuestas: {str(e)}")
            # Respuestas por defecto en caso de error
            return {
                "0": ["¡Hola! ¿En qué puedo ayudarte?"],
                "5": ["Lo siento, no entendí bien tu pregunta."]
            }

    def initialize_topics(self):
        try:
            with open('topic_data.json', 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            for topic_id, topic_data in raw_data['topics'].items():
                # Detectar idioma basado en keywords
                sample_text = ' '.join(topic_data['keywords'][:3])
                language = self.detect_language(sample_text)

                # Crear objeto Topic
                topic = Topic(
                    id=topic_id,
                    primary_keywords=topic_data.get('primary_keywords', []),
                    keywords=topic_data['keywords'],
                    labels=topic_data.get('labels', []),
                    subtopics=topic_data.get('subtopics', []),
                    language=language
                )

                # Almacenar tema
                self.topics[topic_id] = topic
                self.language_specific_topics[language].append(topic_id)

                # Crear embedding para el tema
                topic_text = ' '.join(topic_data['keywords'])
                self.topic_embeddings[topic_id] = self.sentence_model.encode(topic_text)

        except Exception as e:
            print(f"Error al inicializar temas: {str(e)}")
            self.topics = {}
            self.topic_embeddings = {}


    def save_model(self):
        """Guarda el modelo y el tokenizer"""
        try:
            # Guardar modelo
            self.model.save('chatbot_model.h5')
            
            # Guardar tokenizer
            tokenizer_json = self.tokenizer.to_json()
            with open('tokenizer.json', 'w', encoding='utf-8') as f:
                f.write(tokenizer_json)
            
            return True
        except Exception as e:
            print(f"Error al guardar modelo o tokenizer: {str(e)}")
            return False

    def load_model(self):
        """Carga el modelo desde el archivo h5"""
        try:
            if os.path.exists('chatbot_model.h5'):
                self.model = tf.keras.models.load_model('chatbot_model.h5')
                print("Cargando tokenizer...")
                with open('tokenizer.json', 'r', encoding='utf-8') as f:
                    tokenizer_json = f.read()
                    self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
                    return True
        except Exception as e:
            print(f"Error al cargar modelo: {str(e)}")
            return False

    def build_model(self, num_classes: int):
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            Conv1D(64, 3, activation='relu', padding='same'),
            Conv1D(64, 5, activation='relu', padding='same'),
            GlobalMaxPooling1D(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def get_topic_similarity(self, text: str, topic_id: str) -> float:
        text_embedding = self.sentence_model.encode(text)
        topic_embedding = self.topic_embeddings[topic_id]
        return float(cosine_similarity([text_embedding], [topic_embedding])[0][0])

    def identify_topic(self, text: str) -> Tuple[Optional[str], float]:
        """Identifica el tema más relevante para el texto dado, priorizando primary_keywords"""
        language = self.detect_language(text)
        text = text.lower().strip()
        words = set(text.split())

        # Priorizar la coincidencia con primary_keywords
        for topic_id, topic in self.topics.items():
            primary_keywords = topic.primary_keywords
            if any(pk in words for pk in primary_keywords):
                return topic_id, 1.0  # Máxima confianza para coincidencias con primary_keywords

        # Si no hay coincidencia con primary_keywords, buscar en keywords
        for topic_id, topic in self.topics.items():
            topic_keywords = set(topic.keywords)
            if any(keyword in words for keyword in topic_keywords):
                return topic_id, 0.7  # Confianza moderada para coincidencias en keywords

        # Si no hay coincidencia directa, usar embeddings
        relevant_topics = self.language_specific_topics[language]
        if not relevant_topics:
            return None, 0.0

        text_embedding = self.sentence_model.encode(text)
        similarities = {
            topic_id: cosine_similarity([text_embedding], [self.topic_embeddings[topic_id]])[0][0]
            for topic_id in relevant_topics
        }

        best_topic = max(similarities.items(), key=lambda x: x[1])
        return best_topic if best_topic[1] >= self.topic_threshold else (None, 0.0)


    def preprocess_text(self, text: str) -> Dict:
        """Preprocesamiento mejorado del texto"""
        # Normalización básica
        text = text.lower().strip()
        
        # Detectar idioma
        language = self.detect_language(text)
        
        # Normalizar según idioma
        if language == 'es':
            chars_to_replace = {
                'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                'ü': 'u', 'ñ': 'n', '¿': '', '¡': ''
            }
            for old, new in chars_to_replace.items():
                text = text.replace(old, new)

        # Extraer keywords
        words = word_tokenize(text)
        keywords = [w for w in words if len(w) > 2 and not w.isdigit()]
        
        # Identificar tema con el nuevo método
        topic, confidence = self.identify_topic(text)
        
        # Añadir contexto de la conversación
        if self.conversation_history:
            last_exchange = self.conversation_history[-1]
            last_topic = last_exchange.get('topic')
            if last_topic and last_topic in self.topics:
                # Verificar si el tema actual está relacionado con el anterior
                if last_topic in self.topics and topic in self.topics[last_topic].subtopics:
                    confidence += 0.1
        
        return {
            'processed_text': text,
            'language': language,
            'topic': topic,
            'confidence': confidence,
            'keywords': keywords
        }

    def train_model(self):
        print("Cargando datos de entrenamiento...")
        
        with open('data.json', 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        processed_data = []
        for item in training_data:
            proc_result = self.preprocess_text(item['input'])
            processed_data.append({
                'text': proc_result['processed_text'],
                'label': item['label'],
                'language': proc_result['language'],
                'keywords': proc_result['keywords']
            })

        texts = [item['text'] for item in processed_data]
        labels = [item['label'] for item in processed_data]
        
        print("Tokenizando textos...")
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        categorical_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        
        X_train, X_val, y_train, y_val = train_test_split(
            padded, categorical_labels,
            test_size=self.validation_split,
            random_state=42,
            stratify=labels
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        ]
        
        print("Construyendo y entrenando modelo...")
        self.model = self.build_model(num_classes)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Guardando modelo...")
        self.save_model()
        
        return history

    def get_response(self, text: str) -> Tuple[str, float]:
        """Genera una respuesta mejorada"""
        proc_result = self.preprocess_text(text)
        
        # Si no hay tema identificado, intentar usar el contexto
        if not proc_result['topic'] and self.conversation_history:
            last_exchange = self.conversation_history[-1]
            last_topic = last_exchange.get('topic')
            if last_topic:
                proc_result['topic'] = last_topic
                proc_result['confidence'] = 0.3
        
        if not proc_result['topic']:
            default_responses = {
                'es': "Lo siento, no estoy seguro de cómo responder a eso.",
                'en': "I'm sorry, I'm not sure how to respond to that."
            }
            return default_responses[proc_result['language']], 0.0
        
        sequence = self.tokenizer.texts_to_sequences([proc_result['processed_text']])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
        prediction = self.model.predict(padded, verbose=0)[0]
        top_3_classes = np.argsort(prediction)[-3:][::-1]
        
        best_response = None
        best_confidence = 0
        
        for pred_class in top_3_classes:
            confidence = float(prediction[pred_class])
            
            if str(pred_class) in self.responses:
                # Ajustar confianza basada en tema y contexto
                if proc_result['topic']:
                    topic = self.topics[proc_result['topic']]
                    if str(pred_class) in topic.labels:
                        confidence += 0.3
                    if topic.language == proc_result['language']:
                        confidence += 0.1
                        
                # Usar contexto de la conversación
                if self.conversation_history:
                    last_exchange = self.conversation_history[-1]
                    if str(pred_class) == str(last_exchange.get('label')):
                        confidence += 0.1
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    responses = self.responses[str(pred_class)]
                    best_response = random.choice(responses)
        
        if not best_response:
            default_responses = {
                'es': "Lo siento, no estoy seguro de cómo responder a eso.",
                'en': "I'm sorry, I'm not sure how to respond to that."
            }
            best_response = default_responses[proc_result['language']]
            best_confidence = 0.0
        
        # Actualizar historial
        self.conversation_history.append({
            'input': text,
            'processed_text': proc_result['processed_text'],
            'language': proc_result['language'],
            'topic': proc_result['topic'],
            'response': best_response,
            'confidence': best_confidence,
            'label': str(np.argmax(prediction))
        })
        
        return best_response, best_confidence

def main():
    print("Iniciando chatbot mejorado...")
    chatbot = EnhancedTopicAwareChatbot()
    
    if chatbot.load_model():
        print("Modelo cargado exitosamente")
    else:
        print("Entrenando nuevo modelo...")
        chatbot.train_model()
    
    print("\n¡Listo para chatear! (escribe 'salir' para terminar)")
    print("El chatbot puede responder en español e inglés\n")
    
    while True:
        text = input("\nTú: ")
        
        if text.lower() in ['salir', 'exit', 'quit']:
            print("\n¡Hasta luego! Gracias por chatear.")
            break
            
        response, confidence = chatbot.get_response(text)
        print(f"\nBot: {response}")
        if confidence > 0:
            print(f"(Confianza: {confidence:.2f})")

if __name__ == "__main__":
    main()