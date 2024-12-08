import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
import tensorflow as tf
from collections import deque

class EnhancedChatbot:
    def __init__(self):
        self.vocab_size = 2000
        self.max_length = 15
        self.embedding_dim = 32
        
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = None
        self.responses = None
        self.conversation_memory = deque(maxlen=5)
        self.context = {}
    
    def preprocess_text(self, text):
        return text.lower().strip()
    
    def train(self):
        # Cargar datos
        with open('data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open('response.json', 'r', encoding='utf-8') as f:
            self.responses = json.load(f)
        
        # Preparar datos
        texts = [self.preprocess_text(item['input']) for item in data]
        labels = [item['label'] for item in data]
        
        # Tokenización
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Convertir labels
        num_classes = max(labels) + 1
        categorical_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        
        # Modelo simplificado pero mejorado
        self.model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compilar
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenar
        history = self.model.fit(
            padded,
            categorical_labels,
            epochs=100,
            batch_size=32,
            validation_split=0.2
        )
        
        return history
    
    def get_response(self, text):
        processed_text = self.preprocess_text(text)
        self.conversation_memory.append(processed_text)
        
        test_seq = self.tokenizer.texts_to_sequences([processed_text])
        test_padded = pad_sequences(test_seq, maxlen=self.max_length, padding='post')
        
        prediction = self.model.predict(test_padded, verbose=0)[0]
        label = np.argmax(prediction)
        confidence = float(prediction[label])
        
        if confidence > 0.3 and str(label) in self.responses:
            possible_responses = self.responses[str(label)]
            
            if len(self.conversation_memory) > 1:
                prev_message = self.conversation_memory[-2]
                response = self.select_contextual_response(possible_responses, prev_message)
            else:
                response = np.random.choice(possible_responses)
            
            self.update_context(label)
        else:
            response = "No estoy seguro. ¿Podrías reformular?"
            confidence = 0.3
            
        return response, confidence
    
    def select_contextual_response(self, responses, prev_message):
        return np.random.choice(responses)
    
    def update_context(self, label):
        if str(label) not in self.context:
            self.context[str(label)] = 1
        else:
            self.context[str(label)] += 1

def main():
    print("Iniciando chatbot mejorado...")
    bot = EnhancedChatbot()
    
    print("\nEntrenando modelo...")
    history = bot.train()
    
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"\nPrecisión final: {final_accuracy:.4f}")
    print(f"Precisión de validación final: {final_val_accuracy:.4f}")
    
    print("\nListo para chatear! (escribe 'salir' para terminar)\n")
    
    while True:
        text = input("Tú: ")
        
        if text.lower() == 'salir':
            break
            
        response, confidence = bot.get_response(text)
        print(f"Bot: {response}")
        print(f"(confianza: {confidence:.2f})")
        print()

if __name__ == "__main__":
    main()