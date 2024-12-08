import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
import tensorflow as tf
from collections import deque
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

class EnhancedConversationalChatbot:
    def __init__(self):
        self.vocab_size = 2000
        self.max_length = 20
        self.embedding_dim = 64
        self.memory_size = 5

        self.model_dir = 'saved_model'
        self.keras_model_path = os.path.join('model', 'chatbot_model.h5')
        self.tokenizer_path = os.path.join('public', 'tokenizer.json')
        self.responses_path = 'response.json'  # Añadir path de respuestas
        
        # Inicialización
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = None
        self.responses = None
        self.conversation_history = deque(maxlen=self.memory_size)

        # Parámetros de entrenamiento
        self.batch_size = 16
        self.epochs = 50
        self.validation_split = 0.2
        
        # Cargar respuestas al inicializar
        self.load_responses()
        
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
        """Guarda el modelo y el tokenizer"""
        try:
            # Crear directorios si no existen
            os.makedirs(os.path.dirname(self.keras_model_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.tokenizer_path), exist_ok=True)

            # 1. Guardar modelo en formato H5
            print("Guardando modelo en formato H5...")
            self.model.save(self.keras_model_path, save_format='h5')
            print(f"Modelo guardado en: {self.keras_model_path}")
            
            # 2. Guardar tokenizer
            print("Guardando tokenizer...")
            tokenizer_json = self.tokenizer.to_json()
            with open(self.tokenizer_path, 'w', encoding='utf-8') as f:
                f.write(tokenizer_json)
            print(f"Tokenizer guardado en: {self.tokenizer_path}")
            
            # 3. Guardar configuración del modelo para referencia
            model_config = {
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'embedding_dim': self.embedding_dim
            }
            config_path = os.path.join('public', 'model_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(model_config, f, indent=2)
            print(f"Configuración guardada en: {config_path}")
            
            return True
            
        except Exception as e:
            print(f"Error al guardar el modelo: {str(e)}")
            return False
            
    def load_model(self):
        """Carga el modelo y el tokenizer guardados"""
        try:
            if os.path.exists(self.keras_model_path) and os.path.exists(self.tokenizer_path):
                print("Cargando modelo guardado...")
                # Cargar modelo
                self.model = tf.keras.models.load_model(self.keras_model_path)
                
                print("Cargando tokenizer...")
                # Cargar tokenizer
                with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
                    tokenizer_json = f.read()
                    self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
                
                # Asegurar que las respuestas estén cargadas
                if self.responses is None:
                    self.load_responses()
                
                print("Modelo y tokenizer cargados exitosamente!")
                return True
            return False
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            return False

    def build_enhanced_model(self, num_classes):
        """Construye un modelo más simple pero efectivo"""
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(32, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(16)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        # Optimizador con learning rate más bajo
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def preprocess_text(self, text):
        """Preprocesamiento mejorado"""
        # Normalización básica
        text = text.lower().strip()
        
        # Normalizar caracteres especiales
        text = text.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        
        # Eliminar puntuación excepto signos de pregunta
        text = ''.join([char for char in text if char.isalnum() or char.isspace() or char in '¿?'])
        
        return text

    def train_enhanced_model(self):
        """Entrena el modelo con mejor manejo de datos"""
        print("Cargando y preparando datos...")
        
        # Cargar datos
        with open('data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open('response.json', 'r', encoding='utf-8') as f:
            self.responses = json.load(f)
        
        # Preparar datos
        texts = [self.preprocess_text(item['input']) for item in data]
        labels = [item['label'] for item in data]
        
        # Verificar distribución de clases
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        print(f"\nNúmero de clases: {num_classes}")
        for label in unique_labels:
            count = labels.count(label)
            print(f"Clase {label}: {count} ejemplos")
        
        # Tokenización
        print("\nRealizando tokenización...")
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Convertir etiquetas
        categorical_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        
        # Split de datos
        X_train, X_val, y_train, y_val = train_test_split(
            padded, categorical_labels, 
            test_size=self.validation_split, 
            random_state=42,
            stratify=labels  # Asegura distribución balanceada
        )
        
        print(f"\nDatos de entrenamiento: {X_train.shape[0]} ejemplos")
        print(f"Datos de validación: {X_val.shape[0]} ejemplos")
        
        # Callbacks para mejor entrenamiento
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=0.0001
            )
        ]
        
        # Construir modelo
        print("\nConstruyendo modelo...")
        self.model = self.build_enhanced_model(num_classes)
        
        # Entrenar modelo
        print("\nIniciando entrenamiento...")
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar modelo
        print("\nEvaluando modelo...")
        test_loss, test_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"\nPrecisión final en validación: {test_accuracy:.4f}")
        print(f"Pérdida final en validación: {test_loss:.4f}")
        print("\nGuardando modelo y tokenizer...")
        if self.save_model():
            print("¡Guardado completado exitosamente!")
        else:
            print("Hubo problemas al guardar el modelo.")
        
        return history

    def get_response(self, text):
        """Genera una respuesta usando el modelo entrenado"""
        # Preprocesar entrada
        processed_text = self.preprocess_text(text)
        
        # Convertir a secuencia
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
        # Predicción
        prediction = self.model.predict(padded, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])
        
        # Obtener respuesta
        if str(predicted_class) in self.responses:
            response = random.choice(self.responses[str(predicted_class)])
        else:
            response = "Lo siento, no estoy seguro de cómo responder a eso."
            confidence = 0.0
        
        # Actualizar historial
        self.conversation_history.append({
            'input': text,
            'response': response,
            'confidence': confidence
        })
        
        return response, confidence

def main():
    print("Iniciando chatbot conversacional mejorado...")
    bot = EnhancedConversationalChatbot()
    
    # Intentar cargar modelo existente
    if bot.load_model():
        print("\nModelo cargado exitosamente!")
    else:
        print("\nEntrenando nuevo modelo...")
        history = bot.train_enhanced_model()
    
    print("\n¡Listo para chatear! (escribe 'salir' para terminar)\n")
    
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