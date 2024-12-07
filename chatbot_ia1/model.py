import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.cuda.amp import autocast, GradScaler
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from datetime import datetime
import os
import random
from torch.utils.data import Dataset, DataLoader
from collections import deque
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
import logging
from sklearn.metrics.pairwise import cosine_similarity

class ContextualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, nhead=8, num_layers=4):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layers = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.context_layer = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        x = self.pos_encoder(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = self.transformer_encoder(x)
        x = self.context_layer(x)
        return self.norm(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DynamicMemoryNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_classes=12):  # Cambiado a 12
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = ContextualEncoder(embed_dim, hidden_dim)
        
        # Memoria episódica
        self.memory_size = 1000
        self.memory_dim = hidden_dim
        self.memory = nn.Parameter(torch.randn(self.memory_size, self.memory_dim))
        self.memory_key = nn.Linear(hidden_dim, self.memory_dim)
        
        # Clasificador contextual
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Estimador de confianza mejorado
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Análisis de contexto
        self.context_analyzer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Relevancia, Novedad, Importancia
        )
    
    def forward(self, input_ids, attention_mask=None, context_state=None):
        # Embedding inicial
        x = self.embedding(input_ids)
        
        # Codificación contextual
        encoded = self.encoder(x, attention_mask)
        
        # Atención sobre la memoria
        memory_keys = self.memory_key(encoded)
        attention_weights = torch.matmul(memory_keys, self.memory.T)
        attention_weights = F.softmax(attention_weights / np.sqrt(self.memory_dim), dim=-1)
        memory_output = torch.matmul(attention_weights, self.memory)
        
        # Combinar con el estado actual
        if attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1)
            encoded = encoded * mask
            pooled = encoded.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = encoded.mean(dim=1)
        
        # Concatenar con memoria
        combined = torch.cat([pooled, memory_output.mean(dim=1)], dim=-1)
        
        # Salidas
        logits = self.classifier(combined)
        confidence = self.confidence(combined)
        context_analysis = self.context_analyzer(combined)
        
        return logits, confidence, context_analysis

class EnhancedConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=128):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        input_text = conv['input']
        
        # Procesar contexto si existe
        if 'context' in conv and conv['context']:
            context = ' | '.join(conv['context'])
            input_text = f"{context} >>> {input_text}"
        
        # Tokenización simplificada
        inputs = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(conv['label'], dtype=torch.long)
        }
    
class ContextManager:
    def __init__(self, max_history=50):
        self.history = deque(maxlen=max_history)
        self.context_embeddings = {}
        
    def add_interaction(self, input_text, response, embedding, metadata=None):
        interaction = {
            'input': input_text,
            'response': response,
            'embedding': embedding.detach().cpu().numpy(),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.history.append(interaction)
        
    def get_relevant_context(self, current_embedding, k=3):
        if not self.history:
            return []
        
        # Calcular similitudes
        embeddings = np.stack([h['embedding'] for h in self.history])
        similarities = cosine_similarity(current_embedding.detach().cpu().numpy().reshape(1, -1), embeddings)[0]
        
        # Obtener los k más relevantes
        most_relevant_idx = np.argsort(similarities)[-k:]
        
        return [list(self.history)[i] for i in most_relevant_idx]
    
class ImprovedTrainer:
    def __init__(self, model_dir='model', min_confidence=0.6):
      self.model_dir = model_dir
      self.min_confidence = min_confidence
      
      # Configuración GPU
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      if torch.cuda.is_available():
          torch.backends.cudnn.benchmark = True
      
      # Componentes principales
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
      self.model = DynamicMemoryNetwork(self.tokenizer.vocab_size)
      self.model.to(self.device)
      
      # Optimización
      self.optimizer = torch.optim.AdamW(
          self.model.parameters(),
          lr=3e-4,
          weight_decay=0.01,
          betas=(0.9, 0.999)
      )
      
      # Learning rate scheduling
      self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
          self.optimizer,
          max_lr=3e-4,
          epochs=50,
          steps_per_epoch=100
      )
      
      # Gestión de contexto y memoria
      self.context_manager = ContextManager()
      
      # Configurar mixed precision solo si hay GPU disponible
      if torch.cuda.is_available():
          self.scaler = torch.amp.GradScaler()
      else:
          self.scaler = None
          print("CUDA no disponible. Mixed precision desactivada.")
      self.responses = self.load_default_responses()

      self._initialize_system()
      checkpoint_path = os.path.join(self.model_dir, 'model.pth')
      if not os.path.exists(checkpoint_path):
          self.train_initial_model()
    
    def train_iteration(self, batch):
      self.model.train()
      self.optimizer.zero_grad()
      
      input_ids = batch['input_ids'].to(self.device)
      attention_mask = batch['attention_mask'].to(self.device)
      labels = batch['label'].to(self.device)
      
      # Usar mixed precision solo si hay GPU disponible
      if self.scaler:
          with torch.cuda.amp.autocast():
              logits, confidence, context_analysis = self.model(input_ids, attention_mask)
              classification_loss = F.cross_entropy(logits, labels)
              confidence_loss = F.mse_loss(confidence.squeeze(), torch.ones_like(confidence.squeeze()) * 0.9)
              total_loss = classification_loss + 0.1 * confidence_loss
          
          self.scaler.scale(total_loss).backward()
          self.scaler.step(self.optimizer)
          self.scaler.update()
      else:
          # Entrenamiento normal sin mixed precision
          logits, confidence, context_analysis = self.model(input_ids, attention_mask)
          classification_loss = F.cross_entropy(logits, labels)
          confidence_loss = F.mse_loss(confidence.squeeze(), torch.ones_like(confidence.squeeze()) * 0.9)
          total_loss = classification_loss + 0.1 * confidence_loss
          
          total_loss.backward()
          self.optimizer.step()
      
      return total_loss.item()

    def _initialize_system(self):
        """Inicializar el sistema y cargar checkpoint si existe"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        loaded = self.load_checkpoint()
        if not loaded:
            print("No se encontró checkpoint existente. Iniciando con modelo nuevo.")

    def process_conversation(self, text, context=None):
        self.model.eval()
        with torch.no_grad():
            # Preparar input
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Forward pass
            logits, confidence, context_analysis = self.model(
                input_ids, 
                attention_mask
            )
            
            # Procesar resultados
            probs = F.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            conf_value = confidence.item()
            
            # Análisis de contexto
            relevance, novelty, importance = context_analysis[0].tolist()
            
            return {
                'prediction': prediction,
                'confidence': conf_value,
                'context_analysis': {
                    'relevance': relevance,
                    'novelty': novelty,
                    'importance': importance
                }
            }

    def save_checkpoint(self):
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'context_manager': self.context_manager.history,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, os.path.join(self.model_dir, 'model.pth'))
    def load_checkpoint(self):
      """Cargar checkpoint si existe"""
      try:
          checkpoint_path = os.path.join(self.model_dir, 'model.pth')
          if os.path.exists(checkpoint_path):
              print("Cargando checkpoint existente...")
              checkpoint = torch.load(checkpoint_path, map_location=self.device)
              
              # Cargar estado del modelo
              self.model.load_state_dict(checkpoint['model_state'])
              
              # Cargar estado del optimizador
              if 'optimizer_state' in checkpoint:
                  self.optimizer.load_state_dict(checkpoint['optimizer_state'])
              
              # Cargar estado del scheduler
              if 'scheduler_state' in checkpoint:
                  self.scheduler.load_state_dict(checkpoint['scheduler_state'])
              
              # Cargar historial de contexto
              if 'context_manager' in checkpoint:
                  self.context_manager.history = deque(
                      checkpoint['context_manager'],
                      maxlen=self.context_manager.history.maxlen
                  )
              
              print("Checkpoint cargado correctamente")
              return True
          return False
      except Exception as e:
          print(f"Error cargando checkpoint: {e}")
          return False
    def load_default_responses(self):
      """Cargar respuestas predeterminadas con soporte extendido"""
      # Respuestas predeterminadas
      default_responses = {
          "0": [  # saludos
              "¡Hola! ¿En qué puedo ayudarte?",
              "¡Buen día! ¿Cómo estás?",
              "¡Hola! Me alegro de verte.",
              "¡Saludos! ¿Cómo puedo ayudarte hoy?",
              "¡Hola! Espero que estés teniendo un excelente día."
          ],
          "1": [  # despedidas
              "¡Hasta luego! Que tengas un excelente día.",
              "¡Adiós! Espero verte pronto.",
              "¡Nos vemos! Ha sido un placer charlar contigo.",
              "¡Que estés bien! Hasta la próxima."
          ],
          "2": [  # agradecimientos
              "¡De nada! Me alegro de poder ayudar.",
              "¡Es un placer! Estoy aquí para lo que necesites.",
              "¡No hay de qué! Tu satisfacción es importante.",
              "Siempre es un gusto poder ayudarte."
          ],
          "3": [  # estado
              "¡Muy bien, gracias por preguntar! ¿Y tú qué tal?",
              "¡Excelente! ¿Cómo va tu día?",
              "¡Todo bien! ¿Qué tal te encuentras tú?",
              "Estoy aquí para ayudarte. ¿Qué necesitas?"
          ],
          "4": [  # ayuda
              "¿En qué puedo ayudarte?",
              "Cuéntame qué necesitas.",
              "Estoy aquí para asistirte. ¿Qué necesitas?",
              "Dime en qué te puedo apoyar, estoy para ayudarte."
          ],
          "5": [  # opiniones/consultas
              "Es una buena pregunta. ¿Qué piensas tú al respecto?",
              "Creo que podríamos analizarlo más a fondo.",
              "Es un tema interesante, dime más para profundizar.",
              "¿Te gustaría que investigue algo más sobre esto?"
          ],
          "6": [  # preguntas sobre el bot
              "Soy un chatbot diseñado para ayudarte. ¿Cómo puedo asistirte?",
              "Puedo responder preguntas, ofrecer información y mucho más.",
              "¡Sí! Estoy aquí para lo que necesites. ¿Qué deseas saber?",
              "Soy un modelo de IA. ¿Qué más te gustaría saber sobre mí?"
          ],
          "7": [  # preguntas técnicas
              "Eso depende del contexto. ¿Puedes darme más detalles?",
              "Puedo explicártelo. ¿Quieres un resumen o algo más específico?",
              "Es un concepto fascinante. Déjame darte una explicación.",
              "Aquí tienes una introducción: ¿te gustaría que profundice?"
          ],
          "8": [  # humor
              "¿Quieres escuchar un chiste? Aquí va: ¿Por qué los programadores odian la naturaleza? ¡Porque tiene demasiados bugs!",
              "¿Sabías que los chats como yo nunca tienen sueño? Porque estamos en modo 'awake' todo el tiempo.",
              "Dicen que el código perfecto no existe... pero ¡yo me acerco bastante!",
              "¡Espero que te haya hecho sonreír! Si quieres otro chiste, solo dilo."
          ],
          "9": [  # consultas sobre tiempo/fecha
              "Hoy es un gran día. ¿Hay algo en específico que quieras saber?",
              "Puedo ayudarte a verificar la fecha o la hora. Solo pregunta.",
              "Dime qué necesitas saber sobre el día y te ayudo con gusto.",
              "El tiempo es oro, ¡y aquí estoy para ayudarte a aprovecharlo!"
          ],
          "10": [  # pronóstico del tiempo
              "Según el pronóstico, parece que el día estará despejado.",
              "Parece que podría llover más tarde. ¿Tienes un paraguas a mano?",
              "La temperatura hoy es agradable. ¿Planeas salir?",
              "¡Hoy hace calor! Recuerda mantenerte hidratado."
          ],
          "11": [  # sugerencias/ideas
              "¿Qué tal un poco de lectura o aprender algo nuevo hoy?",
              "Puedes relajarte viendo una película o dando un paseo.",
              "Siempre es buen momento para aprender algo nuevo. ¿Te interesa la programación o algún otro tema?",
              "¿Qué te parece organizar tus ideas y trabajar en un proyecto personal?"
          ]
      }

      # Ruta del archivo JSON con respuestas personalizadas
      response_path = os.path.join(self.model_dir, 'responses.json')

      # Intentar cargar respuestas personalizadas desde un archivo JSON
      if os.path.exists(response_path):
          try:
              with open(response_path, 'r', encoding='utf-8') as f:
                  loaded_responses = json.load(f)
                  # Actualizar las respuestas predeterminadas con las personalizadas
                  default_responses.update(loaded_responses)
                  print(f"Respuestas personalizadas cargadas desde {response_path}.")
          except Exception as e:
              print(f"Error al cargar respuestas personalizadas: {e}")

      # Retornar respuestas predeterminadas (y personalizadas si se cargaron correctamente)
      return default_responses


    def initialize_training_data(self, json_path="data.json"):
      """Carga datos de entrenamiento desde un archivo JSON."""
      if not os.path.exists(json_path):
          raise FileNotFoundError(f"No se encontró el archivo: {json_path}")
      with open(json_path, "r", encoding="utf-8") as f:
          training_data = json.load(f)
      return training_data

    def train_initial_model(self, json_path="data.json", epochs=100, batch_size=4):
      """Entrenar modelo con datos cargados desde un archivo JSON."""
      print("\nIniciando entrenamiento con datos desde JSON...")
      
      # Cargar datos desde el archivo JSON
      training_data = self.initialize_training_data(json_path)
      dataset = EnhancedConversationDataset(training_data, self.tokenizer)
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
      
      self.model.train()
      
      for epoch in range(epochs):
          total_loss = 0
          correct = 0
          total = 0
          
          progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
          for batch in progress_bar:
              input_ids = batch['input_ids'].to(self.device)
              attention_mask = batch['attention_mask'].to(self.device)
              labels = batch['label'].to(self.device)
              
              self.optimizer.zero_grad()
              
              # Forward pass
              logits, confidence, _ = self.model(input_ids, attention_mask)
              
              # Calcular pérdida
              classification_loss = F.cross_entropy(logits, labels)
              confidence_loss = F.mse_loss(confidence.squeeze(), torch.ones_like(confidence.squeeze()) * 0.9)
              total_loss_batch = classification_loss + 0.1 * confidence_loss
              
              # Backward pass
              total_loss_batch.backward()
              self.optimizer.step()
              
              # Métricas
              total_loss += total_loss_batch.item()
              _, predicted = torch.max(logits, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
              
              # Actualizar barra de progreso
              progress_bar.set_postfix({
                  'loss': f'{classification_loss.item():.4f}',
                  'acc': f'{100 * correct / total:.2f}%'
              })
          
          print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")
      
      print("\nEntrenamiento completado.")
      self.save_checkpoint()

    def get_response(self, prediction, confidence, context_analysis):
        """Obtener respuesta basada en predicción y contexto"""
        if confidence < self.min_confidence:
            if context_analysis['novelty'] > 0.5:
                return "Interesante. ¿Podrías darme más detalles al respecto?"
            else:
                return "No estoy seguro de entender. ¿Podrías reformular tu mensaje?"
        
        # Obtener respuesta del diccionario de respuestas
        responses = self.responses.get(str(prediction), ["Entiendo. ¿Podrías elaborar más?"])
        
        # Seleccionar respuesta considerando el contexto
        if context_analysis['relevance'] > 0.3:
            # Priorizar respuestas más elaboradas para contextos relevantes
            longer_responses = [r for r in responses if len(r.split()) > 5]
            if longer_responses:
                responses = longer_responses
        
        return random.choice(responses)
    
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar caracteres especiales
    text = re.sub(r'\s+', ' ', text)  # Reemplazar múltiples espacios por uno
    return text.strip()

def main():
    trainer = ImprovedTrainer()
    
    print("\n=== Sistema de Chat Mejorado con Memoria Dinámica ===")
    print(f"Dispositivo: {trainer.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    while True:
        try:
            user_input = input("\nTú: ").strip()
            
            if user_input.lower() == 'salir':
                trainer.save_checkpoint()
                break
            
            # Procesar entrada
            result = trainer.process_conversation(user_input)
            
            # Generar respuesta
            response = trainer.get_response(
                result['prediction'],
                result['confidence'],
                result['context_analysis']
            )
            
            print(f"Bot: {response}")
            cleaned_input = clean_text(user_input)
            # Guardar interacción en el contexto
            trainer.context_manager.add_interaction(
                cleaned_input,
                response,
                trainer.model.encoder(
                    trainer.model.embedding(
                        trainer.tokenizer(
                            cleaned_input,
                            return_tensors='pt',
                            truncation=True,
                            padding=True,
                            max_length=128
                        )['input_ids'].to(trainer.device)
                    )
                ).mean(dim=1)
            )
            
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()