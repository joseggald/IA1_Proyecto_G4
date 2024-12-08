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
from collections import deque, defaultdict
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
class SemanticAnalyzer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.semantic_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Análisis de significados múltiples
        self.meaning_analyzer = nn.ModuleList([
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
            for _ in range(4)  # Diferentes aspectos del significado
        ])
        
    def forward(self, x):
        semantic_features = self.semantic_encoder(x)
        meanings = [analyzer(semantic_features) for analyzer in self.meaning_analyzer]
        return torch.cat(meanings, dim=-1)

class ContextHierarchyAnalyzer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(3)  # Niveles jerárquicos de contexto
        ])
        
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        context_levels = [level(x) for level in self.levels]
        return self.context_fusion(torch.cat(context_levels, dim=-1))
class HierarchicalMemory(nn.Module):
    def __init__(self, hidden_dim, memory_size=128):  # Cambiado a 128 por defecto
        super().__init__()
        self.memory = nn.Parameter(torch.randn(1, memory_size, hidden_dim))
        self.memory_size = memory_size
        self.hidden_dim = hidden_dim
        
        # Añadir una capa de proyección
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        memory = self.memory.expand(batch_size, -1, -1)
        return self.projection(memory)
    
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        return x + self.layers(x)
class DynamicMemoryNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_classes=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = ContextualEncoder(embed_dim, hidden_dim)
        
        # Análisis semántico y contextual
        self.semantic_analyzer = SemanticAnalyzer(hidden_dim)
        self.context_analyzer = ContextHierarchyAnalyzer(hidden_dim)
        
        # Memoria jerárquica mejorada
        self.hierarchical_memories = nn.ModuleList([
            HierarchicalMemory(hidden_dim, memory_size=128)  # Ajustado a 128 para coincidir
            for _ in range(3)
        ])
        
        # Atención multi-nivel
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            for _ in range(3)
        ])
        
        # Capa de adaptación de dimensiones
        self.dimension_adapter = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Integración de contexto y significado
        self.context_integration = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # Ajustado para las nuevas dimensiones
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Estimador de confianza mejorado
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        
        # Codificación inicial
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        encoded = self.encoder(embedded, attention_mask)  # [batch_size, seq_len, hidden_dim]
        
        # Análisis semántico y contextual
        semantic_features = self.semantic_analyzer(encoded)  # [batch_size, seq_len, hidden_dim]
        contextual_features = self.context_analyzer(encoded)  # [batch_size, seq_len, hidden_dim]
        
        # Aplicar pooling para obtener representaciones de secuencia fija
        semantic_pooled = semantic_features.mean(dim=1)  # [batch_size, hidden_dim]
        contextual_pooled = contextual_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Procesamiento de memoria jerárquica
        memory_outputs = []
        for memory_module, attention in zip(self.hierarchical_memories, self.attention_layers):
            memory = memory_module(encoded)  # [batch_size, memory_size, hidden_dim]
            memory_output, _ = attention(encoded, memory, memory)  # [batch_size, seq_len, hidden_dim]
            # Aplicar pooling para obtener una representación fija
            memory_output = memory_output.mean(dim=1)  # [batch_size, hidden_dim]
            memory_outputs.append(memory_output)
        
        # Combinar salidas de memoria
        combined_memory = torch.cat(memory_outputs, dim=-1)  # [batch_size, hidden_dim * 3]
        combined_memory = self.dimension_adapter(combined_memory)  # [batch_size, hidden_dim]
        
        # Integración de características - ahora todas las tensores tienen la misma dimensionalidad
        integrated_features = self.context_integration(torch.cat([
            semantic_pooled,      # [batch_size, hidden_dim]
            contextual_pooled,    # [batch_size, hidden_dim]
            combined_memory       # [batch_size, hidden_dim]
        ], dim=-1))
        
        # Clasificación y confianza
        logits = self.classifier(integrated_features)
        confidence = self.confidence_estimator(integrated_features)
        
        # Análisis contextual
        context_analysis = {
            'semantic_richness': semantic_pooled.mean(dim=-1),
            'context_depth': contextual_pooled.mean(dim=-1),
            'memory_relevance': combined_memory.mean(dim=-1)
        }
        
        return logits, confidence, context_analysis
class ContextualLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, logits, confidence, context_analysis, labels):
        classification_loss = F.cross_entropy(logits, labels)
        
        # Pérdida de consistencia contextual
        semantic_consistency = torch.mean((context_analysis['semantic_richness'] - 
                                         context_analysis['context_depth']).abs())
        
        # Pérdida de relevancia de memoria
        memory_relevance_loss = torch.mean((1.0 - context_analysis['memory_relevance']))
        
        total_loss = classification_loss + \
                    self.alpha * semantic_consistency + \
                    self.beta * memory_relevance_loss
                    
        return total_loss
class ContextAnalyzer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.analyzer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 5)  # [relevancia, novedad, importancia, emoción, complejidad]
        )
        
    def forward(self, x):
        return self.analyzer(x)

class LongTermMemory(nn.Module):
    def __init__(self, hidden_dim, memory_size=1000):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.importance_estimator = nn.Linear(hidden_dim, 1)
        
    def update(self, encoding, context_state):
        importance = self.importance_estimator(encoding).sigmoid()
        update_mask = importance > 0.5
        self.memory[update_mask] = encoding[update_mask]
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
    
class EnhancedContextManager(ContextManager):
    def __init__(self, max_history=50, decay_factor=0.9):
        super().__init__(max_history)
        self.decay_factor = decay_factor
    
    def get_relevant_context(self, current_embedding, k=3):
        # Añadir decaimiento temporal a la relevancia
        current_time = datetime.now()
        
        def compute_decayed_similarity(interaction):
            time_diff = (current_time - datetime.fromisoformat(interaction['timestamp'])).total_seconds() / 3600  # horas
            temporal_decay = self.decay_factor ** time_diff
            
            embedding_sim = cosine_similarity(
                current_embedding.detach().cpu().numpy().reshape(1, -1), 
                interaction['embedding'].reshape(1, -1)
            )[0][0]
            
            return embedding_sim * temporal_decay
        
        # Ordenar por similitud con decaimiento temporal
        sorted_interactions = sorted(
            self.history, 
            key=compute_decayed_similarity, 
            reverse=True
        )
        
        return sorted_interactions[:k]    
class ImprovedTrainer:
    def __init__(self, model_dir='model', min_confidence=0.6):
        self.model_dir = model_dir
        self.min_confidence = min_confidence
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Inicializar componentes principales
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        # Inicializar el modelo primero
        self.model = DynamicMemoryNetwork(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=256,
            hidden_dim=512,
            num_classes=12
        )
        self.model.to(self.device)
        
        # Configuración de entrenamiento continuo
        self.buffer_size = 10000
        self.experience_buffer = deque(maxlen=self.buffer_size)
        self.warmup_steps = 100
        self.accumulation_steps = 4
        self.max_grad_norm = 1.0
        
        # Optimizador mejorado
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=2e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler modificado usando OneCycleLR
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=2e-5,
            epochs=80,
            steps_per_epoch=100,  # Ajustar según el tamaño de tu dataset
            pct_start=0.1,  # 10% del entrenamiento para warmup
            cycle_momentum=False
        )
        
        # Sistema de carga dinámica de datos
        self.data_loaders = {}
        
        # Gestión de contexto y memoria
        self.context_manager = EnhancedContextManager()
        
        # Configurar mixed precision solo si hay GPU disponible
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None
            print("CUDA no disponible. Mixed precision desactivada.")
            
        self.responses = self.load_default_responses()
        
        # Inicializar el sistema y cargar checkpoint si existe
        self._initialize_system()
        checkpoint_path = os.path.join(self.model_dir, 'model.pth')
        if not os.path.exists(checkpoint_path):
            print("No se encontró checkpoint existente. Se iniciará el entrenamiento inicial.")
            self.train_initial_model()
    
    def train_iteration(self, batch):
        """Versión mejorada del entrenamiento por iteración"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # Usar mixed precision si está disponible
        if self.scaler:
            with torch.cuda.amp.autocast():
                logits, confidence, context_analysis = self.model(input_ids, attention_mask)
                classification_loss = F.cross_entropy(logits, labels)
                confidence_loss = F.mse_loss(confidence.squeeze(), torch.ones_like(confidence.squeeze()) * 0.9)
                
                # Añadir pérdida de contexto
                context_loss = torch.mean(torch.abs(context_analysis['semantic_richness'] - 
                                                  context_analysis['context_depth']))
                
                total_loss = classification_loss + 0.1 * confidence_loss + 0.1 * context_loss
                
            # Normalizar pérdida por acumulación de gradientes
            total_loss = total_loss / self.accumulation_steps
            
            self.scaler.scale(total_loss).backward()
            
            # Aplicar gradient clipping
            if (batch['batch_idx'] + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            # Entrenamiento sin mixed precision
            logits, confidence, context_analysis = self.model(input_ids, attention_mask)
            classification_loss = F.cross_entropy(logits, labels)
            confidence_loss = F.mse_loss(confidence.squeeze(), torch.ones_like(confidence.squeeze()) * 0.9)
            context_loss = torch.mean(torch.abs(context_analysis['semantic_richness'] - 
                                              context_analysis['context_depth']))
            
            total_loss = classification_loss + 0.1 * confidence_loss + 0.1 * context_loss
            total_loss = total_loss / self.accumulation_steps
            
            total_loss.backward()
            
            if (batch['batch_idx'] + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        return {
            'loss': total_loss.item() * self.accumulation_steps,
            'classification_loss': classification_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'context_loss': context_loss.item()
        }
    
    def load_multiple_json(self, json_paths):
        """Cargar múltiples archivos JSON de entrenamiento"""
        all_data = []
        for path in json_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
                    print(f"Datos cargados de {path}: {len(data)} ejemplos")
            except Exception as e:
                print(f"Error cargando {path}: {e}")
        return all_data
    def continuous_learning(self, new_data, epochs=5):
        """Aprendizaje continuo con nuevos datos"""
        # Combinar datos nuevos con buffer de experiencia
        self.experience_buffer.extend(new_data)
        
        # Crear dataset temporal
        temp_dataset = EnhancedConversationDataset(
            list(self.experience_buffer),
            self.tokenizer
        )
        
        # Entrenar con los datos combinados
        dataloader = DataLoader(
            temp_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )
        
        for epoch in range(epochs):
            self.train_epoch(dataloader, epoch)
            
        # Guardar checkpoint
        self.save_checkpoint()

    def train_epoch(self, dataloader, epoch):
        """Entrenamiento de una época con métricas detalladas"""
        self.model.train()
        total_loss = 0
        metrics = defaultdict(float)
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            loss, batch_metrics = self.train_step(batch)
            total_loss += loss
            
            for k, v in batch_metrics.items():
                metrics[k] += v
                
        # Imprimir métricas
        metrics = {k: v / len(dataloader) for k, v in metrics.items()}
        print(f"Epoch {epoch + 1} - Loss: {total_loss / len(dataloader):.4f}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    def _initialize_system(self):
        """Inicializar el sistema y cargar checkpoint si existe"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        loaded = self.load_checkpoint()
        if not loaded:
            print("No se encontró checkpoint existente. Iniciando con modelo nuevo.")

    def process_conversation(self, text, context=None):
        """Método mejorado para procesar conversaciones con mejor manejo de errores"""
        try:
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
                conf_value = confidence.squeeze().item() if confidence.numel() > 0 else 0.0
                
                # Extraer valores del análisis de contexto
                context_values = {
                    'relevance': float(context_analysis['semantic_richness'].mean().item()),
                    'novelty': float(context_analysis['context_depth'].mean().item()),
                    'importance': float(context_analysis['memory_relevance'].mean().item())
                }
                
                return {
                    'prediction': prediction,
                    'confidence': conf_value,
                    'context_analysis': context_values
                }
                
        except Exception as e:
            logging.error(f"Error en process_conversation: {str(e)}")
            # Retornar valores por defecto en caso de error
            return {
                'prediction': 0,  # Categoría por defecto (saludos)
                'confidence': 0.0,
                'context_analysis': {
                    'relevance': 0.0,
                    'novelty': 0.0,
                    'importance': 0.0
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

    def train_initial_model(self, json_path="data.json", epochs=100, batch_size=8):
        """Versión mejorada del entrenamiento inicial"""
        print("\nIniciando entrenamiento con datos desde JSON...")
        
        # Cargar y aumentar datos
        training_data = self.initialize_training_data(json_path)
        augmented_data = self._augment_training_data(training_data)
        
        dataset = EnhancedConversationDataset(augmented_data, self.tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
            for batch_idx, batch in progress_bar:
                batch['batch_idx'] = batch_idx  # Añadir índice de batch
                metrics = self.train_iteration(batch)
                
                total_loss += metrics['loss']
                
                # Calcular accuracy
                logits = self.model(batch['input_ids'].to(self.device), 
                                  batch['attention_mask'].to(self.device))[0]
                _, predicted = torch.max(logits, 1)
                total += batch['label'].size(0)
                correct += (predicted == batch['label'].to(self.device)).sum().item()
                
                # Actualizar barra de progreso
                progress_bar.set_postfix({
                    'epoch': f'{epoch+1}/{epochs}',
                    'loss': f'{metrics["loss"]:.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
            
            epoch_loss = total_loss / len(dataloader)
            epoch_acc = 100 * correct / total
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_checkpoint()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        print("\nEntrenamiento completado.")
        return best_loss
    def _augment_training_data(self, data):
        """Método nuevo para aumentación de datos"""
        augmented_data = []
        for item in data:
            # Añadir ejemplo original
            augmented_data.append(item)
            
            # Añadir variaciones
            text = item['input']
            
            # Variación 1: Añadir prefijos comunes
            prefixes = ["por favor ", "disculpa ", "me gustaría saber ", "quisiera preguntar "]
            for prefix in prefixes:
                if random.random() < 0.3:  # 30% de probabilidad
                    augmented_data.append({
                        'input': prefix + text,
                        'label': item['label']
                    })
            
            # Variación 2: Cambiar orden de palabras en preguntas
            if '?' in text and len(text.split()) > 3:
                words = text.split()
                if len(words) > 3:
                    i, j = random.sample(range(len(words)-1), 2)
                    words[i], words[j] = words[j], words[i]
                    augmented_data.append({
                        'input': ' '.join(words),
                        'label': item['label']
                    })
        
        return augmented_data
    def get_response(self, prediction, confidence, context_analysis):
        """Método mejorado para obtener respuestas"""
        try:
            # Convertir prediction a string y validar
            prediction_key = str(prediction)
            if prediction_key not in self.responses:
                logging.warning(f"Predicción no encontrada: {prediction_key}")
                return "Lo siento, no tengo una respuesta apropiada para eso. ¿Podrías reformular tu mensaje?"
                
            # Verificar confianza
            if confidence < self.min_confidence:
                if context_analysis.get('novelty', 0) > 0.5:
                    return "Interesante. ¿Podrías darme más detalles al respecto?"
                else:
                    return "No estoy seguro de entender. ¿Podrías reformular tu mensaje?"
            
            # Obtener y validar respuestas
            responses = self.responses[prediction_key]
            if not responses:
                return "Entiendo. ¿Podrías elaborar más?"
                
            # Seleccionar respuesta considerando el contexto
            try:
                relevance = float(context_analysis.get('relevance', 0))
                if relevance > 0.3:
                    longer_responses = [r for r in responses if len(r.split()) > 5]
                    if longer_responses:
                        responses = longer_responses
            except (TypeError, ValueError) as e:
                logging.warning(f"Error procesando relevancia: {e}")
            
            return random.choice(responses)
            
        except Exception as e:
            logging.error(f"Error en get_response: {str(e)}")
            return "Lo siento, hubo un error generando la respuesta. ¿Podrías intentarlo de nuevo?"
    
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
            
            if not user_input:
                continue
                
            # Procesar entrada
            try:
                result = trainer.process_conversation(user_input)
                
                # Validar resultado
                if not isinstance(result, dict) or 'prediction' not in result:
                    print("Error: Resultado de procesamiento inválido")
                    continue
                    
                # Generar respuesta
                response = trainer.get_response(
                    result['prediction'],
                    result.get('confidence', 0.0),
                    result.get('context_analysis', {
                        'relevance': 0.0,
                        'novelty': 0.0,
                        'importance': 0.0
                    })
                )
                
                if response:
                    print(f"Bot: {response}")
                else:
                    print("Bot: Lo siento, no pude generar una respuesta apropiada.")
                
                # Guardar interacción en el contexto
                cleaned_input = clean_text(user_input)
                encoded_input = trainer.model.encoder(
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
                
                trainer.context_manager.add_interaction(
                    cleaned_input,
                    response,
                    encoded_input
                )
                
            except Exception as processing_error:
                print(f"Error procesando mensaje: {str(processing_error)}")
                print("Bot: Lo siento, hubo un error procesando tu mensaje. ¿Podrías intentarlo de nuevo?")
                
        except KeyboardInterrupt:
            print("\nSesión finalizada por el usuario.")
            trainer.save_checkpoint()
            break
        except Exception as e:
            print(f"Error inesperado: {str(e)}")
            print("Bot: Lo siento, ocurrió un error. Por favor, intenta de nuevo.")
            continue

if __name__ == "__main__":
    main()