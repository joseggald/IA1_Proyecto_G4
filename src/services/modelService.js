import * as tf from '@tensorflow/tfjs';

class ModelService {
  constructor() {
    this.model = null;
    this.tokenizer = {
      word_index: {},
      index_word: {},
      word_counts: {},
      num_words: 2000,
      oov_token: "<OOV>"
    };
    this.config = null;
    this.max_length = 20;
    this.vocab_size = 2000;
    this.initialized = false;
    this.responses = null;
    this.confidenceThreshold = 0.15;
    this.topicBoostFactor = 0.2;
    this.debugMode = true;
    this.greetingThreshold = 0.1;
    // Nuevas propiedades para tópicos
    this.topicData = null;
    this.conversationHistory = [];
    this.memorySize = 10;
  }
  async verifyTopicData() {
    try {
      if (!this.topicData || typeof this.topicData !== 'object') {
        console.error('Topic data not loaded or invalid');
        return false;
      }
  
      // Verificar estructura básica
      const hasTopics = this.topicData.topics && typeof this.topicData.topics === 'object';
      const hasKeywords = this.topicData.keywords && typeof this.topicData.keywords === 'object';
      const hasRelations = this.topicData.relations && typeof this.topicData.relations === 'object';
  
      // Verificar contenido
      const topicsCount = Object.keys(this.topicData.topics || {}).length;
      const keywordsCount = Object.keys(this.topicData.keywords || {}).length;
  
      // Verificar integridad de los datos
      let validTopics = 0;
      for (const [topic, data] of Object.entries(this.topicData.topics)) {
        if (Array.isArray(data.keywords) && Array.isArray(data.labels)) {
          validTopics++;
        }
      }
  
      const verificationResult = {
        structureValid: hasTopics && hasKeywords && hasRelations,
        topicsCount,
        keywordsCount,
        validTopics,
        sampleTopics: Object.keys(this.topicData.topics || {}).slice(0, 3),
        sampleKeywords: Object.keys(this.topicData.keywords || {}).slice(0, 3)
      };
  
      console.log('Topic data verification details:', verificationResult);
  
      // Considerar válido si al menos hay un tópico con datos válidos
      return validTopics > 0;
    } catch (error) {
      console.error('Error during topic verification:', error);
      return false;
    }
  }

  extractKeywords(text) {
    const words = text.split(/\s+/).filter(word => word.length > 2 && !/^\d+$/.test(word));
    if (this.debugMode) {
      console.log('Extracted keywords:', words);
    }
    return [...new Set(words)];
  }

  identifyTopic(text, keywords) {
    if (!this.topicData?.topics) {
      if (this.debugMode) {
        console.log('No topic data available for identification');
      }
      return null;
    }

    const topicScores = {};
    
    for (const [topic, data] of Object.entries(this.topicData.topics)) {
      let score = 0;
      
      // Coincidencia de keywords
      const matches = keywords.filter(keyword => 
        data.keywords.some(topicKeyword => 
          topicKeyword.includes(keyword) || keyword.includes(topicKeyword)
        )
      );
      score += matches.length * 2;

      if (this.debugMode && matches.length > 0) {
        console.log(`Topic ${topic} keyword matches:`, matches);
      }

      // Contexto de conversación
      if (this.conversationHistory.length > 0) {
        const lastTopic = this.conversationHistory[this.conversationHistory.length - 1]?.topic;
        if (lastTopic === topic) {
          score += 1;
        } else if (this.topicData.relations[lastTopic]?.includes(topic)) {
          score += 0.5;
        }
      }

      topicScores[topic] = score;
    }

    if (this.debugMode) {
      console.log('Topic scores:', topicScores);
    }

    const entries = Object.entries(topicScores);
    if (entries.length === 0) return null;
    
    const bestTopic = entries.reduce((a, b) => b[1] > a[1] ? b : a);
    return bestTopic[1] > 0 ? bestTopic[0] : null;
  }
  async loadFile(path) {
    try {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} on path: ${path}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`Error loading file from ${path}:`, error);
      throw error;
    }
  }

  async verifyTokenizer() {
    try {
      if (!this.tokenizer || !this.tokenizer.word_index || Object.keys(this.tokenizer.word_index).length === 0) {
        console.error('Tokenizer not properly initialized or word_index is empty');
        return false;
      }

      const vocabSize = Object.keys(this.tokenizer.word_index).length;
      const hasOOVToken = Object.prototype.hasOwnProperty.call(this.tokenizer.word_index, '<OOV>');
      const firstTokenValue = this.tokenizer.word_index['<OOV>'];

      console.log('Tokenizer verification:', {
        vocabularySize: vocabSize,
        sampleWords: Object.entries(this.tokenizer.word_index).slice(0, 5),
        hasOOVToken,
        firstTokenValue,
        oovTokenCorrect: firstTokenValue === 1
      });

      if (vocabSize === 0) {
        console.error('Empty vocabulary');
        return false;
      }

      if (!hasOOVToken) {
        console.warn('OOV token not found - attempting to add it');
        this.tokenizer.word_index['<OOV>'] = 1;
        this.tokenizer.index_word['1'] = '<OOV>';
      }

      return true;
    } catch (error) {
      console.error('Error verifying tokenizer:', error);
      return false;
    }
  }

  async verifyResponses() {
    try {
      if (!this.responses) {
        console.error('Responses not loaded');
        return false;
      }

      console.log('Response verification:', {
        numberOfClasses: Object.keys(this.responses).length,
        classes: Object.keys(this.responses),
        sampleResponses: Object.entries(this.responses).slice(0, 2)
      });

      return Object.keys(this.responses).length > 0;
    } catch (error) {
      console.error('Error verifying responses:', error);
      return false;
    }
  }

  preprocess_text(text) {
    if (!text) return '';
    
    let processed = text.toLowerCase().trim();
    
    processed = processed
      .replace(/á/g, 'a')
      .replace(/é/g, 'e')
      .replace(/í/g, 'i')
      .replace(/ó/g, 'o')
      .replace(/ú/g, 'u')
      .replace(/ñ/g, 'n');
    
    processed = processed.replace(/[^\w\s¿?]/g, '');
    
    return processed;
  }

  texts_to_sequences(text) {
    if (!this.tokenizer.word_index || Object.keys(this.tokenizer.word_index).length === 0) {
      throw new Error('Tokenizer not initialized or word_index is empty');
    }

    const words = text.split(/\s+/).filter(word => word.length > 0);
    console.log('Words to tokenize:', words);
    
    const sequence = words.map(word => {
      const token = this.tokenizer.word_index[word];
      return token || this.tokenizer.word_index['<OOV>'] || 1;
    });

    return sequence;
  }

  pad_sequences(sequence) {
    const padded = Array(this.max_length).fill(0);
    sequence.slice(0, this.max_length).forEach((value, index) => {
      padded[index] = value;
    });
    return padded;
  }

  async init() {
    try {
      if (this.initialized) {
        return true;
      }

      console.log('Iniciando carga de archivos...');

      // Cargar configuración
      this.config = await this.loadFile('./model_config.json');
      this.max_length = this.config.max_length;
      this.vocab_size = this.config.vocab_size;
      this.memorySize = this.config.memory_size || 10;

      console.log('Cargando datos de tópicos...');
      this.topicData = await this.loadFile('./topic_data.json');
      const topicStatus = await this.verifyTopicData();
      console.log('Estado de carga de tópicos:', topicStatus);

      // Cargar tokenizer
      console.log('Cargando tokenizer...');
      const tokenizerData = await this.loadFile('./tokenizer.json');

      if (tokenizerData && tokenizerData.config) {
        const tokenizerConfig = tokenizerData.config;
        this.tokenizer.word_index = JSON.parse(tokenizerConfig.word_index);
        this.tokenizer.index_word = JSON.parse(tokenizerConfig.index_word);
        this.tokenizer.word_counts = JSON.parse(tokenizerConfig.word_counts);
        this.tokenizer.index_docs = JSON.parse(tokenizerConfig.index_docs);
        this.tokenizer.word_docs = JSON.parse(tokenizerConfig.word_docs);
      } else {
        throw new Error('Tokenizer data is invalid or missing.');
      }

      if (!this.tokenizer.word_index['<OOV>']) {
        this.tokenizer.word_index['<OOV>'] = 1;
      }

      this.tokenizer.index_word = Object.entries(this.tokenizer.word_index)
        .reduce((acc, [word, index]) => {
          acc[index] = word;
          return acc;
        }, {});

      // Cargar modelo
      console.log('Cargando modelo...');
      const modelJson = await this.loadFile('./tfjs_model/model.json');

      // Asegurarse de que la topología del modelo tenga la configuración correcta
      if (!modelJson.modelTopology.config) {
        modelJson.modelTopology.config = {};
      }

      // Corregir la configuración de la capa de entrada
      if (!modelJson.modelTopology.config.layers) {
        modelJson.modelTopology.config.layers = [];
      }

      // Asegurar que la primera capa sea una capa de entrada correctamente configurada
      const inputLayer = {
        class_name: "InputLayer",
        config: {
          batch_input_shape: [null, this.max_length],
          dtype: "float32",
          sparse: false,
          ragged: false,
          name: "input_1"
        }
      };

      // Si no hay capa de entrada, agregarla al principio
      if (modelJson.modelTopology.config.layers[0]?.class_name !== "InputLayer") {
        modelJson.modelTopology.config.layers.unshift(inputLayer);
      } else {
        // Si existe, actualizar su configuración
        modelJson.modelTopology.config.layers[0] = inputLayer;
      }

      // Verificar y ajustar la siguiente capa (Embedding)
      if (modelJson.modelTopology.config.layers[1]?.class_name === "Embedding") {
        modelJson.modelTopology.config.layers[1].config.input_dim = this.vocab_size;
        modelJson.modelTopology.config.layers[1].config.output_dim = 64; // embedding_dim
        modelJson.modelTopology.config.layers[1].config.input_length = this.max_length;
      }

      try {
        this.model = await tf.loadLayersModel(
          tf.io.fromMemory(modelJson),
          {
            strict: false
          }
        );
      } catch (modelError) {
        console.error('Error loading model:', modelError);
        throw new Error(`Failed to load model: ${modelError.message}`);
      }

      // Cargar respuestas
      console.log('Cargando respuestas...');
      this.responses = await this.loadFile('./responses.json');

      this.initialized = true;
      console.log('Modelo inicializado correctamente');
      return true;
    } catch (error) {
      console.error('Error detallado durante la inicialización:', error);
      this.initialized = false;
      throw error;
    }
  }

  async processMessage(text) {
    try {
      if (!this.initialized) {
        await this.init();
      }
  
      const processed_text = this.preprocess_text(text);
      if (!processed_text) {
        throw new Error('Empty text after preprocessing');
      }
  
      // Análisis de tópicos mejorado
      const keywords = this.extractKeywords(processed_text);
      const currentTopic = this.identifyTopic(processed_text, keywords);
      
      if (this.debugMode) {
        console.log('Topic analysis:', {
          text: processed_text,
          keywords: keywords,
          identifiedTopic: currentTopic
        });
      }
  
      const sequence = this.texts_to_sequences(processed_text);
      if (sequence.length === 0) {
        throw new Error('No valid tokens found in input text');
      }
  
      const padded = this.pad_sequences(sequence);
      const tensorInput = tf.tensor2d([padded], [1, this.max_length]);
  
      // Predicción con ajuste de tópicos
      const prediction = await tf.tidy(() => {
        const pred = this.model.predict(tensorInput);
        return pred.arraySync()[0];
      });
  
      tensorInput.dispose();
  
      // Procesamiento de predicción con tópicos
      let predictedClass = prediction.indexOf(Math.max(...prediction));
      let confidence = prediction[predictedClass];
      
      // Ajuste de confianza basado en tópicos
      if (currentTopic && this.topicData.topics[currentTopic]) {
        const topicLabels = this.topicData.topics[currentTopic].labels;
        
        // Buscar la mejor predicción que coincida con el tópico
        const topicAdjustedPredictions = prediction.map((conf, idx) => {
          if (topicLabels.includes(String(idx))) {
            return conf + this.topicBoostFactor;
          }
          return conf;
        });

        const newPredictedClass = topicAdjustedPredictions.indexOf(Math.max(...topicAdjustedPredictions));
        const newConfidence = topicAdjustedPredictions[newPredictedClass];

        if (this.debugMode) {
          console.log('Topic adjustment:', {
            originalClass: predictedClass,
            originalConfidence: confidence,
            adjustedClass: newPredictedClass,
            adjustedConfidence: newConfidence,
            topicLabels: topicLabels
          });
        }

        predictedClass = newPredictedClass;
        confidence = newConfidence;
      }
  
      // Seleccionar respuesta
      let responseText;
      if (confidence > this.confidenceThreshold && this.responses[predictedClass]) {
        const possibleResponses = this.responses[predictedClass];
        responseText = possibleResponses[Math.floor(Math.random() * possibleResponses.length)];
      } else {
        responseText = "Lo siento, no estoy seguro de cómo responder a eso. ¿Podrías dar más detalles?";
      }
  
      // Actualizar historial
      const conversationEntry = {
        input: text,
        processed_text: processed_text,
        topic: currentTopic,
        keywords: keywords,
        response: responseText,
        confidence: confidence,
        timestamp: new Date().toISOString()
      };
  
      this.conversationHistory.push(conversationEntry);
      if (this.conversationHistory.length > this.memorySize) {
        this.conversationHistory.shift();
      }
  
      return {
        prediction: predictedClass,
        confidence: confidence,
        text: responseText,
        contextAnalysis: {
          processed_text: processed_text,
          topic: currentTopic,
          keywords: keywords,
          sequence_length: sequence.length,
          confidence_score: confidence,
          input_tokens: sequence,
          conversation_history: this.conversationHistory,
          topic_adjusted: currentTopic !== null
        }
      };
    } catch (error) {
      console.error('Error processing message:', error);
      throw new Error(`Error processing message: ${error.message}`);
    }
  }
}

export default new ModelService();