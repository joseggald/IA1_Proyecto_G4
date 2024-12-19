import * as tf from '@tensorflow/tfjs';

class ModelService {
  constructor() {
    // Configuración básica
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
    this.topicData = null;
    this.conversationHistory = [];
    this.memorySize = 10;

    // Tokens para detección de idioma
    this.languageTokens = {
      es: {
        greetings: ['hola', 'buenos', 'dias', 'tardes', 'noches', 'saludos', 'que tal'],
        common: ['que', 'como', 'estas', 'bien', 'gracias', 'por', 'favor', 'de', 'la', 'el', 'lo', 'las', 'los'],
        questions: ['que', 'cual', 'como', 'donde', 'cuando', 'por que', 'quién'],
        responses: ['si', 'no', 'tal vez', 'quizas', 'claro', 'vale', 'bueno'],
        commands: ['dime', 'explicame', 'cuentame', 'ayudame', 'dame'],
        technical: ['programacion', 'variable', 'funcion', 'tabla', 'datos', 'sql', 'bases']
      },
      en: {
        greetings: ['hi', 'hello', 'hey', 'good', 'morning', 'afternoon', 'evening', "what's up"],
        common: ['the', 'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should', 'have', 'has'],
        questions: ['what', 'which', 'how', 'where', 'when', 'why', 'who'],
        responses: ['yes', 'no', 'maybe', 'perhaps', 'sure', 'okay', 'alright'],
        commands: ['tell', 'explain', 'help', 'show', 'give'],
        technical: ['programming', 'variable', 'function', 'table', 'data', 'sql', 'database']
      }
    };
  }

  detectLanguage(text) {
    if (!text || !text.trim()) return 'es';

    const normalizedText = this.preprocess_text(text);
    const words = normalizedText.split(/\s+/);
    
    let scores = { es: 0, en: 0 };

    words.forEach(word => {
      const tokenId = this.tokenizer.word_index[word];
      if (tokenId) {
        for (const [lang, categories] of Object.entries(this.languageTokens)) {
          for (const [category, words] of Object.entries(categories)) {
            if (words.includes(word)) {
              const weight = (category === 'greetings' || category === 'questions') ? 2 : 1;
              scores[lang] += weight;
            }
          }
        }
      }
    });

    if (text.match(/^(que|como|cual|donde|cuando|por que|es|un|uno|la|los|el|ella|en|tambien|seria)\b/i)) scores.es += 5;
    if (text.match(/^(what|how|which|where|when|why|is|a|an|the|it|on|in|are)\b/i)) scores.en += 5;

    if (this.debugMode) {
      console.log('Language detection:', {
        text: normalizedText,
        words: words,
        scores: scores,
        tokenMatches: words.map(w => ({
          word: w,
          tokenId: this.tokenizer.word_index[w]
        }))
      });
    }

    return scores.en > scores.es ? 'en' : 'es';
  }

  extractKeywords(text) {
    const words = text.split(/\s+/).filter(word => word.length > 2 && !/^\d+$/.test(word));
    return [...new Set(words)];
  }

  identifyTopic(text, keywords, language) {
    if (!this.topicData?.topics) return null;

    const topicScores = {};
    
    for (const [topic, data] of Object.entries(this.topicData.topics)) {
        // Check if the topic matches the detected language
        if (language === "en" && !topic.endsWith("-en")) continue;
        if (language !== "en" && topic.endsWith("-en")) continue;

        let score = 0;
        
        const matches = keywords.filter(keyword => 
            data.keywords.some(topicKeyword => 
                topicKeyword.includes(keyword) || keyword.includes(topicKeyword)
            )
        );
        score += matches.length * 2;

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

    const entries = Object.entries(topicScores);
    if (entries.length === 0) return null;

    const bestTopic = entries.reduce((a, b) => b[1] > a[1] ? b : a);
    return bestTopic[1] > 0 ? bestTopic[0] : null;
  }


  preprocess_text(text) {
    if (!text) return '';
    
    let processed = text.toLowerCase().trim();
    
    const normalizations = {
      'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
      'ü': 'u', 'ñ': 'n', '¿': '', '¡': ''
    };
    
    for (const [key, value] of Object.entries(normalizations)) {
      processed = processed.replace(new RegExp(key, 'g'), value);
    }
    
    processed = processed.replace(/[^\w\s?!.,]/g, '');
    
    return processed;
  }

  texts_to_sequences(text) {
    if (!this.tokenizer.word_index || !Object.keys(this.tokenizer.word_index).length) {
      throw new Error('Tokenizer not initialized');
    }

    const words = text.split(/\s+/).filter(word => word.length > 0);
    return words.map(word => this.tokenizer.word_index[word] || this.tokenizer.word_index['<OOV>'] || 1);
  }

  pad_sequences(sequence) {
    const padded = Array(this.max_length).fill(0);
    sequence.slice(0, this.max_length).forEach((value, index) => {
      padded[index] = value;
    });
    return padded;
  }

  async loadFile(path) {
    try {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error(`Error loading file from ${path}:`, error);
      throw error;
    }
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
      if (!this.initialized) await this.init();

      // Detectar idioma
      const language = this.detectLanguage(text);
      
      // Preprocesar texto
      const processed_text = this.preprocess_text(text);
      if (!processed_text) throw new Error('Empty text after preprocessing');

      // Análisis de tópicos
      const keywords = this.extractKeywords(processed_text);
      const currentTopic = this.identifyTopic(processed_text, keywords,language);

      // Tokenización y padding
      const sequence = this.texts_to_sequences(processed_text);
      if (!sequence.length) throw new Error('No valid tokens found');

      const padded = this.pad_sequences(sequence);
      const tensorInput = tf.tensor2d([padded], [1, this.max_length]);

      // Predicción
      const prediction = await tf.tidy(() => {
        return this.model.predict(tensorInput).arraySync()[0];
      });
      tensorInput.dispose();

      // Procesar predicción
      let predictedClass = prediction.indexOf(Math.max(...prediction));
      let confidence = prediction[predictedClass];

      // Ajustar por tópico
      if (currentTopic && this.topicData.topics[currentTopic]) {
        const topicLabels = this.topicData.topics[currentTopic].labels;
        const adjustedPredictions = prediction.map((conf, idx) => 
          topicLabels.includes(String(idx)) ? conf + this.topicBoostFactor : conf
        );

        predictedClass = adjustedPredictions.indexOf(Math.max(...adjustedPredictions));
        confidence = adjustedPredictions[predictedClass];
      }

      // Seleccionar respuesta
      let responseText;
      if (confidence > this.confidenceThreshold && this.responses[predictedClass]) {
        const possibleResponses = this.responses[predictedClass];
        responseText = possibleResponses[Math.floor(Math.random() * possibleResponses.length)];
      } else {
        responseText = language === 'es' 
          ? "Lo siento, no estoy seguro de cómo responder a eso."
          : "Sorry, I'm not sure how to respond to that.";
        confidence = 0.0;
      }

      // Actualizar historial
      const entry = {
        input: text,
        processed_text,
        language,
        topic: currentTopic,
        keywords,
        response: responseText,
        confidence,
        timestamp: new Date().toISOString()
      };

      this.conversationHistory.push(entry);
      if (this.conversationHistory.length > this.memorySize) {
        this.conversationHistory.shift();
      }

      return {
        text: responseText,
        confidence,
        language,
        topic: currentTopic
      };

    } catch (error) {
      console.error('Error processing message:', error);
      throw error;
    }
  }
}

export default new ModelService();