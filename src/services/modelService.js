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
    this.confidenceThreshold = 0.1;
    this.greetingThreshold = 0.1;
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

      const commonWords = ['hola', 'gracias', 'ayuda', 'por'];
      const foundCommonWords = commonWords.filter(word => 
        Object.prototype.hasOwnProperty.call(this.tokenizer.word_index, word)
      );

      console.log('Common words check:', {
        searched: commonWords,
        found: foundCommonWords
      });

      if (foundCommonWords.length === 0) {
        console.error('No common words found in vocabulary');
        return false;
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
      console.log(`Word: "${word}", Found in vocabulary: ${token !== undefined}, Token: ${token || 1}`);
      return token || this.tokenizer.word_index['<OOV>'] || 1;
    });

    console.log('Final sequence:', sequence);
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

      this.config = await this.loadFile('/model_config.json');
      this.max_length = this.config.max_length;
      this.vocab_size = this.config.vocab_size;

      console.log('Cargando tokenizer...');
      const tokenizerData = await this.loadFile('/tokenizer.json');

      // Verificar y acceder correctamente a la propiedad 'config' en tokenizerData
      console.log("Tokenizer data loaded:", tokenizerData);

      if (tokenizerData && tokenizerData.config) {
        // Acceder a los datos del tokenizer desde la propiedad config
        const tokenizerConfig = tokenizerData.config;

        // Parsear los datos JSON si son cadenas
        this.tokenizer.word_index = JSON.parse(tokenizerConfig.word_index);
        this.tokenizer.index_word = JSON.parse(tokenizerConfig.index_word);
        this.tokenizer.word_counts = JSON.parse(tokenizerConfig.word_counts);
        this.tokenizer.index_docs = JSON.parse(tokenizerConfig.index_docs);
        this.tokenizer.word_docs = JSON.parse(tokenizerConfig.word_docs);
      } else {
        console.error('Tokenizer data is invalid or missing. Expected structure: { word_index: {...}, index_word: {...}, ... }');
        throw new Error('Tokenizer data is invalid or missing.');
      }

      // Verificar si <OOV> existe en word_index
      if (!this.tokenizer.word_index['<OOV>']) {
        this.tokenizer.word_index['<OOV>'] = 1;
      }

      this.tokenizer.index_word = Object.entries(this.tokenizer.word_index)
        .reduce((acc, [word, index]) => {
          acc[index] = word;
          return acc;
        }, {});

      console.log('Tokenizer procesado:', {
        vocabularySize: Object.keys(this.tokenizer.word_index).length,
        sampleWords: Object.keys(this.tokenizer.word_index).slice(0, 5)
      });

      console.log('Cargando modelo...');
      const modelJson = await this.loadFile('/tfjs_model/model.json');

      if (!modelJson.modelTopology.config) {
        modelJson.modelTopology.config = {};
      }

      modelJson.modelTopology.config.layers = [{
        class_name: "InputLayer",
        config: {
          batch_input_shape: [null, this.max_length],
          dtype: "float32",
          sparse: false,
          ragged: false,
          name: "input_1"
        }
      }, ...modelJson.modelTopology.config.layers.slice(1)];

      this.model = await tf.loadLayersModel(
        tf.io.fromMemory(modelJson),
        {
          strict: false,
          customObjects: {},
          metrics: ['accuracy']
        }
      );

      console.log('Cargando respuestas...');
      this.responses = await this.loadFile('/responses.json');

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
  
      const words = processed_text.split(/\s+/).filter(word => word.length > 0);
      const sequence = this.texts_to_sequences(processed_text);
      
      if (sequence.length === 0) {
        throw new Error('No valid tokens found in input text');
      }
  
      console.log('Processing details:', {
        originalText: text,
        processedText: processed_text,
        tokenSequence: sequence,
        vocabularySize: Object.keys(this.tokenizer.word_index).length
      });
  
      const padded = this.pad_sequences(sequence);
      const tensorInput = tf.tensor2d([padded], [1, this.max_length]);
  
      // Making the prediction
      const prediction = await tf.tidy(() => {
        const pred = this.model.predict(tensorInput);
        return pred.arraySync()[0];
      });
  
      tensorInput.dispose();
  
      // Determine the predicted class and its confidence
      const predictedClass = prediction.indexOf(Math.max(...prediction));
      const confidence = prediction[predictedClass];
  
      // Check for greetings in the input
      const greetingWords = new Set(['hola', 'saludos', 'buenos', 'buenas', 'hey']);
      const isGreeting = words.some(word => greetingWords.has(word));
  
      let responseText;
      if (isGreeting) {
        responseText = "¡Hola! ¿En qué puedo ayudarte?";
      } else if (confidence > this.confidenceThreshold && this.responses && this.responses[predictedClass]) {
        // Select a random response from possible responses
        const possibleResponses = this.responses[predictedClass];
        responseText = possibleResponses[Math.floor(Math.random() * possibleResponses.length)];
      } else {
        // Default fallback response if confidence is below threshold
        responseText = "Lo siento, no estoy seguro de cómo responder a eso. ¿Podrías dar más detalles?";
      }
  
      return {
        prediction: predictedClass,
        confidence: confidence,
        text: responseText,
        contextAnalysis: {
          processed_text: processed_text,
          sequence_length: sequence.length,
          confidence_score: confidence,
          input_tokens: sequence,
          vocabulary_hits: words.filter(word => this.tokenizer.word_index[word] !== undefined).length,
          is_greeting: isGreeting,
          words: words,
          threshold_used: isGreeting ? this.greetingThreshold : this.confidenceThreshold
        }
      };
    } catch (error) {
      console.error('Error processing message:', error);
      throw new Error(`Error processing message: ${error.message}`);
    }
  }
  
}

export default new ModelService();
