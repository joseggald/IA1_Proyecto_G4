import * as tf from '@tensorflow/tfjs';

class ModelService {
  constructor() {
    this.model = null;
    this.config = null;
    this.modelLoaded = false;
    this.contextMemory = [];
    this.maxContextLength = 10;
    this.baseUrl = this.getBaseUrl();
    this.languageProcessor = this.initializeLanguageProcessor();
  }

  getBaseUrl() {
    // Ruta base para cargar los modelos desde el servidor
    return '/web_model'; // Debe estar en la carpeta pública
  }

  initializeLanguageProcessor() {
    return {
      synonyms: {
        'hola': new Set(['saludos', 'buenos días', 'hey', 'hi', 'qué tal']),
        'gracias': new Set(['agradecido', 'te agradezco', 'thanks']),
        // Agregar más sinónimos y palabras emocionales
      },
      stopWords: new Set(['de', 'la', 'que', 'el', 'en', 'y']),
      emotionalWords: {
        'feliz': 1.0,
        'triste': -1.0,
        // Agregar más palabras emocionales
      }
    };
  }

  async validateFile(url, description) {
    try {
      console.log(`Cargando ${description} desde: ${url}`);
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Error HTTP: ${response.status}`);
      }
      
      const contentType = response.headers.get('content-type');
      if (contentType && !contentType.includes('application/json')) {
        console.warn(`Advertencia: Tipo de contenido inesperado: ${contentType}`);
      }
      
      return response;
    } catch (error) {
      console.error(`Error cargando ${description}:`, error);
      throw new Error(`Error al cargar ${description}: ${error.message}`);
    }
  }

  async init() {
    try {
      console.log('Inicializando modelo...');

      // Cargar configuración
      const configPath = `${this.baseUrl}/config.json`;
      const configResponse = await this.validateFile(configPath, 'configuración');
      const configText = await configResponse.text();
      this.config = JSON.parse(configText);

      console.log('Configuración cargada:', this.config);

      // Cargar modelo
      const modelPath = `${this.baseUrl}/tfjs/model.json`;
      this.model = await tf.loadLayersModel(modelPath, {
        onProgress: (fraction) => {
          console.log(`Cargando modelo: ${(fraction * 100).toFixed(1)}%`);
        }
      });

      this.modelLoaded = true;
      console.log('Modelo cargado exitosamente');
      return true;
    } catch (error) {
      console.error('Error de inicialización:', error);
      throw new Error('Error al cargar el modelo TensorFlow.js');
    }
  }

  cleanText(text) {
    text = text.toLowerCase().trim();
    text = text.replace(/[^\w\s¿?¡!.,]/g, '');  // Limpiar texto de caracteres no deseados
    text = text.replace(/\s+/g, ' ');  // Eliminar espacios extra
    return text;
  }

  tokenizeText(text) {
    const cleanedText = this.cleanText(text);
    const words = cleanedText.split(' ');
    const tokens = new Array(128).fill(0);

    for (let i = 0; i < Math.min(words.length, 128); i++) {
      tokens[i] = this.wordToId(words[i]);
    }

    return tokens;
  }

  wordToId(word) {
    // Mapear palabras a un ID usando la lógica de sinónimos y stopwords
    if (this.languageProcessor.synonyms[word]) {
      return 1;  // Por ejemplo, asignar un ID específico
    } else if (this.languageProcessor.stopWords.has(word)) {
      return 0;  // Detenerse si es una stopword
    } else {
      return 2;  // ID para palabras no reconocidas
    }
  }

  async predict(inputText) {
    if (!this.modelLoaded) {
      console.error('El modelo no ha sido cargado aún');
      return;
    }

    // Tokenizar el texto
    const tokens = this.tokenizeText(inputText);

    // Predecir utilizando el modelo cargado
    const inputTensor = tf.tensor([tokens]);  // Convertir tokens a tensor
    const prediction = await this.model.predict(inputTensor).data();

    console.log('Predicción:', prediction);
    return prediction;
  }
}

export default new ModelService();
