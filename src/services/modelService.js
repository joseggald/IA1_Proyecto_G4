import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

// Constants
const VOCAB_URL = 'https://tfhub.dev/tensorflow/tfjs-model/mobilebert/1/processed_vocab.json?tfjs-format=file';
const MODEL_URL = 'https://tfhub.dev/tensorflow/tfjs-model/mobilebert/1';
const MAX_SEQUENCE_LENGTH = 128;
const LEARNING_RATE = 0.001;

class ChatbotModel {
  constructor() {
    this.tokenizer = null;
    this.baseModel = null;
    this.model = null;
    this.vocab = null;
    this.labelMap = {};
    this.reverseLabels = {};
  }

  async loadVocabulary() {
    try {
      const response = await fetch(VOCAB_URL);
      this.vocab = await response.json();
      console.log('Vocabulary loaded successfully');
    } catch (error) {
      console.error('Error loading vocabulary:', error);
      throw error;
    }
  }

  async loadBaseModel() {
    try {
      this.baseModel = await tf.loadLayersModel(MODEL_URL);
      console.log('Base BERT model loaded successfully');
    } catch (error) {
      console.error('Error loading base model:', error);
      throw error;
    }
  }

  // Tokenize input text
  tokenize(text) {
    // Normalize and clean text
    text = text.toLowerCase().trim();
    
    // Split into tokens
    const tokens = text.split(/\s+/);
    
    // Convert tokens to IDs using vocabulary
    const tokenIds = tokens.map(token => {
      const id = this.vocab.indexOf(token);
      return id !== -1 ? id : this.vocab.indexOf('[UNK]');
    });

    // Pad or truncate to MAX_SEQUENCE_LENGTH
    while (tokenIds.length < MAX_SEQUENCE_LENGTH) {
      tokenIds.push(0); // Padding token
    }
    
    return tokenIds.slice(0, MAX_SEQUENCE_LENGTH);
  }

  // Create and compile the model
  createModel(numLabels) {
    const input = tf.input({shape: [MAX_SEQUENCE_LENGTH]});
    
    // Embedding layer
    const embedding = tf.layers.embedding({
      inputDim: this.vocab.length,
      outputDim: 128,
      inputLength: MAX_SEQUENCE_LENGTH
    }).apply(input);

    // LSTM layers
    const lstm1 = tf.layers.lstm({
      units: 64,
      returnSequences: true
    }).apply(embedding);

    const lstm2 = tf.layers.lstm({
      units: 32
    }).apply(lstm1);

    // Dense layers
    const dense1 = tf.layers.dense({
      units: 64,
      activation: 'relu'
    }).apply(lstm2);

    const output = tf.layers.dense({
      units: numLabels,
      activation: 'softmax'
    }).apply(dense1);

    this.model = tf.model({inputs: input, outputs: output});

    this.model.compile({
      optimizer: tf.train.adam(LEARNING_RATE),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    console.log('Model created and compiled successfully');
    return this.model;
  }

  // Prepare training data
  prepareTrainingData(data) {
    // Create label mapping
    const uniqueLabels = [...new Set(data.map(item => item.label))];
    this.labelMap = {};
    this.reverseLabels = {};
    uniqueLabels.forEach((label, index) => {
      this.labelMap[label] = index;
      this.reverseLabels[index] = label;
    });

    // Prepare features and labels
    const features = data.map(item => this.tokenize(item.input));
    const labels = data.map(item => this.labelMap[item.label]);

    // Convert to tensors
    const xs = tf.tensor2d(features, [features.length, MAX_SEQUENCE_LENGTH]);
    const ys = tf.oneHot(labels, Object.keys(this.labelMap).length);

    return [xs, ys];
  }

  // Train the model
  async train(trainData, epochs = 10, batchSize = 32) {
    const [xs, ys] = this.prepareTrainingData(trainData);

    try {
      const history = await this.model.fit(xs, ys, {
        epochs,
        batchSize,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
          }
        }
      });

      console.log('Training completed successfully');
      return history;
    } catch (error) {
      console.error('Error during training:', error);
      throw error;
    }
  }

  // Predict response for input text
  async predict(text) {
    const tokenized = this.tokenize(text);
    const inputTensor = tf.tensor2d([tokenized], [1, MAX_SEQUENCE_LENGTH]);

    const prediction = await this.model.predict(inputTensor).array();
    const labelIndex = prediction[0].indexOf(Math.max(...prediction[0]));
    
    return this.reverseLabels[labelIndex];
  }

  // Initialize the chatbot
  async initialize(trainData) {
    await this.loadVocabulary();
    await this.loadBaseModel();
    this.createModel(new Set(trainData.map(item => item.label)).size);
    await this.train(trainData);
    console.log('Chatbot initialized successfully');
  }
}

// Example usage
async function initializeChatbot(trainData) {
  const chatbot = new ChatbotModel();
  await chatbot.initialize(trainData);
  return chatbot;
}

export { ChatbotModel, initializeChatbot };