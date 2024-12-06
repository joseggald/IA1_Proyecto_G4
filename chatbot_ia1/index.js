import * as tf from '@tensorflow/tfjs';
import natural from 'natural';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import path from 'path';

const { WordTokenizer } = natural;

export class ImprovedChatbot {
    constructor(config = {}) {
        this.config = {
            embeddingDim: 128,          // Reducido para evitar matrices muy grandes
            hiddenUnits: 256,        // Mayor capacidad de procesamiento
            encoderLayers: 3,         // Múltiples capas para mejor comprensión
            dropoutRate: 0.3,         // Mejor regularización
            learningRate: 0.001,
            minConfidence: 0.6,       // Mayor umbral de confianza
            batchSize: 32,            // Batch size optimizado
            maxSeqLength: 50,         // Secuencias más largas
            vocabularySize: 10000,    // Vocabulario más grande
            contextWindow: 3,         // Ventana de contexto para respuestas
            ...config
        };

        this.tokenizer = new WordTokenizer();
        this.vocabulario = ['<PAD>', '<UNK>', '<START>', '<END>', '<MASK>'];
        this.palabra_indice = {};
        this.indice_palabra = {};
        this.model = null;
        this.responseMap = new Map();
        this.conversationHistory = [];
        
        // Inicializar vocabulario base
        this.vocabulario.forEach((token, index) => {
            this.palabra_indice[token] = index;
            this.indice_palabra[index] = token;
        });
    }

    preprocessText(text) {
        // Mejor preprocesamiento de texto
        return text.toLowerCase()
                  .replace(/[^\w\s¿?¡!.,]/g, '')
                  .replace(/\s+/g, ' ')
                  .trim()
                  .split(' ')
                  .filter(word => word.length > 0);
    }

    buildVocabulary(sentences) {
        const wordFreq = new Map();
        
        // Contar frecuencias de palabras
        sentences.forEach(sentence => {
            const tokens = this.preprocessText(sentence);
            tokens.forEach(word => {
                wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
            });
        });

        // Filtrar y ordenar por frecuencia
        const sortedWords = Array.from(wordFreq.entries())
            .filter(([_, freq]) => freq >= 2) // Eliminar palabras muy poco frecuentes
            .sort((a, b) => b[1] - a[1])
            .slice(0, this.config.vocabularySize - this.vocabulario.length)
            .map(([word]) => word);

        this.vocabulario = [...this.vocabulario, ...sortedWords];
        
        // Actualizar índices
        this.vocabulario.forEach((word, index) => {
            this.palabra_indice[word] = index;
            this.indice_palabra[index] = word;
        });

        console.log(`Vocabulario construido: ${this.vocabulario.length} palabras`);
    }

    prepareData(data) {
        // Construir mapa de respuestas con contexto
        this.responseMap.clear();
        data.forEach(item => {
            const inputKey = item.input.toLowerCase();
            const outputValue = item.output.toLowerCase();
            
            if (!this.responseMap.has(inputKey)) {
                this.responseMap.set(inputKey, []);
            }
            this.responseMap.get(inputKey).push(outputValue);
        });

        // Preparar datos de entrenamiento
        const processedData = data.map(item => {
            const input = this.preprocessText(item.input);
            const output = this.preprocessText(item.output);
            return { input, output };
        });

        // Vectorización mejorada
        const inputs = [];
        const outputs = [];

        processedData.forEach(({ input, output }) => {
            // Entrada con padding/truncating
            const inputVector = input
                .slice(0, this.config.maxSeqLength)
                .map(word => this.palabra_indice[word] || this.palabra_indice['<UNK>']);
            
            while (inputVector.length < this.config.maxSeqLength) {
                inputVector.push(this.palabra_indice['<PAD>']);
            }

            // Salida como secuencia completa
            const outputVector = new Array(this.vocabulario.length).fill(0);
            output.forEach(word => {
                const idx = this.palabra_indice[word] || this.palabra_indice['<UNK>'];
                outputVector[idx] = 1;
            });

            inputs.push(inputVector);
            outputs.push(outputVector);
        });

        return {
            inputs: tf.tensor2d(inputs),
            outputs: tf.tensor2d(outputs)
        };
    }

    buildModel() {
        const model = tf.sequential();
    
        model.add(tf.layers.embedding({
            inputDim: this.vocabulario.length,
            outputDim: this.config.embeddingDim,
            maskZero: true,
            inputLength: this.config.maxSeqLength
        }));
    
        for (let i = 0; i < this.config.encoderLayers; i++) {
            model.add(tf.layers.bidirectional({
                layer: tf.layers.lstm({
                    units: this.config.hiddenUnits,
                    returnSequences: i < this.config.encoderLayers - 1,
                    recurrentDropout: 0.1
                })
            }));
            model.add(tf.layers.dropout({ rate: this.config.dropoutRate }));
        }
    
        model.add(tf.layers.dense({
            units: this.config.hiddenUnits,
            activation: 'tanh'
        }));
    
        model.add(tf.layers.dense({
            units: this.config.hiddenUnits / 2,
            activation: 'relu'
        }));
    
        model.add(tf.layers.dropout({ rate: this.config.dropoutRate }));
    
        model.add(tf.layers.dense({
            units: this.vocabulario.length,
            activation: 'softmax'
        }));
    
        const optimizer = tf.train.adam(this.config.learningRate, 0.9, 0.999, 1e-7);
    
        model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
    
        console.log('\nArquitectura del modelo:');
        model.summary();
    
        this.model = model;
        return model;
    }

    async train(data, epochs = 30) {
        try {
            console.log('Preparando datos...');
            const { inputs, outputs } = this.prepareData(data);
            
            if (!this.model) {
                console.log('Construyendo modelo...');
                this.buildModel();
            }
            
            console.log('Iniciando entrenamiento...');
            const history = await this.model.fit(inputs, outputs, {
                epochs: epochs,
                batchSize: this.config.batchSize,
                validationSplit: 0.2,
                shuffle: true,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        // Verificamos si cada métrica existe antes de usarla
                        const loss = logs.loss ? logs.loss.toFixed(4) : 'N/A';
                        const acc = logs.acc ? logs.acc.toFixed(4) : 
                                  (logs.accuracy ? logs.accuracy.toFixed(4) : 'N/A');
                        const valLoss = logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A';
                        const valAcc = logs.val_acc ? logs.val_acc.toFixed(4) : 
                                     (logs.val_accuracy ? logs.val_accuracy.toFixed(4) : 'N/A');
    
                        console.log(
                            `Época ${epoch + 1}/${epochs}: ` +
                            `pérdida = ${loss}, ` +
                            `precisión = ${acc}` +
                            (valLoss !== 'N/A' ? `, val_loss = ${valLoss}` : '') +
                            (valAcc !== 'N/A' ? `, val_accuracy = ${valAcc}` : '')
                        );
                    }
                }
            });
            
            console.log('Entrenamiento completado');
            return history;
        } catch (error) {
            console.error('Error en entrenamiento:', error);
            throw error;
        }
    }

    async generateResponse(input) {
        try {
            // Búsqueda exacta con contexto
            const exactMatches = this.responseMap.get(input.toLowerCase());
            if (exactMatches && exactMatches.length > 0) {
                return exactMatches[Math.floor(Math.random() * exactMatches.length)];
            }

            // Preparar input
            const processedInput = this.preprocessText(input);
            const inputVector = processedInput
                .slice(0, this.config.maxSeqLength)
                .map(word => this.palabra_indice[word] || this.palabra_indice['<UNK>']);
            
            while (inputVector.length < this.config.maxSeqLength) {
                inputVector.push(this.palabra_indice['<PAD>']);
            }

            // Generar predicción
            const inputTensor = tf.tensor2d([inputVector]);
            const predictions = await this.model.predict(inputTensor).array();
            
            // Procesar predicciones
            const probs = predictions[0];
            const validIndices = Array.from(probs.keys())
                .filter(idx => 
                    this.indice_palabra[idx] && 
                    !['<PAD>', '<UNK>', '<START>', '<END>', '<MASK>'].includes(this.indice_palabra[idx])
                )
                .sort((a, b) => probs[b] - probs[a]);
            
            // Seleccionar mejores candidatos
            const topK = 5;
            const candidates = validIndices
                .slice(0, topK)
                .filter(idx => probs[idx] > this.config.minConfidence);
            
            if (candidates.length === 0) {
                return "Lo siento, no estoy seguro de cómo responder a eso.";
            }
            
            // Construir respuesta con las palabras más probables
            const response = candidates
                .map(idx => this.indice_palabra[idx])
                .join(' ');
            
            return response;

        } catch (error) {
            console.error('Error generando respuesta:', error);
            return 'Lo siento, ha ocurrido un error al procesar tu mensaje.';
        }
    }

    // Mantener tus métodos originales de save/load
    async saveModel(modelPath) {
        try {
            const absolutePath = path.resolve(modelPath);
            
            if (!existsSync(absolutePath)) {
                mkdirSync(absolutePath, { recursive: true });
            }
    
            // Guardar metadatos mejorados
            const metadata = {
                vocabulario: this.vocabulario,
                palabra_indice: this.palabra_indice,
                indice_palabra: this.indice_palabra,
                config: this.config,
                modelVersion: '2.0',
                timestamp: new Date().toISOString(),
                vocabularySize: this.vocabulario.length,
                modelArchitecture: {
                    layers: this.model.layers.map(layer => layer.getConfig())
                }
            };
    
            const metadataPath = path.join(absolutePath, 'metadata.json');
            writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
            console.log('Metadatos guardados en:', metadataPath);
    
            // Guardar configuración del modelo
            const modelConfig = {
                config: {
                    layers: this.model.layers.map(layer => ({
                        class_name: layer.constructor.name,
                        config: layer.getConfig()
                    }))
                }
            };
            
            const modelConfigPath = path.join(absolutePath, 'model-config.json');
            writeFileSync(modelConfigPath, JSON.stringify(modelConfig, null, 2));
            console.log('Configuración del modelo guardada');
    
            // Guardar pesos con compresión
            const weights = this.model.getWeights();
            const weightData = await Promise.all(weights.map(w => w.array()));
            const weightsPath = path.join(absolutePath, 'model-weights.json');
            writeFileSync(weightsPath, JSON.stringify(weightData));
            console.log('Pesos del modelo guardados');
    
            console.log('Modelo guardado completamente en:', absolutePath);
            
        } catch (error) {
            console.error('Error al guardar el modelo:', error);
            throw error;
        }
    }

    async loadModel(modelPath) {
        try {
            const absolutePath = path.resolve(modelPath);
            if (!existsSync(absolutePath)) {
                throw new Error(`El directorio ${absolutePath} no existe`);
            }
    
            // Cargar metadatos y configuración
            const metadata = JSON.parse(readFileSync(path.join(absolutePath, 'metadata.json'), 'utf-8'));
            const modelConfig = JSON.parse(readFileSync(path.join(absolutePath, 'model-config.json'), 'utf-8'));
            
            this.vocabulario = metadata.vocabulario;
            this.palabra_indice = metadata.palabra_indice;
            this.indice_palabra = metadata.indice_palabra;
            this.config = metadata.config;
            
            this.model = tf.sequential();
            
            // Reconstruir capas
            for (const layerConfig of modelConfig.config.layers) {
                let layer;
                
                if (layerConfig.class_name === 'Bidirectional') {
                    // Crear capa LSTM primero
                    const lstmConfig = layerConfig.config.layer.config;
                    const lstm = tf.layers.lstm({
                        units: lstmConfig.units,
                        returnSequences: lstmConfig.returnSequences,
                        recurrentDropout: lstmConfig.recurrentDropout || 0
                    });
                    
                    // Crear capa bidireccional
                    layer = tf.layers.bidirectional({
                        layer: lstm,
                        mergeMode: layerConfig.config.mergeMode || 'concat'
                    });
                } else {
                    // Para otras capas, usar el nombre en minúsculas sin 'Layer'
                    const layerClassName = layerConfig.class_name.toLowerCase().replace('layer', '');
                    if (tf.layers[layerClassName]) {
                        layer = tf.layers[layerClassName](layerConfig.config);
                    } else {
                        throw new Error(`Tipo de capa no soportado: ${layerConfig.class_name}`);
                    }
                }
                
                this.model.add(layer);
            }
    
            // Compilar modelo
            this.model.compile({
                optimizer: tf.train.adam(this.config.learningRate),
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
    
            // Cargar pesos
            const weightsData = JSON.parse(readFileSync(path.join(absolutePath, 'model-weights.json'), 'utf-8'));
            const weights = weightsData.map(w => tf.tensor(w));
            this.model.setWeights(weights);
            
            console.log('Modelo cargado exitosamente');
            this.model.summary();
            
        } catch (error) {
            console.error('Error al cargar el modelo:', error);
            throw error;
        }
    }

    async verifyModel() {
        // Verificación básica del modelo
        try {
            const testInput = new Array(this.config.maxSeqLength).fill(this.palabra_indice['<PAD>']);
            const testTensor = tf.tensor2d([testInput]);
            
            const prediction = await this.model.predict(testTensor);
            
            if (!prediction || !prediction.shape || prediction.shape[1] !== this.vocabulario.length) {
                throw new Error('La salida del modelo no tiene las dimensiones esperadas');
            }

            prediction.dispose();
            testTensor.dispose();
            
            console.log('Verificación del modelo completada con éxito');
            return true;
        } catch (error) {
            console.error('Error en la verificación del modelo:', error);
            throw new Error('El modelo no pasó la verificación: ' + error.message);
        }
    }

    // Nuevos métodos para mejor manejo de conversaciones

    addToConversationHistory(input, response) {
        this.conversationHistory.push({
            input,
            response,
            timestamp: new Date().toISOString()
        });

        // Mantener solo las últimas N conversaciones
        if (this.conversationHistory.length > this.config.contextWindow) {
            this.conversationHistory.shift();
        }
    }

    getContextualResponse(input) {
        // Usar el historial de conversación para mejorar la respuesta
        const recentHistory = this.conversationHistory
            .slice(-this.config.contextWindow)
            .map(h => h.input + ' ' + h.response)
            .join(' ');

        return recentHistory;
    }

    async evaluateModel(testData) {
        const results = {
            accuracy: 0,
            totalTests: testData.length,
            successes: 0,
            failures: 0,
            confusionMatrix: new Map()
        };

        for (const test of testData) {
            const response = await this.generateResponse(test.input);
            const expected = test.output.toLowerCase();
            const actual = response.toLowerCase();

            if (actual === expected) {
                results.successes++;
            } else {
                results.failures++;
                
                // Registrar confusión
                const key = `${expected} -> ${actual}`;
                results.confusionMatrix.set(
                    key, 
                    (results.confusionMatrix.get(key) || 0) + 1
                );
            }
        }

        results.accuracy = results.successes / results.totalTests;
        return results;
    }

    getModelStats() {
        return {
            vocabularySize: this.vocabulario.length,
            modelConfig: this.config,
            layerCount: this.model.layers.length,
            parameterCount: this.model.countParams(),
            conversationHistoryLength: this.conversationHistory.length
        };
    }

    cleanupMemory() {
        // Liberar tensores no utilizados
        tf.disposeVariables();
        tf.engine().endScope();
        tf.engine().startScope();
    }
}