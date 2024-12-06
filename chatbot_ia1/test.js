import '@tensorflow/tfjs';  // Acelera el entrenamiento
import { ImprovedChatbot } from './index.js';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { mkdirSync, existsSync, readFileSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function test() {
    try {
        console.log('Iniciando chatbot con datos desde JSON...\n');
        
        // Cargar datos
        const rawData = readFileSync(path.join(__dirname, 'datos.json'), 'utf8');
        const data = JSON.parse(rawData);
        
        // Configuración optimizada
        const chatbot = new ImprovedChatbot({
            embeddingDim: 128,
            hiddenUnits: 256,
            encoderLayers: 2,       // Reducido para mejor rendimiento
            dropoutRate: 0.3,
            learningRate: 0.001,
            minConfidence: 0.6,
            batchSize: 32,
            maxSeqLength: 30,
            contextWindow: 3
        });

        const modelPath = path.join(__dirname, 'saved_model');
        if (!existsSync(modelPath)) {
            mkdirSync(modelPath, { recursive: true });
        }

        // Entrenamiento
        console.log('=== Iniciando Entrenamiento ===\n');
        const startTime = new Date();
        await chatbot.train(data.conversations, 30);
        const endTime = new Date();
        const trainingTime = (endTime - startTime) / 1000;

        // Mostrar resumen del entrenamiento
        console.log('\n=== Resumen del Entrenamiento ===');
        console.log(`Tiempo total: ${trainingTime.toFixed(2)} segundos`);
        console.log(`Tiempo promedio por época: ${(trainingTime / 30).toFixed(2)} segundos`);

        // Pruebas básicas
        console.log('\n=== Pruebas Básicas ===');
        const testCases = [
            'hola',
            'cómo estás',
            'bien gracias',
            'adiós',
            'gracias',
            'quién eres',
            'ayuda'
        ];

        for (const input of testCases) {
            const response = await chatbot.generateResponse(input);
            console.log(`Usuario: "${input}"\nBot: "${response}"\n`);
        }

        // Prueba de conversación continua
        console.log('\n=== Prueba de Conversación ===');
        let conversation = 'hola';
        for (let i = 0; i < 3; i++) {
            const response = await chatbot.generateResponse(conversation);
            console.log(`Usuario: "${conversation}"\nBot: "${response}"\n`);
            conversation = response;
        }

        // Guardar modelo
        console.log('\n=== Guardando Modelo ===');
        await chatbot.saveModel(modelPath);

        // Probar modelo guardado
        console.log('\n=== Probando Modelo Guardado ===');
        const chatbot2 = new ImprovedChatbot();
        await chatbot2.loadModel(modelPath);

        const testInputs = ['hola', 'qué tal', 'adiós'];
        for (const input of testInputs) {
            const response = await chatbot2.generateResponse(input);
            console.log(`Usuario: "${input}"\nBot: "${response}"\n`);
        }

        // Estadísticas
        console.log('\n=== Estadísticas ===');
        const stats = chatbot.getModelStats();
        console.log('Tamaño del vocabulario:', stats.vocabularySize);
        console.log('Número de capas:', stats.layerCount);
        console.log('Total de parámetros:', stats.parameterCount.toLocaleString());
        console.log('Ejemplos de entrenamiento:', data.conversations.length);
        console.log('Entradas únicas:', new Set(data.conversations.map(d => d.input)).size);

        // Evaluar modelo
        console.log('\n=== Evaluación del Modelo ===');
        const evalResults = await chatbot.evaluateModel(data.conversations.slice(0, 10)); // Evaluamos con los primeros 10 ejemplos
        console.log('Precisión en pruebas:', (evalResults.accuracy * 100).toFixed(2) + '%');
        console.log('Tests exitosos:', evalResults.successes);
        console.log('Tests fallidos:', evalResults.failures);

        // Limpiar memoria
        chatbot.cleanupMemory();
        chatbot2.cleanupMemory();

    } catch (error) {
        console.error('\nError en las pruebas:', error);
        if (error.stack) {
            console.error('Stack:', error.stack);
        }
    }
}

test();