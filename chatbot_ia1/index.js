const fs = require('fs')
const natural = require('natural')
const tf = require('@tensorflow/tfjs')

const datos = JSON.parse(fs.readFileSync('datos.json','utf-8'))

const convertidor = new natural.WordTokenizer()

const listaPalabras = (entrada) =>{
    const tokens = entrada.flatMap(texto => convertidor.tokenize(texto.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")))
    return [...new Set(tokens)]
}

const entradas = datos.map(d => convertidor.tokenize(d.input.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")))
const salidas = datos.map(d => convertidor.tokenize(d.output.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")))


const vocabulario = listaPalabras(['Desconocido',...entradas.flat(),...salidas.flat()])
const palabra_indice = Object.fromEntries(vocabulario.map((palabra, i) => [palabra, i]));
const indice_palabra = Object.fromEntries(Object.entries(palabra_indice).map(([ind,palabra]) => [palabra,ind]))


const vectorizar = (secuencia) => secuencia.map(palabra => palabra_indice[palabra] || palabra_indice['Desconocido'])
const inputs = entradas.map(input => vectorizar(input,vocabulario))
const outputs = salidas.map(output => vectorizar(output,vocabulario))

const longitud_maxima = Math.max(...[...inputs,...outputs].map(secuencia => secuencia.length))



const rellenar_secuencia = (secuencias, longitud_maxima) =>
    secuencias.map(secuencia =>
        Array(longitud_maxima).fill(0).map((_, i) => secuencia[i] || 0)
    );

const inputs_rellenos = rellenar_secuencia(inputs,longitud_maxima)
const outputs_rellenos = rellenar_secuencia(outputs,longitud_maxima)


const tensorEntradas = tf.tensor(inputs_rellenos)
const tensorSalidas = tf.tensor(outputs_rellenos)

const modelo = tf.sequential();
modelo.add(tf.layers.embedding({ inputDim: vocabulario.length, outputDim: 128, inputLength: longitud_maxima }));
modelo.add(tf.layers.lstm({ units: 128, returnSequences: true}));
modelo.add(tf.layers.dropout({ rate: 0.2 }));
modelo.add(tf.layers.lstm({ units: 128 }));
modelo.add(tf.layers.dense({ units: vocabulario.length, activation: 'softmax' }));


modelo.compile({optimizer:'adam',loss:'sparseCategoricalCrossentropy',metrics:['accuracy']});



const generarRespuesta = (entrada) => {
    const tokens_entrada = convertidor.tokenize(entrada.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, ""));
    const vector_entrada = rellenar_secuencia([vectorizar(tokens_entrada)], longitud_maxima);
    const prediccion = modelo.predict(tf.tensor(vector_entrada));
    const indices_prediccion = Array.from(prediccion.argMax(1).dataSync());
    return indices_prediccion.map(indice => indice_palabra[indice] || '').join(' ');
};

(async() => {
    console.log("entrenando modelo ...")
    modelo.fit(tensorEntradas,tensorSalidas,{epochs: 100})
    console.log("modelo entrenado")
    const testInput = "hola";
    const response = generarRespuesta(testInput);
    console.log(`Entrada: ${testInput}`);
    console.log(`Respuesta: ${response}`);
})()

