import tensorflow as tf
import json
import numpy as np
import os
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter

def analyze_training_data(training_data):
    """Analiza los datos de entrenamiento para extraer patrones"""
    label_patterns = {}
    for item in training_data:
        label = str(item['label'])
        text = item['input'].lower().strip()
        if label not in label_patterns:
            label_patterns[label] = {'texts': [], 'words': Counter()}
        
        label_patterns[label]['texts'].append(text)
        label_patterns[label]['words'].update(text.split())
    
    return label_patterns

def convert_h5_to_json(h5_path, output_dir):
    """
    Converts H5 model to TensorFlow.js format with comprehensive topic support
    """
    # Ensure NLTK punkt is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('public', exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    with open('data.json', 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    # Get number of classes
    labels = [item['label'] for item in training_data]
    num_classes = len(set(labels))
    print(f"Number of classes detected: {num_classes}")
    
    # Analyze patterns in training data
    label_patterns = analyze_training_data(training_data)
    
    # Process topics and keywords
    print("\nProcessing topics and keywords...")
    topic_data = {
        'topics': {},
        'keywords': {},
        'relations': {}
    }
    
    def extract_keywords(text):
        """Extrae palabras clave del texto"""
        words = word_tokenize(text.lower())
        keywords = [w for w in words if len(w) > 2 and not w.isdigit()]
        return keywords

    # Define topic mappings and keywords
    topic_mappings = {
        '0': {
            'name': 'saludos',
            'keywords': ['hola', 'buenos', 'dias', 'tardes', 'noches', 'saludos', 'que tal', 'hey', 'hi'],
            'description': 'Saludos y bienvenidas'
        },
        '1': {
            'name': 'confirmacion',
            'keywords': ['si', 'claro', 'por supuesto', 'desde luego', 'exacto', 'correcto', 'efectivamente'],
            'description': 'Confirmaciones y afirmaciones'
        },
        '2': {
            'name': 'agradecimiento',
            'keywords': ['gracias', 'agradezco', 'te agradezco', 'muchas gracias', 'thank', 'thanks'],
            'description': 'Expresiones de gratitud'
        },
        '3': {
            'name': 'despedida',
            'keywords': ['adios', 'hasta luego', 'nos vemos', 'chao', 'hasta pronto', 'bye', 'goodbye'],
            'description': 'Despedidas y cierres'
        },
        '4': {
            'name': 'cortesia',
            'keywords': ['por favor', 'disculpa', 'perdón', 'amable', 'gentil', 'please'],
            'description': 'Expresiones de cortesía'
        }
    }

    # Create base topics
    for label, mapping in topic_mappings.items():
        topic_name = mapping['name']
        topic_data['topics'][topic_name] = {
            'keywords': mapping['keywords'].copy(),
            'labels': [label],
            'description': mapping['description']
        }
        
        # Add keywords from training data
        if label in label_patterns:
            common_words = [word for word, count in label_patterns[label]['words'].most_common(10)]
            topic_data['topics'][topic_name]['keywords'].extend(common_words)
            topic_data['topics'][topic_name]['keywords'] = list(set(topic_data['topics'][topic_name]['keywords']))

    # Build keyword index
    for topic, data in topic_data['topics'].items():
        for keyword in data['keywords']:
            if keyword not in topic_data['keywords']:
                topic_data['keywords'][keyword] = []
            if topic not in topic_data['keywords'][keyword]:
                topic_data['keywords'][keyword].append(topic)

    # Build topic relations
    for topic1 in topic_data['topics']:
        topic_data['relations'][topic1] = []
        keywords1 = set(topic_data['topics'][topic1]['keywords'])
        for topic2 in topic_data['topics']:
            if topic1 != topic2:
                keywords2 = set(topic_data['topics'][topic2]['keywords'])
                # If they share keywords or are commonly used together
                if len(keywords1.intersection(keywords2)) > 0:
                    topic_data['relations'][topic1].append(topic2)

    # Save topic data
    topic_data_path = os.path.join('public', 'topic_data.json')
    with open(topic_data_path, 'w', encoding='utf-8') as f:
        json.dump(topic_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Topic data saved to: {topic_data_path}")
    
    # Model parameters
    vocab_size = 2000
    sequence_length = 20
    embedding_dim = 128
    
    # Create model with explicit input shape
    print("\nCreating model with explicit input shape...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(sequence_length,), dtype='int32', name='input_1'),
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name='embedding'
        ),
        tf.keras.layers.Conv1D(64, 3, activation='relu', name='conv1d'),
        tf.keras.layers.GlobalMaxPooling1D(name='global_max_pooling1d'),
        tf.keras.layers.Dense(32, activation='relu', name='dense'),
        tf.keras.layers.Dropout(0.5, name='dropout'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Load original model and copy weights
    print("\nCopying weights from original model...")
    original_model = tf.keras.models.load_model(h5_path)
    
    for i, (orig_layer, new_layer) in enumerate(zip(original_model.layers, model.layers)):
        try:
            if len(orig_layer.get_weights()) > 0:
                new_layer.set_weights(orig_layer.get_weights())
                print(f"✓ Successfully copied weights for layer: {orig_layer.name}")
        except Exception as e:
            print(f"✗ Failed to copy weights for layer {orig_layer.name}: {str(e)}")
    
    # Create and save tokenizer
    print("\nCreating tokenizer...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>"
    )
    
    texts = [item['input'].lower().strip() for item in training_data]
    tokenizer.fit_on_texts(texts)
    
    tokenizer_json = tokenizer.to_json()
    tokenizer_path = os.path.join('public', 'tokenizer.json')
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    print(f"✓ Tokenizer saved to: {tokenizer_path}")
    
    # Save configuration
    config = {
        'vocab_size': vocab_size,
        'max_length': sequence_length,
        'embedding_dim': embedding_dim,
        'input_shape': [sequence_length],
        'batch_size': None,
        'num_classes': num_classes,
        'memory_size': 10,
        'topic_aware': True
    }
    
    config_path = os.path.join('public', 'model_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved to: {config_path}")
    
    # Convert to TensorFlow.js format
    print("\nConverting to TensorFlow.js format...")
    model_json = {
        'format': 'layers-model',
        'generatedBy': 'keras-js-converter',
        'convertedBy': 'manual-converter',
        'modelTopology': json.loads(model.to_json()),
        'weightsManifest': [{
            'paths': ['group1-shard1of1.bin'],
            'weights': []
        }]
    }
    
    # Process weights
    weight_data = []
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for i, weight in enumerate(weights):
                weight_data.append(weight.astype(np.float32))
                shape = list(weight.shape)
                name = f'{layer.name}/weights_{i}' if i == 0 else f'{layer.name}/bias_{i-1}'
                model_json['weightsManifest'][0]['weights'].append({
                    'name': name,
                    'shape': shape,
                    'dtype': 'float32'
                })
    
    # Save model.json
    model_path = os.path.join(output_dir, 'model.json')
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(model_json, f)
    print(f"✓ Model JSON saved to: {model_path}")
    
    # Save weights binary
    weights_path = os.path.join(output_dir, 'group1-shard1of1.bin')
    with open(weights_path, 'wb') as f:
        for weight_array in weight_data:
            f.write(weight_array.tobytes())
    print(f"✓ Weights binary saved to: {weights_path}")
    
    # Save responses
    try:
        with open('response.json', 'r', encoding='utf-8') as f:
            responses = json.load(f)
        responses_path = os.path.join('public', 'responses.json')
        with open(responses_path, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=2)
        print(f"✓ Responses saved to: {responses_path}")
    except FileNotFoundError:
        print("No response.json file found, skipping responses export")
    
    return True

if __name__ == "__main__":
    try:
        h5_path = "model/chatbot_model.h5"
        output_dir = "public/tfjs_model"
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Model file not found: {h5_path}")
        
        success = convert_h5_to_json(h5_path, output_dir)
        
        if success:
            print("\n✓ Conversion completed successfully")
            required_files = [
                "public/model_config.json",
                "public/tokenizer.json",
                "public/topic_data.json",
                "public/tfjs_model/model.json",
                "public/tfjs_model/group1-shard1of1.bin"
            ]
            
            print("\nVerifying output files...")
            for file_path in required_files:
                if os.path.exists(file_path):
                    print(f"✓ Verified: {file_path}")
                else:
                    print(f"Warning: File not found: {file_path}")
        else:
            print("\n✗ Conversion failed")
                
    except Exception as e:
        print(f"\n✗ Error during conversion: {str(e)}")
        import traceback
        print("\nFull stack trace:")
        print(traceback.format_exc())