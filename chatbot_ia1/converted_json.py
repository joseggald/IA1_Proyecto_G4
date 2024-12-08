import tensorflow as tf
import json
import numpy as np
import os

def convert_h5_to_json(h5_path, output_dir):
    """
    Converts H5 model to TensorFlow.js format manually
    """
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
    
    # Model parameters
    vocab_size = 2000
    sequence_length = 20
    embedding_dim = 64
    
    # Create model with explicit input shape
    print("\nCreating model with explicit input shape...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(sequence_length,), dtype='int32', name='input_1'),
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name='embedding'
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True),
            name='bidirectional'
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(16),
            name='bidirectional_1'
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
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
        'num_classes': num_classes
    }
    
    config_path = os.path.join('public', 'model_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Configuration saved to: {config_path}")
    
    # Manual conversion to TensorFlow.js format
    print("\nConverting to TensorFlow.js format...")
    
    # Get model topology
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
    
    # Save weights binary file
    weights_path = os.path.join(output_dir, 'group1-shard1of1.bin')
    with open(weights_path, 'wb') as f:
        for weight_array in weight_data:
            f.write(weight_array.tobytes())
    print(f"✓ Weights binary saved to: {weights_path}")
    
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
            
            # Verify files
            required_files = [
                "public/model_config.json",
                "public/tokenizer.json",
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